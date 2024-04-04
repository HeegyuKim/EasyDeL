from typing import Optional
from dataclasses import dataclass
import os 

from datasets import Dataset, DatasetDict, concatenate_datasets, IterableDataset, interleave_datasets
from .chat.train_templates import find_template

from .registry import Registry


NUM_PROC = max(1, min(16, os.cpu_count() // 2))

@dataclass
class DatasetArguments():
    dataset: Optional[str] = None
    eval_dataset: Optional[str] = None
    load_eval: bool = True
    dataset_streaming: bool = False
    max_length: int = 2048

    limit: Optional[int] = None
    eval_limit: Optional[int] = None
    train_only_response: bool = False
    chat_template: Optional[str] = None
    train_prompt_prefix: Optional[str] = None
    streaming: bool = False
    keep_in_memory: bool = False
    packing: bool = False
    load_from_cache_file: Optional[bool] = False
    interleaving_strategy: str = "first_exhausted" # or all_exhausted

datasources = Registry("datasource")


class DataSource:

    def load(self, args: DatasetArguments, split: str) -> Dataset:
        pass
    
    def num_items(self, split: str) -> int:
        pass

    def map_dataset(self, args: DatasetArguments, ds: Dataset, func, batched=False) -> Dataset:
        if args.limit:
            ds = ds.select(range(self.args.limit))
        return ds.map(func, num_proc=NUM_PROC, load_from_cache_file=args.load_from_cache_file, batched=batched)
    

class DatasetLoader:
    
    def __init__(self, args: DatasetArguments, tokenizer) -> None:
        self.args = args
        self.tokenizer = tokenizer
        self.train_template = find_template(args.chat_template)(self.tokenizer)
        self.prepare_dataset(args)
    
    def encode_item(self, item):
        input_ids = self.tokenizer.encode(
            item["text"], 
            add_special_tokens=True,
            truncation=True,
            max_length=self.args.max_length,
            padding="max_length" if not self.args.packing else False
            )
        labels = [x if x != self.tokenizer.pad_token_id else -100 for x in input_ids]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }
    
    def encode_item_batch(self, batch):
        keys = list(batch.keys())

        required_fields = ["input_ids", "attention_mask", "labels"]

        outputs = {k:[] for k in required_fields}
        batch_size = len(batch[keys[0]])

        for i in range(batch_size):
            item = {k: v[i] for k, v in batch.items()}
            output = self.encode_item(item)
            for k, v in output.items():
                output[k].append(v)

        return outputs
        
    def prepare_dataset(self, args: DatasetArguments):
        dd = {
            "train": self.get_sources(args, args.dataset, "train")
        }
        if args.load_eval:
            test_set = self.get_sources(args, args.eval_dataset or args.dataset, "test")
            if test_set is not None:
                dd["test"] = test_set

        self.dataset = dd
        print(dd)
        self.train_dataset = dd['train']
        self.test_dataset = dd.get('test')

    def _pack(self, items):
        outputs = dict(
            input_ids=[],
            attention_mask=[],
            labels=[]
        )
        accum_len = 0

        batch_len = self.args.max_length
        all_input_ids = items["input_ids"]
        all_attention_mask = items.get("attention_mask")
        all_labels = items["labels"]

        batch_ids, batch_mask, batch_labels = [], [], []

        for ids, mask, labels in zip(all_input_ids, all_attention_mask, all_labels):
            accum_len += len(ids)

            batch_ids.extend(ids)
            if all_attention_mask is not None:
                batch_mask.extend(mask)
            batch_labels.extend(labels)

            while accum_len >= batch_len:
                outputs["input_ids"].append(batch_ids[:batch_len])
                if all_attention_mask is not None:
                    outputs["attention_mask"].append(batch_mask[:batch_len])
                # outputs["labels"].append(batch_labels[1:batch_len + 1])
                outputs["labels"].append(batch_labels[:batch_len])

                batch_ids, batch_labels = batch_ids[batch_len:], batch_labels[batch_len:]
                if all_attention_mask is not None:
                    batch_mask = batch_mask[batch_len:]
                accum_len -= batch_len
        
        if all_attention_mask is None:
            outputs.pop("attention_mask")
        
        return outputs

    def _pack_iter(self, dataset):
        accum_len = 0

        batch_len = self.args.max_length
        batch_ids, batch_mask, batch_labels = [], [], []

        for item in dataset:
            item = self.encode_item(item)
            ids = item['input_ids']
            mask = item.get('attention_mask')
            labels = item['labels']
            accum_len += len(ids)

            batch_ids.extend(ids)
            if mask is not None:
                batch_mask.extend(mask)
            batch_labels.extend(labels)

            while accum_len > batch_len:
                batch = dict(
                    input_ids=batch_ids[:batch_len],
                    labels=batch_labels[:batch_len]
                )
                batch_ids, batch_labels = batch_ids[batch_len:], batch_labels[batch_len:]
                if mask:
                    batch["attention_mask"] = batch_mask[:batch_len]
                    batch_mask = batch_mask[batch_len:]

                accum_len -= batch_len

                yield batch
    
    def get_sources(self, args, names, split):
        names = names.split(",")
        dataset_classes = [c for x in names for c in datasources.search(x)]
        sources = []

        for source_cls in dataset_classes:
            try:
                source = source_cls()
                ds = source.load(args, split)
            except:
                print(f"Failed to load dataset {source_cls.__class__.__name__}")
                raise
        
            if args.limit:
                if ds is not None and len(ds) > args.limit:
                    ds = ds.select(range(args.limit))
            
            if not args.streaming:
                num_proc = max(1, min(len(ds) // 2000, NUM_PROC))
                kwargs = dict()
                kwargs["num_proc"] = num_proc
                kwargs["load_from_cache_file"] = args.load_from_cache_file
                kwargs["keep_in_memory"] = args.keep_in_memory

                required_fields = ["input_ids", "attention_mask", "labels"]
                cols = set(ds.column_names) - set(required_fields)
                ds = ds.map(
                    self.encode_item,
                    remove_columns=cols, 
                    **kwargs
                    )

                if self.args.packing:
                    ds = ds.map(
                        self._pack, 
                        batched=True,
                        **kwargs
                        )
            else:
                if args.packing:
                    ds = IterableDataset.from_generator(
                        self._pack_iter,
                        gen_kwargs={"dataset": ds}
                    )
                else:
                    ds = ds.with_transform(self.encode_item_batch)
                
            if ds is not None:
                sources.append(ds)

        if sources:
            return interleave_datasets(sources, stopping_strategy=args.interleaving_strategy) if len(sources) > 1 else sources[0]
        else:
            return None
    
    # @property
    # def train_dataset(self):
    #     return self.dataset['train']

    # @property
    # def test_dataset(self):
    #     return self.dataset.get('test')
    

class ChatDatasetLoader(DatasetLoader):
    
    def __init__(self, args: DatasetArguments, tokenizer) -> None:
        self.train_template = find_template(args.chat_template)(tokenizer)
        super().__init__(args, tokenizer)
    
    def encode_item(self, item):
        conversation = item["conversations"]
        concat_inputs, concat_labels = [], []
        
        if self.args.train_prompt_prefix:
            ids = self.tokenizer.encode(self.args.train_prompt_prefix, add_special_tokens=False)
            concat_inputs.extend(ids)
            concat_labels.extend([-100] * len(concat_inputs) if self.args.train_only_response else ids)

        for i, uttr in enumerate(conversation):
            content, trainable = self.train_template.handle_utterance(uttr, i)
            if i + 1 != len(conversation):
                content += self.train_template.TURN_SEPERATOR

            input_ids = self.tokenizer.encode(content, add_special_tokens=False)

            if not self.args.train_only_response or trainable:
                labels = input_ids
            else:
                labels = [-100] * len(input_ids)

            concat_inputs.extend(input_ids)
            concat_labels.extend(labels)

        if not self.args.packing:
            if len(concat_inputs) < self.args.max_length:
                concat_inputs += [self.tokenizer.pad_token_id] * (self.args.max_length - len(concat_inputs))
                concat_labels += [-100] * (self.args.max_length - len(concat_labels))

        return {
            "input_ids": concat_inputs,
            "attention_mask": [1] * len(concat_inputs),
            "labels": concat_labels,
        }