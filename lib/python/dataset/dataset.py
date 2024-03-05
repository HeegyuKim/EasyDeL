from typing import Optional
from dataclasses import dataclass
import os 

from datasets import Dataset, DatasetDict, concatenate_datasets, IterableDataset
from .chat.train_templates import find_template

from .registry import Registry


NUM_PROC = max(1, min(16, os.cpu_count() // 2))

@dataclass
class DatasetArguments():
    dataset: Optional[str] = None
    eval_dataset: Optional[str] = None
    dataset_streaming: bool = False
    max_length: int = 2048

    limit: Optional[int] = None
    eval_limit: Optional[int] = None
    train_only_response = False
    chat_template: Optional[str] = None
    train_prompt_prefix: Optional[str] = None
    streaming: bool = False
    keep_in_memory: bool = False
    packing: bool = False

datasources = Registry("datasource")


class DataSource:

    def load(self, args: DatasetArguments, split: str) -> Dataset:
        pass
    
    def num_items(self, split: str) -> int:
        pass

    def map_dataset(self, args: DatasetArguments, ds: Dataset, func, batched=False) -> Dataset:
        if args.limit:
            ds = ds.select(range(self.args.limit))
        return ds.map(func, num_proc=NUM_PROC, load_from_cache_file=False, batched=batched)
    

class DatasetLoader:
    
    def __init__(self, args: DatasetArguments, tokenizer) -> None:
        self.args = args
        self.tokenizer = tokenizer
        self.train_template = find_template(args.chat_template)(self.tokenizer)
        self.dataset = self.prepare_dataset(args)
    
    def encode_item(self, item):
        conversation = item["conversations"]
        concat_inputs, concat_labels, concat_vals = [], [], []
        
        if self.args.train_prompt_prefix:
            ids = self.tokenizer.encode(self.args.train_prompt_prefix, add_special_tokens=False)
            concat_inputs.extend(ids)
            concat_labels.extend([-100] * len(concat_inputs) if self.args.train_only_response else ids)

        for i, uttr in enumerate(conversation):
            content, trainable = self.train_template.handle_utterance(uttr, i)

            input_ids = self.tokenizer.encode(content, add_special_tokens=False)

            if not self.args.train_only_response or trainable:
                vals = [1] * len(input_ids)
            else:
                vals = [0] * len(input_ids)

            concat_inputs.extend(input_ids)
            concat_vals.extend(vals)

        return {
            "input_ids": concat_inputs,
            "attention_mask": [1] * len(concat_inputs),
            "valid_loss": concat_vals,
        }
    
    def encode_item_batch(self, batch):
        keys = list(batch.keys())

        required_fields = ["input_ids", "attention_mask", "valid_loss"]

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
        test_set = self.get_sources(args, args.eval_dataset or args.dataset, "test")
        if test_set is not None:
            dd["test"] = test_set

        if args.limit:
            for k in dd.keys():
                if dd[k] is not None and len(dd[k]) > args.limit:
                    dd[k] = dd[k].select(range(args.limit))
        
        if not args.streaming:
            num_proc = max(1, min(len(dd['train']) // 2000, NUM_PROC))
            kwargs = dict()
            kwargs["num_proc"] = num_proc
            kwargs["load_from_cache_file"] = False
            kwargs["keep_in_memory"] = args.keep_in_memory

            for k in dd.keys():
                dd[k] = dd[k].map(
                    self.encode_item,
                    **kwargs
                    )

                if self.args.packing:
                    required_fields = ["input_ids", "attention_mask", "valid_loss"]
                    cols = set(dd["train"].column_names) - set(required_fields)
                    dd[k] = dd[k].map(
                        self._pack, 
                        batched=True, 
                        remove_columns=cols, 
                        **kwargs
                        )
        else:
            for k in dd.keys():
                if args.streaming:
                    dd[k] = IterableDataset.from_generator(
                        self._pack_iter,
                        gen_kwargs={"dataset": dd[k]}
                    )
                else:
                    dd[k] = dd[k].with_transform(self.encode_item_batch)
            
        self.dataset = dd
        print(dd)
        self.train_dataset = dd['train']
        self.test_dataset = dd.get('test')

    def _pack(self, items):
        outputs = dict(
            input_ids=[],
            attention_mask=[],
            valid_loss=[]
        )
        accum_len = 0

        batch_len = self.args.max_length
        all_input_ids = items["input_ids"]
        all_attention_mask = items.get("attention_mask")
        all_vals = items["valid_loss"]

        batch_ids, batch_mask, batch_vals = [], [], []

        for ids, mask, labels in zip(all_input_ids, all_attention_mask, all_vals):
            accum_len += len(ids)

            batch_ids.extend(ids)
            if all_attention_mask is not None:
                batch_mask.extend(mask)
            batch_vals.extend(labels)

            while accum_len > batch_len:
                outputs["input_ids"].append(batch_ids[:batch_len])
                if all_attention_mask is not None:
                    outputs["attention_mask"].append(batch_mask[:batch_len])
                outputs["valid_loss"].append(batch_vals[1:batch_len + 1])

                batch_ids, batch_vals = batch_ids[batch_len:], batch_vals[batch_len:]
                if all_attention_mask is not None:
                    batch_mask = batch_mask[batch_len:]
                accum_len -= batch_len
        
        if all_attention_mask is None:
            outputs.pop("attention_mask")
        
        return outputs

    def _pack_iter(self, dataset):
        accum_len = 0

        batch_len = self.args.max_length
        batch_ids, batch_mask, batch_vals = [], [], []

        for item in dataset:
            item = self.encode_item(item)
            ids = item['input_ids']
            mask = item.get('attention_mask')
            vals = item['valid_loss']
            accum_len += len(ids)

            batch_ids.extend(ids)
            if mask is not None:
                batch_mask.extend(mask)
            batch_vals.extend(vals)

            while accum_len > batch_len:
                batch = dict(
                    input_ids=batch_ids[:batch_len],
                    valid_loss=batch_vals[1:batch_len + 1]
                )
                batch_ids, batch_vals = batch_ids[batch_len:], batch_vals[batch_len:]
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
                if ds is not None:
                    sources.append(ds)
            except:
                print(f"Failed to load dataset {source_cls.__class__.__name__}")
                raise
        
        if sources:
            return concatenate_datasets(sources) if len(sources) > 1 else sources[0]
        else:
            return None
    
    # @property
    # def train_dataset(self):
    #     return self.dataset['train']

    # @property
    # def test_dataset(self):
    #     return self.dataset.get('test')
    
