
from .datasets import ChatDataSource, BaseAlpacaDataSource, datasources, DatasetArguments, VicunaChatDataSource
from datasets import load_dataset, Dataset


@datasources.register("tatsu-lab/alpaca")
class AlpacaChat(BaseAlpacaDataSource):
    dataset_path = "tatsu-lab/alpaca"

@datasources("nvidia/OpenMathInstruct-1")
class OpenMathInstruct(BaseAlpacaDataSource):
    instruction_key = "question"
    output_key = "generated_solution"
    dataset_path = "nvidia/OpenMathInstruct-1"

    def num_items(self, split: str) -> int:
        if split == "train":
            return 5700000
        elif split == "test":
            return 1130000
        
@datasources("CohereForAI/aya_dataset")
class Aya(BaseAlpacaDataSource):
    instruction_key = "inputs"
    output_key = "targets"
    dataset_path = "CohereForAI/aya_dataset"
    

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        ds = load_dataset(self.dataset_path, "default", split=split, streaming=args.streaming)
        return ds
    
@datasources.register("nampdn-ai/tiny-codes")
class TinyCodes(BaseAlpacaDataSource):
    instruction_key = "prompt"
    output_key = "response"
    dataset_path = "nampdn-ai/tiny-codes"

@datasources.register("Locutusque/hercules-v1.0")    
class Hercules(BaseAlpacaDataSource):
    dataset_path = "Locutusque/hercules-v1.0"

@datasources.register("HuggingFaceH4/ultrachat_200k")
class UltraChat(ChatDataSource):
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"{split}_sft", streaming=args.streaming)
        ds = ds.rename_column("messages", "conversations").select_columns(["conversations"])
        return ds


@datasources.register("Open-Orca/SlimOrca-Dedup")
class SlimOrcaDedup(VicunaChatDataSource):
    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("Open-Orca/SlimOrca-Dedup", split=split, streaming=args.streaming)
        
        return ds


@datasources.register("GAIR/lima")
class Lima(ChatDataSource):
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        if split != "train":
            return None
        ds = load_dataset("GAIR/lima", split=split, streaming=args.streaming)
        ds = ds.map(self._map_conv, load_from_cache_file=False, desc="Converting a GAIR/lima dataset")
        
        return ds

    def _map_conv(self, item):
        convs = []

        for i, conv in enumerate(item["conversations"]):
            convs.append(dict(
                role="user" if i % 2 == 0 else "assistant",
                content=conv
            ))

        return {
            "conversations": convs
        }
