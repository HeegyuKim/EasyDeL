from datasets import Dataset, IterableDataset, load_dataset, interleave_datasets
from ..dataset import DataSource, datasources, DatasetArguments, NUM_PROC


class PretrainingDataSource(DataSource):
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        pass
    

@datasources("koreans")
class Koreans(PretrainingDataSource):
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        datasets = [
            # (("uonlp/CulturaX", "ko"), 50),
            (("maywell/korean_textbooks", "tiny-textbooks"), 1)
        ]
        probs = [d[1] for d in datasets] 
        prob_sum = sum(d[1] for d in datasets)
        probs = [p / prob_sum for p in probs]
        
        datasets = [load_dataset(*d[0], streaming=args.streaming, split=split, token=True) for d in datasets]
        return interleave_datasets(
            datasets,
            probabilities=probs,
            stopping_strategy="all_exhausted"
        )
    


@datasources("cerebras/SlimPajama-627B")
class SlimPajama_627B(PretrainingDataSource):
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        return load_dataset("cerebras/SlimPajama-627B", streaming=args.streaming, split=split)
    

@datasources("allenai/dolma")
class Dolma(PretrainingDataSource):
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        return load_dataset("allenai/dolma", streaming=args.streaming, split=split, token=True)

@datasources("Locutusque/UltraTextbooks")
class UltraTextbooks(PretrainingDataSource):
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        return load_dataset("Locutusque/UltraTextbooks", streaming=args.streaming, split=split)

@datasources("HuggingFaceTB/cosmopedia")
class Cosmopedia(PretrainingDataSource):
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        subsets = [
            "auto_math_text",
            "khanacademy",
            "openstax",
            "stanford",
            "stories",
            "wikihow",
            "web_samples_v1",
            "web_samples_v2",
        ]
        datasets = [load_dataset("HuggingFaceTB/cosmopedia", subset, streaming=args.streaming, split=split) for subset in subsets]
        return interleave_datasets(
            datasets,
            stopping_strategy="all_exhausted"
        )