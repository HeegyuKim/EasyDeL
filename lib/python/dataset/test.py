
from .dataset import ChatDatasetLoader, DatasetArguments
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("beomi/gemma-ko-2b")
args = DatasetArguments(
    dataset="changpt/ko-lima-vicuna",
    load_eval=False,
    train_only_response=True,
    chat_template="gemma"
)
loader = ChatDatasetLoader(args, tokenizer)

print(loader.dataset)


item = loader.train_dataset[0]
print(item)
for k, v in item.items():
    print(k, len(v))

print("=====")
print(tokenizer.decode(item["input_ids"]))