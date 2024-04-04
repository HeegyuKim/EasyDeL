
from .dataset import ChatDatasetLoader, DatasetArguments
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("beomi/gemma-ko-7b")
args = DatasetArguments(
    dataset="heegyu/ko-openchat-0404-test",
    load_eval=False,
    train_only_response=True,
    chat_template="gemma",
    packing=True,
    limit=128
)
loader = ChatDatasetLoader(args, tokenizer)

# print(loader.dataset)


item = next(iter(loader.train_dataset))
print(item)
for k, v in item.items():
    print(k, len(v))

print("=====")
print(tokenizer.decode(item["input_ids"]))