
from .dataset import ChatDatasetLoader, DatasetArguments
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/mistral-7b-sft-beta")
args = DatasetArguments(
    dataset="heegyu/ultrafeedback_binarized_feedback:self-feedback",
    load_eval=False,
    train_only_response=True,
    chat_template="zephyr"
)
loader = ChatDatasetLoader(args, tokenizer)

print(loader.dataset)


item = loader.train_dataset[0]
print(item)
for k, v in item.items():
    print(k, len(v))

print("=====")
print(tokenizer.decode(item["input_ids"]))