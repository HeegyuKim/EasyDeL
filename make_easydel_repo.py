import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# import os
# os.environ['HF_HOME'] = '/data-plm/.cache/'

model_name = "heegyu/TinyLlama-augesc-context-strategy"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    revision="epoch-3",
    torch_dtype=torch.float16,
    cache_dir='/data-plm/.cache/'
    )


tokenizer.push_to_hub(model_name)
model.push_to_hub(model_name)
