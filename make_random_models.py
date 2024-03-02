from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

tokenizer = "beomi/OPEN-SOLAR-KO-10.7B"
tokenizer = AutoTokenizer.from_pretrained(tokenizer)

def build_config(seq_len, hidden_size, num_layers, head_dim, intermediate_size=None):
    num_heads = hidden_size // head_dim
    config = LlamaConfig(
        vocab_size=46160, #tokenizer.vocab_size,
        n_positions=seq_len,
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4 if intermediate_size is None else intermediate_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads // 4,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    return AutoModelForCausalLM.from_config(config)

def count_parameters(model):
    return sum(p.numel() for p in model.model.parameters())

model_name = "heegyu/ko-llama-46M-random"
model = build_config(2048, 512, 6, 64)
print(count_parameters(model) / 1e+9)
tokenizer.push_to_hub(model_name)
model.push_to_hub(model_name)

# model_name = "heegyu/ko-llama-230M-random"
# model = build_config(2048, 1024, 12, 64)
# print(count_parameters(model) / 1e+9)
# tokenizer.push_to_hub(model_name)
# model.push_to_hub(model_name)

# model_name = "heegyu/ko-llama-412M-random"
# model = build_config(2048, 1024, 24, 64)
# print(count_parameters(model) / 1e+9)
# tokenizer.push_to_hub(model_name)
# model.push_to_hub(model_name)

# model_name = "heegyu/ko-llama-1B-random"
# model = build_config(2048, 2048, 16, 64)#, intermediate_size=5632)
# print(count_parameters(model) / 1e+9)
# tokenizer.push_to_hub(model_name)
# model.push_to_hub(model_name)