


python -m examples.jax_serve_example \
    --pretrained_model_name_or_path peft:heegyu/HuggingFaceH4__mistral-7b-sft-beta-self-feedback@epoch-3  \
    --prompter_type zephyr

# python -m examples.jax_serve_example \
#     --pretrained_model_name_or_path peft:heegyu/HuggingFaceH4__mistral-7b-sft-beta@epoch-3  \
#     --prompter_type zephyr
    