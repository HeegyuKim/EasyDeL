
export USE_TORCH=True
export HF_HOME=/data/hf-models/

python -m examples.jax_serve_example_hf \
    --pretrained_model_name_or_path heegyu/mistral-7b-sft-beta-self-feedback-0402@steps-61008  \
    --tokenizer HuggingFaceH4/mistral-7b-sft-beta \
    --prompter_type zephyr

# python -m examples.jax_serve_example \
#     --pretrained_model_name_or_path peft:heegyu/HuggingFaceH4__mistral-7b-sft-beta@epoch-3  \
#     --prompter_type zephyr
    