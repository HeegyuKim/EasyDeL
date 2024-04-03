
export USE_TORCH=True
export HF_HOME=/data/hf-models/

python -m examples.jax_serve_example \
    --pretrained_model_name_or_path heegyu/gemma-2b-it-kor-openorca-platypus-v3@epoch-3  \
    --prompter_type zephyr

# python -m examples.jax_serve_example \
#     --pretrained_model_name_or_path peft:heegyu/HuggingFaceH4__mistral-7b-sft-beta@epoch-3  \
#     --prompter_type zephyr
    