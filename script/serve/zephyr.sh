export USE_TORCH=True


python -m examples.jax_serve_example_hf \
    --pretrained_model_name_or_path HuggingFaceH4/mistral-7b-sft-beta  \
    --prompter_type zephyr
    

