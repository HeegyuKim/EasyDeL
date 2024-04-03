
export USE_TORCH=True
export HF_HOME=/data/hf-models/

python -m examples.jax_serve_example \
    --pretrained_model_name_or_path heegyu/42dot-PLM-1.3b-2e-5-lima@steps-789  \
    --tokenizer 42dot/42dot_LLM-PLM-1.3B \
    --prompter_type "42dot"


    