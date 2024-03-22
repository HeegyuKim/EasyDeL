export HF_HOME=/data-plm/hf-caches

python -m examples.jax_serve_example \
    --pretrained_model_name_or_path UCLA-AGI/zephyr-7b-sft-full-SPIN-iter3  \
    --prompter_type zephyr