
python -m examples.jax_serve_example \
    --pretrained_model_name_or_path kaist-ai/prometheus-7b-v1.0 \
    --prompter_type prometheus \
    --max_sequence_length 4096 \
    --prompt_length 3584