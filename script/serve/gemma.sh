
# python -m examples.jax_serve_example \
#     --pretrained_model_name_or_path heegyu/gemma-ko-2b-kovast-0402@epoch-2 \
#     --prompter_type chatml
#     --eos_token_id 107 \


python -m examples.jax_serve_example_hf \
    --pretrained_model_name_or_path heegyu/gemma-ko-2b-0404@steps-600000 \
    --eos_token_id 107 \
    --prompter_type gemma