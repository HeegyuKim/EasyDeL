size="${1:-2b}"
datasets="nampdn-ai/tiny-codes"

model="google/gemma-$size"
run_name="gemma-$size-code"
batch_size=1

python finetune.py \
    --run_name "$run_name" \
    --model_id "$model" \
    --datasets "$datasets" \
    --max_steps 1000000 \
    --save_steps 100000 \
    --max_length 2048 \
    --step_batch_size $batch_size \
    --total_batch_size 128 \
    --chat_template gemma \
    --streaming \
    --lr 2e-5
