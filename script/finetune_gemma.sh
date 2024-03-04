size="${1:-2b}"
datasets=$2

model="google/gemma-$size"
run_name="gemma-$size-lima"
batch_size=1

python finetune.py \
    --run_name "$run_name" \
    --model_id "$model" \
    --datasets "$datasets" \
    --epoch 3 \
    --max_length 2048 \
    --step_batch_size $batch_size \
    --total_batch_size 128 \
    --chat_template gemma \
    --lr 5e-5
