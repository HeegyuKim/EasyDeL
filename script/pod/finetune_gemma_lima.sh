size="${1:-2b}"
model="google/gemma-$size"
run_name="gemma-$size-lima"
batch_size=1
datasets="GAIR/lima"

python finetune.py \
    --run_name "$run_name" \
    --model_id "$model" \
    --datasets "$datasets" \
    --epoch 10 \
    --sharding mp \
    --save_epochs 5 \
    --max_length 1024 \
    --distributed \
    --step_batch_size $batch_size \
    --total_batch_size 32 \
    --chat_template gemma \
    --lr 1e-5
