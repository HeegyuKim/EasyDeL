model="mistralai/Mistral-7B-v0.1"
run_name="mistral-7b-v0.1-lima"
batch_size=1
datasets="GAIR/lima"

python finetune.py \
    --run_name "$run_name" \
    --model_id "$model" \
    --datasets "$datasets" \
    --epoch 10 \
    --save_epochs 5 \
    --max_length 1024 \
    --step_batch_size $batch_size \
    --total_batch_size 32 \
    --chat_template default \
    --lr 1e-5
