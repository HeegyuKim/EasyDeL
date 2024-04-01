# ../../script/koopenchat/lima.sh 2b 4

size="${1:-2b}"
datasets="changpt/ko-lima-vicuna"
batch_size="${2:-1}"

echo "Batch size: $batch_size"

export WANDB_PROJECT=ko-openchat-$size
lr=1e-4
model="beomi/gemma-ko-$size"
run_name="gemma-$size-$lr-$datasets"


python finetune.py \
    --sharding mp \
    --load_eval False \
    --run_name "$run_name" \
    --model_id "$model" \
    --datasets "$datasets" \
    --epoch 50 \
    --packing \
    --max_length 1024 \
    --step_batch_size $batch_size \
    --warmup_steps 128 \
    --total_batch_size 32 \
    --chat_template gemma \
    --save_epochs 10 \
    --lr $lr
