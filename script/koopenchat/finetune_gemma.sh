# ../../script/koopenchat/alpaca.sh
wandb online

batch_size="${1:-4}"

echo "Batch size: $batch_size"

export WANDB_PROJECT=ko-openchat-$size

model="beomi/gemma-ko-2b"
datasets="heegyu/ko-openchat-0406"
run_name="gemma-ko-2b-0406"
chat_template="gemma"
lr=1e-5


python finetune_hf.py \
    --sharding mp \
    --load_eval False \
    --run_name "$run_name" \
    --model_id "$model" \
    --datasets "$datasets" \
    --max_steps 500000 \
    --warmup_steps 10000 \
    --max_scheduler_steps 50000 \
    --save_steps 100000 \
    --packing \
    --streaming \
    --max_length 2048 \
    --step_batch_size $batch_size \
    --total_batch_size 128 \
    --chat_template $chat_template \
    --lr $lr
