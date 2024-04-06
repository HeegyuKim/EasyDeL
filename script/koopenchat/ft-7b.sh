# ../../script/koopenchat/alpaca.sh
wandb online

batch_size="${1:-2}"

echo "Batch size: $batch_size"

export WANDB_PROJECT=ko-openchat-$size

model="beomi/gemma-ko-7b"
datasets="heegyu/ko-openchat-0404-test"
run_name="gemma-ko-7b-0405-test"
chat_template="gemma"
lr=1e-5


python finetune_hf.py \
    --sharding mp \
    --load_eval False \
    --run_name "$run_name" \
    --model_id "$model" \
    --datasets "$datasets" \
    --epoch 3 \
    --warmup_ratio 0.033 \
    --packing False \
    --max_length 1024 \
    --step_batch_size $batch_size \
    --total_batch_size 32 \
    --chat_template $chat_template \
    --save_epochs 1 \
    --lr $lr
