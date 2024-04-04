# ../../script/koopenchat/alpaca.sh
wandb offline

batch_size="${1:-32}"

echo "Batch size: $batch_size"

export WANDB_PROJECT=ko-openchat-$size

model="Locutusque/TinyMistral-248M-v2.5"
datasets="tatsu-lab/alpaca"
run_name="TinyMistral-248M-v2.5-alpaca"
chat_template="default:bos"
lr=1e-4


python finetune_hf.py \
    --sharding mp \
    --load_eval False \
    --run_name "$run_name" \
    --model_id "$model" \
    --datasets "$datasets" \
    --epoch 3 \
    --packing False \
    --max_length 512 \
    --step_batch_size $batch_size \
    --total_batch_size 32 \
    --chat_template $chat_template \
    --save_epochs 1 \
    --lr $lr
