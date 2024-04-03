# ../../script/koopenchat/ft-42dot.sh 4
export HF_HOME="/data/hf-home/"

size="1.3b"
batch_size="${1:-1}"

echo "Batch size: $batch_size"

export WANDB_PROJECT=ko-openchat-$size
lr=2e-5
model="42dot/42dot_LLM-PLM-1.3B"
datasets="changpt/ko-lima-vicuna"
run_name="42dot-PLM-$size-$lr-lima"


python finetune.py \
    --sharding mp \
    --load_eval False \
    --run_name "$run_name" \
    --model_id "$model" \
    --datasets "$datasets" \
    --push_to_hub_id "heegyu/$run_name" \
    --epoch 10 \
    --packing \
    --max_length 1024 \
    --step_batch_size $batch_size \
    --total_batch_size 32 \
    --chat_template "42dot" \
    --save_epochs 5 \
    --lr $lr
