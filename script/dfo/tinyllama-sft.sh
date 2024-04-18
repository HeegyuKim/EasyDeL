# model="openlm-research/open_llama_3b_v2"
# run_name="open_llama_3b_v2-sft"
model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
run_name="tinyllama-1.1b-sft"
batch_size=1
datasets="HuggingFaceH4/ultrachat_200k"
# datasets="GAIR/lima"

export WANDB_PROJECT=feedback-tree-sft

batch_size="${1:-4}"
lr=2e-5

python finetune_hf.py \
    --sharding fsdp \
    --load_eval False \
    --run_name "$run_name" \
    --model_id "$model" \
    --datasets "$datasets" \
    --epoch 3 \
    --warmup_ratio 0.033 \
    --packing False \
    --max_length 2048 \
    --step_batch_size $batch_size \
    --total_batch_size 32 \
    --chat_template zephyr \
    --save_epochs 1 \
    --lr $lr

