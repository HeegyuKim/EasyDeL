export HF_HOME="/data/hf-home/"
export WANDB_PROJECT=dfo

batch_size="${1:-1}"
model="HuggingFaceH4/mistral-7b-sft-beta"
run_name="mistral-7b-sft-beta-self-feedback-0406"
datasets="heegyu/ultrafeedback_binarized_feedback:self-feedback"
lr=1e-5

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

