export HF_HOME="/data/hf-home/"
export WANDB_PROJECT=dfo

model="HuggingFaceH4/mistral-7b-sft-beta"
run_name="mistral-7b-sft-beta-self-feedback-0402"
batch_size=2
datasets="heegyu/ultrafeedback_binarized_feedback:self-feedback"
lr=1e-5

python finetune.py \
    --sharding mp \
    --load_eval False \
    --run_name "$run_name" \
    --model_id "$model" \
    --datasets "$datasets" \
    --push_to_hub_id "heegyu/$run_name" \
    --epoch 3 \
    --packing False \
    --max_length 2048 \
    --step_batch_size $batch_size \
    --total_batch_size 32 \
    --chat_template "zephyr" \
    --save_epochs 1 \
    --lr $lr

