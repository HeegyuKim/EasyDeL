# ../../script/koopenchat/lima.sh 2b 4
wandb offline

size="${1:-2b}"
datasets="changpt/ko-lima-vicuna"
batch_size="${2:-1}"

echo "Batch size: $batch_size"

export WANDB_PROJECT=ko-openchat-$size
lr=2e-5
model="beomi/gemma-ko-$size"
# model="google/gemma-$size-it"
# model="HuggingFaceM4/tiny-random-LlamaForCausalLM"
# model="Locutusque/TinyMistral-248M-v2.5"
run_name="gemma-$size-it-$lr-$datasets"


python finetune_hf.py \
    --sharding mp \
    --load_eval False \
    --run_name "$run_name" \
    --model_id "$model" \
    --datasets "$datasets" \
    --epoch 10 \
    --packing \
    --max_length 1024 \
    --step_batch_size $batch_size \
    --total_batch_size 32 \
    --chat_template gemma \
    --save_epochs 5 \
    --lr $lr
