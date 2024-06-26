# ../../script/koopenchat/lima.sh 2b 4
wandb offline

size="${1:-2b}"
datasets="changpt/ko-lima-vicuna"
batch_size="${2:-4}"

echo "Batch size: $batch_size"

export WANDB_PROJECT=ko-openchat-$size
lr=5e-5
# model="beomi/gemma-ko-$size"
# run_name="gemma-$size-it-$lr-$datasets"
# model="google/gemma-$size-it"
# model="HuggingFaceM4/tiny-random-LlamaForCausalLM"
chat_template="gemma"

model="Locutusque/TinyMistral-248M-v2.5"
dataset="tatsu-lab/alpaca"
run_name="TinyMistral-248M-v2.5-alpaca"
chat_template="default"
lr=1e-4


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
    --chat_template $chat_template \
    --save_epochs 5 \
    --lr $lr
