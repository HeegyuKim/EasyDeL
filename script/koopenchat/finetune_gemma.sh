
# ../../script/koopenchat/finetune_gemma.sh 7b "maywell/koVast" 2

# ../../script/koopenchat/finetune_gemma.sh 2b "maywell/koVast" 4
# ../../script/koopenchat/finetune_gemma.sh 2b "heegyu/KoCommercial-Dataset" 4
# ../../script/koopenchat/finetune_gemma.sh 2b "squarelike/OpenOrca-gugugo-ko" 4

size="${1:-2b}"
datasets=$2
batch_size=${3:-1}

echo "Batch size: $batch_size"

export WANDB_PROJECT=ko-openchat-$size
model="beomi/gemma-ko-$size"
run_name="gemma-$size-$datasets"
lr=2e-5


python finetune.py \
    --sharding mp \
    --load_eval False \
    --run_name "$run_name" \
    --model_id "$model" \
    --datasets "$datasets" \
    --epoch 3 \
    --packing \
    --max_length 2048 \
    --load_from_cache_file \
    --step_batch_size $batch_size \
    --total_batch_size 32 \
    --chat_template gemma \
    --save_epochs 1 \
    --lr $lr