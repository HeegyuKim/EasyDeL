# ../../script/koopenchat/finetune_gemma.sh 7b "maywell/koVast" ko-openchat-7b-0331
# ../../script/koopenchat/finetune_gemma.sh 2b "maywell/koVast" ko-openchat-2b-0331

size="${1:-2b}"
datasets=$2


# model="google/gemma-$size"
model="beomi/gemma-ko-$size"
run_name="gemma-$size-$3"
batch_size=1

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
    --total_batch_size 128 \
    --chat_template gemma \
    --lr 1e-4
