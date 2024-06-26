# ../../script/finetune_gemma.sh 2b "Open-Orca/SlimOrca-Dedup" aya-slimorca-kocomm
export HF_DATASETS_CACHE='/data-plm/hf-datasets'

size="${1:-2b}"
datasets=$2

model="google/gemma-$size"
run_name="gemma-$size-$3"
batch_size=1

python finetune.py \
    --sharding mp \
    --run_name "$run_name" \
    --model_id "$model" \
    --datasets "$datasets" \
    --epoch 3 \
    --packing False \
    --max_length 1024 \
    --step_batch_size $batch_size \
    --total_batch_size 128 \
    --chat_template gemma \
    --lr 2e-5
