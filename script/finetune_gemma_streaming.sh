# ../../script/finetune_gemma_streaming.sh 2b "CohereForAI/aya_dataset,Open-Orca/SlimOrca-Dedup,MarkrAI/KoCommercial-Dataset" aya-slimorca-kocomm
# ../../script/finetune_gemma_streaming.sh 2b "Open-Orca/SlimOrca-Dedup,MarkrAI/KoCommercial-Dataset" slimorca-kocomm
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
    --save_steps 50000 \
    --max_steps 250000 \
    --packing \
    --streaming \
    --max_length 2048 \
    --step_batch_size $batch_size \
    --total_batch_size 128 \
    --chat_template gemma \
    --lr 2e-5
