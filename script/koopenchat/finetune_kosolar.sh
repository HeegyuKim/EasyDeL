wandb online
export WANDB_PROJECT=ko-openchat-10.8b

# ../../script/koopenchat/finetune_kosolar.sh "changpt/ko-lima-vicuna" ko-lima-10.8b-0331
# changpt/ko-lima-vicuna

datasets=$1

model="yanolja/EEVE-Korean-10.8B-v1.0"
run_name="eeve-10.8B-$2"
batch_size=1

python finetune.py \
    --sharding mp \
    --load_eval False \
    --run_name "$run_name" \
    --model_id "$model" \
    --datasets "$datasets" \
    --epoch 10 \
    --packing \
    --max_length 2048 \
    --load_from_cache_file \
    --step_batch_size $batch_size \
    --total_batch_size 32 \
    --chat_template default \
    --save_epochs 2 \
    --lr 2e-5
