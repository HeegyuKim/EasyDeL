export USE_TORCH=True

size="${1:-46M}"
sharding="dp"

if [ "$size" = "46M" ]; then
    lr="6e-4"
    batch_size=32
elif [ "$size" = "230M" ]; then
    lr="4e-4"
    batch_size=8
elif [ "$size" = "412M" ]; then
    lr="3e-4"
    batch_size=8
    sharding="fsdp"
elif [ "$size" = "1B" ]; then
    lr="1e-4"
    batch_size=4
    sharding="fsdp"
else
    echo "Invalid size"
    exit 1
fi

datasets="HuggingFaceTB/cosmopedia,koreans"

model="heegyu/ko-llama-$size-random"
run_name="ko-llama-$size"

python pretrain.py \
    --sharding $sharding \
    --run_name "$run_name" \
    --model_id "$model" \
    --datasets "$datasets" \
    --max_length 2048 \
    --batch_size $batch_size \
    --lr $lr
