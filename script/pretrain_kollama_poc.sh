
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



python pretrain_kollama_poc.py \
    --size "$size" \
    --batch_size $batch_size \
    --lr $lr \
    --sharding $sharding \
    --epoch 4 \
    --save_dir ./tiny-kollama-ckpt/$size