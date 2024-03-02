
size="46M"
lr="3e-4"
batch_size=8



python pretrain_kollama.py \
    --size "46M" \
    --batch_size $batch_size \
    --lr $lr \
    --save_dir ./tiny-kollama-ckpt/$size