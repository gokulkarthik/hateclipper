python3 main.py --dataset original \
    --image_pair caption \
    --clip_pretrained_model "openai/clip-vit-large-patch14"  \
    --gpus '1' \
    --batch_size 32 \
    --lr 0.0001 \
    --max_epochs 10