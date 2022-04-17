python3 main.py --dataset original \
    --image_pair text \
    --clip_pretrained_model "openai/clip-vit-large-patch14"  \
    --gpus '0' \
    --batch_size 16 \
    --lr 0.0001 \
    --max_epochs 10