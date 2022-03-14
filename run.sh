python3 main.py --dataset original \
    --clip_pretrained_model "openai/clip-vit-large-patch14"  \
    --head concat \
    --map_size 1024 \
    --num_mapping_layers 1 \
    --use_pretrained_map f \
    --drop_probs 0.1 0.4 0.2 \
    --freeze_image_encoder t \
    --freeze_text_encoder t \
    --gpus '0' \
    --batch_size 64 \
    --max_steps 5000 \
    --remove_matches f \
    