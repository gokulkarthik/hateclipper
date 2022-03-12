python3 main.py  --head clip \
    --map_size 1024 \
    --num_mapping_layers 1 \
    --freeze_image_encoder t \
    --freeze_text_encoder t \
    --gpus '1' \
    --batch_size 64 \
    --max_steps 5000 \
    --remove_matches f \
    