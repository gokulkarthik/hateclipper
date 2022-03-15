python3 main.py --dataset original \
    --labels fine_grained \
    --clip_pretrained_model "openai/clip-vit-large-patch14"  \
    --use_pretrained_map f \
    --num_mapping_layers 1 \
    --map_dim 1024 \
    --head concat \
    --num_pre_output_layers 3 \
    --drop_probs 0.1 0.4 0.2 \
    --freeze_image_encoder t \
    --freeze_text_encoder t \
    --gpus '2' \
    --batch_size 64 \
    --lr 0.0001 \
    --weight_fine_grained_loss 1 \
    --max_epochs 40 \
    --remove_matches f \
    --eval_split test_seen 
    