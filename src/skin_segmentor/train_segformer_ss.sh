accelerate launch src/skin_segmentor/train_segformer_ss.py \
    --model_name_or_path nvidia/mit-b5 \
    --dataset_name thisiswooyeol/skin_acne \
    --do_reduce_labels \
    --output_dir checkpoints/segformer-b5-acne-reduce-labels-focal+dice \
    --num_train_epochs 30 \
    --seed 42 \
    --with_tracking \
    --report_to wandb 