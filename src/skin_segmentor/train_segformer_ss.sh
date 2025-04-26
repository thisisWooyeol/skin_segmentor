# Run acne segmentation training
accelerate launch src/skin_segmentor/train_segformer_ss.py \
    --model_name_or_path nvidia/mit-b5 \
    --dataset_name thisiswooyeol/skin_acne \
    --do_reduce_labels \
    --output_dir checkpoints/segformer-b5-acne-reduce-labels-focal+dice \
    --num_train_epochs 100 \
    --seed 42 \
    --with_tracking \
    --report_to wandb 


# Run hemoglobin segmentation training
accelerate launch src/skin_segmentor/train_segformer_ss.py \
    --model_name_or_path nvidia/mit-b5 \
    --dataset_name thisiswooyeol/skin_hemo \
    --do_reduce_labels \
    --output_dir checkpoints/segformer-b5-hemo-reduce-labels-focal+dice \
    --num_train_epochs 100 \
    --seed 42 \
    --with_tracking \
    --report_to wandb 


# Run melanin segmentation training
accelerate launch src/skin_segmentor/train_segformer_ss.py \
    --model_name_or_path nvidia/mit-b5 \
    --dataset_name thisiswooyeol/skin_mela \
    --do_reduce_labels \
    --output_dir checkpoints/segformer-b5-mela-reduce-labels-focal+dice \
    --num_train_epochs 100 \
    --seed 42 \
    --with_tracking \
    --report_to wandb