# Run acne segmentation training
accelerate launch src/skin_segmentor/train_segformer_ss.py \
    --model_name_or_path nvidia/mit-b5 \
    --dataset_name ../dataset/acne \
    --do_reduce_labels \
    --output_dir checkpoints/segformer-b5-focal+dice-acne50kdata-10ksteps \
    --max_train_steps 10000 \
    --checkpointing_steps epoch \
    --seed 42 \
    --with_tracking \
    --report_to wandb 


# Run hemoglobin segmentation training
accelerate launch src/skin_segmentor/train_segformer_ss.py \
    --model_name_or_path nvidia/mit-b5 \
    --dataset_name ../dataset/hemo \
    --do_reduce_labels \
    --output_dir checkpoints/segformer-b5-focal+dice-hemo6.6kdata-10ksteps \
    --max_train_steps 10000 \
    --checkpointing_steps epoch \
    --seed 42 \
    --with_tracking \
    --report_to wandb 


# Run melanin segmentation training
accelerate launch src/skin_segmentor/train_segformer_ss.py \
    --model_name_or_path nvidia/mit-b5 \
    --dataset_name ../dataset/mela \
    --do_reduce_labels \
    --output_dir checkpoints/segformer-b5-focal+dice-mela32kdata-10ksteps \
    --max_train_steps 10000 \
    --checkpointing_steps epoch \
    --seed 42 \
    --with_tracking \
    --report_to wandb
