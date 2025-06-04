# Skin Segmentor with Mask2Former

This repository provides a skin segmentor based on the SegFormer model. As an extension of the original SegFormer, this implementation is specifically designed for three classes: acne, hemoglobin and melanin. 

<br>

# TODO
- [x] work with bigger dataset (currently used toy dataset)
- [x] use larger batch size & more epochs
- [ ] add quantitative evaluation scripts

<br>

# Get Started
## 0. Installation

Please install `uv` from [here](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already.

```bash
git clone https://github.com/thisisWooyeol/skin_segmentor.git

cd skin_segmentor
uv sync
```

## 1. Preprocess mask

Recommended threshold value for each class is as follows:
- acne: 20
- hemo: 10
- mela: 22

```bash
python src/utils/generate_mask.py --dataset_dir <dataset_dir> --threshold <threshold>
```

## 2. Create custom skin dataset

```bash
python src/utils/create_skin_dataset.py --dataset_dir <dataset_dir> --output_dir <output_dir>
```

Then create `id2label.json` file and upload it to the HF dataset repo. It should look like follows, where `<type>` is either `acne`, `hemo` or `mela`.

```json
{
    "0": "background",
    "1": <type>,
}
```

## 3. Train

Example scripts are provided in `src/skin_segmentor/train_segformer_ss.sh`. Or you can run the training script directly as follows:

```bash
accelerate launch src/skin_segmentor/train_segformer_ss.py \
    --model_name_or_path nvidia/mit-b5 \
    --dataset_path ../dataset/acne \
    --do_reduce_labels \
    --output_dir checkpoints/segformer-b5-focal+dice-acne50kdata-10ksteps \
    --max_train_steps 10000 \
    --checkpointing_steps epoch \
    --seed 42 \
    --with_tracking \
    --report_to wandb 
```

## 4. Evaluate

TODO: add quantitative evaluation

<br>

# Interactive demo

By running gradio app, you can run inference with your own image.

```bash
python demo/app.py
```

<br>

# References

[1] [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
](https://arxiv.org/abs/2105.15203)
[2] [Semantic Segmentation examples with Hugging Face Transformers](https://github.com/huggingface/transformers/tree/main/examples/pytorch/semantic-segmentation)

<br>

# Acknowledgements

2025 Spring, Creative Integrated Design 2 class, Seoul National University, South Korea.
This project is supported by the SNU Creative Integrated Design 2 class and [Aram Huvis Co., Ltd](https://www.aramhuvis.com/). Collaborative effort were made with the following students:

- Minseo Kim
- Byeongho Park

Training and evaluation scripts are based on transformers examples. Thanks to Hugging Face for providing the codebase.
