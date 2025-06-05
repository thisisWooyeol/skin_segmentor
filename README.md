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

Here we used `../data_raw/<type>` for `--dataset_dir` argument.


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

Here we used `../data_raw/<type>` for `--dataset_dir` argument and `../dataset/<type>` for `--output_dir` argument.


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

### A. Create GT labels

If there is no GT annotations available, make it with annotation gui:

```bash
python src/annotation_utils/annotation_gui.py --image_folder ../testcases/acne_test/image/

python src/annotation_utils/annotation_gui.py --image_folder ../testcases/hemo_test/image/ 

python src/annotation_utils/annotation_gui.py --image_folder ../testcases/mela_test/image/
```

Grayscale segmentation masks will be saved at `testcases/<type>_test/label/` folder.


### B. Create Ours predictions

Run

```bash
python src/skin_segmentor/inference_segformer_ss.py \
  --checkpoint checkpoints/segformer-b5-focal+dice-acne50kdata-10ksteps \
  --input_dir ../testcases/acne_test/image \
  --output_dir ../testcases/acne_test/ours_pred

python src/skin_segmentor/inference_segformer_ss.py \
  --checkpoint checkpoints/segformer-b5-focal+dice-hemo6.6kdata-10ksteps \
  --input_dir ../testcases/hemo_test/image \
  --output_dir ../testcases/hemo_test/ours_pred

python src/skin_segmentor/inference_segformer_ss.py \
  --checkpoint checkpoints/segformer-b5-focal+dice-mela32kdata-10ksteps \
  --input_dir ../testcases/mela_test/image \
  --output_dir ../testcases/mela_test/ours_pred
```

### C. Get evaluation metrics

To evalutate ours:

```bash
python src/evaluation_utils/evaluate.py \
  --gt_dir ../testcases/acne_test/label/ \
  --pred_dir ../testcases/acne_test/ours_pred/

python src/evaluation_utils/evaluate.py \
  --gt_dir ../testcases/hemo_test/label/ \
  --pred_dir ../testcases/hemo_test/ours_pred/

python src/evaluation_utils/evaluate.py \
  --gt_dir ../testcases/mela_test/label/ \
  --pred_dir ../testcases/mela_test/ours_pred/
````

To evaluate baseline:

```
python src/evaluation_utils/evaluate.py \
  --gt_dir ../testcases/acne_test/label/ \
  --pred_dir ../testcases/acne_test/baseline_pred/

python src/evaluation_utils/evaluate.py \
  --gt_dir ../testcases/hemo_test/label/ \
  --pred_dir ../testcases/hemo_test/baseline_pred/

python src/evaluation_utils/evaluate.py \
  --gt_dir ../testcases/mela_test/label/ \
  --pred_dir ../testcases/mela_test/baseline_pred/
```

### D. Quantitative results

The following table summarizes the evaluation metrics for our model compared to the baseline on the three test sets.

| Dataset | Precision (Ours) | Recall (Ours) | Accuracy (Ours) | Dice (Ours) | IoU (Ours) | Precision (Baseline) | Recall (Baseline) | Accuracy (Baseline) | Dice (Baseline) | IoU (Baseline) |
|---|---|---|---|---|---|---|---|---|---|---|
| Acne | 0.7284 | 0.1677 | 0.9597 | 0.2726 | 0.1578 | 0.6624 | 0.1690 | 0.9587 | 0.2693 | 0.1556 |
| Hemo | 0.7357 | 0.4021 | 0.8281 | 0.5200 | 0.3514 | 0.7340 | 0.3857 | 0.8254 | 0.5057 | 0.3384 |
| Mela | 0.2026 | 0.3731 | 0.8916 | 0.2626 | 0.1512 | 0.2237 | 0.2469 | 0.9167 | 0.2347 | 0.1330 |

Overall, our approach achieves higher Dice and IoU scores across all three datasets compared to the baseline. The improvements are modest for the acne set (+0.003 Dice, +0.002 IoU) but more noticeable for the hemo (+0.014 Dice, +0.013 IoU) and mela (+0.028 Dice, +0.018 IoU) sets.

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
