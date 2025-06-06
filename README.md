# Skin Segmentor with Mask2Former

This repository provides a skin segmentor based on the SegFormer model. As an extension of the original SegFormer, this implementation is specifically designed for three classes: acne, hemoglobin and melanin. 


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
    "1": "<type>",
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


<br>

# Quantitative results

The following table summarizes the evaluation metrics for our model compared to the baseline on the three test sets: Acne, Hemo (hemoglobin), and Mela (melanin).

<table border="1" cellpadding="6" cellspacing="0">
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th colspan="5">Ours</th>
      <th colspan="5">Baseline</th>
    </tr>
    <tr>
      <th>Precision</th>
      <th>Recall</th>
      <th>Accuracy</th>
      <th>Dice</th>
      <th>IoU</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>Accuracy</th>
      <th>Dice</th>
      <th>IoU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Acne</td>
      <td>0.7284</td>
      <td>0.1677</td>
      <td>0.9597</td>
      <td>0.2726</td>
      <td>0.1578</td>
      <td>0.6624</td>
      <td>0.1690</td>
      <td>0.9587</td>
      <td>0.2693</td>
      <td>0.1556</td>
    </tr>
    <tr>
      <td>Hemo</td>
      <td>0.7357</td>
      <td>0.4021</td>
      <td>0.8281</td>
      <td>0.5200</td>
      <td>0.3514</td>
      <td>0.7340</td>
      <td>0.3857</td>
      <td>0.8254</td>
      <td>0.5057</td>
      <td>0.3384</td>
    </tr>
    <tr>
      <td>Mela</td>
      <td>0.2026</td>
      <td>0.3731</td>
      <td>0.8916</td>
      <td>0.2626</td>
      <td>0.1512</td>
      <td>0.2237</td>
      <td>0.2469</td>
      <td>0.9167</td>
      <td>0.2347</td>
      <td>0.1330</td>
    </tr>
  </tbody>
</table>

Overall, our approach achieves higher Dice and IoU scores across all three datasets compared to the baseline. These two metrics, which measure the overlap between the prediction and the ground truth, are primary indicators of segmentation quality. The improvements are modest for the acne set (+0.003 Dice, +0.002 IoU) but more noticeable for the hemo (+0.014 Dice, +0.013 IoU) and mela (+0.028 Dice, +0.018 IoU) sets.

A closer analysis reveals how these improvements were achieved for each specific class:

- Acne: The primary performance gain for the "Ours" model is driven by a significant increase in precision (0.7284 vs. 0.6624 for the baseline), indicating that our model is more accurate in its positive predictions and makes fewer false positive errors.

- Hemo & Mela: For these datasets, the performance improvement is led by an increase in recall. This is especially noticeable in the Mela set, where recall jumped to 0.3731 from the baseline's 0.2469. This shows that the "Ours" model is substantially better at detecting the positive regions (True Positives), which was the key factor in improving the overall Dice and IoU scores for these more challenging segmentation tasks.

<br>

# Interactive demo

By running gradio app, you can run inference with your own image.

```bash
python demo/app.py
```

<br>

# References

[1] [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
](https://arxiv.org/abs/2105.15203) \
[2] [Semantic Segmentation examples with Hugging Face Transformers](https://github.com/huggingface/transformers/tree/main/examples/pytorch/semantic-segmentation)

<br>

# Acknowledgements

2025 Spring, Creative Integrated Design 2 class, Seoul National University, South Korea.
This project is supported by the SNU Creative Integrated Design 2 class and [Aram Huvis Co., Ltd](https://www.aramhuvis.com/). Collaborative effort were made with the following students:

- Minseo Kim
- Byeongho Park

Training and evaluation scripts are based on transformers examples. Thanks to Hugging Face for providing the codebase.
