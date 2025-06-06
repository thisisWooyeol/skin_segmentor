import logging
from typing import Literal
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)-18s - %(levelname)-8s - %(message)s",
)
logger = logging.getLogger(__name__)

SKIN_PALETTE = [
    [120, 120, 120],  # background
    [220, 220, 220],  # skin
]
TASK_TYPE = Literal["acne", "hemo", "mela"]

ASSETS_ROOT = Path("assets")
DATASET_NAMES = {
    "acne": "acne_test",
    "hemo": "hemo_test",
    "mela": "mela_test",
}
DATASET_ITEMS = {
    "acne": [
        "item_00212",
        "item_00236",
        "item_00426",
        "item_00538",
        "item_00570",
        "item_00596",
        "item_00652",
        "item_00663",
        "item_00768",
        "item_00958",
    ],
    "hemo": [
        "item_00001",
        "item_00003",
        "item_00012",
        "item_00013",
        "item_00021",
        "item_00033",
        "item_00036",
        "item_00039",
        "item_00046",
        "item_00054",
    ],
    "mela": [
        "item_00006",
        "item_00013",
        "item_00055",
        "item_00090",
        "item_00138",
        "item_00209",
        "item_00216",
        "item_00227",
        "item_00270",
        "item_00306",
    ],
}


def load_mask(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return (mask > 0).astype(np.uint8)


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    pred_flat = pred.reshape(-1)
    gt_flat = gt.reshape(-1)
    tp = int(np.logical_and(pred_flat == 1, gt_flat == 1).sum())
    fp = int(np.logical_and(pred_flat == 1, gt_flat == 0).sum())
    fn = int(np.logical_and(pred_flat == 0, gt_flat == 1).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return precision, recall, dice


def overlay_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    color_seg = np.zeros((*mask.shape, 3), dtype=np.uint8)
    palette = np.array(SKIN_PALETTE)
    for label, color in enumerate(palette):
        color_seg[mask == label] = color
    color_seg = color_seg[..., ::-1]
    img = np.array(image) * 0.5 + color_seg * 0.5
    return Image.fromarray(img.astype(np.uint8))


def predict_mask(image: Image.Image, model, image_processor) -> np.ndarray:
    inputs = image_processor(images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    seg = image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    seg = seg.cpu().numpy()
    return (seg == 1).astype(np.uint8)


def load_model(task_type: TASK_TYPE):
    if task_type == "acne":
        checkpoint = "checkpoints/segformer-b5-focal+dice-acne50kdata-10ksteps"
    elif task_type == "hemo":
        checkpoint = "checkpoints/segformer-b5-focal+dice-hemo6.6kdata-10ksteps"
    else:
        checkpoint = "checkpoints/segformer-b5-focal+dice-mela32kdata-10ksteps"

    device = "cuda"
    model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, device_map=device)
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    logger.info(f"Loaded model from {checkpoint}")
    return model, image_processor


def run_comparison(task_type: TASK_TYPE, item: str):
    dataset = DATASET_NAMES[task_type]
    image_path = ASSETS_ROOT / dataset / "image" / f"{item}.png"
    gt_path = ASSETS_ROOT / dataset / "label" / f"{item}_mask.png"
    base_path = ASSETS_ROOT / dataset / "baseline_pred" / f"{item}_mask.png"

    image = Image.open(image_path).convert("RGB")
    gt_mask = load_mask(gt_path)
    base_mask = load_mask(base_path)
    base_metrics = compute_metrics(base_mask, gt_mask)

    model, proc = load_model(task_type)
    pred_mask = predict_mask(image, model, proc)
    ours_metrics = compute_metrics(pred_mask, gt_mask)

    orig_img = image
    gt_img = overlay_mask(image, gt_mask)
    base_img = overlay_mask(image, base_mask)
    ours_img = overlay_mask(image, pred_mask)

    metrics_data = [
        [round(base_metrics[0], 3), round(base_metrics[1], 3), round(base_metrics[2], 3)],
        [round(ours_metrics[0], 3), round(ours_metrics[1], 3), round(ours_metrics[2], 3)],
    ]
    return orig_img, gt_img, base_img, ours_img, metrics_data


def update_items(task_type: TASK_TYPE):
    items = DATASET_ITEMS[task_type]
    return gr.Dropdown.update(choices=items, value=items[0])


def main():
    theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="stone",
        font=[gr.themes.GoogleFont("Source Sans 3", weights=(400, 600)), "arial"],
    )
    with gr.Blocks(theme=theme) as demo:
        with gr.Column():
            gr.Markdown("# 🔍 Aramhuvis x SNU: Segment Your Skin!")
            gr.Markdown("### Comparison with Baseline")
        with gr.Row():
            task_radio = gr.Radio(["acne", "hemo", "mela"], value="acne", label="Select Task")
            item_dd = gr.Dropdown(DATASET_ITEMS["acne"], value=DATASET_ITEMS["acne"][0], label="Image")
        task_radio.change(fn=update_items, inputs=task_radio, outputs=item_dd)
        run_btn = gr.Button("Run")
        with gr.Row():
            with gr.Column():
                out_orig = gr.Image(label="Original Image")
                out_gt = gr.Image(label="GT Mask")
            with gr.Column():
                out_base = gr.Image(label="Baseline Prediction")
                out_ours = gr.Image(label="Ours Prediction")
        out_metrics = gr.Dataframe(headers=["Precision", "Recall", "Dice"], row_count=2)
        run_btn.click(fn=run_comparison, inputs=[task_radio, item_dd], outputs=[out_orig, out_gt, out_base, out_ours, out_metrics])
    return demo


if __name__ == "__main__":
    main().queue().launch(share=True)
