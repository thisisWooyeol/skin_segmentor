import logging
from pathlib import Path
from typing import Literal

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

ASSETS_ROOT = Path("demo/assets")
DATASET_NAMES = {
    "acne": "acne_test",
    "hemo": "hemo_test",
    "mela": "mela_test",
}


def load_dataset_items(assets_root_path: Path, dataset_names_map: dict) -> dict:
    """Loads dataset item names by scanning the image directories."""
    dataset_items = {}
    for task_type, dataset_folder_name in dataset_names_map.items():
        image_dir = assets_root_path / dataset_folder_name / "image"
        if image_dir.exists() and image_dir.is_dir():
            items = sorted([p.stem for p in image_dir.glob("*.png")])
            dataset_items[task_type] = items
            logger.info(
                f"Found {len(items)} items for task '{task_type}' in {image_dir}"
            )
        else:
            logger.warning(
                f"Image directory not found for task '{task_type}': {image_dir}"
            )
            dataset_items[task_type] = []
    return dataset_items


DATASET_ITEMS = load_dataset_items(ASSETS_ROOT, DATASET_NAMES)

# Global model and processor
_GLOBAL_MODEL = None
_GLOBAL_PROCESSOR = None
_CURRENT_TASK_TYPE = None


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
    inputs = image_processor(images=[image], return_tensors="pt").to("mps")
    with torch.no_grad():
        outputs = model(**inputs)
    seg = image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    seg = seg.cpu().numpy()
    return (seg == 1).astype(np.uint8)


def get_model_and_processor(task_type: TASK_TYPE):
    global _GLOBAL_MODEL, _GLOBAL_PROCESSOR, _CURRENT_TASK_TYPE

    if task_type == _CURRENT_TASK_TYPE and _GLOBAL_MODEL is not None:
        logger.info(f"Using cached model for task: {task_type}")
        return _GLOBAL_MODEL, _GLOBAL_PROCESSOR

    logger.info(f"Loading model for task: {task_type}")
    if task_type == "acne":
        checkpoint = "checkpoints/segformer-b5-focal+dice-acne50kdata-10ksteps"
    elif task_type == "hemo":
        checkpoint = "checkpoints/segformer-b5-focal+dice-hemo6.6kdata-10ksteps"
    else:  # mela
        checkpoint = "checkpoints/segformer-b5-focal+dice-mela32kdata-10ksteps"

    device = "cuda"
    try:
        model = AutoModelForSemanticSegmentation.from_pretrained(
            checkpoint, device_map=device
        )
        image_processor = AutoImageProcessor.from_pretrained(checkpoint)
        _GLOBAL_MODEL = model
        _GLOBAL_PROCESSOR = image_processor
        _CURRENT_TASK_TYPE = task_type
        logger.info(f"Successfully loaded model from {checkpoint} for task {task_type}")
    except Exception as e:
        logger.error(f"Error loading model for task {task_type} from {checkpoint}: {e}")
        # Fallback or re-raise, depending on desired error handling
        # For now, if a model fails to load, subsequent calls might try again or fail
        # Setting them to None ensures a reload attempt if task type changes or on next call
        _GLOBAL_MODEL = None
        _GLOBAL_PROCESSOR = None
        _CURRENT_TASK_TYPE = None
        raise  # Re-raise the exception to make the failure visible

    return _GLOBAL_MODEL, _GLOBAL_PROCESSOR


def run_comparison(task_type: TASK_TYPE, item: str):
    dataset = DATASET_NAMES[task_type]
    image_path = ASSETS_ROOT / dataset / "image" / f"{item}.png"
    gt_path = ASSETS_ROOT / dataset / "label" / f"{item}_mask.png"
    base_path = ASSETS_ROOT / dataset / "baseline_pred" / f"{item}_mask.png"

    image = Image.open(image_path).convert("RGB")
    gt_mask = load_mask(gt_path)
    base_mask = load_mask(base_path)
    base_metrics = compute_metrics(base_mask, gt_mask)

    model, proc = get_model_and_processor(task_type)
    if model is None or proc is None:
        # Handle case where model loading failed
        # For Gradio, you might want to return an error message or placeholder images
        error_message = "Model failed to load. Please check logs."
        # Create a dummy PIL Image with error text
        error_img = Image.new("RGB", (256, 256), color="red")
        from PIL import ImageDraw

        d = ImageDraw.Draw(error_img)
        d.text((10, 10), error_message, fill=(255, 255, 0))
        return (
            image,
            error_img,
            error_img,
            error_img,
            [["Error", "-", "-", error_message]],
        )

    pred_mask = predict_mask(image, model, proc)
    ours_metrics = compute_metrics(pred_mask, gt_mask)

    orig_img = image
    gt_img = overlay_mask(image, gt_mask)
    base_img = overlay_mask(image, base_mask)
    ours_img = overlay_mask(image, pred_mask)

    metrics_data = [
        [
            "Baseline",
            round(base_metrics[0], 3),
            round(base_metrics[1], 3),
            round(base_metrics[2], 3),
        ],
        [
            "Ours",
            round(ours_metrics[0], 3),
            round(ours_metrics[1], 3),
            round(ours_metrics[2], 3),
        ],
    ]
    return orig_img, gt_img, base_img, ours_img, metrics_data


def update_items(task_type: TASK_TYPE):
    items = DATASET_ITEMS[task_type]
    return gr.Dropdown(choices=items, value=items[0])


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
            task_radio = gr.Radio(
                ["acne", "hemo", "mela"], value="acne", label="Select Task"
            )
            item_dd = gr.Dropdown(
                DATASET_ITEMS["acne"], value=DATASET_ITEMS["acne"][0], label="Image"
            )
        task_radio.change(fn=update_items, inputs=task_radio, outputs=item_dd)
        run_btn = gr.Button("Run")
        with gr.Row():
            with gr.Column():
                out_orig = gr.Image(label="Original Image")
                out_gt = gr.Image(label="GT Mask")
            with gr.Column():
                out_base = gr.Image(label="Baseline Prediction")
                out_ours = gr.Image(label="Ours Prediction")
        out_metrics = gr.Dataframe(
            headers=["Method", "Precision", "Recall", "Dice"], row_count=2
        )
        run_btn.click(
            fn=run_comparison,
            inputs=[task_radio, item_dd],
            outputs=[out_orig, out_gt, out_base, out_ours, out_metrics],
        )

    # Pre-load the model for the default task type
    default_task_type = "acne"  # Matches the initial value of task_radio
    logger.info(f"Pre-loading model for default task: {default_task_type}")
    try:
        get_model_and_processor(default_task_type)
    except Exception as e:
        logger.error(
            f"Failed to pre-load model for default task {default_task_type}: {e}"
        )
        # The app will still launch, but the first run might be slow or fail if model remains unloaded.

    return demo


if __name__ == "__main__":
    main().queue().launch(share=True)
