import logging
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
    # Skin palette that maps each class to RGB values.
    [120, 120, 120],  # background
    [220, 220, 220],  # skin
]
TASK_TYPE = Literal["acne", "hemo", "mela"]


def run_inference(image: Image.Image, task_type: TASK_TYPE):
    model, image_processor = _load_model(task_type)
    return _process_image(image, model, image_processor)


def _load_model(task_type: TASK_TYPE):
    if task_type == "acne":
#        checkpoint = "checkpoints/segformer-b5-acne-reduce-labels-focal+dice"
        checkpoint = "checkpoints/segformer-b5-focal+dice-acne50kdata-10ksteps"
    elif task_type == "hemo":
#        checkpoint = "checkpoints/segformer-b5-hemo-reduce-labels-focal+dice"
        checkpoint = "checkpoints/segformer-b5-focal+dice-hemo6.6kdata-10ksteps" # hemo with threshold 10
    elif task_type == "mela":
#        checkpoint = "checkpoints/segformer-b5-mela-reduce-labels-focal+dice"
        checkpoint = "checkpoints/segformer-b5-focal+dice-mela32kdata-10ksteps"  # mela with threshold 22

    device = "cuda"
    model = AutoModelForSemanticSegmentation.from_pretrained(
        checkpoint, device_map=device
    )
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)

    logger.info(f"Loading model from {checkpoint}...")
    logger.info(f"Successfully loaded model: {model.__class__.__name__}")
    logger.info(f"Model size: {_count_params(model)}M")

    return model, image_processor


def _process_image(image: Image.Image, model, image_processor):
    logger.info("Running inference...")

    inputs = image_processor(images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_segmentation_map = image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()
    logger.info(f"Predicted segmentation map shape: {predicted_segmentation_map}")

    color_seg = np.zeros(
        (predicted_segmentation_map.shape[0], predicted_segmentation_map.shape[1], 3),
        dtype=np.uint8,
    )  # height, width, 3

    palette = np.array(SKIN_PALETTE)
    for label, color in enumerate(palette):
        color_seg[predicted_segmentation_map == label, :] = color
    # Convert to BGR
    color_seg = color_seg[..., ::-1]

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    logger.info("Inference completed.\n")
    return img


def _count_params(model):
    num_params = sum(p.numel() for p in model.parameters())
    return num_params / 1e6  # Convert to millions


theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="stone",
    font=[gr.themes.GoogleFont("Source Sans 3", weights=(400, 600)), "arial"],
)

with gr.Blocks(theme=theme) as demo:
    with gr.Column(elem_classes="header"):
        gr.Markdown("# 🔍 Aramhuvis x SNU: Segment Your Skin!")
        gr.Markdown("### Wooyeol Lee, Minseo Kim, Byeongho Park")
        gr.Markdown("[[GitHub](https://github.com/thisiswooyeol/skin_segmentor)]")

    with gr.Column(elem_classes="abstract"):
        gr.Markdown(
            "Segementation of skin disease using a fine-tuned SegFormer model. This model runs semantic segmentation on acne/hemo/mela images separately."
        )  # Replace with your abstract text
        gr.Markdown(
            "⚠️ This is a test version of the demo app. We are working on improving the model and the app. Please check back later for updates."
        )

        gr.Interface(
            fn=run_inference,
            inputs=[
                gr.Image(type="pil"),
                gr.Radio(
                    choices=["acne", "hemo", "mela"],
                    value="acne",
                    label="Select Task",
                    interactive=True,
                ),
            ],
            outputs="image",
            examples=[
                ["demo/assets/acne.jpg", "acne"],
                ["demo/assets/acne_from_train.jpg", "acne"],
                ["demo/assets/hemo.jpg", "hemo"],
                ["demo/assets/hemo_from_train.jpg", "hemo"],
                ["demo/assets/mela.jpg", "mela"],
                ["demo/assets/mela_from_train.jpg", "mela"],
            ],
            cache_examples=True,  # Cache examples for faster loading
            cache_mode="lazy",
        )

if __name__ == "__main__":
    demo.queue().launch(share=True)
