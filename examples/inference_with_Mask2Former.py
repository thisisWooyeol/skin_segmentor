import argparse
import colorsys
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)-18s - %(levelname)-8s - %(message)s",
)
logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with Mask2Former")
    parser.add_argument(
        "--image_url",
        type=str,
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        help="URL of the image to be processed",
    )
    parser.add_argument(
        "--backbone_size",
        choices=["tiny", "small", "base", "large"],
        default="base",
        help="Size of the backbone swin model",
    )
    parser.add_argument(
        "--training_dataset",
        choices=["coco", "cityscapes", "ade"],
        default="coco",
        help="Dataset used for training the model",
    )
    parser.add_argument(
        "--training_type",
        choices=["panoptic", "instance", "semantic"],
        default="panoptic",
        help="Type of training for the model",
    )
    parsed_args = parser.parse_args()

    # Filter out invalid model combinations
    if (
        parsed_args.training_dataset == "coco"
        and parsed_args.training_type == "semantic"
    ):
        raise ValueError(
            "The combination of coco dataset and semantic training type is not supported."
        )
    if (
        parsed_args.training_dataset == "ade"
        and parsed_args.training_type != "semantic"
    ):
        raise ValueError("The ADE dataset only supports semantic training type.")

    return parsed_args


if __name__ == "__main__":
    args = parse_args()
    # Load the processor and model
    model_path = f"facebook/mask2former-swin-{args.backbone_size}-{args.training_dataset}-{args.training_type}"
    processor = Mask2FormerImageProcessor.from_pretrained(model_path)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path)
    logger.info(f"model device: {model.device}")

    # Load an image from the internet
    try:
        image = Image.open(requests.get(args.image_url, stream=True).raw)
    except Exception as e:
        logger.error(f"Error loading image from URL: {e}")
        logger.info("Trying to load image from local path...")
        image = Image.open(args.image_url)
        # !NOTE: Image should be loaded from either a URL or a local path

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")
    for k, v in inputs.items():
        logger.info(f"k, v.shape: {k}, {v.shape}")

    # Forward pass
    start_time = time.time()
    with torch.inference_mode():
        outputs = model(**inputs)
    end_time = time.time()
    logger.info(f"Time taken for inference: {end_time - start_time:.2f} seconds")

    # Post-process the outputs
    predicted_map = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[image.size[::-1]],  # type: ignore
    )[0]
    logger.info(predicted_map.shape)

    # Generate a high-contrast complementary palette
    num_labels = len(model.config.id2label)
    # We only need to step through half the hues, since each hue gets a complementary partner
    step = (num_labels + 1) // 2
    hues = np.linspace(0, 1, step, endpoint=False)

    color_palette = []
    for h in hues:
        # primary color in RGB
        r1, g1, b1 = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        color_palette.append([int(r1 * 255), int(g1 * 255), int(b1 * 255)])
        # complementary color (hue + 0.5)
        if len(color_palette) < num_labels:
            r2, g2, b2 = colorsys.hsv_to_rgb((h + 0.5) % 1.0, 1.0, 1.0)
            color_palette.append([int(r2 * 255), int(g2 * 255), int(b2 * 255)])

    # Truncate in case of odd num_labels
    color_palette = color_palette[:num_labels]

    seg = predicted_map
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    palette = np.array(color_palette)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # Convert to BGR for consistency if needed
    color_seg = color_seg[..., ::-1]

    # Create a 1x2 subplot: original image and semantic mask
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(color_seg)
    axes[1].set_title("Semantic Mask")
    axes[1].axis("off")

    fig.tight_layout()

    # Ensure output directory exists
    output_dir = CURRENT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the figure
    output_path = (
        output_dir
        / f"image_{args.backbone_size}_{args.training_dataset}_{args.training_type}.png"
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
