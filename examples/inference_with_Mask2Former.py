import argparse
import logging
import time
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

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
    processor = AutoImageProcessor.from_pretrained(model_path)
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
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    logger.info(predicted_map.shape)

    # Visualize the results
    color_palette = [
        list(np.random.choice(range(256), size=3))
        for _ in range(len(model.config.id2label))
    ]

    seg = predicted_map
    color_seg = np.zeros(
        (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
    )  # height, width, 3
    palette = np.array(color_palette)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # Convert to BGR
    color_seg = color_seg[..., ::-1]
    print(np.unique(color_seg))

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    # plt.figure(figsize=(15, 10))
    # plt.imshow(img)
    # plt.show()

    # Save the image
    output_image = Image.fromarray(img)
    output_image.save(
        f"{CURRENT_DIR}/outputs/image_{args.backbone_size}_{args.training_dataset}_{args.training_type}.png"
    )
