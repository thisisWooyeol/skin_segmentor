import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SegFormer skin segmentation on a folder of images"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint directory containing the model and processor",
    )
    parser.add_argument(
        "--input_dir", required=True, help="Directory with input images"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where predicted mask images will be saved",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device (default: autodetect)",
    )
    return parser.parse_args()


def load_image_paths(input_dir: str):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    paths = [
        p
        for p in sorted(Path(input_dir).iterdir())
        if p.suffix.lower() in exts
    ]
    return paths


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)

    model = AutoModelForSemanticSegmentation.from_pretrained(args.checkpoint).to(
        device
    )
    processor = AutoImageProcessor.from_pretrained(args.checkpoint)

    image_paths = load_image_paths(args.input_dir)
    if not image_paths:
        raise ValueError(f"No images found in {args.input_dir}")

    for img_path in tqdm(image_paths, desc="Processing"):
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        mask = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        mask = (mask.cpu().numpy() * 255).astype("uint8")
        out_path = Path(args.output_dir) / f"{img_path.stem}.png"
        Image.fromarray(mask).save(out_path)


if __name__ == "__main__":
    main()
