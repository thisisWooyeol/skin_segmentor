import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_mask(path: str) -> np.ndarray:
    """Load a grayscale mask and convert values to {0,1}."""
    mask = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return (mask > 0).astype(np.uint8)


def compute_confusion(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) for a pair of masks."""
    pred_flat = pred.reshape(-1)
    gt_flat = gt.reshape(-1)
    tp = int(np.logical_and(pred_flat == 1, gt_flat == 1).sum())
    fp = int(np.logical_and(pred_flat == 1, gt_flat == 0).sum())
    fn = int(np.logical_and(pred_flat == 0, gt_flat == 1).sum())
    tn = int(np.logical_and(pred_flat == 0, gt_flat == 0).sum())
    return tp, fp, fn, tn


def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "dice": dice,
        "iou": iou,
    }


def main(gt_dir: str, pred_dir: str, output_path: str | None) -> None:
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)

    gt_files = {
        f.name: f for f in gt_dir.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    }
    pred_files = {
        f.name: f for f in pred_dir.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    }
    common = sorted(set(gt_files.keys()) & set(pred_files.keys()))
    if not common:
        raise ValueError("No matching files between gt_dir and pred_dir")

    tp = fp = fn = tn = 0
    for name in common:
        gt_mask = load_mask(gt_files[name])
        pred_mask = load_mask(pred_files[name])
        if gt_mask.shape != pred_mask.shape:
            raise ValueError(f"Shape mismatch for '{name}'")
        _tp, _fp, _fn, _tn = compute_confusion(pred_mask, gt_mask)
        tp += _tp
        fp += _fp
        fn += _fn
        tn += _tn

    metrics = compute_metrics(tp, fp, fn, tn)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predicted masks")
    parser.add_argument("--gt_dir", required=True, help="Directory with ground truth masks")
    parser.add_argument("--pred_dir", required=True, help="Directory with predicted masks")
    parser.add_argument(
        "--output", default=None, help="Optional path to save computed metrics as JSON"
    )
    args = parser.parse_args()
    main(args.gt_dir, args.pred_dir, args.output)
