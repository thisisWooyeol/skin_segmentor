import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)-18s - %(levelname)-8s - %(message)s",
)
logger = logging.getLogger(__name__)


def retreive_image_annotation_paths(
    dataset_dir: str,
) -> tuple[list[str], list[str]]:
    """
    Scan a directory for original images (containing "_org_") and match them
    with their corresponding mask images (same filename with "_org_"→"_mask_").

    Returns:
        image_paths: list of paths to original images
        mask_paths:  list of paths to corresponding mask images
    """
    image_paths: list[str] = []
    mask_paths: list[str] = []

    for fname in sorted(os.listdir(dataset_dir)):
        if "_org_" not in fname:
            continue

        img_path = os.path.join(dataset_dir, fname)
        mask_fname = fname.replace("_org_", "_mask_")
        mask_path = os.path.join(dataset_dir, mask_fname)

        if not os.path.isfile(mask_path):
            logger.warning(f"Skipping '{fname}': no mask found at '{mask_fname}'")
            continue

        image_paths.append(img_path)
        mask_paths.append(mask_path)

    return image_paths, mask_paths


if __name__ == "__main__":
    dataset_dir = "data/mela"
    image_paths, mask_paths = retreive_image_annotation_paths(dataset_dir)
    logger.info(f"Found {len(image_paths)} images and {len(mask_paths)} masks.")
    # for img, mask in zip(image_paths, mask_paths):
    #     logger.info(f"Image: {img}, Mask: {mask}")
