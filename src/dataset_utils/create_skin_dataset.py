import logging
import os
from argparse import ArgumentParser

from datasets import Dataset, DatasetDict, Features, Image

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
        mask_fname = fname.replace("_org_", "_mask_").replace(".jpg", ".png")
        mask_path = os.path.join(dataset_dir, mask_fname)

        if not os.path.isfile(mask_path):
            logger.warning(f"Skipping '{fname}': no mask found at '{mask_fname}'")
            continue

        image_paths.append(img_path)
        mask_paths.append(mask_path)

    return image_paths, mask_paths


def read_images_raw_bytes(
    image_paths: list[str], mask_paths: list[str]
) -> tuple[list[bytes], list[bytes]]:
    """
    Read images and masks as raw bytes.

    Returns:
        image_bytes: list of raw bytes of original images
        mask_bytes:  list of raw bytes of corresponding mask images
    """
    image_bytes = [open(path, "rb").read() for path in image_paths]
    mask_bytes = [open(path, "rb").read() for path in mask_paths]

    return image_bytes, mask_bytes


def create_skin_dataset(
    image_paths: list[str],
    mask_paths: list[str],
    output_dir: str,
    validation_split: float = 0.1,
) -> None:
    """
    Create a skin dataset by copying images and masks to a new directory.

    Args:
        image_paths: list of paths to original images
        mask_paths: list of paths to corresponding mask images
        output_dir: name of repo to push the dataset to huggingface.co
    """
    # Read paths as raw bytes
    image_bytes, mask_bytes = read_images_raw_bytes(image_paths, mask_paths)
    data = {"image": image_bytes, "label": mask_bytes}

    # Define the features so 'image' & 'mask' are Image columns
    features = Features(
        {
            "image": Image(),
            "label": Image(),
        }
    )

    # Build and split a DatasetDict
    dataset = Dataset.from_dict(data, features=features)
    split = dataset.train_test_split(test_size=validation_split)
    dataset_dict = DatasetDict({"train": split["train"], "validation": split["test"]})

    # Save dataset to local path
    dataset_dict.save_to_disk(dataset_dict_path=output_dir)
    logger.info(f"Successfully saved dataset into {output_dir}")


def main():
    parser = ArgumentParser(
        description="Create a skin dataset and push it to Hugging Face Hub."
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Path to dataset directory with original/marked images",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Name of local path to save the dataset",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (default: 0.1)",
    )
    args = parser.parse_args()

    # Retrieve image and mask paths
    image_paths, mask_paths = retreive_image_annotation_paths(args.dataset_dir)
    logger.info(f"Found {len(image_paths)} images and {len(mask_paths)} masks.")

    # Create the skin dataset
    create_skin_dataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        output_dir=args.output_dir,
        validation_split=args.validation_split,
    )


if __name__ == "__main__":
    main()
