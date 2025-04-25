import logging
import os
from argparse import ArgumentParser

from datasets import Dataset, DatasetDict, Image

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
    # Split the dataset into training and validation sets
    image_paths_train = image_paths[: int(len(image_paths) * (1 - validation_split))]
    image_paths_val = image_paths[int(len(image_paths) * (1 - validation_split)) :]
    mask_paths_train = mask_paths[: int(len(mask_paths) * (1 - validation_split))]
    mask_paths_val = mask_paths[int(len(mask_paths) * (1 - validation_split)) :]

    # Create a train/validation split
    train_dataset = Dataset.from_dict(
        {
            "image": image_paths_train,
            "label": mask_paths_train,
        }
    )
    train_dataset.cast_column("image", Image())
    train_dataset.cast_column("label", Image())

    val_dataset = Dataset.from_dict(
        {
            "image": image_paths_val,
            "label": mask_paths_val,
        }
    )
    val_dataset.cast_column("image", Image())
    val_dataset.cast_column("label", Image())

    # Create a DatasetDict
    dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset})

    # Push to a private repo on Hugging Face Hub
    commit_info = dataset_dict.push_to_hub(
        repo_id=output_dir,
        private=True,
    )
    logger.info(f"Dataset pushed to Hugging Face Hub: {commit_info}")


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
        help="Name of repo to push the dataset to huggingface.co",
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
