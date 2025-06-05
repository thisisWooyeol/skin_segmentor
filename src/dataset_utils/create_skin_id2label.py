import json
from argparse import ArgumentParser

# After creating the id2label.json file, you can easily upload this by clicking
# on "Add file" in the "Files and versions" tab of your repo on the hub.


def parse_args():
    parser = ArgumentParser(description="Create skin id2label mapping")
    parser.add_argument(
        "--task_type",
        choices=["acne", "hemo", "mela"],
        required=True,
        help="Task type to create id2label mapping",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    id2label = {0: "background", 1: args.task_type}

    with open("id2label.json", "w") as f:
        json.dump(id2label, f)
