import torch
from argparse import ArgumentParser
import re

from my_util import DEVICE
from model import effnet
from infer import find_misclassified

if __name__ == "__main__":
    parser = ArgumentParser(description="EfficientNet inference")
    parser.add_argument("--model", type=str, help="Model Name",
                        default="effnetB0_ghibli")
    parser.add_argument("--test", type=str, help="Test Folder Name",
                        default="on_theme")
    parser.add_argument("--label", type=int, help="True Label")
    args = parser.parse_args()

    MODEL_NAME: str = args.model
    model_fname = f"{MODEL_NAME}.pth"
    match = re.search(r"effnet(\w+)_", MODEL_NAME)
    MODEL_SIZE = str(match.group(1)) if match else "B0"
    dir_path: str = args.test
    true_label: int = args.label

    # Load model
    model = effnet(MODEL_SIZE, num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(model_fname, map_location=DEVICE))
    model.eval()

    find_misclassified(model, dir_path, true_label, True)
