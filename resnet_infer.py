import torch
from argparse import ArgumentParser
from my_util import DEVICE
from model import resnet
from infer import find_misclassified

if __name__ == "__main__":
    parser = ArgumentParser(description="ResNet inference")
    parser.add_argument("--model", type=str, help="Model Name",
                        default="resnet18_ghibli")
    parser.add_argument("--test", type=str, help="Test Folder Name",
                        default="on_theme")
    parser.add_argument("--label", type=int, help="True Label",
                        default="on_theme")
    args = parser.parse_args()

    model_fname = f"{args.model}.pth"
    dir_path: str = args.test
    true_label: int = args.label

    # Load model
    model = resnet(18, num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(model_fname, map_location=DEVICE))
    model.eval()

    find_misclassified(model, dir_path, true_label, True)
