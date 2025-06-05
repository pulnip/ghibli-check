import matplotlib.pyplot as plt
import torch
from argparse import ArgumentParser
import re

from model import resnet, vit

def visualize_conv1_filters(weights, n_col=8):
    """
    Plot convolutional filters given a weight tensor of shape (out_channels, in_channels, H, W).
    """
    # Normalize weights to 0-1 for display
    min_w, max_w = weights.min(), weights.max()
    weights = (weights - min_w) / (max_w - min_w)
    n_filters = weights.shape[0]
    n_row = (n_filters + n_col - 1) // n_col

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col, n_row))
    for i, ax in enumerate(axes.flatten()):
        if i < n_filters:
            f = weights[i].permute(1, 2, 0)  # (H, W, in_channels)
            ax.imshow(f)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser(description="Model Explainability")
    parser.add_argument("--model", type=str, help="Model Name",
                        default="resnet18_ghibli")
    args = parser.parse_args()

    MODEL_NAME: str = args.model

    if "resnet" in MODEL_NAME:
        match = re.search(r"resnet(\d+)", MODEL_NAME)
        MODEL_SIZE = int(match.group(1)) if match else 18
        model = resnet(MODEL_SIZE, num_classes=2)
    elif "vit" in MODEL_NAME:
        match = re.search(r"vit-(\w+)_", MODEL_NAME)
        MODEL_SIZE = str(match.group(1)) if match else "tiny"
        model = vit(MODEL_SIZE, num_classes=2)
        raise RuntimeError(f"f{MODEL_NAME} not available.")
    else:
        raise RuntimeError(f"f{MODEL_NAME} not implemented.")

    model.load_state_dict(torch.load(f"{MODEL_NAME}.pth"))
    model.eval()

    conv1_weights = model.conv1.weight.data.clone().cpu()
    visualize_conv1_filters(conv1_weights)
