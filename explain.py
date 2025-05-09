import matplotlib.pyplot as plt
import torch

from my_util import get_argv
from model import resnet18

def load_model(model_fname: str):
    # pretrained 아니고 너가 학습한 모델 불러오기
    model = resnet18(num_classes=2)
    model.load_state_dict(torch.load(model_fname))

    return model

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
    model_fname = get_argv(1, "resnet18_ghibli.pth")

    model = load_model(model_fname).eval()

    conv1_weights = model.conv1.weight.data.clone().cpu()
    visualize_conv1_filters(conv1_weights)
