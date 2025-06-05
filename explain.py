import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image as Image
from argparse import ArgumentParser
import re
import os

from model import resnet, vit
from my_data import val_transform as transform

def visualize_conv1_filters(weights, n_col=8):
    """
    Plot convolutional filters given a weight tensor of shape (out_channels, in_channels, H, W).
    """
    # Normalize weights to 0-1 for display
    min_w, max_w = weights.min(), weights.max()
    weights = (weights - min_w) / (max_w - min_w)
    n_filters = weights.shape[0]
    n_row = (n_filters + n_col - 1) // n_col

    _, axes = plt.subplots(n_row, n_col, figsize=(n_col, n_row))
    for i, ax in enumerate(axes.flatten()):
        if i < n_filters:
            f = weights[i].permute(1, 2, 0)  # (H, W, in_channels)
            ax.imshow(f)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

class GradCAM:
    def __init__(self, model: nn.Module,
                 main: str, sub: str):
        self.model = model.eval()
        self.register_hook(main, sub)
    
    def register_hook(self, main: str, sub: str):
        for name, module in self.model.named_children():
            if name == main:
                for sub_name, sub_module in module[-1].named_children():
                    if sub_name == sub:
                        sub_module.register_forward_hook(self.forward_hook)
                        sub_module.register_full_backward_hook(self.backward_hook)
    
    def forward_hook(self, module: nn.Module, input, output):
        self.feature_map = output

    def backward_hook(self, module: nn.Module, grad_input, grad_output):
        self.gradient = grad_output[0]

    def __call__(self, x: torch.Tensor):
        output: torch.Tensor = self.model(x)

        index = output.argmax(axis=1)
        one_hot = torch.zeros_like(output)
        for i in range(output.size(0)):
            one_hot[i][index[i]] = 1

        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)

        a_k = torch.mean(self.gradient, dim=(2, 3), keepdim=True)
        grad_cam = torch.sum(a_k * self.feature_map, dim=1)
        grad_cam = torch.relu(grad_cam)
        return grad_cam

if __name__ == "__main__":
    parser = ArgumentParser(description="Model Explainability")
    parser.add_argument("--model", type=str, help="Model Name",
                        default="resnet18_ghibli")
    parser.add_argument("--test", type=str, help="Test Folder Name",
                        default="examples")
    args = parser.parse_args()

    MODEL_NAME: str = args.model
    TARGET_FOLDER: str = args.test

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
    gradcam = GradCAM(model=model, main="layer4", sub="conv2")

    images = [
        Image.open(os.path.join(TARGET_FOLDER, fname))
        for fname in sorted(os.listdir(TARGET_FOLDER))
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ]
    W, H = images[0].size
    tensor = torch.stack([transform(img) for img in images])

    heatmaps = F.interpolate(
        input=gradcam(tensor).unsqueeze(0),
        size=(H, W),
        mode="bilinear"
    ).squeeze().detach().numpy()
    predicted: torch.Tensor = model(tensor)

    for image, heatmap, pred in zip(images, heatmaps, predicted):
        plt.imshow(image)
        plt.imshow(heatmap, cmap="jet", alpha=0.5)
        plt.title("AI" if pred.argmax()==0 else "Real")
        plt.axis("off")
        plt.show()

    # conv1_weights = model.conv1.weight.data.clone().cpu()
    # visualize_conv1_filters(conv1_weights)
