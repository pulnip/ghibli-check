import os
import PIL.Image as Image
from PIL.Image import Image as PIL_Image
import torch
import torch.nn as nn
from pathlib import Path
from my_util import DEVICE
from my_data import val_transform as preprocess
from model import ProtoNet

def infer(model, image_paths: list[str]):
    infer_result = []

    # Inference on each image
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(dim=1).item()
        infer_result.append(pred)
    return infer_result

# Function to find misclassified images in a folder
def find_misclassified(model: nn.Module, folder_path: str,
                       true_label: int, verbose=False):
    """
    Given a folder of images and the true label (1 for real Ghibli, 0 for AI-generated),
    print out any images that the model misclassifies.
    """
    mis_count = 0
    # Gather image file paths
    image_paths = [
        os.path.join(folder_path, fname)
        for fname in sorted(os.listdir(folder_path))
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ]
    # Run inference
    results = infer(model, image_paths)
    # Report misclassifications
    for img_path, pred in zip(image_paths, results):
        if pred != true_label:
            mis_count += 1
            img_name = os.path.basename(img_path)
            # label_desc = "Real Ghibli" if pred == 1 else "AI-generated"
            if verbose:
                print(f"{img_name}")
    print(f"잘못 분류된 이미지 개수: {mis_count}")

def proto_infer(model: ProtoNet, task_paths: list[str]):
    infer_result: dict[str, dict[str, int]] = {}

    for task in task_paths:
        support: dict[int, list[torch.Tensor]] = {}
        queries: dict[str, torch.Tensor] = []

        for class_folder in Path(task).iterdir():
            images: dict[str, torch.Tensor] = {img_path.name: preprocess(Image.open(img_path))
                                               for img_path in list(class_folder.glob("*.*"))}

            if(class_folder.is_dir()):
                if(class_folder.name != "query"):
                    class_name = int(class_folder.name)
                    support[class_name] = list(images.values())
                else:
                    queries = images

        support_x: list[torch.Tensor] = []
        support_y: list[int] = []
        for cls, imgs in support.items():
            support_x += imgs
            support_y += [cls]*len(imgs)
        support_x = torch.stack(support_x).to(DEVICE)
        support_y = torch.tensor(support_y).to(DEVICE)

        query_x = torch.stack(list(queries.values())).to(DEVICE)
        query_y: torch.Tensor = model(support_x, support_y, query_x)
        preds = query_y.argmax(dim=1)

        task_result = {name: int(label.item()) for name, label in zip(queries.keys(), preds)}
        infer_result[task] = task_result
    return infer_result
