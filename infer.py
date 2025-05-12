import os
from PIL import Image
from torchvision import transforms
import torch
from my_util import get_argv
from model import resnet18

# Preprocessing pipeline must match training transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def infer(model, image_paths: list[str]):
    infer_result = []

    # Inference on each image
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        input_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(dim=1).item()
        infer_result.append(pred)
    return infer_result


# Function to find misclassified images in a folder
def find_misclassified(model, folder_path: str, true_label: int, verbose=False):
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

if __name__ == "__main__":
    model_fname = get_argv(1, "resnet18_ghibli.pth")
    # image_paths = get_argvs(2)
    dir_path = get_argv(2, "on_theme")
    true_label = int(get_argv(3, 0))

    # Load model
    device = "mps" if torch.backends.mps.is_available() else \
        "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = resnet18(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_fname, map_location=device))
    model.eval()

    find_misclassified(model, dir_path, true_label, True)
