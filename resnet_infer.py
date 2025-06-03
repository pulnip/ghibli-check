import torch
from my_util import DEVICE, get_argv
from model import resnet
from infer import find_misclassified

if __name__ == "__main__":
    model_fname = get_argv(1, "resnet18_ghibli.pth")
    # image_paths = get_argvs(2)
    dir_path = get_argv(2, "on_theme")
    true_label = int(get_argv(3, 0))

    # Load model
    model = resnet(18, num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(model_fname, map_location=DEVICE))
    model.eval()

    find_misclassified(model, dir_path, true_label, True)
