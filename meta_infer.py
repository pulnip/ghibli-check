import torch
from argparse import ArgumentParser
from my_util import DEVICE
from model import ProtoNet, resnet
from infer import proto_infer
import re

def shorten(text: str, maxlen=12, num_dot=3):
    return text if len(text) <= maxlen \
                else text[:maxlen-3] + "..."


if __name__ == "__main__":
    parser = ArgumentParser(description="ProtoNet inference")
    parser.add_argument("--model", type=str, help="모델 이름",
                        default="meta_ghibli_resnet6")
    parser.add_argument("--task", nargs='+', type=str, help="태스크 폴더")
    args = parser.parse_args()

    model_fname: str = f"{args.model}.pth"
    task_paths: list[str] = args.task

    match = re.search(r"resnet(\d+)", args.model)
    model_size = int(match.group(1)) if match else 6
    embedding = resnet(model_size, num_classes=64)

    # Load model
    model = ProtoNet(num_ways=2, num_shots=3,
                     embedding_net=embedding).to(DEVICE)
    model.load_state_dict(torch.load(model_fname, map_location=DEVICE))
    model.eval()

    result = proto_infer(model, task_paths=task_paths)

    for task_name, task in result.items():
        print(f"{task_name}: ", end='')
        for i, (query_name, label) in enumerate(task.items()):
            label = "AI" if label == 0 else "Real"
            num_space = len(task_name) + 2 if i != 0 else 0
            print(f"{' '*num_space}{shorten(query_name)}: {label}")

