import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from argparse import ArgumentParser
import re
import os

from my_data import meta_dataloaders, visualize_random_episodes
from model import ProtoNet, resnet, SimpleEmbedding
from train import meta_one_epoch, train, report_train_result
from callbacks import EarlyStopping
from my_util import DEVICE

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    parser = ArgumentParser(description="ProtoNet training")
    parser.add_argument("--model", type=str, help="Model Name",
                        default="meta_ghibli_resnet18")
    args = parser.parse_args()

    MODEL_NAME: str = args.model
    match = re.search(r"resnet(\d+)", MODEL_NAME)
    MODEL_SIZE = int(match.group(1)) if match else 18
    match = re.search(r"_(\d+)", MODEL_NAME)
    EMB_SIZE = int(match.group(1)) if match else 64

    BATCH_SIZE = 32
    DATASET_SIZE = 2_000
    NUM_WORKERS = os.cpu_count()

    loaders = meta_dataloaders("pairs.jsonl",
                               num_episodes=DATASET_SIZE,
                               batch_size=BATCH_SIZE,
                               num_workers=NUM_WORKERS)
    # visualize_random_episodes(loaders[0].dataset)

    embedding_net = resnet(MODEL_SIZE, EMB_SIZE).to(DEVICE)
    info = summary(embedding_net, verbose=0)
    with open(f"{MODEL_NAME}_summary.txt", "w") as f:
        f.write(str(info))

    model = ProtoNet(num_ways=2, num_shots=3,
                     embedding_net=embedding_net).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4,
                           weight_decay=1e-4)
    es = EarlyStopping(patience=12, restore_best_weights=True)
    result = train(model, loaders, criterion, optimizer,
                   meta_one_epoch, DEVICE,
                   num_epochs=10000, callbacks=[es])

    torch.save(model.state_dict(), f"{MODEL_NAME}.pth")
    print("Model saved!")
    report_train_result(result, MODEL_NAME)
