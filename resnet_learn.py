import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from argparse import ArgumentParser
import re
import os

from my_data import get_dataloaders, visualize_random_batches
from model import resnet, pretrained_resnet
from train import classifier_one_epoch, train, report_train_result
from callbacks import EarlyStopping
from my_util import DEVICE

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    parser = ArgumentParser(description="ResNet training")
    parser.add_argument("--model", type=str, help="Model Name",
                        default="resnet18_ghibli")
    args = parser.parse_args()

    MODEL_NAME: str = args.model
    match = re.search(r"resnet(\d+)", MODEL_NAME)
    MODEL_SIZE = int(match.group(1)) if match else 18

    BATCH_SIZE = 128
    DATASET_SIZE = 10_000
    NUM_WORKERS = os.cpu_count()

    loaders = get_dataloaders(num_data=DATASET_SIZE,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS)
    # visualize_random_batches(loaders[0], num_batches=5)

    model = resnet(MODEL_SIZE, num_classes=2).to(DEVICE)
    # model = pretrained_resnet(18, num_classes=2).to(DEVICE)
    model_info = summary(model, verbose=0)
    with open(f"{MODEL_NAME}_summary.txt", "w") as f:
        f.write(str(model_info))
    # 1. Using Cross Entropys
    criterion = nn.CrossEntropyLoss()
    # 2. Using Binary Cross Entropy
    # criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=[30, 60, 90],
        gamma=0.1
    )
    es = EarlyStopping(patience=10, restore_best_weights=True)
    result = train(model, loaders, criterion, optimizer,        
                   classifier_one_epoch, DEVICE,
                   num_epochs=10000, callbacks=[es],
                   scheduler=scheduler,)

    torch.save(model.state_dict(), f"{MODEL_NAME}.pth")
    print("Model saved!")
    report_train_result(result, MODEL_NAME)
