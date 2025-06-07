import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from argparse import ArgumentParser
import re
import os

from my_data import get_dataloaders, visualize_random_batches
from model import effnet
from train import classifier_one_epoch, train, report_train_result
from callbacks import EarlyStopping
from my_util import DEVICE

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    parser = ArgumentParser(description="EfficientNet training")
    parser.add_argument("--model", type=str, help="Model Name",
                        default="effnetB0_ghibli")
    args = parser.parse_args()

    MODEL_NAME: str = args.model
    match = re.search(r"effnet(\w+)_", MODEL_NAME)
    MODEL_SIZE = str(match.group(1)) if match else "B0"

    BATCH_SIZE = 64
    DATASET_SIZE = 10_000
    NUM_WORKERS = os.cpu_count()

    loaders = get_dataloaders(num_data=DATASET_SIZE,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS)
    # visualize_random_batches(loaders[0], num_batches=5)

    model = effnet(MODEL_SIZE, num_classes=2).to(DEVICE)
    model_info = summary(model, verbose=0)
    with open(f"{MODEL_NAME}_summary.txt", "w") as f:
        f.write(str(model_info))
    # 1. Using Cross Entropys
    criterion = nn.CrossEntropyLoss()
    # 2. Using Binary Cross Entropy
    # criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.256,
                              alpha=0.9, eps=1.0,
                              weight_decay=1e-5,
                              momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=0.97
    )
    es = EarlyStopping(patience=12, restore_best_weights=True)
    result = train(model, loaders, criterion, optimizer,        
                   classifier_one_epoch, DEVICE,
                   num_epochs=10000, callbacks=[es],
                   scheduler=scheduler,)

    torch.save(model.state_dict(), f"{MODEL_NAME}.pth")
    print("Model saved!")
    report_train_result(result, MODEL_NAME)
