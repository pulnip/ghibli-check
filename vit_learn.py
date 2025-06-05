import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from transformers import get_cosine_schedule_with_warmup
from argparse import ArgumentParser
import re
import os

from my_data import get_dataloaders, visualize_random_batches
from model import vit
from train import classifier_one_epoch, train, report_train_result
from callbacks import EarlyStopping
from my_util import DEVICE

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    parser = ArgumentParser(description="ViT training")
    parser.add_argument("--model", type=str, help="Model Name",
                        default="vit-tiny_ghibli")
    args = parser.parse_args()

    MODEL_NAME: str = args.model
    match = re.search(r"vit-(\w+)_", MODEL_NAME)
    MODEL_SIZE = str(match.group(1)) if match else "tiny"

    BATCH_SIZE = 128
    DATASET_SIZE = 10_000
    NUM_EPOCHS = 100
    NUM_WORKERS = os.cpu_count()

    loaders = get_dataloaders(num_data=DATASET_SIZE,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS)
    # visualize_random_batches(loaders[0], num_batches=5)

    model = vit(MODEL_SIZE, num_classes=2).to(DEVICE)
    model_info = summary(model, verbose=0)
    with open(f"{MODEL_NAME}_summary.txt", "w") as f:
        f.write(str(model_info))
    # 1. Using Cross Entropy
    criterion = nn.CrossEntropyLoss()
    # 2. Using Binary Cross Entropy
    # criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-3,
                            betas=(0.9, 0.999), weight_decay=0.1)
    TOTAL_STEPS = DATASET_SIZE // BATCH_SIZE * NUM_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=10_000,
        num_training_steps=TOTAL_STEPS
    )
    es = EarlyStopping(patience=30, restore_best_weights=True)
    result = train(model, loaders, criterion, optimizer,        
                   classifier_one_epoch, DEVICE,
                   num_epochs=NUM_EPOCHS, callbacks=[es],
                   scheduler=scheduler)

    torch.save(model.state_dict(), f"{MODEL_NAME}.pth")
    print("Model saved!")
    report_train_result(result, MODEL_NAME)
