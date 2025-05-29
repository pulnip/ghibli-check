import torch
import torch.nn as nn
import torch.optim as optim

from my_data import meta_dataloaders, visualize_random_episodes
from model import ProtoNet
from train import meta_one_epoch, train, report_train_result
from callbacks import EarlyStopping
from my_util import DEVICE

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    loaders = meta_dataloaders("pairs.jsonl")
    visualize_random_episodes(loaders[0].dataset)

    MODEL_NAME = "meta_ghibli"
    model = ProtoNet(2, 3).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    es = EarlyStopping(patience=3, restore_best_weights=True)
    result = train(model, loaders, criterion, optimizer,
                   meta_one_epoch, DEVICE,
                   num_epochs=10000, callbacks=[es])

    torch.save(model.state_dict(), f"{MODEL_NAME}.pth")
    print("Model saved!")
    report_train_result(result, MODEL_NAME)
