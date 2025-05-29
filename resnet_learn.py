import torch
import torch.nn as nn
import torch.optim as optim

from my_data import get_dataloaders, visualize_random_batches, transform, val_transform
from model import torch_resnet18, resnet18
from train import classifier_one_epoch, train, report_train_result
from callbacks import EarlyStopping
from my_util import DEVICE, get_argv

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    ai_gen_dirname = get_argv(1, "on_theme")
    model_name = get_argv(2, "resnet_ghibli")

    loaders = get_dataloaders(ai_gen_dirname, transform, ai_mul=1)

    # train_loader, _ = get_dataloaders(ai_gen_dirname, val_transform, ai_mul=1)
    # visualize_random_batches(train_loader, num_batches=5)

    model = (torch_resnet18(num_classes=2) if "torch" in model_name \
        else resnet18(num_classes=2)).to(DEVICE)
    # 1. Using Cross Entropy
    criterion = nn.CrossEntropyLoss()
    # 2. Using Binary Cross Entropy
    # criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    es = EarlyStopping(patience=3, restore_best_weights=True)
    result = train(model, loaders, criterion, optimizer,        
                   classifier_one_epoch, DEVICE,
                   num_epochs=10000, callbacks=[es])

    torch.save(model.state_dict(), f"{model_name}.pth")
    print("Model saved!")
    report_train_result(result, model_name)
