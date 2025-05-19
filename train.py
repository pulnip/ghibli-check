import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from my_util import get_argv
from model import resnet18, torch_resnet18, vit
from my_data import get_dataloaders, transform
from callbacks import Callback, EarlyStopping

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    # progress bar
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        imgs = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        # 1. Using Cross Entropy
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        # 2. Using Binary Cross Entropy
        # raw_outputs = model(imgs).squeeze(1)
        # outputs = torch.sigmoid(raw_outputs)
        # loss = criterion(outputs, labels.float())

        loss.backward()
        optimizer.step()

        # 통계 업데이트
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1) # Using Cross Entropy
        # preds = (outputs > 0.5).long() # Using Binary Cross Entropy
        running_correct += preds.eq(labels).sum().item()
        running_total += imgs.size(0)

        # 배치마다 loss & acc 표시
        avg_loss = running_loss / running_total
        avg_acc  = running_correct / running_total * 100
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.1f}%")

    # epoch 전체 통계 리턴
    epoch_loss = running_loss / running_total
    epoch_acc  = running_correct / running_total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Eval ", leave=False)
        for batch in pbar:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)

            # 1. Using Cross Entropy
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            # 2. Using Binary Cross Entropy
            # raw_outputs = model(imgs).squeeze(1)
            # outputs = torch.sigmoid(raw_outputs)
            # loss = criterion(outputs, labels.float())

            val_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1) # Using Cross Entropy
            # preds = (outputs > 0.5).long() # Using Binary Cross Entropy
            val_correct += preds.eq(labels).sum().item()
            val_total += imgs.size(0)

            avg_loss = val_loss / val_total
            avg_acc  = val_correct / val_total * 100
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.1f}%")

    return val_loss / val_total, val_correct / val_total

def train(model, loader: tuple[DataLoader, DataLoader],
    num_epochs=5, callbacks: list[Callback]=[], verbose=True
):
    train_loader, test_loader = loader
    logs = {}
    for cb in callbacks:
        cb.on_train_begin(logs)
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        logs_epoch = {
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc,
            'model': model
        }
        for cb in callbacks:
            cb.on_epoch_end(epoch, logs_epoch)
        if logs_epoch.get('stop_training'):
            break

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if verbose:
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
            print(f"           Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc*100:.2f}%")
        else:
            print(f"Epoch {epoch} ends.")

    for cb in callbacks:
        cb.on_train_end({})
    return train_losses, train_accs, val_losses, val_accs

def report_train_result(train_result: tuple[list, list, list, list], name: str):
    train_losses, train_accs, val_losses, val_accs = train_result
    assert len(train_losses) == len(train_accs)
    assert len(val_losses) == len(val_accs)
    assert len(train_losses) == len(val_losses)

    num_epochs = len(train_losses)

    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train")
    plt.plot(range(1, num_epochs + 1), val_losses,   label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{name} loss_over_epochs.png")
    plt.close()

    plt.figure()
    plt.plot(range(1, num_epochs + 1), [a*100 for a in train_accs], label="Train")
    plt.plot(range(1, num_epochs + 1), [a*100 for a in val_accs],   label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig(f"{name} accuracy_over_epochs.png")
    plt.close()

if __name__ == "__main__":
    ai_gen_dirname = get_argv(1, "on_theme")
    model_name = get_argv(2, "vit_ghibli")

    device = "mps" if torch.backends.mps.is_available() else \
        "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    loaders = get_dataloaders(ai_gen_dirname, transform, ai_mul=1)
    # model = (torch_resnet18(num_classes=2) if "torch" in model_name \
    #     else resnet18(num_classes=2)).to(device)
    model = vit("tiny", num_classes=2).to(device)
    
    # 1. Using Cross Entropy
    criterion = nn.CrossEntropyLoss()
    # 2. Using Binary Cross Entropy
    # criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    early_stop = EarlyStopping(patience=3, restore_best_weights=True)
    train_result = train(model, loaders, num_epochs=10000, callbacks=[early_stop])

    torch.save(model.state_dict(), f"{model_name}.pth")
    print("Model saved!")
    # report_train_result(train_result, model_name)
