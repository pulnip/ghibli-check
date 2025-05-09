import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from datasets import Dataset as HF_Dataset
from torchvision import transforms
from torchvision.models import resnet18 as _resnet18
from torchvision.models.resnet import ResNet18_Weights
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data import random_split, RandomSampler
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import random

from my_util import get_argv

# Basic Residual Block (for ResNet-18, ResNet-34)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# ResNet Architecture
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# ResNet-18 instance
def resnet18(num_classes=2):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def torch_resnet18(num_classes=2):
    model = _resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, num_classes)

    return model

transform = transforms.Compose([
    # resnet 224x224
    transforms.RandomResizedCrop((224, 224), scale=(0.8,1.0)),   # 랜덤 크롭 + 리사이즈
    transforms.RandomHorizontalFlip(),                    # 좌우 뒤집기
    transforms.RandomRotation(20),                        # ±20도 회전
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.3, hue=0.1),      # 컬러 변형
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

class GhibliTorchDataset(Dataset):
    def __init__(self, hf_ds, transform):
        self.ds = hf_ds
        self.transform = transform
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        img = self.ds[idx]["image"]
        return {"image": self.transform(img), "label": self.ds[idx]["label"]}

class AIDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.paths = list(Path(img_dir).glob("*.*"))  # jpg/png 파일
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        # 매번 다른 augmentation이 적용됨
        return {"image": self.transform(img), "label": 0}

def get_dataloaders(ai_dir, ai_mul=2, batch_size=32, split=0.8, num_workers=4):
    real_hf = load_dataset("Nechintosh/ghibli")["train"].map(lambda x: {"label":1})
    real_ds = GhibliTorchDataset(real_hf, transform)
    ai_ds = AIDataset(ai_dir, transform)
    ai_ds = ConcatDataset([ai_ds] * ai_mul)

    full_ds = ConcatDataset([ai_ds, real_ds])

    train_size = int(split * len(full_ds))
    test_size = len(full_ds) - train_size
    train_ds, test_ds = random_split(full_ds, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    # tqdm으로 progress bar 생성
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        imgs = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        # Using Cross Entropy
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        # Using Binary Cross Entropy
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

            # Using Cross Entropy
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            # Using Binary Cross Entropy
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
    num_epochs=5, verbose=True
):
    train_loader, test_loader = loader
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if verbose:
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
            print(f"           Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc*100:.2f}%")
        else:
            print(f"Epoch {epoch} ends.")
    
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

# Visualize random batches from a loader
def visualize_random_batches(loader, num_batches=5, samples_per_batch=6):
    """
    Display a grid of randomly sampled images from the first num_batches of the loader.
    Titles indicate true label: 1 -> Real Ghibli, 0 -> AI-generated.
    """
    loader_iter = iter(loader)
    for i in range(num_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        images = batch["image"]
        labels = batch["label"]
        count = min(len(images), samples_per_batch)
        idxs = random.sample(range(len(images)), k=count)
        fig, axes = plt.subplots(2, 3, figsize=(9, 6))
        for j, ax in enumerate(axes.flatten()):
            if j < count:
                img = images[idxs[j]].cpu()
                # Unnormalize
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
                img = img.permute(1, 2, 0) * std + mean
                img = img.clamp(0, 1).numpy()
                ax.imshow(img)
                lbl = labels[idxs[j]].item()
                ax.set_title("Real Ghibli" if lbl == 1 else "AI-generated")
            ax.axis("off")
        plt.suptitle(f"Batch {i+1}")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    ai_gen_dirname = get_argv(1, "on_theme")
    model_name = get_argv(2, "resnet18_ghibli")

    device = "mps" if torch.backends.mps.is_available() else \
        "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Using Cross Entropy
    model = resnet18(num_classes=2).to(device)
    # model = torch_resnet18(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()

    # 2. Using Binary Cross Entropy
    # model = resnet18(num_classes=1).to(device)
    # model = torch_resnet18(num_classes=1).to(device)
    # criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loaders = get_dataloaders(ai_gen_dirname, ai_mul=1)
    # visualize_random_batches(loaders[0], num_batches=5)
    # quit()

    train_result = train(model, loaders)

    torch.save(model.state_dict(), f"{model_name}.pth")
    print("Model saved!")
    report_train_result(train_result, model_name)
