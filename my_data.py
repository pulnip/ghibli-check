from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data import random_split
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import random
import json

from my_util import get_argv

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

transform64 = transforms.Compose([
    transforms.RandomResizedCrop((64, 64), scale=(0.8,1.0)),   # 랜덤 크롭 + 리사이즈
    transforms.ToTensor(),
])

class GhibliTorchDataset(Dataset):
    def __init__(self, hf_ds, transform):
        self.ds = hf_ds
        self.transform = transform
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        img = self.ds[idx]["image"]
        # ensure the image is a PIL RGB image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
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

class ProtoEpisodeDataset(Dataset):
    def __init__(self, image_pairs, transform,
                 num_episodes=2000, K=3):
        self.transform = transform
        self.image_pairs = image_pairs

        pair_indexes = list(range(len(image_pairs)))
        self.support_indexes = set()
        while len(self.support_indexes) < num_episodes:
            # Ensure tuple for set (hashable)
            self.support_indexes.add(tuple(random.sample(pair_indexes, K)))
        half = num_episodes // 2
        self.query_labels = [0]*half + [1]*(num_episodes - half)
        random.shuffle(self.query_labels)

    def __len__(self):
        return len(self.support_indexes)

    def __getitem__(self, idx):
        # Convert set to list for indexing, and tuple to list for usage
        support_index = list(self.support_indexes)[idx]

        # support
        real = [self.transform(Image.open(self.image_pairs[i]["real_image"]).convert("RGB")) for i in support_index]
        ai = [self.transform(Image.open(self.image_pairs[i]["ai_image"]).convert("RGB")) for i in support_index]
        support_x = torch.stack(ai + real)
        support_y = torch.tensor([0]*len(ai)+[1]*len(real))

        # query
        all_indexes = list(range(len(self.image_pairs)))
        remaining_indexes = list(set(all_indexes) - set(support_index))

        query_idx = random.choice(remaining_indexes)
        label = self.query_labels[idx]
        query_path = self.image_pairs[query_idx]["real_image"] if label == 1 else self.image_pairs[query_idx]["ai_image"]
        query = self.transform(Image.open(query_path).convert("RGB"))

        return support_x, support_y, query, label

def load_datasets(paths: list[str]):
    return [
        load_dataset(
            path, split="train"
        ).map(lambda _: {"label": 1})
        for path in paths
    ]

def get_dataloaders(ai_dir, transform, ai_mul=2, batch_size=32, split=0.8, num_workers=4):
    real_hf = load_datasets(["pulnip/ghibli-dataset"])
    real_hf = ConcatDataset(real_hf)

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

def meta_dataloaders(filename: str, num_episodes=2000,
                     batch_size=32, split=0.8, num_workers=4):
    with open(filename, "r") as f:
        lines = f.readlines()
        image_pairs = [json.loads(line) for line in lines]

    full_ds = ProtoEpisodeDataset(image_pairs, transform64, num_episodes)
    train_size = int(split * len(full_ds))
    test_size = len(full_ds) - train_size
    train_ds, test_ds = random_split(full_ds, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

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

def visualize_random_episodes(dataset, num_episodes=4):
    s = len(dataset[0][0])
    fig, axes = plt.subplots(num_episodes, s+1, figsize=(1.5*s+1, 1.5 * num_episodes))
    if num_episodes == 1:
        axes = [axes]

    chosen_idxs = random.sample(range(len(dataset)), num_episodes)

    for row, idx in enumerate(chosen_idxs):
        support_x, support_y, query, label = dataset[idx]
        label_text = "AI" if label == 0 else "Real"

        for i in range(s):
            axes[row][i].imshow(F.to_pil_image(support_x[i]))
            axes[row][i].set_title("AI" if support_y[i]==0 else "Real")
            axes[row][i].axis("off")

        # query image
        axes[row][s].imshow(F.to_pil_image(query))
        axes[row][s].set_title(f"Query\n({label_text})")
        axes[row][s].axis("off")

    plt.tight_layout()
    plt.show()
