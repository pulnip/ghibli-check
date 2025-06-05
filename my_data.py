from datasets import load_dataset
from datasets import Dataset as HF_Dataset
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
from PIL.Image import Image as PIL_Image
import matplotlib.pyplot as plt
import random
import json
from collections import Counter
import numpy as np
from typing import Any
from tqdm import tqdm

train_transform = transforms.Compose([
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
    def __init__(self, ds: HF_Dataset, num_data: int,
                 train=True, pbar: tqdm = None):
        super(Dataset, self).__init__()
        self.samples: list[dict[str, Any]] = []

        class_counts: dict[int, int] = dict(Counter(ds["label"]))
        per_class = num_data // len(class_counts)

        aug_info: dict[int, list[int]] = {}
        for cls, count in class_counts.items():
            remaining_indexes = random.sample(range(count), per_class%count)
            inhomogeneity = [1 if i in remaining_indexes else 0 for i in range(count)]
            # for class balanced uniform sampling
            homogeneity = [per_class // count] * count
            aug_info[cls] =  np.add(homogeneity, inhomogeneity)

            class_counts: dict[int, int] = dict(Counter(ds["label"]))

        def ensure_pil_rgb(img):
            if not isinstance(img, PIL_Image):
                img = Image.fromarray(img, mode="RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")
            return img

        is_shared_pbar = pbar is not None
        pbar = pbar if is_shared_pbar \
                    else tqdm(total=num_data,
                              desc=f"Augmenting {"Train" if train else "Validate"} samples")
        transform = train_transform if train else val_transform
        for cls, aug_counts in aug_info.items():
            cls_indexes = [i for i, lbl in enumerate(ds["label"]) if lbl == cls]
            for i, num_aug in zip(cls_indexes, aug_counts):
                for j in range(num_aug):
                    img = ensure_pil_rgb(ds[i]["image"])
                    self.samples.append({"label": cls,
                                         "image": transform(img),})
                    pbar.update(1)
        if not is_shared_pbar:
            pbar.close()

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        return self.samples[index]

# support_x, support_y, query, label
Episode = tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]

class ProtoEpisodeDataset(Dataset):
    def __init__(self, image_pair_paths, transform,
                 num_episodes=2000, K=3):
        super(ProtoEpisodeDataset, self).__init__()
        self.episodes: list[Episode] = []

        pair_indexes = list(range(len(image_pair_paths)))
        support_indexes = set()
        while len(support_indexes) < num_episodes:
            # Ensure tuple for set (hashable)
            support_indexes.add(tuple(random.sample(pair_indexes, K)))

        for support_index in support_indexes:
            support_ai   = [transform(Image.open(image_pair_paths[i]["ai_image"]).convert("RGB"))
                            for i in support_index]
            support_real = [transform(Image.open(image_pair_paths[i]["real_image"]).convert("RGB"))
                            for i in support_index]
            support_x = torch.stack(support_ai+support_real)
            support_y = torch.tensor([0]*K + [1]*K)

            remaining_indexes = list(set(pair_indexes) - set(support_index))
            query_index = random.choice(remaining_indexes)
            label = random.randint(0, 1)
            key = "ai_image" if label == 0 else "real_image"

            query = val_transform(Image.open(image_pair_paths[query_index][key]).convert("RGB")).unsqueeze(0)

            self.episodes.append((support_x, support_y, query, label))

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        return self.episodes[index]

def get_dataloaders(num_data=4000, split=0.8,
                    batch_size=32, num_workers=4):
    ds = load_dataset("pulnip/ghibli-dataset", split="train")
    ds = ds.map(lambda batch: {"label": [1 if label=="real" else 0
                                         for label in batch["label"]],},
                remove_columns = [col for col in ds.column_names 
                                 if col not in ["image", "label"]],
                batched=True)

    pbar = tqdm(total=num_data, desc="Augmenting samples")
    train_size = int(split * num_data)
    val_size   = num_data - train_size
    train_ds = GhibliTorchDataset(ds=ds, num_data=train_size,
                                  train=True, pbar=pbar)
    val_ds   = GhibliTorchDataset(ds=ds, num_data=val_size,
                                  train=True, pbar=pbar)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

def meta_dataloaders(filename: str, num_episodes=2000,
                     batch_size=32, split=0.8, num_workers=4):
    with open(filename, "r") as f:
        lines = f.readlines()
        image_pairs = [json.loads(line) for line in lines]

    full_ds = ProtoEpisodeDataset(image_pairs, train_transform, num_episodes)
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
