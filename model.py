import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 as _resnet18
from torchvision.models.resnet import ResNet18_Weights
from typing import Literal

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

# Basic Patch Embedding and Transformer Encoder Block
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x)                          # (B, D, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)          # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)    # (B, 1, D)
        x = torch.cat((cls, x), dim=1)            # (B, 1+N, D)
        x = x + self.pos_embed
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # self-attention + residual
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2)
        x = x + attn_out
        # MLP + residual
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

# Vision Transformer Architecture
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_chans,
                 num_classes, embed_dim, depth, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size,
                                          in_chans, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)     # (B, 1+N, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls = x[:, 0]               # class token
        return self.head(cls)

# ViT instance
def vit(model_size: Literal["tiny", "small"], num_classes=2):
    if model_size == "tiny":
        embed_dim = 192
        num_heads = 3
    elif model_size == "small":
        embed_dim = 384
        num_heads = 6
    else:
        raise RuntimeError(f"ViT-{model_size} not implemented.")

    return VisionTransformer(
        img_size=224, patch_size=16, in_chans=3,
        num_classes=num_classes,
        embed_dim=embed_dim, depth=12, num_heads=num_heads,
        mlp_ratio=4.0, dropout=0.1
    )

if __name__ == "__main__":
    model_tiny  = vit("tiny", 1000)
    model_small = vit("small", 1000)
    dummy = torch.randn(2, 3, 224, 224)
    print(model_tiny(dummy).shape)   # torch.Size([2, 1000])
    print(model_small(dummy).shape)  # torch.Size([2, 1000])