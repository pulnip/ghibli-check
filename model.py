import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 as _resnet18
from torchvision.models.resnet import ResNet18_Weights
from typing import Literal, Type

# Basic Residual Block (for ResNet-18, ResNet-34)
class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels,
                 stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x: torch.Tensor):
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
    def __init__(self, block: Type[ResidualBlock], layers, num_classes=1000,
                 channels: list[int]=None):
        super(ResNet, self).__init__()
        if channels is None:
            channels = [64, 128, 256, 512]
        assert len(layers) == 4 and len(channels) == 4
        self.in_channels = channels[0]

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, channels[0],  layers[0])
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels * block.expansion, num_classes)

    def _make_layer(self, block: Type[ResidualBlock], out_channels,
                    blocks, stride=1):
        if blocks == 0:
            return nn.Identity()
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

    def forward(self, x: torch.Tensor):
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

# ResNet instance
def resnet(model_size: Literal[6, 8, 10, 14, 18], num_classes=2):
    RESNET_CONFIGS = {
        6:  {"layers": [1, 1, 0, 0]},
        8:  {"layers": [1, 1, 1, 0]},
        10: {"layers": [1, 1, 1, 1]},
        14: {"layers": [2, 1, 1, 1]},
        18: {"layers": [2, 2, 2, 2]},
        34: {"layers": [3, 4, 6, 3]},
    }
    CHANNELS = [64, 128, 256, 512]

    if model_size not in RESNET_CONFIGS:
        raise RuntimeError(f"ResNet-{model_size} not implemented.")
    config = RESNET_CONFIGS[model_size]

    return ResNet(ResidualBlock, layers=config["layers"],
                  num_classes=num_classes, channels=CHANNELS)

def pretrained_resnet(model_size: Literal[6, 8, 10, 14, 18], num_classes=2):
    if model_size == 18:
        model = _resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        raise RuntimeError(f"ResNet-{model_size} not provided.")

    model.fc = nn.Linear(512, num_classes)
    return model

# Squeeze and Excitation Block (for EfficientNet)
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super(SqueezeExcitation, self).__init__()
        squeezed_channels = max(1, int(in_channels * se_ratio))
        self.fc1 = nn.Conv2d(in_channels, squeezed_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(squeezed_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # Global Average Pooling
        se = F.adaptive_avg_pool2d(x, 1)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se

# Inverted Residual + SE + Skip (for EfficientNet)
class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio, stride, kernel_size=3, se_ratio=0.25):
        super(MBConv, self).__init__()
        mid_ch = in_ch * expand_ratio
        self.use_residual = (in_ch == out_ch and stride == 1)

        # 1) Expand phase (pointwise conv)
        self.expand_conv = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(mid_ch)

        # 2) Depthwise conv
        self.dw_conv = nn.Conv2d(mid_ch, mid_ch, kernel_size=kernel_size,
                                 stride=stride, padding=kernel_size//2,
                                 groups=mid_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)

        # 3) Squeeze-and-Excitation
        self.se = SqueezeExcitation(mid_ch, se_ratio=se_ratio)

        # 4) Project phase (pointwise conv)
        self.project_conv = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        out = F.relu6(self.bn0(self.expand_conv(x)))
        out = F.relu6(self.bn1(self.dw_conv(out)))
        out = self.se(out)
        out = self.bn2(self.project_conv(out))
        if self.use_residual:
            return x + out
        else:
            return out

# EfficientNet Architecture
class EfficientNet(nn.Module):
    def __init__(self, stage_params: tuple[int, int, int, int, int],
                 num_classes=2):
        super(EfficientNet, self).__init__()
        # Stem: extend channel to 3, and downsample(stride=2)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        blocks = []
        for in_ch, out_ch, exp, n, s in stage_params:
            # stride = s for first block
            blocks.append(MBConv(in_ch, out_ch, expand_ratio=exp, stride=s))
            # stride = 1 for other block
            for _ in range(1, n):
                blocks.append(MBConv(out_ch, out_ch, expand_ratio=exp, stride=1))
        self.blocks = nn.Sequential(*blocks)

        # Head: extend out_channel to 1280, pooling and fc
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# EfficientNet instance
def effnet(model_size: Literal["BO", "B1", "B2", "B3", "B4", "B5", "B6"],
           num_classes=2):
    # MBConv Stage, (in_ch, out_ch, expand_ratio, repeat, stride)
    EFFNET_CONFIGS = {
        "B0": {
            "stage_params": [
                ( 32,  16, 1, 1, 1),
                ( 16,  24, 6, 2, 2),
                ( 24,  40, 6, 2, 2),
                ( 40,  80, 6, 3, 2),
                ( 80, 112, 6, 3, 1),
                (112, 192, 6, 4, 2),
                (192, 320, 6, 1, 1),
            ]
        }
    }
    if model_size not in EFFNET_CONFIGS:
        raise RuntimeError(f"EfficientNet-{model_size} not implemented.")
    config = EFFNET_CONFIGS[model_size]

    return EfficientNet(stage_params=config["stage_params"],
                        num_classes=num_classes)

# Basic Patch Embedding and Transformer Encoder Block
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        x = self.proj(x)                          # (B, D, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)          # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)    # (B, 1, D)
        x = torch.cat((cls, x), dim=1)            # (B, 1+N, D)
        x = x + self.pos_embed
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
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
    def __init__(self, img_size, patch_size, in_chans, num_classes,
                 embed_dim, depth, num_heads, mlp_ratio, dropout):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size,
                                          in_chans, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)     # (B, 1+N, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls = x[:, 0]               # class token
        return self.head(cls)

# ViT instance
def vit(model_size: Literal["tiny", "small", "base", "large"], num_classes=2):
    VIT_CONFIGS = {
        "tiny": {
            "embed_dim": 192,
            "num_heads": 3,
            "depth": 12,
        },
        "small": {
            "embed_dim": 384,
            "num_heads": 6,
            "depth": 12,
        },
        "base": {
            "embed_dim": 768,
            "num_heads": 12,
            "depth": 12,
        },
        "large": {
            "embed_dim": 1024,
            "num_heads": 16,
            "depth": 24,
        },
    }
    if model_size not in VIT_CONFIGS:
        raise RuntimeError(f"ViT-{model_size} not implemented.")
    config = VIT_CONFIGS[model_size]

    return VisionTransformer(
        img_size=224, patch_size=16, in_chans=3,
        num_classes=num_classes, embed_dim=config["embed_dim"],
        depth=config["depth"], num_heads=config["num_heads"],
        mlp_ratio=4.0, dropout=0.1
    )

class SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=1, momentum=1.0):
        super(SimpleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels,
                                 momentum=momentum,
                                 track_running_stats=False)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        out = self.relu(self.bn(self.conv(x)))
        out = self.maxpool(out)
        return out

class SimpleEmbedding(nn.Module):
    def __init__(self, block: Type[nn.Module]=SimpleBlock, num_classes=64,
                 channels: list[int]=None):
        super(SimpleEmbedding, self).__init__()
        if channels is None:
            channels = [3, 64, 64, 64, 64]
        assert len(channels) == 5
        self.layer1 = block(in_channels=channels[0],
                                  out_channels=channels[1])
        self.layer2 = block(in_channels=channels[1],
                                  out_channels=channels[2])
        self.layer3 = block(in_channels=channels[2],
                                  out_channels=channels[3])
        self.layer4 = block(in_channels=channels[3],
                                  out_channels=channels[4])
        self.fc = nn.Linear(in_features=channels[4]*channels[4],
                            out_features=num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        out = torch.flatten(out, 1)
        return out

class ProtoNet(nn.Module):
    def __init__(self, num_ways: int, num_shots: int,
                 embedding_net: Type[nn.Module]=None, in_channels=3):
        super(ProtoNet, self).__init__()
        if embedding_net is None:
            embedding_net = SimpleEmbedding(SimpleBlock, num_classes=64)
        self.in_channels = in_channels
        self.emb_size = 64
        self.num_ways = num_ways
        self.num_support = num_ways * num_shots
        self.num_query = self.num_support

        self.embedding_net = embedding_net

    def get_prototypes(self, embeddings: torch.Tensor,
                       targets: torch.Tensor) -> torch.Tensor:
        B, _, D = embeddings.shape
        prototypes = []

        for way in range(self.num_ways):
            # extract way class
            mask = (targets == way).unsqueeze(-1).expand(-1, -1, D)  # (B, N, D)
            selected = embeddings[mask].view(B, -1, D)  # (B, num_shots, D)
            proto = selected.mean(dim=1)  # (B, D)
            prototypes.append(proto.unsqueeze(1))  # (B, 1, D)

        return torch.cat(prototypes, dim=1)  # (B, num_ways, D)

    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor,
                query: torch.Tensor) -> torch.Tensor:
        if support_x.ndim == 4:
            support_x = support_x.unsqueeze(0)      # (1, N, C, H, W)
        if support_y.ndim == 1:
            support_y = support_y.unsqueeze(0)      # (1, N)
        if query.ndim == 3:
            query = query.unsqueeze(0).unsqueeze(0) # (1, 1, C, H, W)
        elif query.ndim == 4:
            query = query.unsqueeze(0)              # (1, M, C, H, W)
        B, N, C, H, W = support_x.shape
        _, M, _, _, _ = query.shape
        # Flatten support set to (B*N, C, H, W)
        support_x_flat = support_x.view(B*N, C, H, W)
        support_emb = self.embedding_net(support_x_flat).view(B, N, -1)
        # Flatten queries to (B*M, C, H, W)
        query_flat = query.view(B * M, C, H, W)
        query_emb = self.embedding_net(query_flat).view(B, M, -1)

        proto_emb = self.get_prototypes(support_emb, support_y)
        distance = torch.sum(
            (query_emb.unsqueeze(2)-proto_emb.unsqueeze(1))**2, dim=-1)
        distance = distance if B != 1 else distance.squeeze(0)
        logits = -distance

        return logits if logits.size(1) != 1 else logits.squeeze(1)

if __name__ == "__main__":
    model_tiny  = vit("tiny", 1000)
    model_small = vit("small", 1000)
    dummy = torch.randn(2, 3, 224, 224)
    print(model_tiny(dummy).shape)   # torch.Size([2, 1000])
    print(model_small(dummy).shape)  # torch.Size([2, 1000])