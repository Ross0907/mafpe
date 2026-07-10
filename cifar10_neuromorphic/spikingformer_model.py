from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import DropPath, trunc_normal_


class SpikingMlp(nn.Module):
    """Two 1x1 convolutional feed-forward layers gated by LIF spikes."""

    def __init__(self, channels: int, hidden_channels: int, backend: str) -> None:
        super().__init__()
        self.lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.conv2 = nn.Conv2d(hidden_channels, channels, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.hidden_channels = hidden_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_steps, batch, channels, height, width = x.shape
        x = self.lif1(x).flatten(0, 1)
        x = self.bn1(self.conv1(x)).reshape(time_steps, batch, self.hidden_channels, height, width)
        x = self.lif2(x).flatten(0, 1)
        x = self.bn2(self.conv2(x)).reshape(time_steps, batch, channels, height, width)
        return x


class SpikingSelfAttention(nn.Module):
    """Spiking self-attention from Spikingformer with binary Q/K/V activations."""

    def __init__(self, channels: int, heads: int, backend: str) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.pre_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.q_conv = nn.Conv1d(channels, channels, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(channels)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.k_conv = nn.Conv1d(channels, channels, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(channels)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.v_conv = nn.Conv1d(channels, channels, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(channels)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend=backend)
        self.proj_conv = nn.Conv1d(channels, channels, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_steps, batch, channels, height, width = x.shape
        x = self.pre_lif(x).flatten(3)
        tokens = x.shape[-1]
        qkv_input = x.flatten(0, 1)

        q = self.q_lif(self.q_bn(self.q_conv(qkv_input)).reshape(time_steps, batch, channels, tokens))
        k = self.k_lif(self.k_bn(self.k_conv(qkv_input)).reshape(time_steps, batch, channels, tokens))
        v = self.v_lif(self.v_bn(self.v_conv(qkv_input)).reshape(time_steps, batch, channels, tokens))

        q = q.transpose(-1, -2).reshape(time_steps, batch, tokens, self.heads, self.head_dim)
        q = q.permute(0, 1, 3, 2, 4)
        k = k.transpose(-1, -2).reshape(time_steps, batch, tokens, self.heads, self.head_dim)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.transpose(-1, -2).reshape(time_steps, batch, tokens, self.heads, self.head_dim)
        v = v.permute(0, 1, 3, 2, 4)

        attn = q @ k.transpose(-2, -1)
        x = (attn @ v) * 0.125
        x = x.transpose(3, 4).reshape(time_steps, batch, channels, tokens)
        x = self.attn_lif(x).flatten(0, 1)
        x = self.proj_bn(self.proj_conv(x)).reshape(time_steps, batch, channels, height, width)
        return x


class SpikingTransformerBlock(nn.Module):
    """Residual spiking attention plus residual spiking MLP."""

    def __init__(
        self,
        channels: int,
        heads: int,
        mlp_ratio: int,
        drop_path: float,
        backend: str,
    ) -> None:
        super().__init__()
        self.attn = SpikingSelfAttention(channels=channels, heads=heads, backend=backend)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.mlp = SpikingMlp(
            channels=channels,
            hidden_channels=int(channels * mlp_ratio),
            backend=backend,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SpikingTokenizer(nn.Module):
    """Convolutional spiking tokenizer that maps 32x32 RGB frames to 8x8 tokens."""

    def __init__(self, channels: int, embed_dim: int, backend: str) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(channels, embed_dim // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(embed_dim // 8)
        self.lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.conv1 = nn.Conv2d(embed_dim // 8, embed_dim // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(embed_dim // 4)
        self.lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.conv2 = nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(embed_dim // 2)
        self.lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(embed_dim)
        self.lif4 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_steps, batch, _, height, width = x.shape
        x = self.bn0(self.conv0(x.flatten(0, 1))).reshape(time_steps, batch, -1, height, width)
        x = self.lif1(x).flatten(0, 1)
        x = self.bn1(self.conv1(x)).reshape(time_steps, batch, -1, height, width)
        x = self.lif2(x).flatten(0, 1)
        x = self.bn2(self.conv2(x)).reshape(time_steps, batch, -1, height, width)
        x = self.lif3(x).flatten(0, 1)
        x = self.pool3(x)
        x = self.bn3(self.conv3(x)).reshape(time_steps, batch, -1, height // 2, width // 2)
        x = self.lif4(x).flatten(0, 1)
        x = self.pool4(x)
        x = self.bn4(self.conv4(x)).reshape(time_steps, batch, -1, height // 4, width // 4)
        return x


class SpikingformerCIFAR(nn.Module):
    """CIFAR-10 Spikingformer-4-384 matching the published 95.8% recipe."""

    def __init__(
        self,
        *,
        time_steps: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 384,
        heads: int = 12,
        depth: int = 4,
        mlp_ratio: int = 4,
        drop_path: float = 0.1,
        backend: str = "cupy",
    ) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.embed_dim = embed_dim
        self.depth = depth
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.tokenizer = SpikingTokenizer(channels=in_channels, embed_dim=embed_dim, backend=backend)
        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList([
            SpikingTransformerBlock(
                channels=embed_dim,
                heads=heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[idx],
                backend=backend,
            )
            for idx in range(depth)
        ])
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
            if getattr(module, "weight", None) is not None:
                nn.init.ones_(module.weight)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x)
        for block in self.blocks:
            x = block(x)
        return x.flatten(3).mean(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)
        x = self.forward_features(x)
        return self.head(x.mean(0))


def build_spikingformer_4_384(**kwargs) -> SpikingformerCIFAR:
    defaults = dict(
        time_steps=4,
        in_channels=3,
        num_classes=10,
        embed_dim=384,
        heads=12,
        depth=4,
        mlp_ratio=4,
        drop_path=0.1,
        backend="cupy",
    )
    defaults.update(kwargs)
    return SpikingformerCIFAR(**defaults)
