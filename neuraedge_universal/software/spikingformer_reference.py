"""Checkpoint-faithful Spikingformer inference and hardware preparation.

The upstream CIFAR-10 implementation fixes every neuron to a four-step,
hard-reset leaky integrate-and-fire (LIF) model.  This module reproduces that
inference graph with ordinary PyTorch operators.  It deliberately avoids a
CuPy dependency so the reference remains executable on CPU and on any CUDA
device supported by PyTorch.

Module and parameter names match the upstream checkpoint exactly.  A strict
load therefore detects missing, unexpected, or shape-incompatible tensors
before any accuracy or hardware-equivalence result is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn


TensorObserver = Callable[[str, torch.Tensor, bool], None]


class MultiStepHardResetLIF(nn.Module):
    """Exact inference equation of the upstream ``MultiStepLIFNode``.

    With the checkpoint's ``tau=2``, ``decay_input=True`` and ``v_reset=0``,
    each timestep first computes ``v_pre = v + (x - v) / 2``.  A spike is one
    where ``v_pre >= threshold``.  Spiking lanes are then reset to zero while
    all other lanes retain ``v_pre`` for the next timestep.

    The state is intentionally local to one call.  This is equivalent to the
    upstream validation loop, which invokes ``functional.reset_net`` after
    every image batch.
    """

    def __init__(self, *, tau: float = 2.0, threshold: float = 1.0) -> None:
        super().__init__()
        if tau <= 1.0:
            raise ValueError("tau must be greater than one")
        self.tau = float(tau)
        self.threshold = float(threshold)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim < 2:
            raise ValueError("a multi-step LIF input must have shape [T, ...]")
        voltage = torch.zeros_like(sequence[0])
        spikes: list[torch.Tensor] = []
        for timestep in range(sequence.shape[0]):
            voltage = voltage + (sequence[timestep] - voltage) / self.tau
            spike = (voltage >= self.threshold).to(sequence.dtype)
            voltage = voltage * (1.0 - spike)
            spikes.append(spike)
        return torch.stack(spikes, dim=0)


class MLP(nn.Module):
    """Spikingformer's two 1x1-convolution feed-forward projections."""

    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.mlp1_lif = MultiStepHardResetLIF()
        self.mlp1_conv = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_channels)
        self.mlp2_lif = MultiStepHardResetLIF()
        self.mlp2_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.mlp2_bn = nn.BatchNorm2d(channels)
        self.hidden_channels = hidden_channels

    def forward(
        self,
        value: torch.Tensor,
        *,
        prefix: str,
        observer: TensorObserver | None,
    ) -> torch.Tensor:
        time_steps, batch, channels, height, width = value.shape
        value = self.mlp1_lif(value)
        _observe(observer, f"{prefix}.mlp1_lif", value, True)
        value = self.mlp1_conv(value.flatten(0, 1))
        value = self.mlp1_bn(value).reshape(
            time_steps, batch, self.hidden_channels, height, width)
        _observe(observer, f"{prefix}.mlp1_bn", value, False)
        value = self.mlp2_lif(value)
        _observe(observer, f"{prefix}.mlp2_lif", value, True)
        value = self.mlp2_conv(value.flatten(0, 1))
        value = self.mlp2_bn(value).reshape(
            time_steps, batch, channels, height, width)
        _observe(observer, f"{prefix}.mlp2_bn", value, False)
        return value


class SpikingSelfAttention(nn.Module):
    """Binary Q/K/V self-attention used by the trained checkpoint."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.proj_lif = MultiStepHardResetLIF()
        self.q_conv = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.q_bn = nn.BatchNorm1d(channels)
        self.q_lif = MultiStepHardResetLIF()
        self.k_conv = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.k_bn = nn.BatchNorm1d(channels)
        self.k_lif = MultiStepHardResetLIF()
        self.v_conv = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.v_bn = nn.BatchNorm1d(channels)
        self.v_lif = MultiStepHardResetLIF()
        self.attn_lif = MultiStepHardResetLIF(threshold=0.5)
        self.proj_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.proj_bn = nn.BatchNorm1d(channels)

    def forward(
        self,
        value: torch.Tensor,
        *,
        prefix: str,
        observer: TensorObserver | None,
    ) -> torch.Tensor:
        time_steps, batch, channels, height, width = value.shape
        value = self.proj_lif(value)
        _observe(observer, f"{prefix}.proj_lif", value, True)
        value = value.flatten(3)
        tokens = value.shape[-1]
        projection_input = value.flatten(0, 1)

        query = self.q_bn(self.q_conv(projection_input)).reshape(
            time_steps, batch, channels, tokens)
        query = self.q_lif(query)
        _observe(observer, f"{prefix}.q_lif", query, True)
        key = self.k_bn(self.k_conv(projection_input)).reshape(
            time_steps, batch, channels, tokens)
        key = self.k_lif(key)
        _observe(observer, f"{prefix}.k_lif", key, True)
        value_projection = self.v_bn(self.v_conv(projection_input)).reshape(
            time_steps, batch, channels, tokens)
        value_projection = self.v_lif(value_projection)
        _observe(observer, f"{prefix}.v_lif", value_projection, True)

        query = _split_heads(query, self.heads)
        key = _split_heads(key, self.heads)
        value_projection = _split_heads(value_projection, self.heads)
        attention = query @ key.transpose(-2, -1)
        _observe(observer, f"{prefix}.qk", attention, False)
        value = (attention @ value_projection) * 0.125
        _observe(observer, f"{prefix}.attention_value", value, False)
        value = value.transpose(3, 4).reshape(
            time_steps, batch, channels, tokens)
        value = self.attn_lif(value)
        _observe(observer, f"{prefix}.attn_lif", value, True)
        value = self.proj_bn(self.proj_conv(value.flatten(0, 1)))
        value = value.reshape(time_steps, batch, channels, height, width)
        _observe(observer, f"{prefix}.proj_bn", value, False)
        return value


class SpikingTransformer(nn.Module):
    """One residual attention block followed by one residual spiking MLP."""

    def __init__(self, channels: int, heads: int, mlp_ratio: int) -> None:
        super().__init__()
        # The upstream model creates these LayerNorm parameters but does not
        # apply them in forward.  Keeping them is required for strict loading.
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.attn = SpikingSelfAttention(channels, heads)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = MLP(channels, channels * mlp_ratio)

    def forward(
        self,
        value: torch.Tensor,
        *,
        prefix: str,
        observer: TensorObserver | None,
    ) -> torch.Tensor:
        value = value + self.attn(
            value, prefix=f"{prefix}.attn", observer=observer)
        _observe(observer, f"{prefix}.attention_residual", value, False)
        value = value + self.mlp(
            value, prefix=f"{prefix}.mlp", observer=observer)
        _observe(observer, f"{prefix}.mlp_residual", value, False)
        return value


class SpikingTokenizer(nn.Module):
    """Five 3x3 convolutions that convert 32x32 RGB frames to 8x8 tokens."""

    def __init__(self, input_channels: int, embed_dim: int) -> None:
        super().__init__()
        widths = (embed_dim // 8, embed_dim // 4, embed_dim // 2, embed_dim)
        self.block0_conv = nn.Conv2d(
            input_channels, widths[0], kernel_size=3, padding=1, bias=False)
        self.block0_bn = nn.BatchNorm2d(widths[0])
        self.block1_lif = MultiStepHardResetLIF()
        self.block1_conv = nn.Conv2d(
            widths[0], widths[1], kernel_size=3, padding=1, bias=False)
        self.block1_bn = nn.BatchNorm2d(widths[1])
        self.block2_lif = MultiStepHardResetLIF()
        self.block2_conv = nn.Conv2d(
            widths[1], widths[2], kernel_size=3, padding=1, bias=False)
        self.block2_bn = nn.BatchNorm2d(widths[2])
        self.block3_lif = MultiStepHardResetLIF()
        self.block3_mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block3_conv = nn.Conv2d(
            widths[2], widths[3], kernel_size=3, padding=1, bias=False)
        self.block3_bn = nn.BatchNorm2d(widths[3])
        self.block4_lif = MultiStepHardResetLIF()
        self.block4_mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block4_conv = nn.Conv2d(
            widths[3], widths[3], kernel_size=3, padding=1, bias=False)
        self.block4_bn = nn.BatchNorm2d(widths[3])

    def forward(
        self,
        value: torch.Tensor,
        *,
        observer: TensorObserver | None,
    ) -> torch.Tensor:
        time_steps, batch, _, height, width = value.shape
        value = self.block0_bn(self.block0_conv(value.flatten(0, 1)))
        value = value.reshape(time_steps, batch, -1, height, width)
        _observe(observer, "patch_embed.block0_bn", value, False)

        value = self.block1_lif(value)
        _observe(observer, "patch_embed.block1_lif", value, True)
        value = self.block1_bn(self.block1_conv(value.flatten(0, 1)))
        value = value.reshape(time_steps, batch, -1, height, width)
        _observe(observer, "patch_embed.block1_bn", value, False)

        value = self.block2_lif(value)
        _observe(observer, "patch_embed.block2_lif", value, True)
        value = self.block2_bn(self.block2_conv(value.flatten(0, 1)))
        value = value.reshape(time_steps, batch, -1, height, width)
        _observe(observer, "patch_embed.block2_bn", value, False)

        value = self.block3_lif(value).flatten(0, 1)
        _observe(observer, "patch_embed.block3_lif", value, True)
        value = self.block3_mp(value)
        value = self.block3_bn(self.block3_conv(value))
        value = value.reshape(time_steps, batch, -1, height // 2, width // 2)
        _observe(observer, "patch_embed.block3_bn", value, False)

        value = self.block4_lif(value).flatten(0, 1)
        _observe(observer, "patch_embed.block4_lif", value, True)
        value = self.block4_mp(value)
        value = self.block4_bn(self.block4_conv(value))
        value = value.reshape(time_steps, batch, -1, height // 4, width // 4)
        _observe(observer, "patch_embed.block4_bn", value, False)
        return value


class Spikingformer(nn.Module):
    """The exact Spikingformer-4-384 graph used by the CIFAR-10 run."""

    def __init__(
        self,
        *,
        time_steps: int = 4,
        input_channels: int = 3,
        classes: int = 10,
        embed_dim: int = 384,
        heads: int = 12,
        depth: int = 4,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.patch_embed = SpikingTokenizer(input_channels, embed_dim)
        self.block = nn.ModuleList([
            SpikingTransformer(embed_dim, heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.head = nn.Linear(embed_dim, classes)

    def forward(
        self,
        image: torch.Tensor,
        observer: TensorObserver | None = None,
    ) -> torch.Tensor:
        value = image.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)
        _observe(observer, "input_repeated", value, False)
        value = self.patch_embed(value, observer=observer)
        for index, transformer in enumerate(self.block):
            value = transformer(
                value, prefix=f"block.{index}", observer=observer)
        value = value.flatten(3).mean(3)
        _observe(observer, "global_average_pool", value, False)
        value = self.head(value.mean(0))
        _observe(observer, "head", value, False)
        return value


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Strictly loaded model plus provenance stored by the training run."""

    model: Spikingformer
    epoch: int
    metric: float
    architecture: str
    checkpoint_path: Path


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    """Load the trained checkpoint and reject every state-dictionary mismatch."""

    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint: Mapping[str, Any] = torch.load(path, map_location="cpu")
    arguments = checkpoint.get("args")
    if arguments is None:
        raise ValueError("checkpoint does not contain its training arguments")
    model = Spikingformer(
        time_steps=int(getattr(arguments, "time_step")),
        input_channels=3,
        classes=int(getattr(arguments, "num_classes")),
        embed_dim=int(getattr(arguments, "dim")),
        heads=int(getattr(arguments, "num_heads")),
        depth=int(getattr(arguments, "depths", getattr(arguments, "layer"))),
        mlp_ratio=int(getattr(arguments, "mlp_ratio")),
    )
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint does not contain a state_dict mapping")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return LoadedCheckpoint(
        model=model,
        epoch=int(checkpoint.get("epoch", -1)),
        metric=float(checkpoint.get("metric", float("nan"))),
        architecture=str(checkpoint.get("arch", "unknown")),
        checkpoint_path=path,
    )


def fold_batch_norm(
    convolution: nn.Conv1d | nn.Conv2d,
    batch_norm: nn.BatchNorm1d | nn.BatchNorm2d,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return convolution weight/bias tensors with evaluation BatchNorm folded.

    For each output channel ``o``, BatchNorm applies
    ``gamma[o] * (y - mean[o]) / sqrt(var[o] + eps) + beta[o]``.
    Multiplying the convolution kernel and shifting its bias by those constants
    produces exactly the same evaluation result up to floating-point rounding.
    """

    if convolution.out_channels != batch_norm.num_features:
        raise ValueError("convolution and BatchNorm output widths differ")
    weight = convolution.weight.detach()
    if convolution.bias is None:
        bias = torch.zeros(
            convolution.out_channels, dtype=weight.dtype, device=weight.device)
    else:
        bias = convolution.bias.detach()
    scale = batch_norm.weight.detach() / torch.sqrt(
        batch_norm.running_var.detach() + batch_norm.eps)
    reshape = (scale.shape[0],) + (1,) * (weight.ndim - 1)
    folded_weight = weight * scale.reshape(reshape)
    folded_bias = (bias - batch_norm.running_mean.detach()) * scale
    folded_bias = folded_bias + batch_norm.bias.detach()
    return folded_weight, folded_bias


def _split_heads(value: torch.Tensor, heads: int) -> torch.Tensor:
    time_steps, batch, channels, tokens = value.shape
    head_dim = channels // heads
    value = value.transpose(-1, -2)
    value = value.reshape(time_steps, batch, tokens, heads, head_dim)
    return value.permute(0, 1, 3, 2, 4)


def _observe(
    observer: TensorObserver | None,
    name: str,
    value: torch.Tensor,
    is_spike: bool,
) -> None:
    if observer is not None:
        observer(name, value, is_spike)
