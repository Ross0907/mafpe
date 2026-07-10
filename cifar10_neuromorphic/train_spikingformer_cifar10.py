from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path


def early_setup_cuda_runtime() -> None:
    root = Path(__file__).resolve().parents[1]
    site = root / ".venv" / "Lib" / "site-packages"
    cuda_runtime = site / "nvidia" / "cuda_runtime"
    cuda_nvrtc = site / "nvidia" / "cuda_nvrtc"
    torch_lib = site / "torch" / "lib"
    if cuda_runtime.exists():
        os.environ.setdefault("CUDA_PATH", str(cuda_runtime))
    for path in (cuda_runtime / "bin", cuda_nvrtc / "bin", torch_lib):
        if path.exists():
            try:
                os.add_dll_directory(str(path))
            except (AttributeError, OSError):
                pass
            os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")


early_setup_cuda_runtime()

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.data import create_transform
from timm.data.mixup import Mixup
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from spikingformer_model import build_spikingformer_4_384


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
E_MAC_PJ = 4.6
E_AC_PJ = 0.9


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def setup_cuda_runtime() -> None:
    site = repo_root() / ".venv" / "Lib" / "site-packages"
    cuda_runtime = site / "nvidia" / "cuda_runtime"
    cuda_nvrtc = site / "nvidia" / "cuda_nvrtc"
    torch_lib = site / "torch" / "lib"
    if cuda_runtime.exists():
        os.environ.setdefault("CUDA_PATH", str(cuda_runtime))
    for path in (cuda_runtime / "bin", cuda_nvrtc / "bin", torch_lib):
        if path.exists():
            try:
                os.add_dll_directory(str(path))
            except (AttributeError, OSError):
                pass
            os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate Spikingformer-4-384 on CIFAR-10.")
    parser.add_argument("--data-dir", type=Path, default=repo_root() / "datasets")
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "cifar10_neuromorphic" / "outputs_spikingformer_official")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=6e-2)
    parser.add_argument("--mixup", type=float, default=0.5)
    parser.add_argument("--mixup-off-epoch", type=int, default=200)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--auto-augment", type=str, default="rand-m9-n1-mstd0.4-inc1")
    parser.add_argument("--color-jitter", type=float, default=0.0)
    parser.add_argument("--interpolation", type=str, default="bicubic")
    parser.add_argument("--randaugment-ops", type=int, default=1)
    parser.add_argument("--randaugment-mag", type=int, default=9)
    parser.add_argument("--random-erasing", type=float, default=0.25)
    parser.add_argument("--random-erasing-mode", type=str, default="const")
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--drop-path", type=float, default=0.2)
    parser.add_argument("--backend", choices=["cupy", "torch"], default="cupy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--spike-rate-batches", type=int, default=8)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--pause-file", type=Path, default=None)
    parser.add_argument("--archive-every-epochs", type=int, default=25)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_transform = create_transform(
        input_size=(3, 32, 32),
        is_training=True,
        scale=(1.0, 1.0),
        ratio=(1.0, 1.0),
        hflip=0.5,
        vflip=0.0,
        color_jitter=args.color_jitter,
        auto_augment=args.auto_augment,
        interpolation=args.interpolation,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
        re_prob=args.random_erasing,
        re_mode=args.random_erasing_mode,
        re_count=1,
    )
    test_transform = create_transform(
        input_size=(3, 32, 32),
        is_training=False,
        interpolation=args.interpolation,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
        crop_pct=1.0,
    )
    train_set = datasets.CIFAR10(root=str(args.data_dir), train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=str(args.data_dir), train=False, download=True, transform=test_transform)
    if args.train_limit > 0:
        train_set = Subset(train_set, range(min(args.train_limit, len(train_set))))
    if args.test_limit > 0:
        test_set = Subset(test_set, range(min(args.test_limit, len(test_set))))
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    return train_loader, test_loader


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float, classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        return x, F.one_hot(y, classes).float()
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.shape[0], device=x.device)
    mixed = lam * x + (1.0 - lam) * x[index]
    y1 = F.one_hot(y, classes).float()
    y2 = F.one_hot(y[index], classes).float()
    return mixed, lam * y1 + (1.0 - lam) * y2


def soft_target_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sum(-target * F.log_softmax(logits, dim=-1), dim=-1).mean()


def cosine_lr(epoch: int, args: argparse.Namespace) -> float:
    if epoch < args.warmup_epochs:
        return args.lr * float(epoch + 1) / float(max(1, args.warmup_epochs))
    progress = (epoch - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    backup = path.with_suffix(path.suffix + ".prev")
    if path.exists():
        backup_tmp = backup.with_suffix(backup.suffix + ".tmp")
        shutil.copy2(path, backup_tmp)
        os.replace(backup_tmp, backup)
    with tmp.open("wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return nullcontext()


def build_model(args: argparse.Namespace) -> nn.Module:
    return build_spikingformer_4_384(
        time_steps=args.time_steps,
        embed_dim=args.embed_dim,
        heads=args.heads,
        depth=args.depth,
        mlp_ratio=args.mlp_ratio,
        drop_path=args.drop_path,
        backend=args.backend,
    )


def resolve_resume_candidates(path: Path | None, out_dir: Path) -> list[Path]:
    if path is not None and str(path).lower() != "auto":
        return [path]
    candidates = [
        out_dir / "latest_checkpoint.pt",
        out_dir / "latest_checkpoint.pt.prev",
        out_dir / "best_checkpoint.pt",
        out_dir / "best_checkpoint.pt.prev",
    ]
    archives = sorted(out_dir.glob("checkpoint_epoch_*.pt"), reverse=True)
    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates + archives:
        resolved = candidate.resolve()
        if candidate.exists() and resolved not in seen:
            ordered.append(candidate)
            seen.add(resolved)
    return ordered


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    cleaned = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    restore_rng_state(ckpt.get("rng"))
    return int(ckpt.get("epoch", -1)) + 1, float(ckpt.get("best_acc", 0.0))


def load_first_valid_checkpoint(
    candidates: list[Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, float, Path | None, list[str]]:
    failures: list[str] = []
    for candidate in candidates:
        try:
            start_epoch, best_acc = load_checkpoint(candidate, model, optimizer, scaler)
            return start_epoch, best_acc, candidate, failures
        except Exception as exc:
            failures.append(f"{candidate}: {type(exc).__name__}: {exc}")
            print(f"checkpoint_unusable={candidate} reason={type(exc).__name__}: {exc}", flush=True)
    return 0, 0.0, None, failures


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_acc: float,
    args: argparse.Namespace,
    status: str,
) -> None:
    serializable_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    atomic_torch_save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_acc": best_acc,
            "args": serializable_args,
            "status": status,
            "rng": rng_state(),
        },
        path,
    )


def write_run_state(out_dir: Path, state: dict) -> None:
    payload = state.copy()
    payload["updated_at_unix"] = time.time()
    tmp = out_dir / "run_state.json.tmp"
    state_path = out_dir / "run_state.json"
    backup = out_dir / "run_state.json.prev"
    if state_path.exists():
        backup_tmp = out_dir / "run_state.json.prev.tmp"
        shutil.copy2(state_path, backup_tmp)
        os.replace(backup_tmp, backup)
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, state_path)


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return f'"{pid}"' in result.stdout or f",{pid}," in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def guard_single_active_run(out_dir: Path) -> None:
    state_path = out_dir / "run_state.json"
    if not state_path.exists():
        return
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    status = str(state.get("status", ""))
    pid = int(state.get("pid", 0) or 0)
    active_status = status in {"running", "running_eval"}
    if active_status and pid != os.getpid() and pid_is_alive(pid):
        raise RuntimeError(
            "Another Spikingformer job is already active for this output directory "
            f"(pid={pid}, status={status}). Pause it with pause_spikingformer.ps1 or wait for it before resuming."
        )


def load_existing_history(csv_path: Path, start_epoch: int) -> list[dict[str, float]]:
    if not csv_path.exists() or start_epoch <= 0:
        return []
    rows: list[dict[str, float]] = []
    text = csv_path.read_text(encoding="utf-8", errors="ignore").replace("\x00", "")
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        try:
            epoch = int(float(row["epoch"]))
            parsed = {key: float(value) for key, value in row.items()}
        except (KeyError, TypeError, ValueError):
            continue
        if epoch <= start_epoch:
            rows.append(parsed)
    return rows


def rewrite_history(csv_path: Path, rows: list[dict[str, float]]) -> None:
    fields = ["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_top1", "test_top5", "seconds"]
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, csv_path)


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    _, pred = logits.topk(5, dim=1)
    correct = pred.eq(target.view(-1, 1))
    top1 = correct[:, :1].any(dim=1).float().mean().item() * 100.0
    top5 = correct.any(dim=1).float().mean().item() * 100.0
    return top1, top5


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    use_mixup = args.mixup > 0 and epoch < args.mixup_off_epoch
    mixup_fn = Mixup(
        mixup_alpha=args.mixup,
        cutmix_alpha=0.0,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=args.label_smoothing,
        num_classes=10,
    ) if use_mixup else None
    loss_sum = 0.0
    seen = 0
    correct = 0
    pbar = tqdm(loader, desc=f"train {epoch + 1}", leave=False, disable=not args.progress)
    for batch_idx, (x, y) in enumerate(pbar):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if mixup_fn is not None:
            x, y_soft = mixup_fn(x, y)
        else:
            y_soft = None

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.amp):
            logits = model(x)
            if y_soft is None:
                loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
            else:
                loss = soft_target_loss(logits, y_soft)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        functional.reset_net(model)

        batch = x.shape[0]
        loss_sum += float(loss.detach().cpu()) * batch
        seen += batch
        correct += int((logits.detach().argmax(dim=1) == y).sum().item())
        if batch_idx % args.log_every == 0:
            pbar.set_postfix(loss=loss_sum / max(1, seen), acc=100.0 * correct / max(1, seen))
    return {"loss": loss_sum / seen, "acc": 100.0 * correct / seen}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    loss_sum = 0.0
    seen = 0
    top1_sum = 0.0
    top5_sum = 0.0
    confusion = np.zeros((10, 10), dtype=np.int64)
    for x, y in tqdm(loader, desc="eval", leave=False, disable=not args.progress):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_context(device, args.amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        functional.reset_net(model)
        top1, top5 = accuracy(logits, y)
        batch = x.shape[0]
        loss_sum += float(loss.detach().cpu()) * batch
        top1_sum += top1 * batch
        top5_sum += top5 * batch
        seen += batch
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        labels = y.detach().cpu().numpy()
        for label, pred in zip(labels, preds):
            confusion[int(label), int(pred)] += 1
    return {"loss": loss_sum / seen, "top1": top1_sum / seen, "top5": top5_sum / seen}, confusion


@torch.no_grad()
def collect_spike_rates(model: nn.Module, loader: DataLoader, device: torch.device, batches: int) -> dict[str, float]:
    rates: dict[str, list[float]] = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, MultiStepLIFNode):
            rates[name] = []

            def hook(_module, _inputs, output, key=name):
                if torch.is_tensor(output):
                    rates[key].append(float(output.detach().float().mean().cpu()))

            handles.append(module.register_forward_hook(hook))
    model.eval()
    for batch_idx, (x, _y) in enumerate(loader):
        if batch_idx >= batches:
            break
        _ = model(x.to(device, non_blocking=True))
        functional.reset_net(model)
    for handle in handles:
        handle.remove()
    return {key: float(np.mean(value)) if value else 0.0 for key, value in rates.items()}


def estimate_energy(args: argparse.Namespace, spike_rates: dict[str, float]) -> dict[str, float | dict[str, float]]:
    dim = args.embed_dim
    time_steps = args.time_steps
    depth = args.depth
    tokens = 8 * 8
    conv0_mac = time_steps * 32 * 32 * 3 * (dim // 8) * 3 * 3
    conv1_ops = time_steps * 32 * 32 * (dim // 8) * (dim // 4) * 3 * 3
    conv2_ops = time_steps * 32 * 32 * (dim // 4) * (dim // 2) * 3 * 3
    conv3_ops = time_steps * 16 * 16 * (dim // 2) * dim * 3 * 3
    conv4_ops = time_steps * 8 * 8 * dim * dim * 3 * 3
    qkv_proj_ops = depth * time_steps * 3 * tokens * dim * dim
    attn_qk_ops = depth * time_steps * args.heads * tokens * tokens * (dim // args.heads)
    attn_v_ops = attn_qk_ops
    proj_ops = depth * time_steps * tokens * dim * dim
    mlp_ops = depth * time_steps * 2 * tokens * dim * (dim * args.mlp_ratio)
    head_ops = dim * 10
    dense_ops = sum([conv0_mac, conv1_ops, conv2_ops, conv3_ops, conv4_ops, qkv_proj_ops, attn_qk_ops, attn_v_ops, proj_ops, mlp_ops, head_ops])

    r = lambda key, default=0.05: float(spike_rates.get(key, default))
    gated_ops = conv0_mac
    gated_ops += conv1_ops * r("tokenizer.lif1")
    gated_ops += conv2_ops * r("tokenizer.lif2")
    gated_ops += conv3_ops * r("tokenizer.lif3")
    gated_ops += conv4_ops * r("tokenizer.lif4")
    q_rate = np.mean([r(f"blocks.{i}.attn.q_lif") for i in range(depth)])
    k_rate = np.mean([r(f"blocks.{i}.attn.k_lif") for i in range(depth)])
    v_rate = np.mean([r(f"blocks.{i}.attn.v_lif") for i in range(depth)])
    attn_rate = np.mean([r(f"blocks.{i}.attn.attn_lif") for i in range(depth)])
    mlp1_rate = np.mean([r(f"blocks.{i}.mlp.lif1") for i in range(depth)])
    mlp2_rate = np.mean([r(f"blocks.{i}.mlp.lif2") for i in range(depth)])
    gated_ops += qkv_proj_ops * np.mean([q_rate, k_rate, v_rate])
    gated_ops += attn_qk_ops * q_rate * k_rate
    gated_ops += attn_v_ops * attn_rate * v_rate
    gated_ops += proj_ops * attn_rate
    gated_ops += mlp_ops * np.mean([mlp1_rate, mlp2_rate])
    gated_ops += head_ops

    dense_energy_mj = dense_ops * E_MAC_PJ * 1e-9
    spiking_energy_mj = conv0_mac * E_MAC_PJ * 1e-9 + (gated_ops - conv0_mac) * E_AC_PJ * 1e-9
    return {
        "note": "Compute-energy estimate using measured spike rates, MAC=4.6pJ and AC=0.9pJ; Vivado power is reported separately for RTL.",
        "dense_ops": dense_ops,
        "gated_ops": gated_ops,
        "dense_energy_mj": dense_energy_mj,
        "spiking_energy_mj": spiking_energy_mj,
        "energy_reduction": 1.0 - spiking_energy_mj / dense_energy_mj,
        "mean_q_rate": float(q_rate),
        "mean_k_rate": float(k_rate),
        "mean_v_rate": float(v_rate),
        "mean_attn_rate": float(attn_rate),
        "mean_mlp1_rate": float(mlp1_rate),
        "mean_mlp2_rate": float(mlp2_rate),
    }


def write_plots(history: list[dict[str, float]], confusion: np.ndarray, out_dir: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, [row["train_acc"] for row in history], label="train")
    plt.plot(epochs, [row["test_top1"] for row in history], label="test top1")
    plt.axhline(95.0, color="crimson", linestyle="--", linewidth=1, label="95% target")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "spikingformer_accuracy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["test_loss"] for row in history], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "spikingformer_loss.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.imshow(confusion, cmap="magma")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_dir / "spikingformer_confusion.png", dpi=180)
    plt.close()


def write_report(out_dir: Path, metrics: dict, history: list[dict[str, float]]) -> None:
    with PdfPages(out_dir / "spikingformer_cifar10_report.pdf") as pdf:
        for name in ["spikingformer_accuracy.png", "spikingformer_loss.png", "spikingformer_confusion.png"]:
            img = plt.imread(out_dir / name)
            plt.figure(figsize=(10, 7))
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(bbox_inches="tight")
            plt.close()
        fig = plt.figure(figsize=(10, 7))
        fig.text(0.06, 0.92, "Spikingformer-4-384 CIFAR-10 Run", fontsize=18, weight="bold")
        lines = [
            f"Best top-1: {metrics['best_top1']:.3f}%",
            f"Final top-1: {metrics['final_top1']:.3f}%",
            f"Final top-5: {metrics['final_top5']:.3f}%",
            f"Epochs completed: {len(history)}",
            f"Energy reduction estimate: {100.0 * metrics['energy']['energy_reduction']:.3f}%",
            "Architecture: T=4, tokenizer conv stack, spiking self-attention, spiking MLP, 9.3M parameters.",
            "Research target: Spikingformer-4-384-400E reports 95.81% CIFAR-10 top-1.",
        ]
        fig.text(0.06, 0.82, "\n".join(lines), fontsize=12, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    setup_cuda_runtime()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.pause_file is None:
        args.pause_file = args.output_dir / "PAUSE"
    guard_single_active_run(args.output_dir)
    job_id = f"{int(time.time())}-{os.getpid()}"
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_loader, test_loader = make_loaders(args)
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start_epoch = 0
    best_acc = 0.0
    resume_path: Path | None = None
    resume_failures: list[str] = []
    resume_candidates = [] if args.fresh else resolve_resume_candidates(args.resume, args.output_dir)
    if resume_candidates:
        start_epoch, best_acc, resume_path, resume_failures = load_first_valid_checkpoint(
            resume_candidates,
            model,
            optimizer if not args.eval_only else None,
            scaler,
        )
        if resume_path is None:
            raise RuntimeError("No valid resume checkpoint found. Tried:\n" + "\n".join(resume_failures))
        print(f"resuming_from={resume_path} next_epoch={start_epoch + 1} best_acc={best_acc:.3f}", flush=True)
        if args.eval_only:
            start_epoch = 0

    config = vars(args).copy()
    config["device"] = str(device)
    config["params"] = sum(p.numel() for p in model.parameters())
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    write_run_state(
        args.output_dir,
        {
            "status": "running_eval" if args.eval_only else "running",
            "pid": os.getpid(),
            "job_id": job_id,
            "resume_path": str(resume_path) if resume_path else "",
            "resume_failures": resume_failures,
            "start_epoch": start_epoch,
            "target_epochs": args.epochs,
            "best_acc": best_acc,
            "pause_file": str(args.pause_file),
            "latest_checkpoint": str(args.output_dir / "latest_checkpoint.pt"),
            "best_checkpoint": str(args.output_dir / "best_checkpoint.pt"),
        },
    )

    csv_path = args.output_dir / "history.csv"
    history: list[dict[str, float]] = load_existing_history(csv_path, start_epoch)
    if args.eval_only:
        eval_metrics, confusion = evaluate(model, test_loader, device, args)
        spike_rates = collect_spike_rates(model, test_loader, device, args.spike_rate_batches)
        energy = estimate_energy(args, spike_rates)
        metrics = {
            "best_top1": eval_metrics["top1"],
            "final_top1": eval_metrics["top1"],
            "final_top5": eval_metrics["top5"],
            "eval": eval_metrics,
            "spike_rates": spike_rates,
            "energy": energy,
        }
        write_plots([{"epoch": 1, "train_acc": eval_metrics["top1"], "test_top1": eval_metrics["top1"], "train_loss": eval_metrics["loss"], "test_loss": eval_metrics["loss"]}], confusion, args.output_dir)
        write_report(args.output_dir, metrics, [{"epoch": 1}])
        (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        write_run_state(
            args.output_dir,
            {
                "status": "complete_eval",
                "pid": os.getpid(),
                "job_id": job_id,
                "best_acc": eval_metrics["top1"],
                "final_top1": eval_metrics["top1"],
            },
        )
        return

    rewrite_history(csv_path, history)
    interrupted = False
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_top1", "test_top5", "seconds"])
        try:
            last_complete_epoch = start_epoch - 1
            current_epoch = start_epoch
            for epoch in range(start_epoch, args.epochs):
                current_epoch = epoch
                if args.pause_file.exists():
                    write_run_state(
                        args.output_dir,
                        {
                            "status": "paused_before_epoch",
                            "last_complete_epoch": epoch,
                            "next_epoch": epoch + 1,
                            "pid": os.getpid(),
                            "job_id": job_id,
                            "best_acc": best_acc,
                            "latest_checkpoint": str(args.output_dir / "latest_checkpoint.pt"),
                            "pause_file": str(args.pause_file),
                        },
                    )
                    print(f"pause_requested_before_epoch={epoch + 1}; remove {args.pause_file} and resume.", flush=True)
                    return

                ts = time.time()
                lr = cosine_lr(epoch, args)
                set_optimizer_lr(optimizer, lr)
                train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args)
                eval_metrics, confusion = evaluate(model, test_loader, device, args)
                row = {
                    "epoch": epoch + 1,
                    "lr": lr,
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["acc"],
                    "test_loss": eval_metrics["loss"],
                    "test_top1": eval_metrics["top1"],
                    "test_top5": eval_metrics["top5"],
                    "seconds": time.time() - ts,
                }
                history.append(row)
                writer.writerow(row)
                fh.flush()
                if eval_metrics["top1"] >= best_acc:
                    best_acc = eval_metrics["top1"]
                    save_checkpoint(args.output_dir / "best_checkpoint.pt", model, optimizer, scaler, epoch, best_acc, args, "best")
                save_checkpoint(args.output_dir / "latest_checkpoint.pt", model, optimizer, scaler, epoch, best_acc, args, "latest")
                if args.archive_every_epochs > 0 and ((epoch + 1) % args.archive_every_epochs == 0 or epoch + 1 == args.epochs):
                    shutil.copy2(args.output_dir / "latest_checkpoint.pt", args.output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt")
                write_run_state(
                    args.output_dir,
                    {
                        "status": "running",
                        "pid": os.getpid(),
                        "job_id": job_id,
                        "current_epoch": epoch + 1,
                        "target_epochs": args.epochs,
                        "best_acc": best_acc,
                        "latest_checkpoint": str(args.output_dir / "latest_checkpoint.pt"),
                        "best_checkpoint": str(args.output_dir / "best_checkpoint.pt"),
                        "pause_file": str(args.pause_file),
                    },
                )
                print(
                    f"epoch={epoch + 1:03d} lr={lr:.3e} train_acc={train_metrics['acc']:.2f}% "
                    f"test_top1={eval_metrics['top1']:.2f}% best={best_acc:.2f}% loss={eval_metrics['loss']:.4f}",
                    flush=True,
                )
                if args.pause_file.exists():
                    write_run_state(
                        args.output_dir,
                        {
                            "status": "paused_after_epoch",
                            "pid": os.getpid(),
                            "job_id": job_id,
                            "current_epoch": epoch + 1,
                            "best_acc": best_acc,
                            "latest_checkpoint": str(args.output_dir / "latest_checkpoint.pt"),
                            "pause_file": str(args.pause_file),
                        },
                    )
                    print(f"pause_requested_after_epoch={epoch + 1}; remove {args.pause_file} and resume.", flush=True)
                    return
                last_complete_epoch = epoch
        except KeyboardInterrupt:
            interrupted = True
            save_checkpoint(args.output_dir / "interrupted_checkpoint.pt", model, optimizer, scaler, max(last_complete_epoch, -1), best_acc, args, "interrupted_not_epoch_exact")
            write_run_state(
                args.output_dir,
                {
                    "status": "interrupted",
                    "pid": os.getpid(),
                    "job_id": job_id,
                    "best_acc": best_acc,
                    "interrupted_during_epoch": current_epoch + 1,
                    "last_complete_epoch": last_complete_epoch + 1 if last_complete_epoch >= 0 else 0,
                    "resume_from": str(args.output_dir / "latest_checkpoint.pt"),
                    "emergency_checkpoint": str(args.output_dir / "interrupted_checkpoint.pt"),
                    "note": "latest_checkpoint.pt is the last complete epoch. interrupted_checkpoint.pt is partial and not used by auto resume.",
                },
            )
            print("interrupted; auto-resume will use latest complete epoch checkpoint.", flush=True)

    if interrupted:
        return

    spike_rates = collect_spike_rates(model, test_loader, device, args.spike_rate_batches)
    energy = estimate_energy(args, spike_rates)
    metrics = {
        "best_top1": best_acc,
        "final_top1": history[-1]["test_top1"],
        "final_top5": history[-1]["test_top5"],
        "history": history[-10:],
        "spike_rates": spike_rates,
        "energy": energy,
    }
    write_plots(history, confusion, args.output_dir)
    write_report(args.output_dir, metrics, history)
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_run_state(
        args.output_dir,
        {
            "status": "complete",
            "pid": os.getpid(),
            "job_id": job_id,
            "current_epoch": len(history),
            "target_epochs": args.epochs,
            "best_acc": best_acc,
            "final_top1": history[-1]["test_top1"],
            "latest_checkpoint": str(args.output_dir / "latest_checkpoint.pt"),
            "best_checkpoint": str(args.output_dir / "best_checkpoint.pt"),
        },
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
