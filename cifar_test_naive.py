"""
CIFAR-10 training (VGG4/VGG8/MobileNetV2) with activations (any supported by make_activation):
  - GELU
  - Swish (SiLU)
  - LeakyReLU
  - ReLU
  - Rational (initialized by approx_func)

NO CLI. Just run the file.

PLAIN VGG (VGG4/VGG8):
  - No BatchNorm
  - No Dropout
  - No Label smoothing
  - No RandomErasing
  - No special init
  - No AMP
  - Fixed LR=0.02 (no scheduler)

MobileNetV2:
  - Standard torchvision MobileNetV2 implementation (includes BN/Dropout as defined by the model)
  - Activations inside the model are replaced with make_activation(...) for comparison

Runs are saved under:
  OUT_ROOT/cifar10_<model>_plain_<activation>/seed_<seed>/
"""

import os
import time
import json
import random
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T

# Rational activations
from rational.torch import Rational


# -------------------------
# Fixed config (NO CLI)
# -------------------------
DATA_ROOT    = r"D:\datasets\rational_cifar"     # change if needed
OUT_ROOT     = r"D:\runs\rational_cifar10"       # change if needed

# Plain VGG4 or VGG8 or MobileNetV2 (edit MODEL to switch)
MODEL        = "vgg8"  # "vgg4" or "vgg8" or "mobilenet_v2"
ACTIVATIONS  = ("rational_leaky_relu")

EPOCHS       = 60
BATCH_SIZE   = 128

LR           = 0.02
MOMENTUM     = 0.9
WEIGHT_DECAY = 0
NESTEROV     = True

NUM_WORKERS  = 4
RUNS         = 1
SEED0        = 0

SAVE_METRICS = True


# -------------------------
# Helpers (Windows drive guard)
# -------------------------
def ensure_not_c_drive(path: str) -> None:
    if os.name == "nt":
        drive, _ = os.path.splitdrive(os.path.abspath(path))
        if drive.upper() == "C:":
            raise RuntimeError(
                f"Refusing to use C: drive path: {path}\n"
                f"Set DATA_ROOT/OUT_ROOT to a non-C drive (e.g., D:\\datasets, D:\\runs)."
            )

def mkdir(path: str) -> str:
    ensure_not_c_drive(path)
    os.makedirs(path, exist_ok=True)
    return path

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# -------------------------
# CIFAR-10 stats (standard)
# -------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


# -------------------------
# VGG definitions
# Each block: (num_conv_layers_in_block, in_channels, out_channels)
# conv: 3x3 stride=1 pad=1; after each conv -> activation; then maxpool.
# -------------------------
VGG4_CFG = [(1, 3, 64), (1, 64, 128), (2, 128, 256)]
VGG8_CFG = [(2, 3, 64), (2, 64, 128), (2, 128, 256), (2, 256, 512)]

MODEL_CFGS = {
    "vgg4": VGG4_CFG,
    "vgg8": VGG8_CFG,
}

SUPPORTED_MODELS = tuple(list(MODEL_CFGS.keys()) + ["mobilenet_v2"])


# -------------------------
# Activations
# -------------------------
def make_activation(act: str) -> nn.Module:
    a = act.lower()
    if a == "gelu":
        return nn.GELU()
    if a in ("swish", "silu"):
        return nn.SiLU(inplace=True)
    if a == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    if a == "relu":
        return nn.ReLU(inplace=True)
    if a == "rational_leaky_relu":
        return Rational(approx_func="leaky_relu")
    if a == "rational_swish":
        return Rational(approx_func="swish")
    if a == "rational_gelu":
        return Rational(approx_func="gelu")
    raise ValueError(f"Unknown activation={act}")


# -------------------------
# Model (plain VGG for CIFAR)
# -------------------------
class VGGCIFARSmall(nn.Module):
    def __init__(
        self,
        cfg: List[Tuple[int, int, int]],
        activation: str,
        num_classes: int = 10,
    ):
        super().__init__()
        layers: List[nn.Module] = []

        for (n_conv, in_ch, out_ch) in cfg:
            cur_in = in_ch
            for _ in range(n_conv):
                layers.append(
                    nn.Conv2d(
                        cur_in, out_ch,
                        kernel_size=3, stride=1, padding=1,
                        bias=True,
                    )
                )
                layers.append(make_activation(activation))
                cur_in = out_ch
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*layers)

        num_pools = len(cfg)
        last_out = cfg[-1][2]
        spatial = 32 // (2 ** num_pools)
        if spatial < 1 or (32 % (2 ** num_pools)) != 0:
            raise RuntimeError(f"Unexpected pooling config: num_pools={num_pools} for 32x32 input.")
        flat_dim = last_out * spatial * spatial

        self.classifier = nn.Linear(flat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# -------------------------
# MobileNetV2 activation replacement
# -------------------------
def replace_relu_modules_inplace(module: nn.Module, activation: str) -> None:
    """
    Replace all nn.ReLU / nn.ReLU6 modules in-place with make_activation(activation).
    This keeps the MobileNetV2 architecture intact while swapping activations for comparison.
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.ReLU, nn.ReLU6)):
            setattr(module, name, make_activation(activation))
        else:
            replace_relu_modules_inplace(child, activation)


def build_model(model_name: str, activation: str, num_classes: int = 10) -> nn.Module:
    if model_name in MODEL_CFGS:
        return VGGCIFARSmall(cfg=MODEL_CFGS[model_name], activation=activation, num_classes=num_classes)

    if model_name == "mobilenet_v2":
        m = torchvision.models.mobilenet_v2(weights=None)
        # Adapt classifier to CIFAR-10
        if not hasattr(m, "classifier") or not isinstance(m.classifier, nn.Sequential) or len(m.classifier) < 2:
            raise RuntimeError("Unexpected torchvision MobileNetV2 structure (classifier).")
        if not isinstance(m.classifier[1], nn.Linear):
            raise RuntimeError("Unexpected torchvision MobileNetV2 classifier[1] (expected nn.Linear).")
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

        # Replace activations throughout the model
        replace_relu_modules_inplace(m, activation)
        return m

    raise ValueError(f"Unknown MODEL={model_name}. Supported: {SUPPORTED_MODELS}")


# -------------------------
# Data (plain: no augmentation)
# -------------------------
def build_transforms() -> Tuple[nn.Module, nn.Module]:
    train_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_tf, test_tf

def build_loaders(data_root: str, batch_size: int, num_workers: int, use_cuda: bool):
    train_tf, test_tf = build_transforms()
    tv_root = mkdir(os.path.join(data_root, "torchvision"))

    train_ds = torchvision.datasets.CIFAR10(root=tv_root, train=True, download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(root=tv_root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_cuda
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_cuda
    )
    return train_loader, test_loader


# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += ce(logits, y).item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return loss_sum / max(1, total), correct / max(1, total)


def build_optimizer(model: nn.Module) -> optim.Optimizer:
    return optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=NESTEROV,
    )


def train_one_run(run_seed: int, activation: str) -> Dict[str, Any]:
    set_seed(run_seed)

    if MODEL not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown MODEL={MODEL}. Supported: {SUPPORTED_MODELS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = (device.type == "cuda")

    train_loader, test_loader = build_loaders(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_cuda=use_cuda,
    )

    model = build_model(model_name=MODEL, activation=activation, num_classes=10).to(device)

    n_params = count_params(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model)

    run_group = f"cifar10_{MODEL}_plain_{activation}"
    run_dir = mkdir(os.path.join(OUT_ROOT, run_group, f"seed_{run_seed}"))
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    ckpt_path = os.path.join(run_dir, "best.pt")

    log_f = None
    if SAVE_METRICS:
        try:
            log_f = open(metrics_path, "w", encoding="utf-8", newline="\n")
        except OSError as e:
            print(f"WARNING: cannot open metrics file: {metrics_path}\n{e}\nContinuing without file logging.")
            log_f = None

    best_test_acc = 0.0
    best_epoch = -1
    total_start = time.perf_counter()

    try:
        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.perf_counter()
            model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                logits = model(x)
                loss = criterion(logits, y)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * y.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()

            train_loss = running_loss / max(1, total)
            train_acc = correct / max(1, total)

            test_loss, test_acc = evaluate(model, test_loader, device)

            lr_now = optimizer.param_groups[0]["lr"]
            epoch_time = time.perf_counter() - epoch_start

            row = {
                "dataset": "cifar10",
                "model": MODEL,
                "activation": activation,
                "seed": run_seed,
                "epoch": epoch,
                "lr": float(lr_now),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "epoch_time_sec": float(epoch_time),
                "n_params": int(n_params),
            }

            if log_f is not None:
                try:
                    log_f.write(json.dumps(row) + "\n")
                except OSError as e:
                    print(f"WARNING: metrics write failed (disabling file logging): {e}")
                    try:
                        log_f.close()
                    except OSError:
                        pass
                    log_f = None

            print(
                f"[{activation} | seed {run_seed}] epoch {epoch:03d}/{EPOCHS} | "
                f"lr {lr_now:.5f} | "
                f"train {train_acc*100:.2f}% (loss {train_loss:.4f}) | "
                f"test {test_acc*100:.2f}% (loss {test_loss:.4f}) | "
                f"{epoch_time:.1f}s"
            )

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "test_acc": float(test_acc),
                        "n_params": int(n_params),
                        "config": {
                            "DATA_ROOT": DATA_ROOT,
                            "OUT_ROOT": OUT_ROOT,
                            "MODEL": MODEL,
                            "EPOCHS": EPOCHS,
                            "BATCH_SIZE": BATCH_SIZE,
                            "LR": LR,
                            "MOMENTUM": MOMENTUM,
                            "WEIGHT_DECAY": WEIGHT_DECAY,
                            "NESTEROV": NESTEROV,
                            "NUM_WORKERS": NUM_WORKERS,
                        },
                        "seed": run_seed,
                    },
                    ckpt_path,
                )

    finally:
        if log_f is not None:
            try:
                log_f.close()
            except OSError:
                pass

    total_time = time.perf_counter() - total_start

    return {
        "seed": run_seed,
        "activation": activation,
        "best_test_acc": float(best_test_acc),
        "best_epoch": int(best_epoch),
        "best_ckpt": ckpt_path,
        "n_params": int(n_params),
        "total_time_sec": float(total_time),
        "run_dir": run_dir,
    }


def main():
    mkdir(DATA_ROOT)
    mkdir(OUT_ROOT)

    if MODEL not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown MODEL={MODEL}. Supported: {SUPPORTED_MODELS}")

    config = {
        "dataset": "cifar10",
        "model": MODEL,
        "activations": list(ACTIVATIONS),
        "data_root": DATA_ROOT,
        "out_root": OUT_ROOT,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "nesterov": NESTEROV,
        "num_workers": NUM_WORKERS,
        "runs": RUNS,
        "seed0": SEED0,
    }
    print("Config:", json.dumps(config, indent=2))

    all_results: Dict[str, Any] = {
        "config": config,
        "per_activation": [],
    }

    for activation in ACTIVATIONS:
        print("\n" + "=" * 80)
        print(f"RUNNING: model={MODEL} | activation={activation} | runs={RUNS} | seed0={SEED0}")
        print("=" * 80)

        results = []
        for i in range(RUNS):
            seed = SEED0 + i
            results.append(train_one_run(run_seed=seed, activation=activation))

        bests = np.array([r["best_test_acc"] for r in results], dtype=np.float64)
        mean = float(bests.mean()) if len(bests) else float("nan")
        std = float(bests.std(ddof=1)) if len(bests) > 1 else 0.0

        summary = {
            "dataset": "cifar10",
            "model": MODEL,
            "activation": activation,
            "runs": RUNS,
            "seed0": SEED0,
            "best_test_acc_mean": mean,
            "best_test_acc_std": std,
            "per_run": results,
        }

        summary_path = os.path.join(OUT_ROOT, f"summary_cifar10_{MODEL}_plain_{activation}.json")
        with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, indent=2)

        print("\n=== SUMMARY ===")
        print(f"cifar10 | {MODEL}_plain | {activation} | runs={RUNS}")
        print(f"best test acc: {mean*100:.2f}% ± {std*100:.2f}%")
        print(f"saved summary to: {summary_path}")

        all_results["per_activation"].append({
            "activation": activation,
            "summary_path": summary_path,
            "best_test_acc_mean": mean,
            "best_test_acc_std": std,
        })

    all_summary_path = os.path.join(OUT_ROOT, f"summary_cifar10_{MODEL}_plain_all_activations.json")
    with open(all_summary_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL DONE")
    print(f"Saved overall index to: {all_summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
