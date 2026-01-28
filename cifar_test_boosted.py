"""
Train CIFAR-10 with a fixed configuration (no CLI).

Models:
  - vgg4, vgg6, vgg8 (custom small VGG for 32x32)
  - mobilenet_v2 (torchvision)
  - resnet50 (torchvision)

Activations:
  - gelu, swish (SiLU), leaky_relu, relu
  - rational_gelu, rational_swish, rational_leaky_relu (Rational with approx_func)

Training features implemented in this file:
  - label smoothing via CrossEntropyLoss(label_smoothing=...)
  - mixup in the training loop
  - dropout before the VGG classifier
  - optional GroupNorm after each VGG Conv2d and before activation
  - "weights-only" weight decay: Conv/Linear weights use WEIGHT_DECAY; biases and norm params use 0
  - Rational coefficients are excluded from weight decay and can use an LR multiplier
  - optional one-time LSUV-style data-dependent weight rescaling at initialization
  - AMP uses torch.amp.GradScaler and torch.amp.autocast (no deprecated AMP API)

Edit DATA_ROOT / OUT_ROOT to change data and run directories.
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

from rational.torch import Rational


DATA_ROOT    = r"D:\datasets\rational_cifar"
OUT_ROOT     = r"D:\runs\rational_cifar10"

MODEL        = "vgg8"
ACTIVATIONS  = ("rational_gelu", "rational_leaky_relu", "leaky_relu", "relu", "swish", "gelu")

EPOCHS       = 60
BATCH_SIZE   = 128
LR           = 0.02
RATIONAL_LR_MULT = [0.5]
MOMENTUM     = 0.9
WEIGHT_DECAY = 5e-4
RATIONAL_WEIGHT_DECAY = 0.0
NESTEROV     = True

LR_SCHEDULE      = "step"
STEP_MILESTONES  = (30, 45)
STEP_GAMMA       = 0.1

USE_AUG      = True
NUM_WORKERS  = 4

RUNS         = 5
SEED0        = 0

USE_AMP      = False
GRAD_CLIP    = 0.0

SAVE_METRICS = True

LABEL_SMOOTHING = 0.1
MIXUP_ALPHA     = 0.2
VGG_DROPOUT_P   = 0.3

VGG_USE_GROUPNORM = False
VGG_GN_GROUPS     = 16

USE_LSUV_INIT    = True
LSUV_TARGET_VAR  = 0.125
LSUV_MAX_ITERS   = 10
LSUV_TOL_REL     = 0.10
LSUV_EPS         = 1e-6
LSUV_BATCH       = 64


def ensure_not_c_drive(path: str) -> None:
    """On Windows, reject C: paths to avoid slow or quota-limited system disks."""
    if os.name == "nt":
        drive, _ = os.path.splitdrive(os.path.abspath(path))
        if drive.upper() == "C:":
            raise RuntimeError(
                f"Refusing to use C: drive path: {path}\n"
                f"Set DATA_ROOT/OUT_ROOT to your T9 disk (e.g., D:\\datasets, D:\\runs)."
            )

def mkdir(path: str) -> str:
    """Create a directory (after applying the Windows drive guard) and return the path."""
    ensure_not_c_drive(path)
    os.makedirs(path, exist_ok=True)
    return path

def set_seed(seed: int) -> None:
    """Set Python/NumPy/PyTorch RNG seeds and enable deterministic cuDNN behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_params(model: nn.Module) -> int:
    """Return the total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())

def split_rational_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Partition model parameters into those owned by Rational modules and all remaining parameters."""
    rational_param_ids = set()
    for m in model.modules():
        if isinstance(m, Rational):
            for p in m.parameters():
                rational_param_ids.add(id(p))

    rational_params: List[nn.Parameter] = []
    other_params: List[nn.Parameter] = []
    for p in model.parameters():
        if id(p) in rational_param_ids:
            rational_params.append(p)
        else:
            other_params.append(p)

    return rational_params, other_params

def split_decay_no_decay_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Build two parameter lists for optimizer groups:
      - decay: Conv/Linear weight tensors
      - no_decay: biases, normalization parameters, Rational parameters, and any unclassified parameters
    """
    decay_ids = set()
    no_decay_ids = set()

    norm_types = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.LayerNorm, nn.GroupNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
        nn.LocalResponseNorm,
    )
    weight_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)

    for m in model.modules():
        if isinstance(m, weight_types):
            if getattr(m, "weight", None) is not None:
                decay_ids.add(id(m.weight))
            if getattr(m, "bias", None) is not None:
                no_decay_ids.add(id(m.bias))
        elif isinstance(m, norm_types):
            for p in m.parameters(recurse=False):
                no_decay_ids.add(id(p))
        elif isinstance(m, Rational):
            for p in m.parameters(recurse=False):
                no_decay_ids.add(id(p))

    for p in model.parameters():
        pid = id(p)
        if pid not in decay_ids and pid not in no_decay_ids:
            no_decay_ids.add(pid)

    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
    for p in model.parameters():
        if id(p) in decay_ids:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    return decay_params, no_decay_params

def choose_gn_groups(num_channels: int, max_groups: int) -> int:
    """Return the largest group count <= max_groups that divides num_channels (at least 1)."""
    g = min(int(max_groups), int(num_channels))
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return max(1, g)

def _as_lr_mult_list(x) -> List[float]:
    """Normalize the rational LR multiplier setting to a Python list of floats."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return [float(v) for v in x]
    return [float(x)]

def _lr_mult_tag(m: float) -> str:
    """Convert a float multiplier to a filename-safe tag."""
    mf = float(m)
    if mf.is_integer():
        return str(int(mf))
    return str(mf).replace(".", "p")


def _lsuv_collect_exec_trace(model: nn.Module, x: torch.Tensor):
    """
    Run one forward pass and record outputs (in execution order) for:
      - Conv/Linear modules ("weight")
      - activation modules ("act")
    """
    weight_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
    act_types = (nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.LeakyReLU, Rational)

    trace: List[Dict[str, Any]] = []
    hooks = []

    def _hook_factory(kind: str, mref: nn.Module):
        def _hook(_m, _inp, out):
            if isinstance(out, (tuple, list)):
                return
            if not torch.is_tensor(out):
                return
            trace.append({"module": mref, "type": kind, "out": out})
        return _hook

    for m in model.modules():
        if isinstance(m, weight_types):
            hooks.append(m.register_forward_hook(_hook_factory("weight", m)))
        elif isinstance(m, act_types):
            hooks.append(m.register_forward_hook(_hook_factory("act", m)))

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    return trace

def _lsuv_find_next_act_with_shape(trace, weight_idx: int):
    """Find the next activation output after a given weight-module output that matches its tensor shape."""
    w_out = trace[weight_idx]["out"]
    if not torch.is_tensor(w_out):
        return None
    w_shape = tuple(w_out.shape)
    for j in range(weight_idx + 1, len(trace)):
        if trace[j]["type"] != "act":
            continue
        a_out = trace[j]["out"]
        if not torch.is_tensor(a_out):
            continue
        if tuple(a_out.shape) == w_shape:
            return j
    return None

def lsuv_init(model: nn.Module, x: torch.Tensor) -> None:
    """
    One-time data-dependent weight rescaling:
      - For each executed Conv/Linear module, rescale its weight so the variance of the next same-shape
        activation output (or the module output if no match) moves toward LSUV_TARGET_VAR.
      - If a bias exists, shift it to reduce the pre-activation mean.
    """
    if not USE_LSUV_INIT:
        return

    was_training = model.training
    model.eval()

    if x.dtype != torch.float32:
        x = x.float()
    if x.size(0) > int(LSUV_BATCH):
        x = x[: int(LSUV_BATCH)]

    trace0 = _lsuv_collect_exec_trace(model, x)
    weight_positions = [i for i, e in enumerate(trace0) if e["type"] == "weight"]

    for wi in weight_positions:
        wmod = trace0[wi]["module"]

        for _ in range(int(LSUV_MAX_ITERS)):
            trace = _lsuv_collect_exec_trace(model, x)

            cur_wi = None
            for k, e in enumerate(trace):
                if e["type"] == "weight" and e["module"] is wmod:
                    cur_wi = k
                    break
            if cur_wi is None:
                break

            w_out = trace[cur_wi]["out"]
            if not torch.is_tensor(w_out):
                break

            act_j = _lsuv_find_next_act_with_shape(trace, cur_wi)
            target_tensor = trace[act_j]["out"] if act_j is not None else w_out
            if not torch.is_tensor(target_tensor):
                break

            t = target_tensor
            if t.dtype != torch.float32:
                t = t.float()

            var = torch.var(t, unbiased=False)
            var_val = float(var.item())

            if getattr(wmod, "bias", None) is not None and wmod.bias is not None:
                m = w_out
                if m.dtype != torch.float32:
                    m = m.float()
                mean_val = float(m.mean().item())
                if abs(mean_val) > 1e-3:
                    try:
                        wmod.bias.data = wmod.bias.data - mean_val
                    except Exception:
                        pass

            if not np.isfinite(var_val) or var_val <= 0.0:
                break

            rel_err = abs(var_val - float(LSUV_TARGET_VAR)) / max(float(LSUV_TARGET_VAR), float(LSUV_EPS))
            if rel_err <= float(LSUV_TOL_REL):
                break

            scale = float(np.sqrt(float(LSUV_TARGET_VAR) / (var_val + float(LSUV_EPS))))
            try:
                wmod.weight.data.mul_(scale)
            except Exception:
                break

    if was_training:
        model.train()
    else:
        model.eval()


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup to a batch and return (mixed_x, y_a, y_b, lam)."""
    if alpha <= 0.0:
        return x, y, y, 1.0

    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    x_mix = lam * x + (1.0 - lam) * x[index]
    y_a = y
    y_b = y[index]
    return x_mix, y_a, y_b, lam

def mixup_loss(criterion: nn.Module, logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Compute a convex combination of per-target losses for mixup."""
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


VGG4_CFG = [(1, 3, 64), (1, 64, 128), (2, 128, 256)]
VGG6_CFG = [(1, 3, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512)]
VGG8_CFG = [(2, 3, 64), (2, 64, 128), (2, 128, 256), (2, 256, 512)]

def get_cfg(model_name: str):
    """Return the VGG block configuration for the requested VGG variant."""
    model_name = model_name.lower()
    if model_name == "vgg4":
        return VGG4_CFG
    if model_name == "vgg6":
        return VGG6_CFG
    if model_name == "vgg8":
        return VGG8_CFG
    raise ValueError(f"Unknown MODEL={model_name}")


def make_activation(act: str) -> nn.Module:
    """Map an activation name string to a torch.nn.Module (including Rational approx_func options)."""
    a = act.lower()
    if a == "gelu":
        return nn.GELU()
    if a == "swish":
        return nn.SiLU(inplace=True)
    if a == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    if a == "rational_leaky_relu":
        return Rational(approx_func="leaky_relu")
    if a == "rational_swish":
        return Rational(approx_func="swish")
    if a == "rational_gelu":
        return Rational(approx_func="gelu")
    if a == "relu":
        return nn.ReLU()
    if a == "relu6":
        return nn.ReLU6(inplace=True)
    raise ValueError(f"Unknown activation={act}")


class VGGCIFARSmall(nn.Module):
    """Small VGG-style CIFAR-10 model with optional GroupNorm and dropout before the classifier."""
    def __init__(self, cfg: List[Tuple[int, int, int]], activation: str, num_classes: int = 10, dropout_p: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []

        use_gn_here = bool(VGG_USE_GROUPNORM)

        for (n_conv, in_ch, out_ch) in cfg:
            cur_in = in_ch
            for _ in range(n_conv):
                layers.append(nn.Conv2d(cur_in, out_ch, kernel_size=3, stride=1, padding=1, bias=True))

                if use_gn_here:
                    g = choose_gn_groups(out_ch, VGG_GN_GROUPS)
                    layers.append(nn.GroupNorm(num_groups=g, num_channels=out_ch))

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

        self.dropout = nn.Dropout(p=float(dropout_p)) if dropout_p and dropout_p > 0.0 else nn.Identity()
        self.classifier = nn.Linear(flat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)


def replace_relu_modules_inplace(module: nn.Module, activation: str) -> None:
    """Recursively replace nn.ReLU / nn.ReLU6 children with the requested activation module."""
    for name, child in module.named_children():
        if isinstance(child, (nn.ReLU, nn.ReLU6)):
            setattr(module, name, make_activation(activation))
        else:
            replace_relu_modules_inplace(child, activation)

def build_model(model_name: str, activation: str, num_classes: int = 10) -> nn.Module:
    """Construct the requested model and apply activation swapping for torchvision models when needed."""
    mn = model_name.lower()

    if mn in ("vgg4", "vgg6", "vgg8"):
        cfg = get_cfg(mn)
        return VGGCIFARSmall(cfg, activation=activation, num_classes=num_classes, dropout_p=VGG_DROPOUT_P)

    if mn == "mobilenet_v2":
        try:
            m = torchvision.models.mobilenet_v2(weights=None)
        except TypeError:
            m = torchvision.models.mobilenet_v2(pretrained=False)

        if not hasattr(m, "classifier") or not isinstance(m.classifier, nn.Sequential) or len(m.classifier) < 2:
            raise RuntimeError("Unexpected torchvision MobileNetV2 structure (classifier).")
        if not isinstance(m.classifier[1], nn.Linear):
            raise RuntimeError("Unexpected torchvision MobileNetV2 classifier[1] (expected nn.Linear).")
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

        if activation.lower() != "relu6":
            replace_relu_modules_inplace(m, activation)

        return m

    if mn == "resnet50":
        try:
            m = torchvision.models.resnet50(weights=None)
        except TypeError:
            m = torchvision.models.resnet50(pretrained=False)

        if not hasattr(m, "fc") or not isinstance(m.fc, nn.Linear):
            raise RuntimeError("Unexpected torchvision ResNet50 structure (fc).")
        m.fc = nn.Linear(m.fc.in_features, num_classes)

        if activation.lower() != "relu":
            replace_relu_modules_inplace(m, activation)

        return m

    raise ValueError(f"Unknown MODEL={model_name}. Expected vgg4/vgg6/vgg8/mobilenet_v2/resnet50.")


def build_transforms(use_aug: bool) -> Tuple[nn.Module, nn.Module]:
    """Return (train_transform, test_transform), using random crop/flip only when use_aug is True."""
    if use_aug:
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        train_tf = T.Compose([
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_tf, test_tf

def build_loaders(data_root: str, batch_size: int, num_workers: int, use_aug: bool, use_cuda: bool):
    """Create CIFAR-10 train/test DataLoaders under data_root/torchvision (download enabled)."""
    train_tf, test_tf = build_transforms(use_aug=use_aug)
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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Evaluate sum-reduced cross-entropy loss and accuracy over a dataloader."""
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


def build_optimizer(model: nn.Module, activation: str, rational_lr_mult: float) -> optim.Optimizer:
    """
    Build SGD with explicit parameter groups:
      - Conv/Linear weights: WEIGHT_DECAY
      - all other params: 0
      - if activation is rational_*: Rational params are isolated in their own group (0 WD, LR scaled)
    """
    decay_params, no_decay_params = split_decay_no_decay_params(model)

    if not activation.lower().startswith("rational_"):
        param_groups = [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return optim.SGD(
            param_groups,
            lr=LR,
            momentum=MOMENTUM,
            nesterov=NESTEROV,
            weight_decay=0.0,
        )

    rational_params, _ = split_rational_params(model)
    rational_ids = {id(p) for p in rational_params}

    decay_params = [p for p in decay_params if id(p) not in rational_ids]
    no_decay_params = [p for p in no_decay_params if id(p) not in rational_ids]

    param_groups = [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if len(rational_params) > 0:
        param_groups.append({
            "params": rational_params,
            "weight_decay": RATIONAL_WEIGHT_DECAY,
            "lr": float(LR) * float(rational_lr_mult),
        })

    return optim.SGD(
        param_groups,
        lr=LR,
        momentum=MOMENTUM,
        nesterov=NESTEROV,
        weight_decay=0.0,
    )


def train_one_run(run_seed: int, activation: str, rational_lr_mult: float) -> Dict[str, Any]:
    """Run a single seed training loop and save best checkpoint + per-epoch metrics (optional)."""
    set_seed(run_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = (device.type == "cuda")

    train_loader, test_loader = build_loaders(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_aug=USE_AUG,
        use_cuda=use_cuda,
    )

    model = build_model(MODEL, activation=activation, num_classes=10).to(device)

    if USE_LSUV_INIT:
        try:
            xb, _ = next(iter(train_loader))
            xb = xb.to(device, non_blocking=True)
            lsuv_init(model, xb)
        except Exception as e:
            print(f"WARNING: LSUV init failed (continuing without LSUV): {e}")

    n_params = count_params(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(LABEL_SMOOTHING))

    optimizer = build_optimizer(model, activation, rational_lr_mult=float(rational_lr_mult))

    scheduler = None
    if LR_SCHEDULE == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    elif LR_SCHEDULE == "step":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(STEP_MILESTONES), gamma=STEP_GAMMA
        )
    elif LR_SCHEDULE == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown LR_SCHEDULE={LR_SCHEDULE}")

    amp_device = "cuda" if use_cuda else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=USE_AMP and use_cuda)

    run_group = f"cifar10_{MODEL}_{activation}"
    if activation.lower().startswith("rational_"):
        run_group += "_no_wd_on_coeff"
        run_group += f"_lrmult{_lr_mult_tag(rational_lr_mult)}"
    if USE_LSUV_INIT:
        run_group += "_lsuv"

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

    vgg_gn_used_this_run = bool(MODEL.lower().startswith("vgg")) and bool(VGG_USE_GROUPNORM)
    vgg_gn_groups_this_run = int(VGG_GN_GROUPS) if vgg_gn_used_this_run else 0

    try:
        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.perf_counter()
            model.train()

            running_loss = 0.0
            correct = 0.0
            total = 0

            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                if MIXUP_ALPHA > 0.0:
                    x_in, y_a, y_b, lam = mixup_batch(x, y, alpha=float(MIXUP_ALPHA))
                else:
                    x_in, y_a, y_b, lam = x, y, y, 1.0

                with torch.amp.autocast(amp_device, enabled=scaler.is_enabled()):
                    logits = model(x_in)
                    if MIXUP_ALPHA > 0.0:
                        loss = mixup_loss(criterion, logits, y_a, y_b, lam)
                    else:
                        loss = criterion(logits, y)

                scaler.scale(loss).backward()

                if GRAD_CLIP > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * y.size(0)
                pred = logits.argmax(dim=1)

                if MIXUP_ALPHA > 0.0:
                    correct += lam * (pred == y_a).sum().item() + (1.0 - lam) * (pred == y_b).sum().item()
                else:
                    correct += (pred == y).sum().item()

                total += y.numel()

            if scheduler is not None:
                scheduler.step()

            train_loss = running_loss / max(1, total)
            train_acc = float(correct) / max(1, total)

            test_loss, test_acc = evaluate(model, test_loader, device)

            lr_now = optimizer.param_groups[0]["lr"]
            lr_groups_now = [float(pg["lr"]) for pg in optimizer.param_groups]
            epoch_time = time.perf_counter() - epoch_start

            row = {
                "dataset": "cifar10",
                "model": MODEL,
                "activation": activation,
                "seed": run_seed,
                "epoch": epoch,
                "lr": float(lr_now),
                "lr_groups": lr_groups_now,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "epoch_time_sec": float(epoch_time),
                "n_params": int(n_params),

                "label_smoothing": float(LABEL_SMOOTHING),
                "mixup_alpha": float(MIXUP_ALPHA),
                "vgg_dropout_p": float(VGG_DROPOUT_P) if MODEL.lower().startswith("vgg") else 0.0,

                "weight_decay_mode": "weights_only",

                "vgg_groupnorm": bool(vgg_gn_used_this_run),
                "vgg_gn_groups": int(vgg_gn_groups_this_run),

                "rational_lr_mult": float(rational_lr_mult),

                "lsuv_init": bool(USE_LSUV_INIT),
                "lsuv_target_var": float(LSUV_TARGET_VAR),
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
                            "ACTIVATION": activation,
                            "EPOCHS": EPOCHS,
                            "BATCH_SIZE": BATCH_SIZE,
                            "LR": LR,
                            "RATIONAL_LR_MULT": float(rational_lr_mult),
                            "MOMENTUM": MOMENTUM,
                            "WEIGHT_DECAY": WEIGHT_DECAY,
                            "RATIONAL_WEIGHT_DECAY": RATIONAL_WEIGHT_DECAY,
                            "NESTEROV": NESTEROV,
                            "LR_SCHEDULE": LR_SCHEDULE,
                            "STEP_MILESTONES": STEP_MILESTONES,
                            "STEP_GAMMA": STEP_GAMMA,
                            "USE_AUG": USE_AUG,
                            "NUM_WORKERS": NUM_WORKERS,
                            "USE_AMP": USE_AMP,
                            "GRAD_CLIP": GRAD_CLIP,

                            "LABEL_SMOOTHING": float(LABEL_SMOOTHING),
                            "MIXUP_ALPHA": float(MIXUP_ALPHA),
                            "VGG_DROPOUT_P": float(VGG_DROPOUT_P),

                            "WEIGHT_DECAY_MODE": "weights_only",

                            "VGG_USE_GROUPNORM": bool(vgg_gn_used_this_run),
                            "VGG_GN_GROUPS": int(vgg_gn_groups_this_run),

                            "USE_LSUV_INIT": bool(USE_LSUV_INIT),
                            "LSUV_TARGET_VAR": float(LSUV_TARGET_VAR),
                            "LSUV_MAX_ITERS": int(LSUV_MAX_ITERS),
                            "LSUV_TOL_REL": float(LSUV_TOL_REL),
                            "LSUV_BATCH": int(LSUV_BATCH),
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
        "rational_lr_mult": float(rational_lr_mult),
        "best_test_acc": float(best_test_acc),
        "best_epoch": int(best_epoch),
        "best_ckpt": ckpt_path,
        "n_params": int(n_params),
        "total_time_sec": float(total_time),
        "run_dir": run_dir,
    }


def main():
    """Run RUNS seeds per activation; for rational_* also sweep rational LR multipliers; write per-activation summaries."""
    mkdir(DATA_ROOT)
    mkdir(OUT_ROOT)

    rational_lr_mult_list = _as_lr_mult_list(RATIONAL_LR_MULT)

    config = {
        "dataset": "cifar10",
        "model": MODEL,
        "activations": list(ACTIVATIONS),
        "data_root": DATA_ROOT,
        "out_root": OUT_ROOT,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "rational_lr_mult_list": [float(v) for v in rational_lr_mult_list],
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "rational_weight_decay": RATIONAL_WEIGHT_DECAY,
        "nesterov": NESTEROV,
        "lr_schedule": LR_SCHEDULE,
        "step_milestones": STEP_MILESTONES,
        "step_gamma": STEP_GAMMA,
        "use_aug": USE_AUG,
        "num_workers": NUM_WORKERS,
        "runs": RUNS,
        "seed0": SEED0,
        "amp": USE_AMP,
        "grad_clip": GRAD_CLIP,

        "label_smoothing": float(LABEL_SMOOTHING),
        "mixup_alpha": float(MIXUP_ALPHA),
        "vgg_dropout_p": float(VGG_DROPOUT_P),

        "weight_decay_mode": "weights_only",

        "vgg_use_groupnorm": bool(VGG_USE_GROUPNORM),
        "vgg_gn_groups": int(VGG_GN_GROUPS),

        "use_lsuv_init": bool(USE_LSUV_INIT),
        "lsuv_target_var": float(LSUV_TARGET_VAR),
        "lsuv_max_iters": int(LSUV_MAX_ITERS),
        "lsuv_tol_rel": float(LSUV_TOL_REL),
        "lsuv_batch": int(LSUV_BATCH),
    }
    print("Config:", json.dumps(config, indent=2))

    all_results: Dict[str, Any] = {
        "config": config,
        "per_activation": [],
    }

    for activation in ACTIVATIONS:
        is_rational = activation.lower().startswith("rational_")
        mults = rational_lr_mult_list if is_rational else [1.0]

        for mult in mults:
            activation_tag = activation
            if is_rational:
                activation_tag = f"{activation}_lrmult{_lr_mult_tag(mult)}"

            print("\n" + "=" * 80)
            if is_rational:
                print(f"RUNNING: model={MODEL} | activation={activation} | lr_mult={float(mult)} | runs={RUNS} | seed0={SEED0}")
            else:
                print(f"RUNNING: model={MODEL} | activation={activation} | runs={RUNS} | seed0={SEED0}")
            print("=" * 80)

            results = []
            for i in range(RUNS):
                seed = SEED0 + i
                results.append(train_one_run(run_seed=seed, activation=activation, rational_lr_mult=float(mult)))

            bests = np.array([r["best_test_acc"] for r in results], dtype=np.float64)
            mean = float(bests.mean()) if len(bests) else float("nan")
            std = float(bests.std(ddof=1)) if len(bests) > 1 else 0.0

            summary = {
                "dataset": "cifar10",
                "model": MODEL,
                "activation": activation,
                "activation_tag": activation_tag,
                "rational_lr_mult": float(mult),
                "runs": RUNS,
                "seed0": SEED0,
                "best_test_acc_mean": mean,
                "best_test_acc_std": std,
                "per_run": results,
            }

            suffix = activation_tag
            if activation.lower().startswith("rational_") and "_no_wd_on_coeff" not in suffix:
                suffix += "_no_wd_on_coeff"
            if USE_LSUV_INIT and "_lsuv" not in suffix:
                suffix += "_lsuv"

            summary_path = os.path.join(OUT_ROOT, f"summary_cifar10_{MODEL}_{suffix}.json")
            with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(summary, f, indent=2)

            print("\n=== SUMMARY ===")
            if is_rational:
                print(f"cifar10 | {MODEL} | {activation_tag} | runs={RUNS}")
            else:
                print(f"cifar10 | {MODEL} | {activation} | runs={RUNS}")
            print(f"best test acc: {mean*100:.2f}% ± {std*100:.2f}%")
            print(f"saved summary to: {summary_path}")

            all_results["per_activation"].append({
                "activation": activation,
                "activation_tag": activation_tag,
                "rational_lr_mult": float(mult),
                "summary_path": summary_path,
                "best_test_acc_mean": mean,
                "best_test_acc_std": std,
            })

    all_summary_path = os.path.join(OUT_ROOT, f"summary_cifar10_{MODEL}_all_activations.json")
    with open(all_summary_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL DONE")
    print(f"Saved overall index to: {all_summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
