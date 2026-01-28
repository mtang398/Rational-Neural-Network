"""
Train an image-classification model on Tiny ImageNet (64x64) using Hugging Face datasets caching.

This script:
- Loads "zh-plus/tiny-imagenet" via datasets.load_dataset with caches under DATA_ROOT.
- Wraps HF splits into a PyTorch Dataset that converts images to RGB and applies torchvision transforms.
- Trains a timm model from scratch for NUM_CLASSES with optional repeated-augmentation sampling.
- Supports ACTIVATION in {"gelu", "rational"} by replacing nn.GELU modules with a Rational module wrapper.
- Optionally wraps ViT block LayerNorm modules and forces them disabled when using Rational.
- Uses AdamW with per-group weight decay rules and an optional separate param-group for Rational params.
- Uses cosine LR with warmup computed per training step, optional Mixup/CutMix, optional EMA, and optional AMP.
- Writes config.json, log.csv, last.pth, best.pth, and meta.json under OUT_ROOT per run.

Edit the CONFIG constants below.
"""

VIT_VARIANT = "vit_small_patch8_224"

MODEL_RUNS = [
    "swin_tiny_patch4_window7_224",
]

DATA_ROOT = r"D:\datasets\tiny_imagenet64_hf"
OUT_ROOT  = r"D:\runs\tiny_imagenet64_vit"

EPOCHS = 100

BATCH_SIZE = 128
GRAD_ACCUM_STEPS = 1
NUM_WORKERS = 8
PIN_MEMORY = True

IMG_SIZE = 64
NUM_CLASSES = 200

USE_REPEAT_AUG = True
REPEAT_AUG_REPEATS = 3

ACTIVATION = "rational"

ALWAYS_DISABLE_NORMS_WHEN_RATIONAL = True

RATIONAL_APPROX_FUNC = "gelu"
RATIONAL_DEGREES = (5, 4)
RATIONAL_VERSION = "A"

RATIONAL_LR_MULT = 16.0
RATIONAL_WEIGHT_DECAY = 0

WEIGHT_DECAY = 0.05
BETAS = (0.9, 0.999)
EPS = 1e-8
CLIP_GRAD_NORM = 1.0

BASE_LR = 2.0e-3
MIN_LR  = 1.0e-6
WARMUP_EPOCHS = 5

LABEL_SMOOTHING = 0.1
DROP_PATH_RATE = 0.1
COLOR_JITTER = 0.4

RAND_AUG_N = 2
RAND_AUG_M = 9
REPROB = 0.25

MIXUP_ALPHA = 0.8
CUTMIX_ALPHA = 1.0
MIXUP_PROB = 1.0
SWITCH_PROB = 0.5
MIXUP_MODE = "batch"

USE_EMA = True
EMA_DECAY = 0.999

USE_AMP = False

SEED = 1

RUN_NAME = None
RESUME_FROM_LAST = True

import os
import math
import json
import time
import random
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Iterator, Set

import numpy as np

HF_HOME = os.path.join(DATA_ROOT, "hf_home")
HF_DATASETS_CACHE = os.path.join(DATA_ROOT, "hf_datasets")
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

import timm
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD  = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = int(torch.initial_seed() % (2**32))
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class HFTinyImageNetTorch(Dataset):
    """PyTorch Dataset wrapper over a Hugging Face split with optional torchvision transforms."""
    def __init__(self, hf_split, transform=None):
        self.ds = hf_split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.ds[idx]
        img = item["image"]
        y = int(item["label"])

        if hasattr(img, "mode") and img.mode != "RGB":
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        return img, y


def build_transforms(img_size: int):
    """Return (train_transform, val_transform) for the given square image size."""
    train_tf = T.Compose([
        T.RandomResizedCrop(
            img_size,
            scale=(0.6, 1.0),
            ratio=(0.75, 1.3333),
            interpolation=InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=COLOR_JITTER, contrast=COLOR_JITTER, saturation=COLOR_JITTER, hue=0.1),
        T.RandAugment(num_ops=RAND_AUG_N, magnitude=RAND_AUG_M),
        T.ToTensor(),
        T.Normalize(IMNET_MEAN, IMNET_STD),
        T.RandomErasing(p=REPROB, value="random"),
    ])

    val_tf = T.Compose([
        T.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMNET_MEAN, IMNET_STD),
    ])
    return train_tf, val_tf


def accuracy_top1(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    correct = (pred == target).float().sum().item()
    return 100.0 * correct / max(1, target.numel())


class AverageMeter:
    """Online average accumulator with (sum, count, avg)."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.cnt = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / max(1, self.cnt)


@dataclass
class CosineWarmupLRSchedule:
    """Cosine decay LR schedule with linear warmup over warmup_steps, evaluated per step."""
    base_lr: float
    min_lr: float
    warmup_steps: int
    total_steps: int

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * float(step + 1) / float(max(1, self.warmup_steps))
        t = float(step - self.warmup_steps)
        T = float(max(1, self.total_steps - self.warmup_steps))
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, t / T)))
        return self.min_lr + (self.base_lr - self.min_lr) * cos


def set_optimizer_lr(optimizer: optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        mult = float(pg.get("lr_mult", 1.0))
        pg["lr"] = lr * mult


class ModelEMA:
    """Exponential moving average wrapper that tracks a deep-copied model state."""
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.ema = self._clone_model(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _clone_model(model: nn.Module) -> nn.Module:
        import copy
        return copy.deepcopy(model)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k in esd.keys():
            if k in msd:
                esd[k].mul_(self.decay).add_(msd[k], alpha=(1.0 - self.decay))
        self.ema.load_state_dict(esd, strict=True)


class RepeatAugSampler(Sampler[int]):
    """Sampler that repeats each dataset index num_repeats times per epoch with epoch-seeded shuffling."""
    def __init__(self, data_source: Dataset, num_repeats: int = 3, shuffle: bool = True, seed: int = 0):
        self.data_source = data_source
        self.num_repeats = int(max(1, num_repeats))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.data_source) * self.num_repeats

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            base = torch.randperm(n, generator=g).tolist()
        else:
            base = list(range(n))

        expanded = []
        for idx in base:
            expanded.extend([idx] * self.num_repeats)

        if self.shuffle:
            perm = torch.randperm(len(expanded), generator=g)
            expanded = [expanded[i] for i in perm.tolist()]

        return iter(expanded)


class ToggleLayerNorm(nn.Module):
    """LayerNorm wrapper that can bypass normalization when enabled=False."""
    def __init__(self, ln: nn.LayerNorm, enabled: bool = True):
        super().__init__()
        self.ln = ln
        self.enabled = bool(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            return self.ln(x)
        return x


def wrap_vit_block_norms_for_warmup(model: nn.Module, enabled: bool) -> Tuple[int, int]:
    """Replace attributes named norm1/norm2 that are LayerNorm with ToggleLayerNorm wrappers."""
    c1, c2 = 0, 0
    for m in model.modules():
        if hasattr(m, "norm1"):
            n1 = getattr(m, "norm1")
            if isinstance(n1, nn.LayerNorm):
                setattr(m, "norm1", ToggleLayerNorm(n1, enabled=enabled))
                c1 += 1
            elif isinstance(n1, ToggleLayerNorm):
                n1.enabled = bool(enabled)

        if hasattr(m, "norm2"):
            n2 = getattr(m, "norm2")
            if isinstance(n2, nn.LayerNorm):
                setattr(m, "norm2", ToggleLayerNorm(n2, enabled=enabled))
                c2 += 1
            elif isinstance(n2, ToggleLayerNorm):
                n2.enabled = bool(enabled)

    return c1, c2


def set_vit_block_norms_enabled(model: nn.Module, enabled: bool) -> int:
    """Set ToggleLayerNorm.enabled for all ToggleLayerNorm modules in the model."""
    cnt = 0
    for m in model.modules():
        if isinstance(m, ToggleLayerNorm):
            m.enabled = bool(enabled)
            cnt += 1
    return cnt


USE_RATIONAL_ACTIVATION = (ACTIVATION.lower() == "rational")

Rational = None
if USE_RATIONAL_ACTIVATION:
    try:
        from rational.torch import Rational as _Rational
        Rational = _Rational
    except Exception as e:
        raise RuntimeError(
            "ACTIVATION='rational' but could not import `from rational.torch import Rational`.\n"
            "Install/build rational_activations first (and ensure it's compatible with your torch/cuda).\n"
            f"Original import error: {e}"
        )


class RationalFP32(nn.Module):
    """Wrap Rational so its forward executes in fp32 under autocast and casts output back to input dtype."""
    def __init__(self, approx_func: str, degrees=(5, 4), version="A"):
        super().__init__()
        assert Rational is not None
        self.r = Rational(approx_func=approx_func, degrees=degrees, version=version)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return self.r(x)
        out_dtype = x.dtype
        with torch.amp.autocast(device_type="cuda", enabled=False):
            y = self.r(x.float())
        return y.to(dtype=out_dtype)


def _make_rational_gelu() -> nn.Module:
    assert Rational is not None
    return RationalFP32(approx_func=RATIONAL_APPROX_FUNC, degrees=RATIONAL_DEGREES, version=RATIONAL_VERSION)


def replace_gelu_with_rational(module: nn.Module) -> int:
    """Recursively replace nn.GELU submodules with RationalFP32 configured by RATIONAL_* constants."""
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            setattr(module, name, _make_rational_gelu())
            count += 1
        else:
            count += replace_gelu_with_rational(child)
    return count


def collect_rational_param_ids(model: nn.Module) -> Set[int]:
    """Return a set of parameter id(...) values belonging to RationalFP32 submodules."""
    ids: Set[int] = set()
    for m in model.modules():
        if isinstance(m, RationalFP32):
            for p in m.parameters(recurse=True):
                if p.requires_grad:
                    ids.add(id(p))
    return ids


def param_groups_weight_decay_with_rational(
    model: nn.Module,
    weight_decay: float,
    rat_lr_mult: float,
    rat_weight_decay: float,
) -> List[Dict[str, Any]]:
    """Build AdamW parameter groups with (decay, no_decay, rational) partitioning and lr_mult metadata."""
    rat_ids = collect_rational_param_ids(model)
    decay, no_decay, rat = [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if id(p) in rat_ids:
            rat.append(p)
            continue

        if p.ndim == 1 or name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    groups: List[Dict[str, Any]] = [
        {"params": decay,    "weight_decay": float(weight_decay), "lr_mult": 1.0},
        {"params": no_decay, "weight_decay": 0.0,                 "lr_mult": 1.0},
    ]

    if len(rat) > 0:
        groups.append({"params": rat, "weight_decay": float(rat_weight_decay), "lr_mult": float(rat_lr_mult)})

    return groups


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    ce = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = ce(logits, y)
        acc1 = accuracy_top1(logits, y)

        bs = x.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(acc1, bs)

    return {"loss": float(loss_meter.avg), "acc1": float(acc_meter.avg)}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: CosineWarmupLRSchedule,
    device: torch.device,
    epoch: int,
    mixup_fn: Optional[Mixup],
    scaler: Optional[torch.amp.GradScaler],
    steps_per_epoch: int,
    ema: Optional[ModelEMA],
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING) if LABEL_SMOOTHING > 0 else nn.CrossEntropyLoss()

    if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
        try:
            loader.sampler.set_epoch(epoch)  # type: ignore[attr-defined]
        except Exception:
            pass

    optimizer.zero_grad(set_to_none=True)
    global_step_base = epoch * steps_per_epoch

    autocast_enabled = (scaler is not None and scaler.is_enabled())

    for it, (x, y) in enumerate(loader):
        step = global_step_base + it
        lr = scheduler.lr_at(step)
        set_optimizer_lr(optimizer, lr)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if mixup_fn is not None:
            x, y_mix = mixup_fn(x, y)
            y_for_acc = y
        else:
            y_mix = y
            y_for_acc = y

        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
            logits = model(x)
            loss = criterion(logits, y_mix)
            loss = loss / float(max(1, GRAD_ACCUM_STEPS))

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (it + 1) % GRAD_ACCUM_STEPS == 0:
            if CLIP_GRAD_NORM and CLIP_GRAD_NORM > 0:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)

            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update(model)

        bs = x.size(0)
        loss_meter.update(float(loss.item()) * float(max(1, GRAD_ACCUM_STEPS)), bs)
        acc_meter.update(accuracy_top1(logits.detach(), y_for_acc), bs)

    return {"loss": float(loss_meter.avg), "acc1": float(acc_meter.avg)}


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def try_load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    ema: Optional[ModelEMA],
    strict_model_load: bool = True,
) -> int:
    if not path.exists():
        return 0
    ckpt = torch.load(str(path), map_location="cpu")

    if strict_model_load:
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        model.load_state_dict(ckpt["model"], strict=False)

    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if ema is not None and ckpt.get("ema") is not None:
        if strict_model_load:
            ema.ema.load_state_dict(ckpt["ema"], strict=True)
        else:
            ema.ema.load_state_dict(ckpt["ema"], strict=False)

    return int(ckpt.get("epoch", 0))


def run_single_model(
    model_variant: str,
    run_index: int,
    train_split,
    val_split,
    device: torch.device,
) -> None:
    seed_everything(SEED)

    torch.cuda.empty_cache()

    ts = time.strftime("%Y%m%d_%H%M%S")
    ra_tag = f"_ra{REPEAT_AUG_REPEATS}" if (USE_REPEAT_AUG and REPEAT_AUG_REPEATS > 1) else ""
    act_tag = f"_{ACTIVATION.lower()}"
    norm_tag = "_noNorm12" if (USE_RATIONAL_ACTIVATION and ALWAYS_DISABLE_NORMS_WHEN_RATIONAL) else ""

    if RUN_NAME is None:
        run_name = f"vit_{model_variant}_img{IMG_SIZE}_e{EPOCHS}{ra_tag}{act_tag}{norm_tag}_run{run_index}_{ts}"
    else:
        suffix = f"_run{run_index}" if len(MODEL_RUNS) > 1 else ""
        run_name = f"{RUN_NAME}{suffix}"

    run_dir = Path(OUT_ROOT) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "VIT_VARIANT": model_variant,
        "DATA_ROOT": DATA_ROOT,
        "OUT_ROOT": OUT_ROOT,
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
        "NUM_WORKERS": NUM_WORKERS,
        "IMG_SIZE": IMG_SIZE,
        "NUM_CLASSES": NUM_CLASSES,
        "USE_REPEAT_AUG": USE_REPEAT_AUG,
        "REPEAT_AUG_REPEATS": REPEAT_AUG_REPEATS,
        "ACTIVATION": ACTIVATION,
        "ALWAYS_DISABLE_NORMS_WHEN_RATIONAL": (ALWAYS_DISABLE_NORMS_WHEN_RATIONAL if USE_RATIONAL_ACTIVATION else None),
        "RATIONAL_APPROX_FUNC": (RATIONAL_APPROX_FUNC if USE_RATIONAL_ACTIVATION else None),
        "RATIONAL_DEGREES": (RATIONAL_DEGREES if USE_RATIONAL_ACTIVATION else None),
        "RATIONAL_VERSION": (RATIONAL_VERSION if USE_RATIONAL_ACTIVATION else None),
        "RATIONAL_LR_MULT": (RATIONAL_LR_MULT if USE_RATIONAL_ACTIVATION else None),
        "RATIONAL_WEIGHT_DECAY": (RATIONAL_WEIGHT_DECAY if USE_RATIONAL_ACTIVATION else None),
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "BETAS": BETAS,
        "EPS": EPS,
        "CLIP_GRAD_NORM": CLIP_GRAD_NORM,
        "BASE_LR": BASE_LR,
        "MIN_LR": MIN_LR,
        "WARMUP_EPOCHS": WARMUP_EPOCHS,
        "LABEL_SMOOTHING": LABEL_SMOOTHING,
        "DROP_PATH_RATE": DROP_PATH_RATE,
        "COLOR_JITTER": COLOR_JITTER,
        "RAND_AUG_N": RAND_AUG_N,
        "RAND_AUG_M": RAND_AUG_M,
        "REPROB": REPROB,
        "MIXUP_ALPHA": MIXUP_ALPHA,
        "CUTMIX_ALPHA": CUTMIX_ALPHA,
        "MIXUP_PROB": MIXUP_PROB,
        "SWITCH_PROB": SWITCH_PROB,
        "MIXUP_MODE": MIXUP_MODE,
        "USE_EMA": USE_EMA,
        "EMA_DECAY": EMA_DECAY,
        "USE_AMP": USE_AMP,
        "SEED": SEED,
        "HF_HOME": HF_HOME,
        "HF_DATASETS_CACHE": HF_DATASETS_CACHE,
        "RUN_INDEX": run_index,
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    train_tf, val_tf = build_transforms(IMG_SIZE)
    train_set = HFTinyImageNetTorch(train_split, transform=train_tf)
    val_set = HFTinyImageNetTorch(val_split, transform=val_tf)

    train_sampler = None
    if USE_REPEAT_AUG and REPEAT_AUG_REPEATS > 1:
        train_sampler = RepeatAugSampler(train_set, num_repeats=REPEAT_AUG_REPEATS, shuffle=True, seed=SEED)

    dl_gen = torch.Generator()
    dl_gen.manual_seed(SEED)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=seed_worker,
        generator=dl_gen,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=seed_worker,
        generator=dl_gen,
    )

    model = timm.create_model(
        model_variant,
        pretrained=False,
        num_classes=NUM_CLASSES,
        img_size=IMG_SIZE,
        drop_path_rate=DROP_PATH_RATE,
    )

    if USE_RATIONAL_ACTIVATION:
        n_rep = replace_gelu_with_rational(model)
        print(
            f"[MODEL] Replaced {n_rep} GELU modules with Rational(approx_func='{RATIONAL_APPROX_FUNC}', "
            f"degrees={RATIONAL_DEGREES}, version='{RATIONAL_VERSION}')."
        )

        if ALWAYS_DISABLE_NORMS_WHEN_RATIONAL:
            c1, c2 = wrap_vit_block_norms_for_warmup(model, enabled=False)
            print(f"[MODEL] Disabled block.norm1 in {c1} blocks and block.norm2 in {c2} blocks (always).")

    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {model_variant}  params={n_params/1e6:.2f}M  img_size={IMG_SIZE}  classes={NUM_CLASSES}")
    if USE_RATIONAL_ACTIVATION and ALWAYS_DISABLE_NORMS_WHEN_RATIONAL:
        print("[MODEL] Activation inside ViT MLP blocks is Rational initialized to GELU (norm1/norm2 disabled).")
    elif USE_RATIONAL_ACTIVATION:
        print("[MODEL] Activation inside ViT MLP blocks is Rational initialized to GELU.")
    else:
        print("[MODEL] Activation inside ViT MLP blocks is GELU (standard ViT).")

    ema = ModelEMA(model, EMA_DECAY) if USE_EMA else None

    mixup_active = (MIXUP_ALPHA > 0) or (CUTMIX_ALPHA > 0)
    mixup_fn = None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=MIXUP_ALPHA,
            cutmix_alpha=CUTMIX_ALPHA,
            cutmix_minmax=None,
            prob=MIXUP_PROB,
            switch_prob=SWITCH_PROB,
            mode=MIXUP_MODE,
            label_smoothing=LABEL_SMOOTHING,
            num_classes=NUM_CLASSES,
        )

    eff_batch = BATCH_SIZE * max(1, GRAD_ACCUM_STEPS)
    scaled_lr = BASE_LR * (eff_batch / 512.0)

    if USE_RATIONAL_ACTIVATION:
        print(
            f"[OPT] AdamW  base_lr={BASE_LR:.2e}  scaled_lr={scaled_lr:.2e}  eff_batch={eff_batch} | "
            f"rational_lr_mult={RATIONAL_LR_MULT}  rational_wd={RATIONAL_WEIGHT_DECAY}"
        )
        pg = param_groups_weight_decay_with_rational(
            model,
            weight_decay=WEIGHT_DECAY,
            rat_lr_mult=RATIONAL_LR_MULT,
            rat_weight_decay=RATIONAL_WEIGHT_DECAY,
        )
    else:
        print(f"[OPT] AdamW  base_lr={BASE_LR:.2e}  scaled_lr={scaled_lr:.2e}  eff_batch={eff_batch}")
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        pg = [
            {"params": decay, "weight_decay": WEIGHT_DECAY, "lr_mult": 1.0},
            {"params": no_decay, "weight_decay": 0.0, "lr_mult": 1.0},
        ]

    optimizer = optim.AdamW(
        pg,
        lr=scaled_lr,
        betas=BETAS,
        eps=EPS,
    )

    steps_per_epoch = len(train_loader)
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = int(WARMUP_EPOCHS * steps_per_epoch)
    scheduler = CosineWarmupLRSchedule(
        base_lr=scaled_lr,
        min_lr=MIN_LR,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    scaler: Optional[torch.amp.GradScaler]
    if USE_AMP and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    else:
        scaler = None
    print(
        f"[AMP] USE_AMP={USE_AMP} | scaler_is_none={scaler is None} | "
        f"scaler_enabled={(scaler.is_enabled() if scaler is not None else False)}"
    )

    start_epoch = 0
    best_acc = 0.0
    last_ckpt = run_dir / "last.pth"
    best_ckpt = run_dir / "best.pth"

    if RESUME_FROM_LAST and last_ckpt.exists():
        strict_load = (not USE_RATIONAL_ACTIVATION)
        start_epoch = try_load_checkpoint(last_ckpt, model, optimizer, scaler, ema, strict_model_load=strict_load)

        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            best_acc = float(meta.get("best_acc", 0.0))
        print(f"[RESUME] start_epoch={start_epoch}  best_acc={best_acc:.2f}  strict_load={strict_load}")

    if USE_RATIONAL_ACTIVATION and ALWAYS_DISABLE_NORMS_WHEN_RATIONAL:
        set_vit_block_norms_enabled(model, enabled=False)
        if ema is not None:
            set_vit_block_norms_enabled(ema.ema, enabled=False)

    log_path = run_dir / "log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,lr,train_loss,train_acc1,val_loss,val_acc1,ema_val_acc1,epoch_time_sec\n", encoding="utf-8")

    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            mixup_fn=mixup_fn,
            scaler=scaler,
            steps_per_epoch=steps_per_epoch,
            ema=ema,
        )

        val_stats = evaluate(model, val_loader, device)
        ema_val_acc = float("nan")
        if ema is not None:
            ema_stats = evaluate(ema.ema, val_loader, device)
            ema_val_acc = ema_stats["acc1"]

        end_step = (epoch + 1) * steps_per_epoch - 1
        cur_lr = scheduler.lr_at(end_step)

        dt = time.time() - t0

        score = ema_val_acc if ema is not None else val_stats["acc1"]
        is_best = score > best_acc
        if is_best:
            best_acc = score

        line = f"{epoch+1},{cur_lr:.8f},{train_stats['loss']:.6f},{train_stats['acc1']:.3f},{val_stats['loss']:.6f},{val_stats['acc1']:.3f},{ema_val_acc:.3f},{dt:.2f}\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)

        payload = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "ema": (ema.ema.state_dict() if ema is not None else None),
            "optimizer": optimizer.state_dict(),
            "scaler": (scaler.state_dict() if (scaler is not None and scaler.is_enabled()) else None),
            "best_acc": best_acc,
            "cfg": cfg,
        }
        save_checkpoint(last_ckpt, payload)
        if is_best:
            save_checkpoint(best_ckpt, payload)

        (run_dir / "meta.json").write_text(json.dumps({"best_acc": best_acc}, indent=2), encoding="utf-8")

        print(
            f"[{epoch+1:03d}/{EPOCHS}] "
            f"lr={cur_lr:.2e} | "
            f"train {train_stats['acc1']:.2f}% (loss {train_stats['loss']:.4f}) | "
            f"val {val_stats['acc1']:.2f}% (loss {val_stats['loss']:.4f}) | "
            f"ema_val {ema_val_acc:.2f}% | "
            f"best {best_acc:.2f}% | "
            f"{dt:.1f}s"
        )

    print(f"\nDone. Best score (EMA if enabled) acc@1 = {best_acc:.2f}%")
    print(f"Run dir:  {run_dir}")
    print(f"Log:      {log_path}")
    print(f"Best ckpt: {best_ckpt}")
    print(f"Last ckpt: {last_ckpt}")

    try:
        del train_loader, val_loader, train_set, val_set, train_tf, val_tf
        del model, optimizer, scheduler, scaler, ema, mixup_fn
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    if ACTIVATION.lower() not in ["gelu", "rational"]:
        raise ValueError("ACTIVATION must be either 'gelu' or 'rational'.")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is strongly recommended for ViT training.")

    from datasets import load_dataset
    ds = load_dataset("zh-plus/tiny-imagenet", cache_dir=HF_DATASETS_CACHE)

    train_split = ds["train"]
    if "valid" in ds:
        val_split = ds["valid"]
    elif "validation" in ds:
        val_split = ds["validation"]
    elif "val" in ds:
        val_split = ds["val"]
    else:
        raise RuntimeError(f"Could not find validation split in: {list(ds.keys())}")

    print(f"[DATA] train={len(train_split)}  val={len(val_split)}  cache={HF_DATASETS_CACHE}")

    for i, variant in enumerate(MODEL_RUNS):
        print(f"\n================ RUN {i}: {variant} ================\n")
        run_single_model(
            model_variant=variant,
            run_index=i,
            train_split=train_split,
            val_split=val_split,
            device=device,
        )


if __name__ == "__main__":
    try:
        torch.multiprocessing.freeze_support()
    except Exception:
        pass
    main()
