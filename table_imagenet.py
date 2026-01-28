#!/usr/bin/env python3
"""
Scan Tiny-ImageNet training run folders and summarize:
  - Top-1 EMA validation accuracy (max over epochs) from each log.csv
  - Median epoch time (seconds) from each log.csv

IMPORTANT FIX:
  - Treat rational variants like "rational_noNorm12" as a DISTINCT activation key
    (so you get all runs, not merged into a single "rational" bucket).

ALSO:
  - Plot EMA score vs epoch for ALL runs (single figure, no legend)
  - Add a zoom-in inset over the last ~10-20 epochs (configurable)
  - Use large fonts for title
  - Save figure as a PDF

IMPORTANT PLOT FIX:
  - Do NOT rely on Matplotlib's implicit color cycling / plot order.
  - Instead: enforce a deterministic plot order (model, activation) and assign
    explicit tab10 colors so your LaTeX legend ALWAYS matches the real colors.

Writes:
  - summary_best_ema.csv (per-run)
  - summary_best_ema_by_model_activation.csv (grouped by (model, activation_variant))
  - ema_vs_epoch_all_runs.pdf
"""

import os
import re
import json
import csv
import math
import statistics
from typing import Dict, List, Optional, Tuple, Any

# Headless-friendly plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# CONFIG (NO CLI)
# =========================

def find_base_root() -> str:
    """
    Try a few likely locations for the runs directory.
    Edit candidates if your path is different.
    """
    candidates = [
        r"/mnt/d/runs/rational_tiny_imagenet",      # WSL (your example)
        r"D:\runs\rational_tiny_imagenet",          # Windows
        r"/mnt/d/runs/rational_tiny_imagenet/",     # WSL (with trailing slash)
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return os.getcwd()


BASE_ROOT = find_base_root()

# Output files
OUT_PER_RUN_CSV = os.path.join(BASE_ROOT, "summary_best_ema.csv")
OUT_GROUPED_CSV = os.path.join(BASE_ROOT, "summary_best_ema_by_model_activation.csv")
OUT_PLOT_PDF = os.path.join(BASE_ROOT, "ema_vs_epoch_all_runs.pdf")

# File to look for inside each run directory
LOG_NAME = "log.csv"

# ---- Inset config (last epochs zoom) ----
INSET_LAST_EPOCHS = 15  # last ~10–20 epochs (edit as desired)
# Inset placement in axes coordinates: (x0, y0, w, h)
INSET_RECT = (0.52, 0.18, 0.45, 0.35)  # raised bottom-right, avoids x-label overlap
INSET_ZORDER = 10
INSET_FACE_ALPHA = 0.98
INSET_TICK_FONTSIZE = 10

# ---- Deterministic plot order + explicit colors (tab10) ----
# Matplotlib tab10 RGB (matches the LaTeX \definecolor{mat...}{RGB}{...} you use).
TAB10_NAMED_RGB: List[Tuple[str, Tuple[int, int, int]]] = [
    ("matblue",   (31, 119, 180)),
    ("matorange", (255, 127, 14)),
    ("matgreen",  (44, 160, 44)),
    ("matred",    (214, 39, 40)),
    ("matpurple", (148, 103, 189)),
    ("matbrown",  (140, 86, 75)),
    ("matpink",   (227, 119, 194)),
    ("matgray",   (127, 127, 127)),
    ("matolive",  (188, 189, 34)),
    ("matcyan",   (23, 190, 207)),
]

# This is the legend order you want (model, activation) -> color index in tab10 above.
# If a key is absent in your runs, it is simply skipped.
DESIRED_KEY_ORDER: List[Tuple[str, str]] = [
    ("cait_s24_224", "gelu"),
    ("cait_s24_224", "rational"),
    ("cait_s24_224", "relu"),
    ("swin_tiny_patch4_window7_224", "gelu"),
    ("swin_tiny_patch4_window7_224", "rational"),
    ("swin_tiny_patch4_window7_224", "relu"),
    ("vit_small_patch8_224", "gelu"),
    ("vit_small_patch8_224", "rational"),
    ("vit_small_patch8_224", "rational_nonorm12"),
    ("vit_small_patch8_224", "relu"),
]

MODEL_DISPLAY = {
    "cait_s24_224": "CaiT-S24",
    "swin_tiny_patch4_window7_224": "Swin-Tiny",
    "vit_small_patch8_224": "ViT-Small",
}
ACT_DISPLAY = {
    "gelu": "GELU",
    "relu": "ReLU",
    "silu": "SiLU",
    "swish": "Swish",
    "rational": "Rational",
    "rational_nonorm12": "Rational (noNorm12)",
}


# =========================
# Helpers
# =========================

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return None
        return float(s)
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return None
        return int(float(s))
    except Exception:
        return None


def discover_run_dirs(base_root: str, log_name: str = "log.csv") -> List[str]:
    """
    Recursively find directories containing log.csv.
    """
    run_dirs = []
    for root, _, files in os.walk(base_root):
        if log_name in files:
            run_dirs.append(root)
    run_dirs.sort()
    return run_dirs


def try_read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def parse_activation_variant_from_dirname(run_dir_name: str) -> str:
    """
    Extract the activation token from folder names like:
      ..._ra3_gelu_run1_YYYYMMDD_HHMMSS
      ..._ra3_rational_run0_YYYYMMDD_HHMMSS
      ..._ra3_rational_noNorm12_run0_YYYYMMDD_HHMMSS
      ..._ra3_gelu_YYYYMMDD_HHMMSS

    Returns a LOWERCASED activation variant string (e.g. 'gelu', 'rational', 'rational_nonorm12').
    """
    name = run_dir_name
    lname = name.lower()

    # Prefer parsing the token right after _ra{K}_
    m = re.search(r"_ra\d+_", lname)
    if m:
        start = m.end()
        tail = name[start:]  # keep original case for slicing; we lower at the end

        end = len(tail)

        # Stop before _runX_
        mrun = re.search(r"_run\d+_", tail, flags=re.IGNORECASE)
        if mrun:
            end = min(end, mrun.start())

        # Or stop before trailing _YYYYMMDD_HHMMSS
        mts = re.search(r"_(\d{8})_\d{6}$", tail)
        if mts:
            end = min(end, mts.start())

        token = tail[:end].strip("_")
        if token:
            return token.lower()

    # Fallback: look for common activation substrings
    m2 = re.search(r"_(gelu|relu|silu|swish|leakyrelu|rational)(?:_|$)", lname)
    if m2:
        return m2.group(1).lower()

    return "unknown"


def parse_model_from_dirname(run_dir_name: str) -> str:
    """
    Parse model from folder name like:
      vit_swin_tiny_patch4_window7_224_img64_...
      vit_vit_small_patch8_224_img64_...
      vit_cait_s24_224_img64_...

    Returns the part between leading 'vit_' and '_imgNN'.
    """
    name = run_dir_name
    m = re.search(r"^vit_(.+?)_img\d+", name, re.IGNORECASE)
    if m:
        return m.group(1)
    return re.sub(r"_(\d{8})_\d{6}$", "", name)


def prefer_model_activation_from_config(
    run_dir: str,
    fallback_model: str,
    fallback_act_variant: str
) -> Tuple[str, str]:
    """
    If config.json/meta.json includes model/activation, prefer those.
    BUT: do NOT let a generic 'rational' in config overwrite a more specific
         folder-derived variant like 'rational_nonorm12'.
    """
    cfg = try_read_json(os.path.join(run_dir, "config.json")) or {}
    meta = try_read_json(os.path.join(run_dir, "meta.json")) or {}

    model_keys = ["model", "model_name", "arch", "architecture", "timm_model", "timm_arch", "net"]
    act_keys = ["activation", "act", "act_name", "activation_name"]

    model = None
    for k in model_keys:
        v = cfg.get(k, None)
        if isinstance(v, str) and v.strip():
            model = v.strip()
            break
        v = meta.get(k, None)
        if isinstance(v, str) and v.strip():
            model = v.strip()
            break

    act = None
    for k in act_keys:
        v = cfg.get(k, None)
        if isinstance(v, str) and v.strip():
            act = v.strip().lower()
            break
        v = meta.get(k, None)
        if isinstance(v, str) and v.strip():
            act = v.strip().lower()
            break

    if model is None:
        model = fallback_model

    if act is None:
        act = fallback_act_variant
    else:
        if fallback_act_variant.startswith("rational_") and act == "rational":
            act = fallback_act_variant
        if "rational" in act and fallback_act_variant.startswith("rational_") and act.startswith("rational_"):
            pass

    return model, act


def detect_columns(header: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Return (ema_val_acc_col, val_acc_col, epoch_time_col) from a log.csv header.
    Tries to be robust to slightly different naming.
    """
    cols = [c.strip() for c in header]

    def pick(pred):
        for c in cols:
            if pred(c.lower()):
                return c
        return None

    ema_col = pick(lambda s: ("ema" in s) and ("acc" in s) and ("val" in s))
    if ema_col is None and "ema_val_acc1" in cols:
        ema_col = "ema_val_acc1"

    val_acc_col = pick(lambda s: ("val" in s) and ("acc" in s) and ("ema" not in s))
    if val_acc_col is None and "val_acc1" in cols:
        val_acc_col = "val_acc1"

    t_col = pick(lambda s: ("epoch" in s) and ("time" in s) and ("sec" in s))
    if t_col is None and "epoch_time_sec" in cols:
        t_col = "epoch_time_sec"

    return ema_col, val_acc_col, t_col


def summarize_log_csv(log_path: str) -> Dict[str, Any]:
    """
    Parse log.csv and compute:
      - best_ema_val_acc (max)
      - epoch_of_best_ema
      - val_acc_at_best_epoch (if available)
      - median_epoch_time_sec
      - total_time_sec
      - num_epochs
    """
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []

        ema_col, val_acc_col, t_col = detect_columns(header)

        best_ema = None
        best_epoch = None
        best_val_at_best = None

        epoch_times: List[float] = []
        total_time = 0.0
        num_rows = 0

        for row in reader:
            num_rows += 1

            ep = _safe_int(row.get("epoch", None))
            ema = _safe_float(row.get(ema_col, None)) if ema_col else None
            val_acc = _safe_float(row.get(val_acc_col, None)) if val_acc_col else None
            t = _safe_float(row.get(t_col, None)) if t_col else None

            if t is not None:
                epoch_times.append(t)
                total_time += t

            if ema is not None:
                if (best_ema is None) or (ema > best_ema):
                    best_ema = ema
                    best_epoch = ep
                    best_val_at_best = val_acc

        median_t = statistics.median(epoch_times) if epoch_times else None

    return {
        "best_ema_val_acc": best_ema,
        "best_ema_epoch": best_epoch,
        "val_acc_at_best_ema_epoch": best_val_at_best,
        "median_epoch_time_sec": median_t,
        "total_time_sec": total_time if num_rows > 0 else None,
        "num_epochs": num_rows,
        "ema_column": ema_col,
        "val_acc_column": val_acc_col,
        "time_column": t_col,
    }


def load_ema_curve(log_path: str) -> Tuple[List[int], List[float], Optional[str]]:
    """
    Load (epoch, ema_val_acc) pairs from log.csv for plotting.
    Returns (epochs, emas, ema_column_name).
    If epoch is missing for a row, uses the running index.
    """
    epochs: List[int] = []
    emas: List[float] = []

    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        ema_col, _, _ = detect_columns(header)

        if ema_col is None:
            return [], [], None

        idx = 0
        for row in reader:
            ep = _safe_int(row.get("epoch", None))
            ema = _safe_float(row.get(ema_col, None))
            if ep is None:
                ep = idx
            idx += 1
            if ema is None:
                continue
            epochs.append(ep)
            emas.append(ema)

    # Ensure sorted by epoch (some logs might be unsorted)
    if epochs and emas:
        pairs = sorted(zip(epochs, emas), key=lambda p: p[0])
        epochs = [p[0] for p in pairs]
        emas = [p[1] for p in pairs]

    return epochs, emas, ema_col


def format_hms(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{sec:02d}"
    return f"{m:d}:{sec:02d}"


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _display_label(model: str, activation: str) -> str:
    m = MODEL_DISPLAY.get(model, model)
    a = ACT_DISPLAY.get(activation, activation)
    return f"{m} + {a}"


def _tab10_color(idx: int) -> Tuple[float, float, float]:
    _, (r, g, b) = TAB10_NAMED_RGB[idx % len(TAB10_NAMED_RGB)]
    return (r / 255.0, g / 255.0, b / 255.0)


def plot_all_runs_ema(curves: List[Dict[str, Any]], out_pdf: str) -> None:
    """
    Plot EMA vs epoch for all runs (no legend), add inset over last epochs, save to PDF.
    curves: list of dicts with keys: model, activation, epochs, emas, ...
    IMPORTANT: curves MUST already be in the desired plotting order, and each curve must
               carry an explicit 'plot_color_index' to match your LaTeX legend colors.
    """
    if not curves:
        print("[WARN] No EMA curves available to plot.")
        return

    fig = plt.figure(figsize=(10.5, 6.5))
    ax = plt.gca()

    plotted: List[Dict[str, Any]] = []
    max_epoch_seen: Optional[int] = None

    for c in curves:
        xs = c.get("epochs", [])
        ys = c.get("emas", [])
        if not xs or not ys:
            continue

        color_idx = int(c.get("plot_color_index", 0))
        color = _tab10_color(color_idx)

        ax.plot(xs, ys, linewidth=1.6, alpha=0.90, color=color)

        plotted.append({"xs": xs, "ys": ys, "color": color})
        try:
            xe = max(xs)
            max_epoch_seen = xe if (max_epoch_seen is None or xe > max_epoch_seen) else max_epoch_seen
        except Exception:
            pass

    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("EMA Val Acc", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_title("EMA Score vs Epoch", fontsize=24, pad=12)
    ax.grid(True, linewidth=0.6, alpha=0.35)

    # Inset: last epochs zoom
    if plotted and (max_epoch_seen is not None) and (max_epoch_seen >= 1):
        x_hi = int(max_epoch_seen)
        x_lo = max(1, x_hi - int(INSET_LAST_EPOCHS) + 1)

        inset_ax = ax.inset_axes(INSET_RECT, zorder=INSET_ZORDER)
        inset_ax.set_facecolor("white")
        inset_ax.patch.set_alpha(INSET_FACE_ALPHA)

        inset_yvals: List[float] = []
        for p in plotted:
            xs = p["xs"]
            ys = p["ys"]
            color = p["color"]

            xs2: List[int] = []
            ys2: List[float] = []
            for x, y in zip(xs, ys):
                if x_lo <= x <= x_hi:
                    xs2.append(x)
                    ys2.append(y)

            if not xs2:
                continue

            inset_ax.plot(xs2, ys2, linewidth=1.6, alpha=0.95, color=color)
            inset_yvals.extend(ys2)

        inset_ax.set_xlim(x_lo, x_hi)

        if inset_yvals:
            y0 = min(inset_yvals)
            y1 = max(inset_yvals)
            pad = 0.12 * max(1e-6, (y1 - y0))
            inset_ax.set_ylim(y0 - pad, y1 + pad)

        inset_ax.grid(True, linewidth=0.5, alpha=0.25)
        inset_ax.tick_params(axis="both", labelsize=INSET_TICK_FONTSIZE)
        inset_ax.set_xlabel("")
        inset_ax.set_ylabel("")
        for spine in inset_ax.spines.values():
            spine.set_linewidth(1.0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


# =========================
# Main
# =========================

def main() -> None:
    run_dirs = discover_run_dirs(BASE_ROOT, LOG_NAME)
    if not run_dirs:
        print(f"[WARN] No '{LOG_NAME}' found under: {BASE_ROOT}")
        return

    per_run_rows: List[Dict[str, Any]] = []

    # Collect curves indexed by (model, activation) so we can enforce plot order
    curves_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    extra_curves: List[Dict[str, Any]] = []

    for rd in run_dirs:
        run_name = os.path.basename(rd.rstrip("/\\"))
        log_path = os.path.join(rd, LOG_NAME)

        parsed_model = parse_model_from_dirname(run_name)
        parsed_act_variant = parse_activation_variant_from_dirname(run_name)

        model, act_variant = prefer_model_activation_from_config(
            rd, parsed_model, parsed_act_variant
        )

        summ = summarize_log_csv(log_path)

        best_ema = summ["best_ema_val_acc"]
        best_ep = summ["best_ema_epoch"]
        val_at_best = summ["val_acc_at_best_ema_epoch"]
        med_t = summ["median_epoch_time_sec"]
        tot_t = summ["total_time_sec"]
        n_ep = summ["num_epochs"]

        row = {
            "run_dir": run_name,
            "abs_path": rd,
            "model": model,
            "activation": act_variant,
            "best_ema_val_acc1": best_ema,
            "best_ema_epoch": best_ep,
            "val_acc1_at_best_ema_epoch": val_at_best,
            "median_epoch_time_sec": med_t,
            "median_epoch_time_min": (med_t / 60.0) if med_t is not None else None,
            "total_time_sec": tot_t,
            "total_time_hms": format_hms(tot_t),
            "num_epochs": n_ep,
            "ema_column": summ["ema_column"],
            "time_column": summ["time_column"],
        }
        per_run_rows.append(row)

        # Load curve for plotting (EMA vs epoch)
        epochs, emas, ema_col = load_ema_curve(log_path)
        if ema_col is None or not epochs or not emas:
            print(f"[WARN] Skipping plot curve (no EMA column/data): {run_name}")
            continue

        curve = {
            "run_dir": run_name,
            "abs_path": rd,
            "model": model,
            "activation": act_variant,
            "display_label": _display_label(model, act_variant),
            "ema_column": ema_col,
            "epochs": epochs,
            "emas": emas,
            "plot_color_index": None,  # fill later
        }

        key = (model, act_variant)
        if key in DESIRED_KEY_ORDER:
            curves_by_key.setdefault(key, []).append(curve)
        else:
            extra_curves.append(curve)

    # Sort per-run: model, activation variant, best EMA desc
    def _key_per_run(r: Dict[str, Any]):
        b = r["best_ema_val_acc1"]
        b_sort = -b if isinstance(b, (int, float)) else math.inf
        return (str(r["model"]), str(r["activation"]), b_sort, str(r["run_dir"]))

    per_run_rows.sort(key=_key_per_run)

    # Write per-run CSV (contains ALL runs)
    per_run_fields = [
        "run_dir", "model", "activation",
        "best_ema_val_acc1", "best_ema_epoch", "val_acc1_at_best_ema_epoch",
        "median_epoch_time_sec", "median_epoch_time_min",
        "total_time_sec", "total_time_hms", "num_epochs",
        "ema_column", "time_column",
        "abs_path",
    ]
    write_csv(OUT_PER_RUN_CSV, per_run_rows, per_run_fields)

    # Group by (model, activation_variant): take best run by best_ema_val_acc1
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in per_run_rows:
        k = (str(r["model"]), str(r["activation"]))
        grouped.setdefault(k, []).append(r)

    grouped_rows: List[Dict[str, Any]] = []
    for (model, act), rows in grouped.items():
        best_row = None
        for r in rows:
            b = r["best_ema_val_acc1"]
            if not isinstance(b, (int, float)):
                continue
            if (best_row is None) or (b > best_row["best_ema_val_acc1"]):
                best_row = r

        med_times = [
            r["median_epoch_time_sec"]
            for r in rows
            if isinstance(r["median_epoch_time_sec"], (int, float))
        ]
        med_of_meds = statistics.median(med_times) if med_times else None

        grouped_rows.append({
            "model": model,
            "activation": act,
            "best_ema_val_acc1": best_row["best_ema_val_acc1"] if best_row else None,
            "best_ema_epoch": best_row["best_ema_epoch"] if best_row else None,
            "best_run_dir": best_row["run_dir"] if best_row else None,
            "median_epoch_time_sec_across_runs": med_of_meds,
            "median_epoch_time_min_across_runs": (med_of_meds / 60.0) if med_of_meds is not None else None,
            "num_runs": len(rows),
        })

    grouped_rows.sort(
        key=lambda r: (
            -(r["best_ema_val_acc1"] if isinstance(r["best_ema_val_acc1"], (int, float)) else -1e9),
            r["model"],
            r["activation"],
        )
    )

    grouped_fields = [
        "model", "activation",
        "best_ema_val_acc1", "best_ema_epoch", "best_run_dir",
        "median_epoch_time_sec_across_runs", "median_epoch_time_min_across_runs",
        "num_runs",
    ]
    write_csv(OUT_GROUPED_CSV, grouped_rows, grouped_fields)

    # -------------------------
    # Build curves_for_plot in EXACT legend order and assign explicit tab10 colors
    # -------------------------
    curves_for_plot: List[Dict[str, Any]] = []
    color_mapping_lines: List[str] = []

    color_idx = 0
    for key in DESIRED_KEY_ORDER:
        curves_here = curves_by_key.get(key, [])
        if not curves_here:
            continue

        # If you ever have multiple runs with the same (model, activation), we still plot all of them.
        # They will share the SAME color (since your LaTeX legend is per (model, activation)).
        for c in curves_here:
            c["plot_color_index"] = color_idx
            curves_for_plot.append(c)

        # Record mapping once per key
        model, act = key
        label = _display_label(model, act)
        latex_color_name = TAB10_NAMED_RGB[color_idx % len(TAB10_NAMED_RGB)][0]
        color_mapping_lines.append(f"  {latex_color_name:9s} -> {label}")

        color_idx += 1
        if color_idx >= len(TAB10_NAMED_RGB):
            break

    # Any curves not covered by DESIRED_KEY_ORDER: append after, with remaining colors (still explicit).
    if extra_curves:
        extra_curves.sort(key=lambda c: (str(c["model"]), str(c["activation"]), str(c["run_dir"])))
        for c in extra_curves:
            if color_idx >= len(TAB10_NAMED_RGB):
                break
            c["plot_color_index"] = color_idx
            curves_for_plot.append(c)
            latex_color_name = TAB10_NAMED_RGB[color_idx % len(TAB10_NAMED_RGB)][0]
            color_mapping_lines.append(f"  {latex_color_name:9s} -> {c['display_label']} (extra)")
            color_idx += 1

    # Plot EMA vs epoch for all runs (no legend) + inset -> PDF
    plot_all_runs_ema(curves_for_plot, OUT_PLOT_PDF)

    # Console summary
    print(f"[OK] Scanned {len(per_run_rows)} runs under: {BASE_ROOT}")
    print(f"[OK] Wrote per-run summary (ALL runs): {OUT_PER_RUN_CSV}")
    print(f"[OK] Wrote grouped summary (by model+activation-variant): {OUT_GROUPED_CSV}")
    print(f"[OK] Wrote EMA vs epoch plot (ALL runs, no legend, inset): {OUT_PLOT_PDF}")

    print("\nColor mapping used in the PDF (this MUST match your LaTeX legend):")
    for line in color_mapping_lines:
        print(line)

    print("\nGrouped results:")
    for r in grouped_rows:
        b = r["best_ema_val_acc1"]
        b_str = f"{b:.2f}" if isinstance(b, (int, float)) else "NA"
        print(f"  {r['model']:35s}  {r['activation']:18s}  {b_str:>6s}  {r['best_run_dir']}")

    print("\nPer-run results (ALL runs):")
    for r in per_run_rows:
        b = r["best_ema_val_acc1"]
        b_str = f"{b:.2f}" if isinstance(b, (int, float)) else "NA"
        med = r["median_epoch_time_sec"]
        med_str = f"{med:.2f}s" if isinstance(med, (int, float)) else "NA"
        print(f"  {r['model']:35s}  {r['activation']:18s}  {b_str:>6s}  med_epoch={med_str:>8s}  {r['run_dir']}")


if __name__ == "__main__":
    main()
