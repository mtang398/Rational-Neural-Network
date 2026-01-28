# Summarize per-run best Top-1 test accuracy from metrics.jsonl under BASE_ROOT and write CSV/TXT.
# Optionally generate mean±std test-accuracy-vs-epoch PDFs for selected cases.

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_ROOT = Path(r"D:\runs\rational_cifar10_result")

ROOT_VARIANTS = [
    "cifar10_GN",
    "cifar10_no_GN",
    "cifar10_plain",
]

EXPECTED_SEEDS = [0, 1, 2, 3, 4]

FAIL_ACC_FRAC = 0.10

OUT_CSV = BASE_ROOT / "summary_top1_testacc.csv"
OUT_TXT = BASE_ROOT / "summary_top1_testacc.txt"

PLOT_SCORE_CURVES = True
PLOT_DIR = BASE_ROOT / "score_curves_pdf"
INSET_LAST_EPOCHS = 15
INSET_ONLY_FOR_BOOSTED = True

INSET_RECT = (0.50, 0.20, 0.47, 0.38)
INSET_ZORDER = 10
INSET_FACE_ALPHA = 0.98

LEGEND_FRAME_ALPHA = 0.92

TITLE_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 17
TICK_FONTSIZE = 14
INSET_TICK_FONTSIZE = 10


def _safe_list_subdirs(p: Path) -> List[str]:
    if not p.exists() or not p.is_dir():
        return []
    out = []
    for x in p.iterdir():
        if x.is_dir() and not x.name.startswith("."):
            out.append(x.name)
    out.sort()
    return out


def _normalize_acc_to_frac(v: float) -> float:
    if not math.isfinite(v) or v < 0:
        return FAIL_ACC_FRAC
    if v <= 1.0 + 1e-6:
        return float(v)
    if v <= 100.0 + 1e-6:
        return float(v) / 100.0
    return 1.0


def _mean_std(vals: List[float], sample_std: bool = True) -> Tuple[float, float]:
    n = len(vals)
    if n == 0:
        return FAIL_ACC_FRAC, 0.0
    m = sum(vals) / n
    if n == 1:
        return m, 0.0
    ss = sum((x - m) ** 2 for x in vals)
    denom = (n - 1) if sample_std else n
    if denom <= 0:
        return m, 0.0
    return m, math.sqrt(ss / denom)


def _fmt_pct(frac: float, digits: int = 2) -> str:
    return f"{100.0 * frac:.{digits}f}"


def _read_best_test_acc(metrics_path: Path) -> Tuple[Optional[float], Optional[int], Optional[Dict]]:
    """
    Returns: (best_test_acc_frac, best_epoch, meta_dict_from_first_seen_line)
    If file unreadable or no supported acc key found => (None, None, None)
    """
    if not metrics_path.exists():
        return None, None, None

    best = None
    best_epoch = None
    meta = None

    acc_keys = (
        "test_acc",
        "test_accuracy",
        "acc_test",
        "eval_test_acc",
        "eval_acc",
        "val_acc",
        "val_accuracy",
    )

    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue

                if meta is None:
                    meta = {
                        "dataset": obj.get("dataset"),
                        "model": obj.get("model"),
                        "activation": obj.get("activation"),
                    }

                acc_val = None
                for k in acc_keys:
                    if k in obj:
                        acc_val = obj.get(k)
                        break
                if acc_val is None:
                    continue

                try:
                    acc_val_f = float(acc_val)
                except Exception:
                    continue

                acc_frac = _normalize_acc_to_frac(acc_val_f)

                if best is None or acc_frac > best:
                    best = acc_frac
                    ep = obj.get("epoch", None)
                    try:
                        best_epoch = int(ep) if ep is not None else None
                    except Exception:
                        best_epoch = None

        return best, best_epoch, meta
    except Exception:
        return None, None, None


def _read_epoch_curve(metrics_path: Path) -> Tuple[Optional[Dict[int, float]], Optional[Dict]]:
    """
    Returns:
      curve: dict epoch(int) -> test_acc_frac
      meta: dict with model/activation (first seen)
    If unreadable or no supported test-acc key found => (None, None)
    """
    if not metrics_path.exists():
        return None, None

    acc_keys = (
        "test_acc",
        "test_accuracy",
        "acc_test",
        "eval_test_acc",
        "eval_acc",
        "val_acc",
        "val_accuracy",
    )

    curve: Dict[int, float] = {}
    meta = None
    seen_any = False

    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue

                if meta is None:
                    meta = {
                        "model": obj.get("model"),
                        "activation": obj.get("activation"),
                    }

                ep = obj.get("epoch", None)
                try:
                    ep_i = int(ep)
                except Exception:
                    continue

                acc_val = None
                for k in acc_keys:
                    if k in obj:
                        acc_val = obj.get(k)
                        break
                if acc_val is None:
                    continue

                try:
                    acc_val_f = float(acc_val)
                except Exception:
                    continue

                acc_frac = _normalize_acc_to_frac(acc_val_f)
                curve[ep_i] = acc_frac
                seen_any = True

        if not seen_any:
            return None, None
        return curve, meta
    except Exception:
        return None, None


def _parse_exp_name(exp_name: str) -> Tuple[str, str]:
    toks = exp_name.split("_")
    if len(toks) < 3:
        return "", exp_name

    model = toks[1]
    start_idx = 2
    if len(toks) >= 4 and toks[2] == "plain":
        start_idx = 3
    activation = "_".join(toks[start_idx:]) if start_idx < len(toks) else ""
    return model, activation


def _base_activation_key(act_str: str) -> Optional[str]:
    a = (act_str or "").lower()

    if a.startswith("rational_gelu"):
        return "rational_gelu"
    if a.startswith("rational_leaky_relu"):
        return "rational_leaky_relu"

    if a.startswith("gelu"):
        return "gelu"
    if a.startswith("relu"):
        return "relu"
    if a.startswith("swish") or a.startswith("silu"):
        return "swish"
    if a.startswith("leaky_relu"):
        return "leaky_relu"

    return None


def _display_name(base_key: str) -> str:
    if base_key == "gelu":
        return "GELU"
    if base_key == "relu":
        return "ReLU"
    if base_key == "swish":
        return "Swish"
    if base_key == "leaky_relu":
        return "LeakyReLU"
    if base_key == "rational_leaky_relu":
        return "Rational 1"
    if base_key == "rational_gelu":
        return "Rational 2"
    return base_key


def _case_title(model: str, variant: str) -> str:
    m = model.upper()
    if variant == "plain":
        return f"{m} plain"
    if variant == "boosted_noGN":
        return f"{m} boosted (no GN)"
    if variant == "boosted_GN":
        return f"{m} boosted (GN)"
    return f"{m} {variant}"


def _select_best_folder_for_base_key(
    rv_path: Path,
    exp_folders: List[str],
    base_key: str,
) -> Optional[str]:
    candidates = []
    for exp in exp_folders:
        _model_guess, act_guess = _parse_exp_name(exp)
        bk = _base_activation_key(act_guess)
        if bk != base_key:
            continue
        candidates.append(exp)

    if not candidates:
        return None

    best_exp = None
    best_mean = -1.0
    best_missing = 10**9

    for exp in candidates:
        exp_path = rv_path / exp
        seed_best: List[float] = []
        n_missing = 0

        for s in EXPECTED_SEEDS:
            metrics_path = exp_path / f"seed_{s}" / "metrics.jsonl"
            best, _, _ = _read_best_test_acc(metrics_path)
            if best is None:
                seed_best.append(FAIL_ACC_FRAC)
                n_missing += 1
            else:
                seed_best.append(best)

        mean_acc, _ = _mean_std(seed_best, sample_std=True)

        if (n_missing < best_missing) or (n_missing == best_missing and mean_acc > best_mean):
            best_missing = n_missing
            best_mean = mean_acc
            best_exp = exp

    return best_exp


def _load_mean_std_curve(exp_path: Path) -> Tuple[Optional[List[int]], Optional[List[float]], Optional[List[float]]]:
    seed_curves: List[Dict[int, float]] = []
    max_epoch = 0

    for s in EXPECTED_SEEDS:
        mp = exp_path / f"seed_{s}" / "metrics.jsonl"
        curve, _ = _read_epoch_curve(mp)
        if curve is None or len(curve) == 0:
            continue
        seed_curves.append(curve)
        max_epoch = max(max_epoch, max(curve.keys()))

    if len(seed_curves) == 0 or max_epoch <= 0:
        return None, None, None

    import numpy as np

    S = len(seed_curves)
    E = max_epoch
    mat = np.full((S, E), np.nan, dtype=np.float64)

    for i, curve in enumerate(seed_curves):
        for ep, acc_frac in curve.items():
            if 1 <= ep <= E:
                mat[i, ep - 1] = float(acc_frac)

    mean_frac = np.nanmean(mat, axis=0)
    std_frac = np.zeros(E, dtype=np.float64)
    for j in range(E):
        col = mat[:, j]
        col = col[np.isfinite(col)]
        if col.size >= 2:
            std_frac[j] = float(np.std(col, ddof=1))
        else:
            std_frac[j] = 0.0

    epochs = list(range(1, E + 1))
    mean_pct = (100.0 * mean_frac).tolist()
    std_pct = (100.0 * std_frac).tolist()
    return epochs, mean_pct, std_pct


def _rects_intersect(r1: Tuple[float, float, float, float], r2: Tuple[float, float, float, float]) -> bool:
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    a1x0, a1x1 = x1, x1 + w1
    a1y0, a1y1 = y1, y1 + h1
    a2x0, a2x1 = x2, x2 + w2
    a2y0, a2y1 = y2, y2 + h2
    return (a1x0 < a2x1) and (a1x1 > a2x0) and (a1y0 < a2y1) and (a1y1 > a2y0)


def _legend_bbox_axes(ax, fig, legend) -> Tuple[float, float, float, float]:
    renderer = fig.canvas.get_renderer()
    bb_disp = legend.get_window_extent(renderer=renderer)
    bb_axes = ax.transAxes.inverted().transform_bbox(bb_disp)
    return (float(bb_axes.x0), float(bb_axes.y0), float(bb_axes.width), float(bb_axes.height))


def _legend_overlap_score(ax, fig, legend, all_xy_disp) -> int:
    renderer = fig.canvas.get_renderer()
    bb = legend.get_window_extent(renderer=renderer)
    x0, y0, x1, y1 = bb.x0, bb.y0, bb.x1, bb.y1
    pts = all_xy_disp
    inside = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    return int(inside.sum())


def _place_legend_inside_avoid_inset(ax, fig, inset_rect: Optional[Tuple[float, float, float, float]], all_xy_disp):
    handles, labels = ax.get_legend_handles_labels()

    candidates = [
        "upper left",
        "upper right",
        "upper center",
        "center left",
        "center",
        "lower left",
    ]

    best = None
    best_score = None

    for loc in candidates:
        leg = ax.legend(handles, labels, loc=loc, framealpha=LEGEND_FRAME_ALPHA)

        fig.canvas.draw()

        if inset_rect is not None:
            leg_rect = _legend_bbox_axes(ax, fig, leg)
            if _rects_intersect(leg_rect, inset_rect):
                leg.remove()
                continue

        score = _legend_overlap_score(ax, fig, leg, all_xy_disp)

        if best is None or score < best_score:
            if best is not None:
                best.remove()
            best = leg
            best_score = score
        else:
            leg.remove()

    if best is None:
        best = ax.legend(handles, labels, loc="upper left", framealpha=LEGEND_FRAME_ALPHA)
        fig.canvas.draw()
    return best


def _plot_case(
    rv: str,
    rv_path: Path,
    model: str,
    variant: str,
    out_pdf: Path,
    want_inset: bool,
) -> None:
    exps_all = _safe_list_subdirs(rv_path)
    if rv == "cifar10_plain":
        exps_all = [e for e in exps_all if "plain" in e]
    else:
        exps_all = [e for e in exps_all if "plain" not in e]

    exps = []
    for e in exps_all:
        mg, _ag = _parse_exp_name(e)
        if (mg or "").lower() == model.lower():
            exps.append(e)

    base_keys_present = set()
    for e in exps:
        _, ag = _parse_exp_name(e)
        bk = _base_activation_key(ag)
        if bk is not None:
            base_keys_present.add(bk)

    preferred = ["gelu", "relu", "swish", "leaky_relu", "rational_leaky_relu", "rational_gelu"]
    base_keys = [k for k in preferred if k in base_keys_present]

    if len(base_keys) == 0:
        print(f"[WARN] No activations found for {rv} model={model}. Skipping plot.")
        return

    chosen: Dict[str, str] = {}
    for bk in base_keys:
        best_exp = _select_best_folder_for_base_key(rv_path, exps, bk)
        if best_exp is not None:
            chosen[bk] = best_exp

    if len(chosen) == 0:
        print(f"[WARN] No readable experiments for {rv} model={model}. Skipping plot.")
        return

    curves = []
    for bk in base_keys:
        if bk not in chosen:
            continue
        exp = chosen[bk]
        exp_path = rv_path / exp
        ep, mean_pct, std_pct = _load_mean_std_curve(exp_path)
        if ep is None:
            continue
        curves.append((bk, ep, mean_pct, std_pct, exp))

    if len(curves) == 0:
        print(f"[WARN] No curve data for {rv} model={model}. Skipping plot.")
        return

    import matplotlib as mpl
    mpl.rcParams["pdf.use14corefonts"] = True
    mpl.rcParams["ps.useafm"] = True

    import matplotlib.pyplot as plt
    import numpy as np

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    all_xy = []
    for (bk, ep, mean_pct, std_pct, _expname) in curves:
        x = ep
        y = mean_pct
        s = std_pct

        line, = ax.plot(x, y, linewidth=2.0, label=_display_name(bk))
        c = line.get_color()
        y_lo = [yy - ss for yy, ss in zip(y, s)]
        y_hi = [yy + ss for yy, ss in zip(y, s)]
        ax.fill_between(x, y_lo, y_hi, color=c, alpha=0.18, linewidth=0)

        all_xy.extend(list(zip(x, y)))

    ax.set_title(_case_title(model, variant), fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Epoch", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Test accuracy (%)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    ax.grid(True, alpha=0.25)

    all_y = []
    for (_bk, _ep, mean_pct, _std_pct, _e) in curves:
        all_y.extend(mean_pct)
    if len(all_y) > 0:
        y_min = min(all_y)
        y_max = max(all_y)
        pad = 0.08 * max(1.0, (y_max - y_min))
        ax.set_ylim(y_min - pad, y_max + pad)

    inset_rect_used = None

    if want_inset:
        Emax = max(len(ep) for (_bk, ep, _m, _s, _e) in curves)
        x_hi = Emax
        x_lo = max(1, x_hi - int(INSET_LAST_EPOCHS) + 1)

        inset_ax = ax.inset_axes(INSET_RECT, zorder=INSET_ZORDER)
        inset_ax.set_facecolor("white")
        inset_ax.patch.set_alpha(INSET_FACE_ALPHA)

        inset_all = []
        for (bk, ep, mean_pct, std_pct, _expname) in curves:
            xs = []
            ys = []
            ss = []
            for x, y, s in zip(ep, mean_pct, std_pct):
                if x_lo <= x <= x_hi:
                    xs.append(x)
                    ys.append(y)
                    ss.append(s)

            if len(xs) == 0:
                continue

            line, = inset_ax.plot(xs, ys, linewidth=2.0)
            c = line.get_color()
            y_lo_ = [yy - ss_ for yy, ss_ in zip(ys, ss)]
            y_hi_ = [yy + ss_ for yy, ss_ in zip(ys, ss)]
            inset_ax.fill_between(xs, y_lo_, y_hi_, color=c, alpha=0.18, linewidth=0)
            inset_all.extend(y_lo_)
            inset_all.extend(y_hi_)

        inset_ax.set_xlim(x_lo, x_hi)

        if len(inset_all) > 0:
            iy0 = min(inset_all)
            iy1 = max(inset_all)
            ipad = 0.12 * max(0.2, (iy1 - iy0))
            inset_ax.set_ylim(iy0 - ipad, iy1 + ipad)

        inset_ax.grid(True, alpha=0.20)
        inset_ax.tick_params(axis="both", labelsize=INSET_TICK_FONTSIZE)
        inset_ax.set_xlabel("")
        inset_ax.set_ylabel("")
        inset_ax.set_zorder(INSET_ZORDER)
        for spine in inset_ax.spines.values():
            spine.set_linewidth(1.0)

        inset_rect_used = INSET_RECT

    fig.canvas.draw()
    all_xy = np.array(all_xy, dtype=np.float64)
    all_xy_disp = ax.transData.transform(all_xy) if all_xy.size else np.zeros((0, 2), dtype=np.float64)
    _place_legend_inside_avoid_inset(ax, fig, inset_rect_used, all_xy_disp)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.90)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"[PLOT] {out_pdf.name}")
    for bk in base_keys:
        if bk in chosen:
            print(f"  - {_display_name(bk):>10s}: {chosen[bk]}")


def main() -> None:
    rows: List[Dict[str, str]] = []

    report_lines: List[str] = []
    report_lines.append(f"BASE_ROOT: {str(BASE_ROOT)}")
    report_lines.append(f"FAIL_ACC: {_fmt_pct(FAIL_ACC_FRAC)}% used for missing/failed seeds (missing metrics.jsonl, parse failure, etc.)")
    report_lines.append("")

    for rv in ROOT_VARIANTS:
        rv_path = BASE_ROOT / rv
        exps = _safe_list_subdirs(rv_path)

        if rv == "cifar10_plain":
            exps = [e for e in exps if "plain" in e]
        else:
            exps = [e for e in exps if "plain" not in e]

        report_lines.append("=" * 80)
        report_lines.append(f"{rv}  ({rv_path})")
        report_lines.append(f"Found {len(exps)} experiment folders.")
        report_lines.append("=" * 80)

        if not rv_path.exists():
            report_lines.append(f"[WARN] Missing root directory: {rv_path}")
            report_lines.append("")
            continue

        if len(exps) == 0:
            report_lines.append("[WARN] No experiment folders found.")
            report_lines.append("")
            continue

        for exp in exps:
            exp_path = rv_path / exp
            model_guess, act_guess = _parse_exp_name(exp)

            seed_best: List[float] = []
            seed_epochs: List[Optional[int]] = []
            n_missing_seeds = 0

            meta_model = None
            meta_act = None

            for s in EXPECTED_SEEDS:
                metrics_path = exp_path / f"seed_{s}" / "metrics.jsonl"
                best, best_ep, meta = _read_best_test_acc(metrics_path)

                if best is None:
                    seed_best.append(FAIL_ACC_FRAC)
                    seed_epochs.append(None)
                    n_missing_seeds += 1
                else:
                    seed_best.append(best)
                    seed_epochs.append(best_ep)
                    if meta is not None:
                        if meta_model is None and meta.get("model") is not None:
                            meta_model = str(meta.get("model"))
                        if meta_act is None and meta.get("activation") is not None:
                            meta_act = str(meta.get("activation"))

            model = meta_model if meta_model is not None else model_guess
            activation = meta_act if meta_act is not None else act_guess

            mean_acc, std_acc = _mean_std(seed_best, sample_std=True)

            row = {
                "root_variant": rv,
                "experiment_folder": exp,
                "model": model,
                "activation": activation,
                "seed0_best_pct": _fmt_pct(seed_best[0]),
                "seed1_best_pct": _fmt_pct(seed_best[1]),
                "seed2_best_pct": _fmt_pct(seed_best[2]),
                "seed3_best_pct": _fmt_pct(seed_best[3]),
                "seed4_best_pct": _fmt_pct(seed_best[4]),
                "mean_best_pct": _fmt_pct(mean_acc),
                "std_best_pct": _fmt_pct(std_acc),
                "n_missing_seeds": str(n_missing_seeds),
            }
            rows.append(row)

            seeds_str = ", ".join(_fmt_pct(x) for x in seed_best)
            miss_tag = f"  [missing_seeds={n_missing_seeds}]" if n_missing_seeds > 0 else ""
            report_lines.append(
                f"{exp} | model={model} act={activation} | "
                f"best(test)% seeds=[{seeds_str}] | mean±std={row['mean_best_pct']}±{row['std_best_pct']}{miss_tag}"
            )

        report_lines.append("")

    headers = [
        "root_variant",
        "experiment_folder",
        "model",
        "activation",
        "seed0_best_pct",
        "seed1_best_pct",
        "seed2_best_pct",
        "seed3_best_pct",
        "seed4_best_pct",
        "mean_best_pct",
        "std_best_pct",
        "n_missing_seeds",
    ]

    try:
        BASE_ROOT.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    try:
        import csv
        with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in headers})
    except Exception as e:
        print(f"[WARN] Failed to write CSV to {OUT_CSV}: {e}")

    try:
        with OUT_TXT.open("w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")
    except Exception as e:
        print(f"[WARN] Failed to write TXT to {OUT_TXT}: {e}")

    print("\n".join(report_lines))
    print(f"\n[OK] Wrote CSV: {OUT_CSV}")
    print(f"[OK] Wrote TXT: {OUT_TXT}")

    if PLOT_SCORE_CURVES:
        cases = [
            ("cifar10_plain",  "vgg4", "plain",         "score_curve_vgg4_plain.pdf",              False),
            ("cifar10_plain",  "vgg8", "plain",         "score_curve_vgg8_plain.pdf",              False),

            ("cifar10_no_GN",  "vgg4", "boosted_noGN",  "score_curve_vgg4_boosted_no_gn.pdf",      True),
            ("cifar10_GN",     "vgg4", "boosted_GN",    "score_curve_vgg4_boosted_gn.pdf",         True),

            ("cifar10_no_GN",  "vgg8", "boosted_noGN",  "score_curve_vgg8_boosted_no_gn.pdf",      True),
            ("cifar10_GN",     "vgg8", "boosted_GN",    "score_curve_vgg8_boosted_gn.pdf",         True),
        ]

        for rv, model, variant, fname, inset_flag in cases:
            rv_path = BASE_ROOT / rv
            out_pdf = PLOT_DIR / fname
            want_inset = bool(inset_flag) if INSET_ONLY_FOR_BOOSTED else True
            _plot_case(
                rv=rv,
                rv_path=rv_path,
                model=model,
                variant=variant,
                out_pdf=out_pdf,
                want_inset=want_inset,
            )

        print(f"\n[OK] Wrote 6 PDFs under: {PLOT_DIR}")


if __name__ == "__main__":
    main()
