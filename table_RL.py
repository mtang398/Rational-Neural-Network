import os
import re
import json
import math
import csv
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Configuration values for locating runs, controlling outputs, and formatting tables/plots
BASE_ROOT = r"D:\runs\Rational_Mujoco"  # Root directory containing per-task subdirectories

OUT_DIR_NAME = "_paper_tables_plots"
SAVE_PLOTS = True

PLOT_FORMATS = ("pdf",)
PLOT_DPI = 250
PLOT_LOG_Y = False

PLOT_SCORE_KEY = "eval_v5_normalized"

ACT_ORDER = {"gelu": 0, "relu": 1, "silu": 2, "rational": 3}
ALGO_ORDER = {"iql": 0, "td3bc": 1}

USE_SAMPLE_VARIANCE = False

PRINT_PER_RUN = True

DEC_RET = 2
DEC_NORM = 2

LATEX_BEST_COLOR = "green!18"
LATEX_SECOND_COLOR = "yellow!18"

MY_METHOD_ACT = "rational"


# Regex patterns for parsing run metadata from directory names
RUN_RE = re.compile(r"^(?P<task>.+?)__(?P<algo>td3bc|iql)__(?P<act>[^_]+)__seed(?P<seed>\d+)__")
RUN_TS_RE = re.compile(r"__(?P<dt>\d{8}_\d{6})$")


def mean_and_variance(vals: List[float], sample: bool) -> Tuple[float, float]:
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"))
    mu = sum(vals) / n
    if sample:
        if n < 2:
            return (mu, float("nan"))
        var = sum((x - mu) ** 2 for x in vals) / (n - 1)
    else:
        var = sum((x - mu) ** 2 for x in vals) / n
    return mu, var


def std_from_var(var: float) -> float:
    return math.sqrt(var) if var == var else float("nan")


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
         .replace("_", "\\_")
         .replace("%", "\\%")
         .replace("&", "\\&")
         .replace("#", "\\#")
         .replace("{", "\\{")
         .replace("}", "\\}")
         .replace("$", "\\$")
    )


def _fmt_float(x: float, dec: int) -> str:
    if x != x:
        return "nan"
    return f"{x:.{dec}f}"


def _fmt_pm(mu: float, std: float, dec: int) -> str:
    if (mu != mu) or (std != std):
        return "nan"
    return f"{mu:.{dec}f} ± {std:.{dec}f}"


def _fmt_pm_latex(mu: float, std: float, dec: int) -> str:
    if (mu != mu) or (std != std):
        return "nan"
    return f"{mu:.{dec}f} $\\pm$ {std:.{dec}f}"


def _round_minute_dt(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


def _parse_run_start_from_name(run_name: str) -> Optional[datetime]:
    m = RUN_TS_RE.search(run_name)
    if not m:
        return None
    s = m.group("dt")
    try:
        return datetime.strptime(s, "%Y%m%d_%H%M%S")
    except Exception:
        return None


def _as_float(v) -> Optional[float]:
    try:
        if v is None or isinstance(v, bool):
            return None
        fv = float(v)
        if not math.isfinite(fv):
            return None
        return fv
    except Exception:
        return None


def _as_int(v) -> Optional[int]:
    try:
        if v is None or isinstance(v, bool):
            return None
        return int(v)
    except Exception:
        return None


def _extract_update(obj: dict) -> Optional[int]:
    for k in ("update", "updates", "iter", "iteration", "step", "train_step", "global_step"):
        if k in obj:
            u = _as_int(obj.get(k))
            if u is not None:
                return u
    return None


def _try_parse_iso_datetime(s: str) -> Optional[datetime]:
    try:
        ss = s.strip()
        if ss.endswith("Z"):
            ss = ss[:-1] + "+00:00"
        dt = datetime.fromisoformat(ss)
        if dt.tzinfo is not None:
            dt = dt.astimezone().replace(tzinfo=None)
        return dt
    except Exception:
        return None


def _extract_time_info(obj: dict) -> Tuple[Optional[datetime], Optional[float]]:
    for k in ("timestamp", "time", "wall_time", "walltime", "datetime", "date", "iso_time"):
        if k not in obj:
            continue
        v = obj.get(k)

        if isinstance(v, str):
            dt = _try_parse_iso_datetime(v)
            if dt is not None:
                return dt, None

        fv = _as_float(v)
        if fv is None:
            continue

        if fv > 1e11:
            sec = fv / 1000.0
            dt = datetime.fromtimestamp(sec)
            return dt, None
        if fv > 1e9:
            dt = datetime.fromtimestamp(fv)
            return dt, None

        if fv >= 0.0:
            return None, fv

    return None, None


def _collect_score_series(metrics_path: str, score_key: str) -> Dict[int, float]:
    series: Dict[int, float] = {}
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if score_key not in obj:
                    continue
                u = _extract_update(obj)
                if u is None:
                    continue
                fv = _as_float(obj.get(score_key))
                if fv is None:
                    continue
                series[u] = fv
    except Exception:
        pass
    return series


def parse_metrics_summary(metrics_path: str, run_name: str) -> dict:
    eval_returns: List[float] = []
    eval_norms: List[float] = []

    first_abs: Optional[datetime] = None
    last_abs: Optional[datetime] = None
    first_rel: Optional[float] = None
    last_rel: Optional[float] = None

    final_update: Optional[int] = None

    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            u = _extract_update(obj)
            if u is not None:
                if (final_update is None) or (u > final_update):
                    final_update = u

            if ("eval_return" in obj) and ("eval_v5_normalized" in obj):
                r = _as_float(obj.get("eval_return"))
                n = _as_float(obj.get("eval_v5_normalized"))
                if r is not None and n is not None:
                    eval_returns.append(r)
                    eval_norms.append(n)
                    if len(eval_returns) > 5:
                        eval_returns = eval_returns[-5:]
                        eval_norms = eval_norms[-5:]

            abs_dt, rel_s = _extract_time_info(obj)
            if abs_dt is not None:
                if first_abs is None:
                    first_abs = abs_dt
                last_abs = abs_dt
            if rel_s is not None:
                if first_rel is None:
                    first_rel = rel_s
                last_rel = rel_s

    k = min(len(eval_returns), len(eval_norms))
    eval_returns = eval_returns[-k:]
    eval_norms = eval_norms[-k:]
    mu_r, var_r = mean_and_variance(eval_returns, sample=USE_SAMPLE_VARIANCE)
    mu_n, var_n = mean_and_variance(eval_norms, sample=USE_SAMPLE_VARIANCE)

    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None
    duration_min = float("nan")

    if first_abs is not None and last_abs is not None:
        start_dt = first_abs
        end_dt = last_abs
        duration_min = (end_dt - start_dt).total_seconds() / 60.0
    elif first_rel is not None and last_rel is not None:
        duration_min = (last_rel - first_rel) / 60.0
        start_dt = _parse_run_start_from_name(run_name)
        try:
            end_dt = datetime.fromtimestamp(os.path.getmtime(metrics_path))
        except Exception:
            end_dt = None
        if start_dt is not None and end_dt is not None:
            duration_min = (end_dt - start_dt).total_seconds() / 60.0
    else:
        start_dt = _parse_run_start_from_name(run_name)
        try:
            end_dt = datetime.fromtimestamp(os.path.getmtime(metrics_path))
        except Exception:
            end_dt = None
        if start_dt is not None and end_dt is not None:
            duration_min = (end_dt - start_dt).total_seconds() / 60.0

    score_series = _collect_score_series(metrics_path, PLOT_SCORE_KEY)

    return {
        "k": k,
        "mu_ret": mu_r, "var_ret": var_r,
        "mu_norm": mu_n, "var_norm": var_n,
        "final_update": final_update,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "duration_min": duration_min,
        "score_series": score_series,
    }


def _print_ascii_table(headers: List[str], rows: List[List[str]]) -> None:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))
    sep = "  "
    line = sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    bar = sep.join("-" * widths[i] for i in range(len(headers)))
    print(line)
    print(bar)
    for r in rows:
        print(sep.join(r[i].ljust(widths[i]) for i in range(len(headers))))


def _write_csv(path: str, headers: List[str], rows: List[List[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def _write_latex_table_grouped(
    path: str,
    caption: str,
    label: str,
    headers: List[str],
    rows: List[List[str]],
    group_break_before: List[bool],
) -> None:
    """
    Write a LaTeX table environment to `path` using booktabs rules, with optional group midrules.
    `rows` are expected to already contain LaTeX-safe cell strings, and `group_break_before[i]`
    controls insertion of a \\midrule before row i.
    """
    assert len(rows) == len(group_break_before)

    col_spec = "l" * len(headers)
    with open(path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated.\n")
        f.write("% Requires in preamble:\n")
        f.write("%   \\usepackage{booktabs}\n")
        f.write("%   \\usepackage[table]{xcolor}\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write(f"\\caption{{{_latex_escape(caption)}}}\n")
        f.write(f"\\label{{{_latex_escape(label)}}}\n")
        f.write("\\begin{tabular}{" + col_spec + "}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(_latex_escape(h) for h in headers) + " \\\\\n")
        f.write("\\midrule\n")
        for i, r in enumerate(rows):
            if group_break_before[i]:
                f.write("\\midrule\n")
            f.write(" & ".join(r) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")


def _aggregate_mean_std_by_update(series_list: List[Dict[int, float]], sample: bool) -> Tuple[List[int], List[float], List[float]]:
    if not series_list:
        return [], [], []
    all_updates = set()
    for s in series_list:
        all_updates.update(s.keys())
    xs = sorted(all_updates)
    mus: List[float] = []
    stds: List[float] = []
    for u in xs:
        vals = [s[u] for s in series_list if u in s]
        mu, var = mean_and_variance(vals, sample=sample)
        mus.append(mu)
        stds.append(std_from_var(var))
    return xs, mus, stds


def _plot_task_algo_score(out_dir: str,
                          task: str,
                          algo: str,
                          act_to_series: Dict[str, List[Dict[int, float]]],
                          score_key: str) -> None:
    if not SAVE_PLOTS:
        return

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)

    for act in sorted(act_to_series.keys(), key=lambda a: ACT_ORDER.get(a, 999)):
        series_list = act_to_series[act]
        xs, mu, sd = _aggregate_mean_std_by_update(series_list, sample=USE_SAMPLE_VARIANCE)
        if not xs:
            continue
        ax.plot(xs, mu, linewidth=2.0)
        low = [m - s if (m == m and s == s) else float("nan") for m, s in zip(mu, sd)]
        high = [m + s if (m == m and s == s) else float("nan") for m, s in zip(mu, sd)]
        ax.fill_between(xs, low, high, alpha=0.18, linewidth=0)

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.grid(True, alpha=0.25)

    if PLOT_LOG_Y:
        ax.set_yscale("log")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    safe_task = task.replace(os.sep, "_")
    fname_base = f"score__{safe_task}__{algo}__{score_key}"
    for ext in PLOT_FORMATS:
        fig.savefig(os.path.join(out_dir, f"{fname_base}.{ext}"), dpi=PLOT_DPI)
    plt.close(fig)


def main():
    if not os.path.isdir(BASE_ROOT):
        raise FileNotFoundError(
            f"BASE_ROOT not found:\n  {BASE_ROOT}\n"
            "Edit BASE_ROOT to the correct Windows directory (e.g., D:\\runs\\Rational_Mujoco)."
        )

    out_dir = os.path.join(BASE_ROOT, OUT_DIR_NAME)
    safe_mkdir(out_dir)

    task_dirs = [d for d in os.listdir(BASE_ROOT) if os.path.isdir(os.path.join(BASE_ROOT, d))]
    task_dirs.sort()

    ddof_tag = "sample(ddof=1)" if USE_SAMPLE_VARIANCE else "population(ddof=0)"
    print(f"BASE_ROOT: {BASE_ROOT}")
    print(f"OUT_DIR:   {out_dir}")
    print(f"PLOT_SCORE_KEY: {PLOT_SCORE_KEY}")
    print(f"Variance mode: {ddof_tag}")
    print(f"Discovered task-dirs: {task_dirs}\n")

    run_rows: List[dict] = []
    group_to_runs: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    plot_map: Dict[Tuple[str, str], Dict[str, List[Dict[int, float]]]] = defaultdict(lambda: defaultdict(list))

    for task_dir in task_dirs:
        task_path = os.path.join(BASE_ROOT, task_dir)
        run_names = [n for n in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, n))]

        runs = []
        for name in run_names:
            m = RUN_RE.match(name)
            if not m:
                continue
            act = m.group("act")
            if act not in ACT_ORDER:
                continue
            seed = int(m.group("seed"))
            algo = m.group("algo")
            parsed_task = m.group("task")
            runs.append((parsed_task, algo, act, seed, name))

        runs.sort(key=lambda t: (t[0], ALGO_ORDER.get(t[1], 999), ACT_ORDER.get(t[2], 999), t[3], t[4]))

        if not runs:
            continue

        print("=" * 80)
        print(f"TASK-DIR: {task_dir}")
        print("=" * 80)

        for parsed_task, algo, act, seed, name in runs:
            metrics_path = os.path.join(task_path, name, "metrics.jsonl")
            if not os.path.isfile(metrics_path):
                continue

            summary = parse_metrics_summary(metrics_path, run_name=name)

            k = summary["k"]
            mu_r, var_r = summary["mu_ret"], summary["var_ret"]
            mu_n, var_n = summary["mu_norm"], summary["var_norm"]
            duration_min = summary["duration_min"]
            final_update = summary["final_update"]
            score_series = summary["score_series"]

            if PRINT_PER_RUN:
                print(
                    f"{name}\n"
                    f"  last_eval_points_used={k}\n"
                    f"  eval_return:          mean(last5)={_fmt_float(mu_r, 6)}   var(last5)={_fmt_float(var_r, 6)}\n"
                    f"  eval_v5_normalized:   mean(last5)={_fmt_float(mu_n, 6)}   var(last5)={_fmt_float(var_n, 6)}\n"
                    f"  duration_min={_fmt_float(duration_min, 1)}  final_update={final_update}  score_points={len(score_series)}"
                )

            run_row = {
                "task_dir": task_dir,
                "task": parsed_task,
                "algo": algo,
                "act": act,
                "seed": seed,
                "run_name": name,
                "k_last5": k,
                "mu_ret_last5": mu_r,
                "std_ret_last5": std_from_var(var_r),
                "mu_norm_last5": mu_n,
                "std_norm_last5": std_from_var(var_n),
                "duration_min": duration_min,
                "final_update": final_update,
            }
            run_rows.append(run_row)
            group_to_runs[(parsed_task, algo, act)].append(run_row)

            if score_series:
                plot_map[(parsed_task, algo)][act].append(score_series)

        print()

    if not run_rows:
        print("No runs found. Check BASE_ROOT and run naming / metrics.jsonl presence.")
        return

    group_rows_sorted: List[dict] = []

    for (task, algo, act), rr in group_to_runs.items():
        n_runs = len(rr)

        vals_ret = [r["mu_ret_last5"] for r in rr if r["mu_ret_last5"] == r["mu_ret_last5"]]
        vals_norm = [r["mu_norm_last5"] for r in rr if r["mu_norm_last5"] == r["mu_norm_last5"]]
        vals_time = [r["duration_min"] for r in rr if r["duration_min"] == r["duration_min"] and math.isfinite(r["duration_min"])]

        mu_ret, var_ret = mean_and_variance(vals_ret, sample=USE_SAMPLE_VARIANCE)
        mu_norm, var_norm = mean_and_variance(vals_norm, sample=USE_SAMPLE_VARIANCE)
        mu_time, var_time = mean_and_variance(vals_time, sample=USE_SAMPLE_VARIANCE)

        group_rows_sorted.append({
            "task": task,
            "algo": algo,
            "act": act,
            "runs": n_runs,
            "ret_mu": mu_ret, "ret_std": std_from_var(var_ret),
            "norm_mu": mu_norm, "norm_std": std_from_var(var_norm),
            "time_mu": mu_time, "time_std": std_from_var(var_time),
        })

    group_rows_sorted.sort(key=lambda a: (a["task"], ALGO_ORDER.get(a["algo"], 999), ACT_ORDER.get(a["act"], 999)))

    if PLOT_SCORE_KEY == "eval_return":
        primary_key = "ret_mu"
        primary_dec = DEC_RET
        other_key = "norm_mu"
        other_dec = DEC_NORM
        primary_name = "eval_return"
        other_name = "eval_v5_norm"
    else:
        primary_key = "norm_mu"
        primary_dec = DEC_NORM
        other_key = "ret_mu"
        other_dec = DEC_RET
        primary_name = "eval_v5_norm"
        other_name = "eval_return"

    rank_map: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(dict)
    by_group: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in group_rows_sorted:
        by_group[(row["task"], row["algo"])].append(row)

    for gkey, rows in by_group.items():
        sorted_rows = sorted(
            rows,
            key=lambda r: (float("-inf") if r[primary_key] != r[primary_key] else r[primary_key]),
            reverse=True
        )
        if len(sorted_rows) >= 1:
            rank_map[gkey][sorted_rows[0]["act"]] = 1
        if len(sorted_rows) >= 2:
            rank_map[gkey][sorted_rows[1]["act"]] = 2

    headers = [
        "task", "algo", "act", "runs",
        f"{primary_name} (mean±std)", f"{other_name} (mean±std)",
        "wall_min (mean±std)"
    ]
    rows_console: List[List[str]] = []
    for a in group_rows_sorted:
        mu_p = a[primary_key]
        std_p = a["norm_std"] if primary_key == "norm_mu" else a["ret_std"]
        mu_o = a[other_key]
        std_o = a["ret_std"] if other_key == "ret_mu" else a["norm_std"]

        rows_console.append([
            a["task"],
            a["algo"],
            a["act"],
            str(a["runs"]),
            _fmt_pm(mu_p, std_p, primary_dec),
            _fmt_pm(mu_o, std_o, other_dec),
            _fmt_pm(a["time_mu"], a["time_std"], 0),
        ])

    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY (one row per task × algo × activation; mean±std over seeds)")
    print("=" * 80)
    _print_ascii_table(headers, rows_console)
    print()

    out_tables = os.path.join(out_dir, "tables")
    safe_mkdir(out_tables)
    out_plots = os.path.join(out_dir, "plots")
    safe_mkdir(out_plots)

    agg_csv_path = os.path.join(out_tables, "results_aggregate.csv")
    _write_csv(agg_csv_path, headers, rows_console)

    run_csv_headers = [
        "task_dir", "task", "algo", "act", "seed", "run_name",
        "k_last5",
        "eval_return_last5_mean", "eval_return_last5_std",
        "eval_norm_last5_mean", "eval_norm_last5_std",
        "duration_min", "final_update"
    ]
    run_csv_rows: List[List[str]] = []
    for r in sorted(run_rows, key=lambda x: (x["task"], ALGO_ORDER.get(x["algo"], 999), ACT_ORDER.get(x["act"], 999), x["seed"], x["run_name"])):
        run_csv_rows.append([
            r["task_dir"],
            r["task"],
            r["algo"],
            r["act"],
            str(r["seed"]),
            r["run_name"],
            str(r["k_last5"]),
            _fmt_float(r["mu_ret_last5"], DEC_RET),
            _fmt_float(r["std_ret_last5"], DEC_RET),
            _fmt_float(r["mu_norm_last5"], DEC_NORM),
            _fmt_float(r["std_norm_last5"], DEC_NORM),
            _fmt_float(r["duration_min"], 1),
            str(r["final_update"]) if r["final_update"] is not None else "-",
        ])
    run_csv_path = os.path.join(out_tables, "results_per_run.csv")
    _write_csv(run_csv_path, run_csv_headers, run_csv_rows)

    tex_headers = [
        "Task", "Algo", "Activation", "Runs",
        f"{primary_name} (mean$\\pm$std)", f"{other_name} (mean$\\pm$std)",
        "Wall-min (mean$\\pm$std)"
    ]

    tex_rows: List[List[str]] = []
    group_break_before: List[bool] = []
    prev_group: Optional[Tuple[str, str]] = None

    for a in group_rows_sorted:
        gkey = (a["task"], a["algo"])
        is_break = (prev_group is not None and gkey != prev_group)
        prev_group = gkey
        group_break_before.append(is_break)

        task_cell = _latex_escape(a["task"])
        algo_cell = _latex_escape(a["algo"])

        act_esc = _latex_escape(a["act"])
        if a["act"] == MY_METHOD_ACT:
            act_cell = f"\\textbf{{{act_esc}}}"
        else:
            act_cell = act_esc

        runs_cell = str(a["runs"])

        mu_p = a[primary_key]
        std_p = a["norm_std"] if primary_key == "norm_mu" else a["ret_std"]
        mu_o = a[other_key]
        std_o = a["ret_std"] if other_key == "ret_mu" else a["norm_std"]

        score_cell = _fmt_pm_latex(mu_p, std_p, primary_dec)
        other_cell = _fmt_pm_latex(mu_o, std_o, other_dec)
        time_cell = _fmt_pm_latex(a["time_mu"], a["time_std"], 0)

        rnk = rank_map.get(gkey, {}).get(a["act"], 0)
        if rnk == 1:
            score_cell = f"\\cellcolor{{{LATEX_BEST_COLOR}}}{score_cell}"
        elif rnk == 2:
            score_cell = f"\\cellcolor{{{LATEX_SECOND_COLOR}}}{score_cell}"

        tex_rows.append([
            task_cell,
            algo_cell,
            act_cell,
            runs_cell,
            score_cell,
            other_cell,
            time_cell,
        ])

    agg_tex_path = os.path.join(out_tables, "results_aggregate_icml_colored.tex")
    _write_latex_table_grouped(
        agg_tex_path,
        caption=f"Aggregate results across seeds (last-5 eval mean). Best/second-best highlighted per (task, algo) using {primary_name}.",
        label="tab:results_aggregate",
        headers=tex_headers,
        rows=tex_rows,
        group_break_before=group_break_before,
    )

    if SAVE_PLOTS:
        for (task, algo), act_map in sorted(plot_map.items(), key=lambda kv: (kv[0][0], ALGO_ORDER.get(kv[0][1], 999))):
            if not act_map:
                continue
            _plot_task_algo_score(
                out_dir=out_plots,
                task=task,
                algo=algo,
                act_to_series=act_map,
                score_key=PLOT_SCORE_KEY
            )

    print("Wrote:")
    print(f"  {agg_csv_path}")
    print(f"  {agg_tex_path}")
    print(f"  {run_csv_path}")
    if SAVE_PLOTS:
        print(f"  {out_plots}{os.sep}score__<task>__<algo>__{PLOT_SCORE_KEY}.pdf")
    print()


if __name__ == "__main__":
    main()
