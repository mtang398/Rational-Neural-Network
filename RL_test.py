"""
Offline RL (IQL and TD3+BC) on MuJoCo Minari datasets with an activation sweep:
  - relu, silu, gelu
  - rational (Rational activations with configurable approx_func initialization)

This script:
  - loads Minari datasets into a transition replay buffer
  - trains agents for a fixed number of gradient updates
  - periodically evaluates in a Gymnasium MuJoCo v5 environment
  - optionally computes and caches v5-consistent normalization anchors (random-policy + expert-dataset)
  - optionally uses a separate LR for Rational activation coefficients without touching site-packages
"""

import os
import sys
import math
import time
import json
import shutil
import random
import zipfile
import tempfile
import traceback
import re
import inspect
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Set

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


DATA_ROOT = r"D:\datasets\rational_RL"
OUT_ROOT  = r"D:\runs\rational_RL"

RATIONAL_LR_MULT = 1

TASKS_TO_RUN = [
    "halfcheetah_medium",
    "hopper_medium",
    "walker2d_medium",
]

ALGOS_TO_RUN = [
                "iql",
                "td3bc"
                ]

ACTIVATIONS_TO_RUN = ["relu", "silu", "gelu", "rational"]

RATIONAL_APPROX_FUNC = "relu"

SEEDS = [0, 1, 2, 3, 4]

TOTAL_UPDATES = 300_000
BATCH_SIZE = 256

EVAL_EVERY = 10_000
EVAL_EPISODES = 20
MAX_EPISODE_STEPS = 1000

HIDDEN_DIMS = (256, 256)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOCAL_ASSET_CACHE = os.path.join(DATA_ROOT, "_gymnasium_mujoco_assets_cache")


TASK_SPECS = {
    "halfcheetah_medium": {
        "dataset_id": "mujoco/halfcheetah/medium-v0",
        "env_id": "HalfCheetah-v5",
        "xml": "half_cheetah.xml",
        "d4rl_norm": ("halfcheetah",),
    },
    "hopper_medium": {
        "dataset_id": "mujoco/hopper/medium-v0",
        "env_id": "Hopper-v5",
        "xml": "hopper.xml",
        "d4rl_norm": ("hopper",),
    },
    "walker2d_medium": {
        "dataset_id": "mujoco/walker2d/medium-v0",
        "env_id": "Walker2d-v5",
        "xml": "walker2d.xml",
        "d4rl_norm": ("walker2d",),
    },
    "halfcheetah_medium_replay": {
        "dataset_id": "mujoco/halfcheetah/medium-replay-v0",
        "env_id": "HalfCheetah-v5",
        "xml": "half_cheetah.xml",
        "d4rl_norm": ("halfcheetah",),
    },
    "hopper_medium_replay": {
        "dataset_id": "mujoco/hopper/medium-replay-v0",
        "env_id": "Hopper-v5",
        "xml": "hopper.xml",
        "d4rl_norm": ("hopper",),
    },
    "walker2d_medium_replay": {
        "dataset_id": "mujoco/walker2d/medium-replay-v0",
        "env_id": "Walker2d-v5",
        "xml": "walker2d.xml",
        "d4rl_norm": ("walker2d",),
    },
}

D4RL_REF = {
    "halfcheetah": {"expert": 12135.0, "random": -280.18},
    "hopper": {"expert": 3234.3, "random": -20.27},
    "walker2d": {"expert": 4592.3, "random": 1.63},
}


V5_NORM_CACHE_PATH = os.path.join(DATA_ROOT, "_v5_norm_ref.json")

V5_RANDOM_ANCHOR_EPISODES = 100

V5_EXPERT_DATASETS = {
    "halfcheetah": "mujoco/halfcheetah/expert-v0",
    "hopper": "mujoco/hopper/expert-v0",
    "walker2d": "mujoco/walker2d/expert-v0",
}


def set_seed(seed: int):
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_str():
    """Return a timestamp string used for run directory names."""
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str):
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def _parse_rational_approx_from_activation_name(name: str) -> str:
    """
    Parse a Rational initialization target from an activation name.

    Accepted forms:
      - "rational" -> uses global RATIONAL_APPROX_FUNC
      - "rational:<relu|gelu|swish|silu>"
      - "rational_<relu|gelu|swish|silu>"

    Returns an approx_func value compatible with Rational(approx_func=...).
    """
    n = name.lower().strip()
    approx = None

    if n == "rational":
        approx = RATIONAL_APPROX_FUNC
    elif n.startswith("rational:"):
        approx = n.split(":", 1)[1].strip()
    elif n.startswith("rational_"):
        approx = n.split("_", 1)[1].strip()
    else:
        approx = RATIONAL_APPROX_FUNC

    if approx == "silu":
        approx = "swish"

    if approx not in {"relu", "gelu", "swish"}:
        raise ValueError(
            f"Unsupported Rational init target '{approx}'. "
            f"Use one of: relu, gelu, swish (or silu as alias)."
        )

    return approx


def get_activation(name: str) -> nn.Module:
    """
    Construct an activation module for MLPs used by actors/critics/value nets.

    For "rational*", constructs rational.torch.Rational and sets its initialization
    target according to _parse_rational_approx_from_activation_name.
    """
    name_l = name.lower()
    if name_l == "relu":
        return nn.ReLU(inplace=True)
    if name_l == "silu":
        return nn.SiLU(inplace=True)
    if name_l == "gelu":
        return nn.GELU()

    if name_l.startswith("rational"):
        try:
            from rational.torch import Rational
        except Exception as e:
            raise ImportError(
                "Activation='rational' requires the Rational Activations package. "
                "Install with: pip install rational-activations"
            ) from e

        approx = _parse_rational_approx_from_activation_name(name_l)

        try:
            return Rational(approx_func=approx)
        except TypeError:
            try:
                return Rational(approx)
            except Exception as e:
                raise RuntimeError(
                    f"Could not construct Rational activation with approx_func='{approx}'. "
                    "Please ensure you have a recent 'rational-activations' installation."
                ) from e

    raise ValueError(f"Unknown activation: {name}")


def _unique_params_in_order(params: List[nn.Parameter]) -> List[nn.Parameter]:
    """Deduplicate parameters while preserving first-seen order."""
    seen = set()
    out = []
    for p in params:
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out


def _split_rational_params(modules: List[nn.Module]) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Split parameters into (base_params, rational_params) by detecting Rational submodules.
    If Rational cannot be imported, return (all_params, []).
    """
    all_params = []
    for m in modules:
        all_params.extend(list(m.parameters()))
    all_params = _unique_params_in_order(all_params)

    try:
        from rational.torch import Rational as _Rational
    except Exception:
        return all_params, []

    rat_ids = set()
    for m in modules:
        for sub in m.modules():
            if isinstance(sub, _Rational):
                for p in sub.parameters():
                    rat_ids.add(id(p))

    rat_params = [p for p in all_params if id(p) in rat_ids]
    base_params = [p for p in all_params if id(p) not in rat_ids]
    return base_params, rat_params


def _make_adam_with_rational_lr(modules: List[nn.Module], base_lr: float) -> torch.optim.Optimizer:
    """
    Create Adam optimizer with either:
      - a single parameter group at base_lr when RATIONAL_LR_MULT == 1.0, or
      - two parameter groups when RATIONAL_LR_MULT != 1.0, scaling Rational parameters by RATIONAL_LR_MULT.
    """
    lr_mult = float(RATIONAL_LR_MULT)

    if lr_mult == 1.0:
        params = []
        for m in modules:
            params.extend(list(m.parameters()))
        params = _unique_params_in_order(params)
        return torch.optim.Adam(params, lr=base_lr)

    base_params, rat_params = _split_rational_params(modules)

    if len(rat_params) == 0:
        params = []
        for m in modules:
            params.extend(list(m.parameters()))
        params = _unique_params_in_order(params)
        return torch.optim.Adam(params, lr=base_lr)

    return torch.optim.Adam(
        [
            {"params": base_params, "lr": base_lr},
            {"params": rat_params, "lr": base_lr * lr_mult},
        ]
    )


def _try_import_gymnasium_version() -> str:
    """Best-effort Gymnasium version detection for cache keying and informative logging."""
    try:
        import importlib.metadata as importlib_metadata
        return importlib_metadata.version("gymnasium")
    except Exception:
        try:
            import gymnasium
            return getattr(gymnasium, "__version__", "unknown")
        except Exception:
            return "unknown"


def _download_file(url: str, out_path: str, timeout=60):
    """Download a URL to a local file path using urllib (used for Gymnasium source zips)."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(out_path, "wb") as f:
        shutil.copyfileobj(r, f)


_INCLUDE_RE = re.compile(r"<include\s+file=\"([^\"]+)\"\s*/?>")


def _collect_mjcf_includes_recursive(root_xml_abs: str, base_dir: str) -> Set[str]:
    """Parse MJCF <include file="..."> dependencies recursively starting from a root XML."""
    required: Set[str] = set()
    visited_abs: Set[str] = set()

    def _walk(xml_abs: str):
        if xml_abs in visited_abs:
            return
        visited_abs.add(xml_abs)

        try:
            with open(xml_abs, "r", encoding="utf-8") as f:
                txt = f.read()
        except Exception:
            return

        for m in _INCLUDE_RE.finditer(txt):
            rel = m.group(1).strip()
            if not rel:
                continue
            required.add(rel)
            child_abs = os.path.normpath(os.path.join(base_dir, rel))
            _walk(child_abs)

    _walk(root_xml_abs)
    return required


def _merge_copy_tree(src_dir: str, dst_dir: str):
    """Recursively copy files from src_dir into dst_dir, creating directories as needed."""
    for r, _dnames, fnames in os.walk(src_dir):
        rel = os.path.relpath(r, src_dir)
        dest_r = os.path.join(dst_dir, rel) if rel != "." else dst_dir
        ensure_dir(dest_r)
        for fn in fnames:
            src_f = os.path.join(r, fn)
            dst_f = os.path.join(dest_r, fn)
            shutil.copy2(src_f, dst_f)


def ensure_local_mujoco_assets() -> Optional[str]:
    """
    Build a local cache folder containing Gymnasium mujoco/assets files needed for v5 env XMLs.

    Behavior:
      - checks a versioned cache folder under LOCAL_ASSET_CACHE
      - tries copying from installed gymnasium.envs.mujoco/assets (read-only source)
      - if incomplete, downloads a Gymnasium source zip (tagged version or main) and extracts assets
      - validates dependencies by recursively parsing MJCF <include file="..."> references
    """
    ensure_dir(LOCAL_ASSET_CACHE)

    gym_ver = _try_import_gymnasium_version()
    if gym_ver == "unknown":
        print("[assets] Could not detect gymnasium version. Will try best-effort GitHub main.")
    else:
        print(f"[assets] Detected gymnasium version: {gym_ver}")

    cache_dir = os.path.join(LOCAL_ASSET_CACHE, f"gymnasium_assets_{gym_ver}")
    assets_dir = os.path.join(cache_dir, "assets")
    marker = os.path.join(cache_dir, "_READY")

    root_xmls = ["half_cheetah.xml", "hopper.xml", "walker2d.xml"]

    def _verify_assets_tree() -> Tuple[bool, List[str]]:
        missing: List[str] = []

        for x in root_xmls:
            if not os.path.exists(os.path.join(assets_dir, x)):
                missing.append(os.path.join(assets_dir, x))

        for x in root_xmls:
            root_abs = os.path.join(assets_dir, x)
            if not os.path.exists(root_abs):
                continue
            incs = _collect_mjcf_includes_recursive(root_abs, assets_dir)
            for rel in sorted(incs):
                need_abs = os.path.normpath(os.path.join(assets_dir, rel))
                if not os.path.exists(need_abs):
                    missing.append(need_abs)

        return (len(missing) == 0), missing

    if os.path.exists(marker) and os.path.isdir(assets_dir):
        ok, missing = _verify_assets_tree()
        if ok:
            return assets_dir
        print(f"[assets] Cache marker exists but assets incomplete ({len(missing)} missing). Will refetch.")
        for m in missing[:10]:
            print(f"         - missing: {m}")

    ensure_dir(cache_dir)
    ensure_dir(assets_dir)

    try:
        import gymnasium.envs.mujoco as gym_mujoco
        pkg_dir = os.path.dirname(gym_mujoco.__file__)
        pkg_assets = os.path.join(pkg_dir, "assets")
        if os.path.isdir(pkg_assets):
            print(f"[assets] Copying assets from installed Gymnasium: {pkg_assets}")
            _merge_copy_tree(pkg_assets, assets_dir)
            ok, missing = _verify_assets_tree()
            if ok:
                with open(marker, "w", encoding="utf-8") as f:
                    f.write("ok\n")
                print(f"[assets] Local assets READY at: {assets_dir}")
                return assets_dir
            print(f"[assets] Installed-copy incomplete ({len(missing)} missing). Will try GitHub zip.")
            for m in missing[:10]:
                print(f"         - missing: {m}")
        else:
            print("[assets] Installed gymnasium mujoco/assets directory not found. Will try GitHub zip.")
    except Exception as e:
        print(f"[assets] Could not copy from installed Gymnasium assets: {repr(e)}")
        print("[assets] Will try GitHub zip.")

    urls = []
    if gym_ver != "unknown":
        urls.append(f"https://github.com/Farama-Foundation/Gymnasium/archive/refs/tags/v{gym_ver}.zip")
        urls.append(f"https://github.com/Farama-Foundation/Gymnasium/archive/refs/tags/{gym_ver}.zip")
    urls.append("https://github.com/Farama-Foundation/Gymnasium/archive/refs/heads/main.zip")

    ok = False
    last_err = None

    for url in urls:
        try:
            print(f"[assets] Downloading: {url}")
            with tempfile.TemporaryDirectory() as td:
                zip_path = os.path.join(td, "gymnasium.zip")
                _download_file(url, zip_path)
                extract_root = os.path.join(td, "extract")
                ensure_dir(extract_root)

                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(extract_root)

                found_assets = None
                for root, _dirs, _files in os.walk(extract_root):
                    if root.replace("\\", "/").endswith("gymnasium/envs/mujoco/assets".replace("\\", "/")):
                        found_assets = root
                        break

                if found_assets is None:
                    raise RuntimeError("Could not locate gymnasium/envs/mujoco/assets in downloaded archive.")

                print(f"[assets] Found source assets: {found_assets}")
                _merge_copy_tree(found_assets, assets_dir)

                ok2, missing = _verify_assets_tree()
                if not ok2:
                    raise RuntimeError("Downloaded assets but still missing required dependency files: " + "; ".join(missing[:50]))

                ok = True
                break

        except Exception as e:
            last_err = e
            print(f"[assets] Fetch attempt failed: {repr(e)}")
            continue

    if not ok:
        print("[assets] FAILED to fetch local MuJoCo assets.")
        if last_err is not None:
            print("         Last error:", repr(last_err))
        return None

    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok\n")
    print(f"[assets] Local assets READY at: {assets_dir}")
    return assets_dir


def make_env_fallback(env_id: str, xml_file_abs: Optional[str] = None):
    """Create a Gymnasium environment, preferring an explicit xml_file when provided."""
    import gymnasium as gym
    if xml_file_abs is not None and os.path.exists(xml_file_abs):
        try:
            return gym.make(env_id, xml_file=xml_file_abs)
        except Exception as e:
            print(f"[env] gym.make({env_id}, xml_file=...) failed, falling back to default assets.")
            print(f"      Reason: {repr(e)}")
    return gym.make(env_id)


def _minari_load_dataset_compat(dataset_id: str, data_path: str, download_if_missing: bool = True):
    """
    Load a Minari dataset with compatibility across Minari versions:
      - sets MINARI_DATASETS_PATH
      - inspects minari.load_dataset signature for (data_path=, download=) support
      - retries with download=True when a local dataset is missing and download_if_missing is True
    """
    if data_path is not None:
        os.environ["MINARI_DATASETS_PATH"] = str(data_path)

    import minari

    try:
        sig = inspect.signature(minari.load_dataset)
        has_data_path = ("data_path" in sig.parameters)
        has_download = ("download" in sig.parameters)
    except Exception:
        has_data_path = False
        has_download = True

    def _load(download_flag: bool):
        if has_data_path and has_download:
            return minari.load_dataset(dataset_id, data_path=data_path, download=download_flag)
        if has_data_path and (not has_download):
            return minari.load_dataset(dataset_id, data_path=data_path)
        if (not has_data_path) and has_download:
            return minari.load_dataset(dataset_id, download=download_flag)
        return minari.load_dataset(dataset_id)

    try:
        return _load(download_flag=False)
    except FileNotFoundError as e:
        if not download_if_missing:
            raise
        print(f"[minari] load_dataset failed locally: {repr(e)}")
        print("[minari] Trying download=True ...")
        return _load(download_flag=True)


def _load_v5_norm_cache() -> Dict[str, Dict[str, float]]:
    """Load the cached v5 normalization anchors from disk, returning {} when unavailable or invalid."""
    ensure_dir(DATA_ROOT)
    if os.path.exists(V5_NORM_CACHE_PATH):
        try:
            with open(V5_NORM_CACHE_PATH, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def _save_v5_norm_cache(cache: Dict[str, Dict[str, float]]):
    """Atomically write the v5 normalization anchor cache to disk."""
    ensure_dir(DATA_ROOT)
    tmp = V5_NORM_CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    os.replace(tmp, V5_NORM_CACHE_PATH)


def _estimate_random_return_v5(env_id: str, xml_abs: Optional[str], episodes: int, max_steps: int, seed_base: int = 12345) -> float:
    """Estimate a random-policy return anchor in the evaluation environment by sampling actions."""
    env = make_env_fallback(env_id, xml_abs)
    try:
        rets = []
        for i in range(episodes):
            obs, _info = env.reset(seed=seed_base + i)
            ep_ret = 0.0
            for _t in range(max_steps):
                a = env.action_space.sample()
                obs, rew, terminated, truncated, _info = env.step(a)
                ep_ret += float(rew)
                if terminated or truncated:
                    break
            rets.append(ep_ret)
        return float(np.mean(rets))
    finally:
        try:
            env.close()
        except Exception:
            pass


def _estimate_expert_return_from_minari(expert_dataset_id: str, data_path: str) -> float:
    """Estimate an expert return anchor from episode returns in a Minari expert dataset."""
    ds = _minari_load_dataset_compat(expert_dataset_id, data_path=data_path, download_if_missing=True)
    rets = []
    for ep in ds.iterate_episodes():
        rew = np.asarray(ep.rewards).reshape(-1)
        if rew.size == 0:
            continue
        rets.append(float(np.sum(rew)))
    if len(rets) == 0:
        return float("nan")
    return float(np.mean(rets))


def ensure_v5_norm_anchors(task_key: str, env_id: str, xml_abs: Optional[str], data_path: str, max_steps: int) -> Optional[Dict[str, float]]:
    """
    Compute or load cached normalization anchors for v5 evaluation:
      - "random" is the mean return of a random policy in env_id
      - "expert" is the mean episode return from a Minari expert dataset
    """
    if task_key not in V5_EXPERT_DATASETS:
        return None

    cache = _load_v5_norm_cache()
    gym_ver = _try_import_gymnasium_version()

    cache_key = f"{task_key}||{env_id}||gymnasium={gym_ver}"

    if cache_key in cache:
        obj = cache[cache_key]
        if isinstance(obj, dict) and ("random" in obj) and ("expert" in obj):
            return {"random": float(obj["random"]), "expert": float(obj["expert"])}

    print(f"[norm_v5] Computing v5 anchors for {task_key} using env={env_id} (gymnasium={gym_ver})")
    print(f"[norm_v5]  - random: {V5_RANDOM_ANCHOR_EPISODES} episodes in eval env")
    r_random = _estimate_random_return_v5(env_id, xml_abs, episodes=V5_RANDOM_ANCHOR_EPISODES, max_steps=max_steps)

    expert_ds = V5_EXPERT_DATASETS[task_key]
    print(f"[norm_v5]  - expert: mean episode return from Minari dataset {expert_ds}")
    r_expert = _estimate_expert_return_from_minari(expert_ds, data_path=data_path)

    cache[cache_key] = {
        "task_key": task_key,
        "env_id": env_id,
        "gymnasium_version": gym_ver,
        "random": float(r_random),
        "expert": float(r_expert),
        "random_episodes": int(V5_RANDOM_ANCHOR_EPISODES),
        "max_episode_steps": int(max_steps),
        "expert_dataset_id": expert_ds,
        "ts": now_str(),
    }
    _save_v5_norm_cache(cache)

    print(f"[norm_v5] DONE: random={r_random:.6f}  expert={r_expert:.6f}")
    print(f"[norm_v5] Saved cache: {V5_NORM_CACHE_PATH}")
    return {"random": float(r_random), "expert": float(r_expert)}


def load_minari_transitions(dataset_id: str, data_path: str, normalize_obs: bool = True):
    """
    Load episodes from a Minari dataset and convert them into transitions:
      (obs, act, rew, next_obs, done)

    "done" is derived from terminations (not truncations).
    Also computes and returns:
      - observation normalization statistics (mean/std)
      - an IQL reward scaling factor computed from dataset episode return range
    """
    print(f"\n=== Loading dataset: {dataset_id} ===")
    print(f"Minari datasets path: {data_path}")

    ds = _minari_load_dataset_compat(dataset_id, data_path=data_path, download_if_missing=True)

    obs_list = []
    act_list = []
    rew_list = []
    next_obs_list = []
    done_list = []

    ep_returns = []

    n_ep = 0
    n_tr = 0

    for ep in ds.iterate_episodes():
        obs = np.asarray(ep.observations)
        act = np.asarray(ep.actions)
        rew = np.asarray(ep.rewards).reshape(-1)
        term = np.asarray(ep.terminations).reshape(-1)

        done = term.astype(np.float32)

        if obs.shape[0] == act.shape[0] + 1:
            s = obs[:-1]
            s2 = obs[1:]
        else:
            s = obs
            s2 = obs[1:]
            act = act[: s2.shape[0]]
            rew = rew[: s2.shape[0]]
            done = done[: s2.shape[0]]

        T = min(len(act), len(rew), len(done), len(s2), len(s))
        if T <= 0:
            continue

        ep_returns.append(float(np.sum(rew[:T])))

        obs_list.append(s[:T])
        next_obs_list.append(s2[:T])
        act_list.append(act[:T])
        rew_list.append(rew[:T])
        done_list.append(done[:T])

        n_ep += 1
        n_tr += T
        if n_ep % 200 == 0:
            print(f"  ... episodes: {n_ep}, transitions so far: {n_tr}")

    obs = np.concatenate(obs_list, axis=0).astype(np.float32)
    next_obs = np.concatenate(next_obs_list, axis=0).astype(np.float32)
    acts = np.concatenate(act_list, axis=0).astype(np.float32)
    rews = np.concatenate(rew_list, axis=0).astype(np.float32)
    dones = np.concatenate(done_list, axis=0).astype(np.float32)

    obs_dim = obs.shape[1]
    act_dim = acts.shape[1]
    print(f"Loaded transitions: N={obs.shape[0]}, obs_dim={obs_dim}, act_dim={act_dim}")

    if len(ep_returns) >= 2:
        r_max = float(np.max(ep_returns))
        r_min = float(np.min(ep_returns))
        denom = (r_max - r_min)
        iql_rew_scale = 1.0 / (denom + 1e-9)
    else:
        iql_rew_scale = 1.0

    print(f"[iql] reward_standardize scale = {iql_rew_scale:.6e}  (computed from dataset best-worst episode returns)")

    obs_mean = obs.mean(axis=0) if normalize_obs else np.zeros(obs_dim, dtype=np.float32)
    obs_std = obs.std(axis=0) if normalize_obs else np.ones(obs_dim, dtype=np.float32)
    obs_std = np.maximum(obs_std, 1e-6)

    def norm(x):
        return (x - obs_mean) / obs_std

    obs_n = norm(obs)
    next_obs_n = norm(next_obs)

    norm_stats = {"mean": obs_mean, "std": obs_std, "enabled": normalize_obs, "iql_rew_scale": float(iql_rew_scale)}
    return (obs_n, acts, rews, next_obs_n, dones), norm_stats


class ReplayBuffer:
    """Numpy-backed replay buffer with torch sampling on demand."""
    def __init__(self, obs, act, rew, next_obs, done):
        self.obs = obs
        self.act = act
        self.rew = rew.reshape(-1, 1)
        self.next_obs = next_obs
        self.done = done.reshape(-1, 1)
        self.size = obs.shape[0]

    def sample(self, batch_size: int, device: str, rew_scale: float = 1.0):
        """Sample a random minibatch and return tensors on the requested device."""
        idx = np.random.randint(0, self.size, size=batch_size)
        o = torch.as_tensor(self.obs[idx], device=device, dtype=torch.float32)
        a = torch.as_tensor(self.act[idx], device=device, dtype=torch.float32)
        r = torch.as_tensor(self.rew[idx], device=device, dtype=torch.float32) * float(rew_scale)
        o2 = torch.as_tensor(self.next_obs[idx], device=device, dtype=torch.float32)
        d = torch.as_tensor(self.done[idx], device=device, dtype=torch.float32)
        return o, a, r, o2, d


class MLP(nn.Module):
    """Feedforward MLP used as a building block for actors, critics, and value functions."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dims=(256, 256), activation="relu"):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(get_activation(activation))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def atanh(x: torch.Tensor) -> torch.Tensor:
    """Numerically-stabilized inverse tanh used for squashed Gaussian log-prob computations."""
    return 0.5 * (torch.log1p(x + 1e-6) - torch.log1p(-x + 1e-6))


def gaussian_log_prob(mean, log_std, pre_tanh):
    """Compute per-sample log probability under a diagonal Gaussian at pre_tanh."""
    std = torch.exp(log_std)
    var = std * std
    logp = -0.5 * (((pre_tanh - mean) ** 2) / (var + 1e-8) + 2.0 * log_std + math.log(2.0 * math.pi))
    return logp.sum(dim=-1, keepdim=True)


def tanh_squash_log_det_jacobian(pre_tanh):
    """Compute log|det J| for tanh squashing, summed over action dimensions."""
    t = torch.tanh(pre_tanh)
    return torch.log(1.0 - t * t + 1e-6).sum(dim=-1, keepdim=True)


class GaussianActor(nn.Module):
    """Squashed Gaussian policy with a learned, state-independent log_std vector."""
    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256), activation="relu"):
        super().__init__()
        self.mu_net = MLP(obs_dim, act_dim, hidden_dims=hidden_dims, activation=activation)
        self.log_std_param = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs):
        mu = self.mu_net(obs)
        log_std = self.log_std_param.clamp(LOG_STD_MIN, LOG_STD_MAX).expand_as(mu)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        eps = torch.randn_like(mu)
        pre_tanh = mu + std * eps
        action = torch.tanh(pre_tanh)
        logp = gaussian_log_prob(mu, log_std, pre_tanh) - tanh_squash_log_det_jacobian(pre_tanh)
        return action, logp

    def mode(self, obs):
        mu, _ = self.forward(obs)
        return torch.tanh(mu)

    def log_prob_of(self, obs, action_tanh):
        pre_tanh = atanh(action_tanh.clamp(-0.999, 0.999))
        mu, log_std = self.forward(obs)
        logp = gaussian_log_prob(mu, log_std, pre_tanh) - tanh_squash_log_det_jacobian(pre_tanh)
        return logp


class DeterministicActor(nn.Module):
    """Deterministic policy network with tanh output squashing."""
    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256), activation="relu"):
        super().__init__()
        self.mlp = MLP(obs_dim, act_dim, hidden_dims=hidden_dims, activation=activation)

    def forward(self, obs):
        return torch.tanh(self.mlp(obs))


class Critic(nn.Module):
    """Q-function network Q(s,a) implemented as an MLP over concatenated (s,a)."""
    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256), activation="relu"):
        super().__init__()
        self.q = MLP(obs_dim + act_dim, 1, hidden_dims=hidden_dims, activation=activation)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q(x)


class ValueNet(nn.Module):
    """State value function V(s) implemented as an MLP."""
    def __init__(self, obs_dim, hidden_dims=(256, 256), activation="relu"):
        super().__init__()
        self.v = MLP(obs_dim, 1, hidden_dims=hidden_dims, activation=activation)

    def forward(self, obs):
        return self.v(obs)


@dataclass
class IQLConfig:
    gamma: float = 0.99
    tau: float = 0.005
    expectile: float = 0.7
    beta: float = 3.0
    adv_clip: float = 100.0
    lr: float = 3e-4


class IQLAgent:
    """IQL agent with twin Q networks, a value network, and a squashed Gaussian actor."""
    def __init__(self, obs_dim, act_dim, activation: str, hidden_dims, max_action: float, cfg: IQLConfig):
        self.cfg = cfg
        self.max_action = float(max_action)

        self.q1 = Critic(obs_dim, act_dim, hidden_dims, activation).to(DEVICE)
        self.q2 = Critic(obs_dim, act_dim, hidden_dims, activation).to(DEVICE)
        self.q1_t = Critic(obs_dim, act_dim, hidden_dims, activation).to(DEVICE)
        self.q2_t = Critic(obs_dim, act_dim, hidden_dims, activation).to(DEVICE)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

        self.v = ValueNet(obs_dim, hidden_dims, activation).to(DEVICE)
        self.actor = GaussianActor(obs_dim, act_dim, hidden_dims, activation).to(DEVICE)

        self.q_opt = _make_adam_with_rational_lr([self.q1, self.q2], base_lr=cfg.lr)
        self.v_opt = _make_adam_with_rational_lr([self.v], base_lr=cfg.lr)
        self.pi_opt = _make_adam_with_rational_lr([self.actor], base_lr=cfg.lr)

        self.pi_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.pi_opt, T_max=TOTAL_UPDATES)

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Compute an environment-scaled action from a single normalized observation."""
        obs = torch.as_tensor(obs_np, device=DEVICE, dtype=torch.float32).unsqueeze(0)
        a = self.actor.mode(obs) if deterministic else self.actor.sample(obs)[0]
        a = a.squeeze(0).detach().cpu().numpy()
        return np.clip(a, -1.0, 1.0) * self.max_action

    def _expectile_loss(self, diff: torch.Tensor, expectile: float):
        """Expectile regression loss used to fit V(s) to a quantile-like target."""
        w = torch.where(diff > 0, expectile, 1 - expectile)
        return (w * diff.pow(2)).mean()

    def update(self, batch):
        """Perform one gradient update step for Q, V, and actor, and soft-update target Q networks."""
        o, a, r, o2, d = batch
        gamma = self.cfg.gamma

        with torch.no_grad():
            v_next = self.v(o2)
            q_target = r + gamma * (1.0 - d) * v_next

        q1 = self.q1(o, a)
        q2 = self.q2(o, a)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()

        with torch.no_grad():
            q1t = self.q1_t(o, a)
            q2t = self.q2_t(o, a)
            q_min = torch.minimum(q1t, q2t)

        v = self.v(o)
        v_loss = self._expectile_loss(q_min - v, self.cfg.expectile)

        self.v_opt.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_opt.step()

        with torch.no_grad():
            v_now = self.v(o)
            adv = q_min - v_now
            w = torch.exp(self.cfg.beta * adv).clamp(max=self.cfg.adv_clip)

        logp = self.actor.log_prob_of(o, a)
        pi_loss = -(w * logp).mean()

        self.pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        self.pi_opt.step()
        self.pi_sched.step()

        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1_t.parameters()):
                pt.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
            for p, pt in zip(self.q2.parameters(), self.q2_t.parameters()):
                pt.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

        metrics = {
            "q_loss": float(q_loss.item()),
            "v_loss": float(v_loss.item()),
            "pi_loss": float(pi_loss.item()),
            "q_mean": float(q_min.mean().item()),
            "v_mean": float(v_now.mean().item()),
            "adv_mean": float(adv.mean().item()),
        }
        return metrics

    def state_dict(self):
        """Return a checkpointable state dict for all trainable IQL components."""
        return {
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_t": self.q1_t.state_dict(),
            "q2_t": self.q2_t.state_dict(),
            "v": self.v.state_dict(),
            "actor": self.actor.state_dict(),
        }


@dataclass
class TD3BCConfig:
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    alpha: float = 2.5
    lr: float = 3e-4


class TD3BCAgent:
    """TD3+BC agent with deterministic actor, twin critics, target networks, and delayed policy updates."""
    def __init__(self, obs_dim, act_dim, activation: str, hidden_dims, max_action: float, cfg: TD3BCConfig):
        self.cfg = cfg
        self.max_action = float(max_action)
        self.act_dim = act_dim

        self.actor = DeterministicActor(obs_dim, act_dim, hidden_dims, activation).to(DEVICE)
        self.actor_t = DeterministicActor(obs_dim, act_dim, hidden_dims, activation).to(DEVICE)
        self.actor_t.load_state_dict(self.actor.state_dict())

        self.q1 = Critic(obs_dim, act_dim, hidden_dims, activation).to(DEVICE)
        self.q2 = Critic(obs_dim, act_dim, hidden_dims, activation).to(DEVICE)
        self.q1_t = Critic(obs_dim, act_dim, hidden_dims, activation).to(DEVICE)
        self.q2_t = Critic(obs_dim, act_dim, hidden_dims, activation).to(DEVICE)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

        self.q_opt = _make_adam_with_rational_lr([self.q1, self.q2], base_lr=cfg.lr)
        self.pi_opt = _make_adam_with_rational_lr([self.actor], base_lr=cfg.lr)

        self.total_it = 0

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Compute an environment-scaled action from a single normalized observation."""
        obs = torch.as_tensor(obs_np, device=DEVICE, dtype=torch.float32).unsqueeze(0)
        a = self.actor(obs).squeeze(0).detach().cpu().numpy()
        return np.clip(a, -1.0, 1.0) * self.max_action

    def update(self, batch):
        """Perform one TD3+BC update step for critics and (periodically) the actor and target networks."""
        self.total_it += 1
        o, a, r, o2, d = batch
        cfg = self.cfg

        with torch.no_grad():
            noise = (torch.randn((o2.shape[0], self.act_dim), device=DEVICE) * cfg.policy_noise).clamp(
                -cfg.noise_clip, cfg.noise_clip
            )
            a2 = (self.actor_t(o2) + noise).clamp(-1.0, 1.0)
            q_t = torch.minimum(self.q1_t(o2, a2), self.q2_t(o2, a2))
            target = r + cfg.gamma * (1.0 - d) * q_t

        q1 = self.q1(o, a)
        q2 = self.q2(o, a)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()

        pi_loss = None
        if self.total_it % cfg.policy_freq == 0:
            pi = self.actor(o)
            q_pi = self.q1(o, pi)
            lmbda = cfg.alpha / (q_pi.abs().mean().detach() + 1e-6)
            bc = F.mse_loss(pi, a)
            pi_loss_t = -lmbda * q_pi.mean() + bc

            self.pi_opt.zero_grad(set_to_none=True)
            pi_loss_t.backward()
            self.pi_opt.step()

            with torch.no_grad():
                for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
                    pt.data.mul_(1 - cfg.tau).add_(cfg.tau * p.data)
                for p, pt in zip(self.q1.parameters(), self.q1_t.parameters()):
                    pt.data.mul_(1 - cfg.tau).add_(cfg.tau * p.data)
                for p, pt in zip(self.q2.parameters(), self.q2_t.parameters()):
                    pt.data.mul_(1 - cfg.tau).add_(cfg.tau * p.data)

            pi_loss = float(pi_loss_t.item())

        metrics = {
            "q_loss": float(q_loss.item()),
            "pi_loss": pi_loss,
            "q_mean": float(torch.minimum(q1, q2).mean().item()),
        }
        return metrics

    def state_dict(self):
        """Return a checkpointable state dict for all trainable TD3+BC components."""
        return {
            "actor": self.actor.state_dict(),
            "actor_t": self.actor_t.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_t": self.q1_t.state_dict(),
            "q2_t": self.q2_t.state_dict(),
            "total_it": self.total_it,
        }


def v5_normalized_score(task_key: str, return_sum: float, v5_ref: Optional[Dict[str, float]]) -> Optional[float]:
    """Compute a D4RL-style normalized score using v5-consistent random/expert anchors when available."""
    if v5_ref is None:
        return None
    mx = float(v5_ref["expert"])
    mn = float(v5_ref["random"])
    return 100.0 * (float(return_sum) - mn) / (mx - mn + 1e-9)


@torch.no_grad()
def evaluate_policy(agent, env, obs_mean, obs_std, episodes=10, max_steps=1000) -> float:
    """Run deterministic evaluation episodes in env using observation normalization stats."""
    returns = []
    for _ in range(episodes):
        obs, _info = env.reset()
        ep_ret = 0.0
        for _t in range(max_steps):
            obs_n = (np.asarray(obs, dtype=np.float32) - obs_mean) / obs_std
            act = agent.act(obs_n, deterministic=True)
            obs, rew, terminated, truncated, _info = env.step(act)
            ep_ret += float(rew)
            if terminated or truncated:
                break
        returns.append(ep_ret)
    return float(np.mean(returns))


def main():
    """Run the configured sweep across tasks, algorithms, activations, and seeds, writing metrics and checkpoints."""
    print(f"Device: {DEVICE}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"OUT_ROOT: {OUT_ROOT}")
    print(f"Selected tasks: {TASKS_TO_RUN}")
    print(f"Selected algos: {ALGOS_TO_RUN}")
    print(f"Selected activations: {ACTIVATIONS_TO_RUN}")
    print(f"RATIONAL_APPROX_FUNC (default): {RATIONAL_APPROX_FUNC}")
    print(f"RATIONAL_LR_MULT: {RATIONAL_LR_MULT}")
    print(f"Seeds: {SEEDS}")
    print(f"TOTAL_UPDATES: {TOTAL_UPDATES} BATCH_SIZE: {BATCH_SIZE}")

    ensure_dir(OUT_ROOT)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    assets_dir = ensure_local_mujoco_assets()
    if assets_dir is None:
        print("[assets] WARNING: No local assets. Will fall back to default Gymnasium assets for env creation.")

    for task_name in TASKS_TO_RUN:
        if task_name not in TASK_SPECS:
            print(f"[skip] Unknown task: {task_name}")
            continue

        spec = TASK_SPECS[task_name]
        dataset_id = spec["dataset_id"]
        env_id = spec["env_id"]
        xml_name = spec["xml"]
        d4rl_key = spec["d4rl_norm"][0]

        xml_abs_for_norm = os.path.join(assets_dir, xml_name) if assets_dir is not None else None
        v5_ref = ensure_v5_norm_anchors(
            task_key=d4rl_key,
            env_id=env_id,
            xml_abs=xml_abs_for_norm,
            data_path=DATA_ROOT,
            max_steps=MAX_EPISODE_STEPS,
        )

        env = None
        max_action = 1.0
        try:
            xml_abs = os.path.join(assets_dir, xml_name) if assets_dir is not None else None
            env = make_env_fallback(env_id, xml_abs)
            max_action = float(np.max(np.abs(env.action_space.high)))
            if xml_abs is not None:
                print(f"Env: {env_id} | max_action={max_action:.3f} | xml={xml_abs}")
            else:
                print(f"Env: {env_id} | max_action={max_action:.3f} | xml=<default>")
        except Exception as e:
            print(f"[eval] Could not create env={env_id} at startup. Will attempt later.")
            print("       Reason:", repr(e))
            env = None

        (obs, acts_raw, rews, next_obs, dones), norm_stats = load_minari_transitions(
            dataset_id, data_path=DATA_ROOT, normalize_obs=True
        )
        obs_mean = norm_stats["mean"].astype(np.float32)
        obs_std = norm_stats["std"].astype(np.float32)
        iql_rew_scale = float(norm_stats.get("iql_rew_scale", 1.0))

        if env is None:
            inferred = float(np.max(np.abs(acts_raw)))
            max_action = max(1.0, inferred)
            print(f"[action] Env not available. Inferred max_action from dataset: {max_action:.3f}")

        acts = (acts_raw / max_action).astype(np.float32)
        acts = np.clip(acts, -1.0, 1.0)

        rb = ReplayBuffer(obs, acts, rews, next_obs, dones)
        obs_dim = obs.shape[1]
        act_dim = acts.shape[1]

        for algo in ALGOS_TO_RUN:
            for act_name in ACTIVATIONS_TO_RUN:
                for seed in SEEDS:
                    set_seed(seed)

                    run_id = f"{task_name}__{algo}__{act_name}__seed{seed}__{now_str()}"
                    run_dir = os.path.join(OUT_ROOT, run_id)
                    ensure_dir(run_dir)

                    print("\n" + "=" * 100)
                    print(f"RUN: task={task_name}  algo={algo}  act={act_name}  seed={seed}")
                    print("=" * 100)

                    if algo == "iql":
                        agent = IQLAgent(
                            obs_dim, act_dim, activation=act_name, hidden_dims=HIDDEN_DIMS,
                            max_action=max_action, cfg=IQLConfig()
                        )
                        rew_scale = iql_rew_scale
                    elif algo == "td3bc":
                        agent = TD3BCAgent(
                            obs_dim, act_dim, activation=act_name, hidden_dims=HIDDEN_DIMS,
                            max_action=max_action, cfg=TD3BCConfig()
                        )
                        rew_scale = 1.0
                    else:
                        print(f"[skip] Unknown algo: {algo}")
                        continue

                    eval_env = None
                    try:
                        xml_abs = os.path.join(assets_dir, xml_name) if assets_dir is not None else None
                        eval_env = make_env_fallback(env_id, xml_abs)
                    except Exception as e:
                        print(f"[eval] Could not create env={env_id}. Will train without evaluation.")
                        print("       Reason:", repr(e))
                        eval_env = None

                    metrics_path = os.path.join(run_dir, "metrics.jsonl")
                    ckpt_path = os.path.join(run_dir, "checkpoint.pt")

                    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "task": task_name,
                                "dataset_id": dataset_id,
                                "env_id": env_id,
                                "algo": algo,
                                "activation": act_name,
                                "seed": seed,
                                "total_updates": TOTAL_UPDATES,
                                "batch_size": BATCH_SIZE,
                                "hidden_dims": HIDDEN_DIMS,
                                "max_action": max_action,
                                "obs_norm": {"enabled": True},
                                "assets_dir": assets_dir,
                                "v5_norm_cache": V5_NORM_CACHE_PATH,
                                "v5_norm_ref": v5_ref,
                                "v5_random_anchor_episodes": V5_RANDOM_ANCHOR_EPISODES,
                                "rational_default_approx_func": RATIONAL_APPROX_FUNC,
                                "rational_lr_mult": float(RATIONAL_LR_MULT),
                                "iql_reward_scale": float(iql_rew_scale),
                            },
                            f,
                            indent=2,
                        )

                    t0 = time.time()

                    with open(metrics_path, "a", encoding="utf-8") as mf:
                        for step in range(1, TOTAL_UPDATES + 1):
                            batch = rb.sample(BATCH_SIZE, DEVICE, rew_scale=rew_scale)
                            m = agent.update(batch)

                            if (step % EVAL_EVERY == 0) or (step == 1):
                                elapsed = time.time() - t0
                                sps = step / max(elapsed, 1e-6)

                                out = {
                                    "step": step,
                                    "elapsed_sec": elapsed,
                                    "steps_per_sec": sps,
                                    **{k: (v if v is None else float(v)) for k, v in m.items()},
                                }

                                if eval_env is not None and step % EVAL_EVERY == 0:
                                    try:
                                        avg_return = evaluate_policy(
                                            agent, eval_env, obs_mean, obs_std,
                                            episodes=EVAL_EPISODES, max_steps=MAX_EPISODE_STEPS
                                        )
                                        out["eval_return"] = float(avg_return)

                                        ns = v5_normalized_score(d4rl_key, avg_return, v5_ref)
                                        if ns is not None:
                                            out["eval_v5_normalized"] = float(ns)
                                    except Exception as e:
                                        out["eval_error"] = repr(e)

                                mf.write(json.dumps(out) + "\n")
                                mf.flush()

                                msg = (
                                    f"[{run_id}] step={step}  "
                                    f"q_loss={out.get('q_loss', None)}  "
                                    f"pi_loss={out.get('pi_loss', None)}  "
                                    f"q_mean={out.get('q_mean', None)}  "
                                )
                                if "v_loss" in out:
                                    msg += f"v_loss={out.get('v_loss', None)}  v_mean={out.get('v_mean', None)}  "
                                if "eval_return" in out:
                                    msg += f"eval_return={out['eval_return']:.2f}  "
                                if "eval_v5_normalized" in out:
                                    msg += f"norm_v5={out['eval_v5_normalized']:.2f}  "
                                msg += f"(elapsed={elapsed:.1f}s, {sps:.1f} upd/s)"
                                print(msg)

                            if step % (EVAL_EVERY * 2) == 0 or step == TOTAL_UPDATES:
                                torch.save(
                                    {
                                        "agent": agent.state_dict(),
                                        "norm": {"mean": obs_mean, "std": obs_std},
                                        "max_action": max_action,
                                        "step": step,
                                        "v5_norm_ref": v5_ref,
                                    },
                                    ckpt_path,
                                )

                    if eval_env is not None:
                        try:
                            eval_env.close()
                        except Exception:
                            pass

    print("\nAll runs done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("FATAL ERROR:", repr(e))
        traceback.print_exc()
        sys.exit(1)
