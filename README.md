# Rational activations — experiment scripts

This repo contains the scripts we used to run and summarize experiments across:
- CIFAR-10 (plain + “boosted” training)
- Tiny-ImageNet (ViT-family via `timm`, with EMA logging)
- Offline MuJoCo RL (Minari datasets; IQL / TD3+BC)
- Result aggregation into CSV/LaTeX tables and paper-ready plots

## Rational activations (upstream code)

We use the rational activation implementation from:

https://github.com/ml-research/rational_activations

Follow their setup / install instructions to enable rational activations in our scripts (we intentionally do **not** provide a one-line `pip install ...` here because the upstream project documents the correct setup).

## Environment setup (non-trivial installs)

These scripts assume you already have a working Python environment and a compatible PyTorch install.

### PyTorch
Install PyTorch following the official instructions for your OS/CUDA version:
- https://pytorch.org/get-started/locally/

### Vision models + datasets
```bash
pip install timm datasets torchvision
```

### Offline RL (MuJoCo + Minari)
```bash
pip install gymnasium minari mujoco
```

Notes:
- `mujoco` may require system-specific dependencies (see Gymnasium/MuJoCo docs if you hit install/runtime issues).
- Some scripts also use plotting/tabulation libraries; install them only if you run the table/figure scripts.

## What’s in this repo

- `cifar_test_naive.py`  
  CIFAR-10 training loop; runs multiple seeds and logs `metrics.jsonl` plus a best-checkpoint per run.

- `cifar_test_boosted.py`  
  “Boosted” CIFAR-10 training and support for treating Rational parameters as a distinct optimizer group (e.g., LR multipliers).

- `imagenet_test.py`  
  Tiny-ImageNet training (HF datasets wrapper + timm models). Produces per-run directories with configs/logs/checkpoints.

- `RL_test.py`  
  Offline RL on Minari tasks with activation sweeps (includes rational) and optional separate LR scaling for rational parameters.

- `cifar_table.py`  
  Summarize CIFAR runs under a `BASE_ROOT` into CSV/TXT and generate paper-ready figures.

- `table_imagenet.py`  
  Scan Tiny-ImageNet run folders, summarize best EMA validation accuracy + epoch-time stats, and generate a single “all runs” EMA curve PDF (with inset).

- `table_RL.py`  
  Aggregate RL runs into per-run + grouped summaries, export CSV and LaTeX-friendly tables, and generate summary plots.

- `Section_A_Demo_ICML_final.ipynb`  
  Notebook used for paper/demo figures.

## Quickstart

Most scripts are “edit constants at the top, then run the file”.

### CIFAR-10 (plain)
1. Edit paths and experiment knobs in `cifar_test_naive.py`
2. Run:
```bash
python cifar_test_naive.py
```

### CIFAR-10 (boosted)
1. Edit paths / knobs in `cifar_test_boosted.py`
2. Run:
```bash
python cifar_test_boosted.py
```

### Tiny-ImageNet
1. Edit `DATA_ROOT` / `OUT_ROOT`, model list, and activation selection in `imagenet_test.py`
2. Run:
```bash
python imagenet_test.py
```

### Offline RL (Minari)
1. Edit task list / output root / hyperparameters in `RL_test.py`
2. Run:
```bash
python RL_test.py
```

## Summaries / tables / plots

### CIFAR summaries
```bash
python cifar_table.py
```

### Tiny-ImageNet summaries
```bash
python table_imagenet.py
```

### Offline RL summaries
```bash
python table_RL.py
```

## Notes

- Run directory names encode model/activation variants so the table scripts can parse them consistently.
- If you add new activation variants, update the parsing rules in the corresponding `table_*.py` script.

## Acknowledgements

Rational activation code is based on: https://github.com/ml-research/rational_activations
