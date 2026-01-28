# Rational activations — experiment scripts

This repo contains the scripts we used to run and summarize experiments across:
- CIFAR-10 (plain + “boosted” training)
- Tiny-ImageNet (ViT-family via `timm`, with EMA logging)
- Offline MuJoCo RL (Minari datasets; IQL / TD3+BC)
- Result aggregation into CSV/LaTeX tables and paper-ready plots

## Rational activations (upstream code)

We use the rational activation implementation from:

https://github.com/ml-research/rational_activations

Follow their setup / install instructions to enable rational activations in our scripts

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

All scripts are “edit constants at the top, then run the file”.

## Acknowledgements

Rational activation code is based on: https://github.com/ml-research/rational_activations
