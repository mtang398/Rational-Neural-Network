# Rational activations — experiment scripts

This repo contains the scripts we used to run and summarize experiments across:
- CIFAR-10 (plain + “boosted” training)
- Tiny-ImageNet (ViT-family via `timm`, with EMA logging)
- Offline MuJoCo RL (Minari datasets; IQL / TD3+BC)
- Result aggregation into CSV/LaTeX tables and paper-ready plots

## Rational activations

We use the rational activation implementation from:

https://github.com/ml-research/rational_activations

Follow their setup / install instructions to enable rational activations in our scripts

## Quickstart

All scripts are “edit constants at the top, then run the file”.

## Acknowledgements

Rational activation code is based on: https://github.com/ml-research/rational_activations
