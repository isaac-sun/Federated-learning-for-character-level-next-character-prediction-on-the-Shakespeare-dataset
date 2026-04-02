<div align="center">

# 🎭 FedBard

**Robust Federated Learning with Shapley-Value Defense on Shakespeare**

PyTorch · Flower · FedAvg · SVRFL

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.6%2B-green.svg)](https://flower.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey.svg)]()

[中文文档](README_CN.md)

</div>

---

## Overview

FedBard is a federated learning research platform for **next-character prediction** on Shakespeare, featuring a faithful reproduction of the **SVRFL** (Shapley Value-based Robust Federated Learning) defense from:

> *"Robust and Fair Federated Learning Based on Model-Agnostic Shapley Value"*

Each Shakespeare speaking role is an independent FL client. The system supports both a **FedAvg baseline** and the full **SVRFL defense pipeline** — including Monte Carlo Shapley estimation, free-rider detection, poisonous update mitigation, and reputation management — under multiple adversarial attack scenarios.

### Key Features

| | Feature | Description |
|:---|:---|:---|
| 🔁 | **FedAvg Baseline** | Weighted parameter aggregation across 500+ clients (LSTM) or 20 selected clients (GRU) |
| 🛡️ | **SVRFL Defense** | Shapley-based free-rider detection + binary utility filtering for poisoning mitigation |
| ⚔️ | **Attack Suite** | DFR, SDFR, AFR (free-rider) and SF (sign-flipping poisoning), plus concurrent attack mode |
| 🧠 | **Dual Models** | CharLSTM (baseline) and CharGRU (SVRFL, matching the paper) |
| 📦 | **Auto Dataset** | Downloads [TinyShakespeare](https://github.com/karpathy/char-rnn), parses by character, with synthetic fallback |
| 🖥️ | **Cross-Platform GPU** | Auto-detects CUDA (NVIDIA) · MPS (Apple Silicon) · CPU; supports `cuda:N` for multi-GPU |
| 🌸 | **Flower Compatible** | Implements `NumPyClient` interface |
| 📈 | **Rich Visualization** | Loss, perplexity, accuracy, reputation trajectories, detection metrics, utility scores |
| ⚙️ | **Fully Configurable** | All hyperparameters tunable via CLI |

---

## Project Structure

```
federated_shakespeare/
├── attacks/
│   ├── __init__.py              # Attack type constants
│   ├── free_rider.py            # DFR, SDFR, AFR attacks
│   └── poisoning.py             # SF (sign-flipping) attack
├── data/
│   └── shakespeare_loader.py    # Dataset download, parsing, client partitioning,
│                                # server validation set construction
├── models/
│   ├── lstm_model.py            # CharLSTM (baseline)
│   └── gru_model.py             # CharGRU (SVRFL, paper-faithful)
├── federated/
│   ├── client.py                # Flower NumPyClient
│   ├── server.py                # FedAvg aggregation loop
│   └── svrfl_server.py          # SVRFL experiment runner
├── utils/
│   ├── preprocessing.py         # Character vocabulary and sequence generation
│   ├── metrics.py               # Metrics computation and plotting
│   ├── shapley.py               # Monte Carlo Shapley value estimation
│   ├── svrfl.py                 # SVRFL defense logic (detection + mitigation)
│   └── device.py                # Cross-platform device resolution and GPU info
├── experiments/
│   ├── train.py                 # FedAvg baseline entry point
│   ├── train_svrfl.py           # SVRFL experiment entry point
│   └── run_all.py               # One-click full experiment suite runner
├── requirements.txt
└── .gitignore
```

---

## Environment Setup

> **TL;DR:** use `--device auto` (default) and FedBard will pick the best available device automatically.

### System Requirements

| Component | Minimum | Recommended |
|:---|:---|:---|
| Python | 3.10 | 3.11 |
| RAM | 8 GB | 16 GB |
| Storage | 500 MB | 1 GB |
| GPU | — | NVIDIA (CUDA 12+) or Apple Silicon |

---

### 🍎 macOS — Apple Silicon (M1/M2/M3/M4)

Apple Silicon Macs use **MPS** (Metal Performance Shaders) for GPU acceleration.

```bash
# 1. Create environment
conda create -n fedbard python=3.10
conda activate fedbard

# 2. Install PyTorch with MPS support (included in standard macOS wheels)
pip install torch torchvision

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Run — MPS is selected automatically
python experiments/train_svrfl.py --defense svrfl --attack dfr
# Or force MPS explicitly:
python experiments/train_svrfl.py --device mps --defense svrfl --attack dfr
```

> **Note:** MPS support requires macOS 12.3+. Verify with `python -c "import torch; print(torch.backends.mps.is_available())"`.

---

### 🍎 macOS — Intel Mac

Intel Macs do not have MPS. The device falls back to CPU automatically.

```bash
conda create -n fedbard python=3.10 && conda activate fedbard
pip install torch torchvision
pip install -r requirements.txt

python experiments/train_svrfl.py --device cpu --defense svrfl --attack dfr
```

---

### 🪟 Windows — NVIDIA GPU (Recommended)

Windows with an NVIDIA GPU requires a **CUDA-enabled PyTorch build**. The standard `pip install torch` installs the CPU-only version on Windows.

**Step 1 — Check your CUDA version:**
```powershell
nvidia-smi
# Look for "CUDA Version: XX.X" in the top-right corner
```

**Step 2 — Install CUDA-enabled PyTorch:**

| CUDA Version | Install Command |
|:---:|:---|
| 12.4 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| 12.1 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| 11.8 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |

> Find all available wheels at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

**Step 3 — Install remaining dependencies:**
```powershell
conda create -n fedbard python=3.10
conda activate fedbard
# (install CUDA PyTorch as above first)
pip install -r requirements.txt
```

**Step 4 — Run:**
```powershell
# Auto-detect NVIDIA GPU
python experiments/train_svrfl.py --defense svrfl --attack dfr

# Explicitly target GPU 0
python experiments/train_svrfl.py --device cuda:0 --defense svrfl --attack dfr

# Multi-GPU: target a specific card
python experiments/train_svrfl.py --device cuda:1 --defense svrfl --attack sf
```

**Verify CUDA is available:**
```python
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce RTX 5070 Ti
```

---

### 🐧 Linux — NVIDIA GPU

```bash
conda create -n fedbard python=3.10 && conda activate fedbard

# Install CUDA PyTorch (adjust cu124 to match your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python experiments/train_svrfl.py --defense svrfl --attack dfr
```

---

### Device Selection Reference

The `--device` flag is supported by all entry points (`train.py`, `train_svrfl.py`, `run_all.py`).

| Value | Hardware | Notes |
|:---:|:---|:---|
| `auto` | Best available | **Default.** CUDA → MPS → CPU |
| `cuda` | NVIDIA GPU 0 | Requires CUDA PyTorch build |
| `cuda:0` | NVIDIA GPU 0 | Explicit index (same as `cuda`) |
| `cuda:1` | NVIDIA GPU 1 | For multi-GPU systems |
| `mps` | Apple Silicon GPU | macOS 12.3+ only |
| `cpu` | CPU only | Always available |

At startup, FedBard prints a device info block:
```
  🖥️  Device / 计算设备
     Type    : cuda
     GPU     : NVIDIA GeForce RTX 5070 Ti
     VRAM    : 16.0 GB
     CUDA    : 12.4
```

---

## Quick Start

### ▶ Recommended: One-Click Full Experiment Suite

Run all 12 experiment combinations (both defenses × all attacks) with a single command:

```bash
cd federated_shakespeare

# Full suite — saves all results + summary.csv (~6–12 hours on GPU)
python experiments/run_all.py

# Quick smoke test — reduced rounds, ideal for verifying your setup
python experiments/run_all.py --quick

# Run on a specific device
python experiments/run_all.py --device cuda        # NVIDIA GPU
python experiments/run_all.py --device mps         # Apple Silicon
python experiments/run_all.py --device cpu         # CPU only

# Run only a subset of attacks
python experiments/run_all.py --attacks dfr sf concurrent
```

Results land in `results/` with a `summary.csv` comparing all configurations side-by-side.

---

### Individual Experiments

#### FedAvg Baseline

```bash
python experiments/train.py                                    # 25 rounds, 5 clients/round
python experiments/train.py --num-rounds 50 --clients-per-round 10
python experiments/train.py --device cuda --num-rounds 50
```

#### SVRFL Defense Experiments

```bash
# No attack baseline
python experiments/train_svrfl.py --defense fedavg --attack none

# SVRFL vs. individual attacks
python experiments/train_svrfl.py --defense svrfl --attack dfr
python experiments/train_svrfl.py --defense svrfl --attack sdfr
python experiments/train_svrfl.py --defense svrfl --attack afr
python experiments/train_svrfl.py --defense svrfl --attack sf

# SVRFL vs. concurrent attack (3 free-riders + 2 poisoners)
python experiments/train_svrfl.py --defense svrfl --attack concurrent

# FedAvg under attack (no defense — for ablation comparison)
python experiments/train_svrfl.py --defense fedavg --attack sf
```

---

## SVRFL Defense

The SVRFL pipeline implements two defense branches that execute every FL round:

### A. Free-Rider Detection

1. Estimate per-client **Shapley values** via Monte Carlo permutations over the server validation set
2. Compute **feature values**: $d_i = |\text{sv}_i| \;/\; (L_{\cos,i}^2 + \varepsilon)$, where $L_{\cos,i} = 1 - \cos(w_i, w_g)$
3. Run **KMeans(k=2)** on feature values; flag the high-centroid cluster if centroid ratio exceeds threshold $h$
4. Penalize detected free-riders: decrease reputation, zero out their Shapley values for this round

### B. Poisonous Update Mitigation

1. Maintain **exponential moving average utility** scores: $u_i = \alpha \cdot u_i + (1-\alpha) \cdot \text{sv}_i$
2. **Binary filtering**: only aggregate updates from clients where $u_i > 0$
3. Update **reputation** scores based on aggregation participation

### Attacks Implemented

| Attack | Type | Description |
|:---|:---|:---|
| **DFR** | Free-rider | Decaying Gaussian noise: $g = \sigma \cdot t^{-\gamma} \cdot \varepsilon$ |
| **SDFR** | Free-rider | Mimics previous global update direction and scale |
| **AFR** | Free-rider | SDFR + Gaussian noise on random 10% of coordinates |
| **SF** | Poisoning | Sign-flipping: $g_{\text{attack}} = -g_{\text{honest}}$ |
| **Concurrent** | Mixed | 1 DFR + 1 SDFR + 1 AFR + 2 SF attackers simultaneously |

---

## Configuration

### FedAvg Baseline (`train.py`)

| Category | Flag | Default | Description |
|:---|:---|:---:|:---|
| **FL** | `--num-rounds` | `25` | Communication rounds |
| | `--clients-per-round` | `5` | Clients sampled per round |
| | `--local-epochs` | `1` | Local training epochs |
| **Model** | `--embed-dim` | `64` | Embedding dimension |
| | `--hidden-dim` | `128` | LSTM hidden dimension |
| | `--num-layers` | `2` | Number of LSTM layers |
| **Training** | `--batch-size` | `16` | Batch size |
| | `--lr` | `0.001` | Learning rate (Adam) |
| | `--seq-length` | `50` | Sequence length |
| **System** | `--device` | `auto` | `auto` / `cuda` / `cuda:N` / `mps` / `cpu` |

### SVRFL Experiments (`train_svrfl.py`)

Includes all baseline flags above, plus:

| Category | Flag | Default | Description |
|:---|:---|:---:|:---|
| **Experiment** | `--defense` | `svrfl` | `fedavg` or `svrfl` |
| | `--attack` | `none` | `none` / `dfr` / `sdfr` / `afr` / `sf` / `concurrent` |
| **SVRFL** | `--shapley-mc-samples` | `50` | Monte Carlo permutations for Shapley estimation |
| | `--alpha` | `0.9` | EMA smoothing factor for utility scores |
| | `--threshold-h` | `200` | Free-rider detection KMeans centroid ratio threshold |
| | `--eta-server` | `1.0` | Server-side aggregation learning rate |
| **Scale** | `--num-clients` | `20` | Total FL client pool size |
| | `--clients-per-round` | `10` | Clients sampled per round |
| | `--val-samples` | `1000` | Server-side validation set size |
| **System** | `--device` | `auto` | `auto` / `cuda` / `cuda:N` / `mps` / `cpu` |

Run `python experiments/train_svrfl.py --help` for the complete parameter list.

---

## Model Architectures

```
CharLSTM (baseline)                    CharGRU (SVRFL, paper-faithful)
─────────────────────────────          ──────────────────────────────────
Embedding(vocab=67, dim=64)            Embedding(vocab=67, dim=64)
LSTM(input=64, hidden=128, layers=2)   GRU(input=64, hidden=128, layers=2)
Dropout(p=0.3)                         Dropout(p=0.3)
Linear(128 → 67)                       Linear(128 → 67)
≈ 244K parameters                      ≈ 186K parameters
```

The vocabulary size (67) is determined automatically from the dataset. Both models operate on character-level sequences of length 50.

---

## Outputs

### FedAvg Baseline → `results/`

| File | Description |
|:---|:---|
| `metrics.json` | Per-round loss, perplexity, accuracy |
| `loss_vs_rounds.png` | Training loss curve |
| `perplexity_vs_rounds.png` | Perplexity curve |
| `accuracy_vs_rounds.png` | Character-level accuracy curve |
| `combined_metrics.png` | Side-by-side loss & perplexity |
| `global_model.pt` | Final global model checkpoint |

### SVRFL Experiments → `results/<defense>_<attack>_r<rounds>/`

All of the above, plus:

| File | Description |
|:---|:---|
| `round_logs.json` | Per-round Shapley values, reputations, detections, utility scores |
| `config.json` | Full experiment configuration snapshot |
| `reputation_trajectories.png` | Benign vs. malicious client reputations over rounds |
| `freerider_detection.png` | Detection precision and FRDR per round |
| `utility_trajectories.png` | Client utility score evolution over rounds |

---

## Algorithms

### FedAvg — [McMahan et al., 2017](https://arxiv.org/abs/1602.05629)

```
for round r = 1 .. R:
    S ← sample K clients
    for client k ∈ S:
        w_k ← LocalTrain(w_global, data_k)
    w_global ← Σ (n_k / n) · w_k
```

### SVRFL

```
for round r = 1 .. R:
    I_r ← sample K clients with reputation ≥ 0
    g_i = w_global - w_i          for each i ∈ I_r   # compute updates
    sv_i ← MonteCarloShapley(g, D_val)                # Shapley estimation
    d_i = |sv_i| / (L_cos_i² + ε)                     # feature values
    FR ← KMeans(d_i, k=2) if centroid_ratio > h        # free-rider detection
    u_i = α·u_i + (1-α)·sv_i                          # utility EMA update
    w_global -= η · mean{ g_i : u_i > 0 }             # binary-filtered aggregation
    r_i ← update reputation for each i ∈ I_r
```

---

## Metrics

| Metric | Formula | Notes |
|:---|:---|:---|
| **Cross-Entropy Loss** | $-\frac{1}{N}\sum \log p(c_t)$ | Evaluated on server validation set |
| **Perplexity** | $e^{\text{loss}}$ | Lower is better |
| **Accuracy** | correct / total | Character-level next-char prediction |
| **FRDR** | TP / (TP + FN) | Free-rider detection recall |
| **Precision** | TP / (TP + FP) | Free-rider detection precision |

---

## Performance Notes

Experiment runtimes depend heavily on hardware. Approximate benchmarks for a **standard full run** (25 rounds, 10 clients/round, R=50 Shapley samples):

| Hardware | Approx. Time per Round | Full 25 Rounds |
|:---|:---:|:---:|
| NVIDIA RTX 4090 / 5070 Ti | ~1–2 min | ~30–50 min |
| Apple M2 Pro (MPS) | ~3–4 min | ~80–100 min |
| CPU only (8-core) | ~8–12 min | ~3–5 hours |

> **Tip:** Use `--shapley-mc-samples 20` for faster iteration during development; increase to `50`–`100` for final experiments.

---

## References

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017. [[arXiv]](https://arxiv.org/abs/1602.05629)
- *Robust and Fair Federated Learning Based on Model-Agnostic Shapley Value*

---

## License

This project is released under the [MIT License](LICENSE).
