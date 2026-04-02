<div align="center">

# 🎭 FedBard

**Robust Federated Learning with Shapley-Value Defense on Shakespeare**

PyTorch · Flower · FedAvg · SVRFL

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.6%2B-green.svg)](https://flower.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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
| 🌸 | **Flower Compatible** | Implements `NumPyClient` interface |
| 📈 | **Rich Visualization** | Loss, perplexity, accuracy, reputation trajectories, detection metrics, utility scores |
| ⚙️ | **Fully Configurable** | All hyperparameters tunable via CLI |

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
│   └── svrfl.py                 # SVRFL defense logic (detection + mitigation)
├── experiments/
│   ├── train.py                 # FedAvg baseline entry point
│   └── train_svrfl.py           # SVRFL experiment entry point
├── requirements.txt
└── .gitignore
```

## Getting Started

### Prerequisites

- Python 3.10+
- (Optional) CUDA / Apple MPS for GPU acceleration

### Installation

```bash
git clone https://github.com/<your-username>/FedBard.git
cd FedBard/federated_shakespeare

# Create environment (pick one)
conda create -n fedbard python=3.10 && conda activate fedbard
# or
python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

### Quick Start — FedAvg Baseline

```bash
python experiments/train.py                           # default: 25 rounds, 5 clients/round, LSTM
python experiments/train.py --num-rounds 50 --clients-per-round 10
```

### Quick Start — SVRFL Defense

```bash
# SVRFL defending against DFR free-riders
python experiments/train_svrfl.py --defense svrfl --attack dfr

# SVRFL defending against sign-flipping poisoning
python experiments/train_svrfl.py --defense svrfl --attack sf

# SVRFL defending against concurrent attack (3 free-riders + 2 poisoners)
python experiments/train_svrfl.py --defense svrfl --attack concurrent

# FedAvg under attack (no defense, for comparison)
python experiments/train_svrfl.py --defense fedavg --attack dfr

# No attack baseline
python experiments/train_svrfl.py --defense fedavg --attack none
```

## SVRFL Defense

The SVRFL pipeline implements two defense branches that execute every FL round:

### A. Free-Rider Detection

1. Estimate per-client **Shapley values** via Monte Carlo permutations
2. Compute **feature values**: $d_i = |\text{sv}_i| \;/\; (L_{\cos,i}^2 + \varepsilon)$ where $L_{\cos,i} = 1 - \cos(w_i, w_g)$
3. Run **KMeans(k=2)** on feature values; flag the high-centroid cluster if centroid ratio exceeds threshold $h$
4. Penalize detected free-riders: decrease reputation, zero out their Shapley values

### B. Poisonous Update Mitigation

1. Maintain **exponential moving average utility** scores: $u_i = \alpha \cdot u_i + (1-\alpha) \cdot \text{sv}_i$
2. **Binary filtering**: only aggregate updates from clients with $u_i > 0$
3. Update **reputation** based on aggregation participation

### Attacks Implemented

| Attack | Type | Description |
|:---|:---|:---|
| **DFR** | Free-rider | Decaying Gaussian noise: $g = \sigma \cdot t^{-\gamma} \cdot \varepsilon$ |
| **SDFR** | Free-rider | Mimics previous global update direction and scale |
| **AFR** | Free-rider | SDFR + Gaussian noise on random 10% of coordinates |
| **SF** | Poisoning | Sign-flipping: $g_{\text{attack}} = -g_{\text{honest}}$ |
| **Concurrent** | Mixed | 1 DFR + 1 SDFR + 1 AFR + 2 SF attackers |

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

### SVRFL Experiments (`train_svrfl.py`)

Includes all baseline flags above, plus:

| Category | Flag | Default | Description |
|:---|:---|:---:|:---|
| **Experiment** | `--defense` | `svrfl` | `fedavg` or `svrfl` |
| | `--attack` | `none` | `none` / `dfr` / `sdfr` / `afr` / `sf` / `concurrent` |
| **SVRFL** | `--shapley-mc-samples` | `50` | Monte Carlo permutations for Shapley estimation |
| | `--alpha` | `0.9` | EMA smoothing factor for utility scores |
| | `--threshold-h` | `200` | Free-rider detection KMeans threshold |
| | `--eta-server` | `1.0` | Server-side learning rate |
| **Scale** | `--num-clients` | `20` | Total FL client pool size |
| | `--clients-per-round` | `10` | Clients sampled per round |
| | `--val-samples` | `1000` | Server-side validation set size |

Run `python experiments/train_svrfl.py --help` for the full list.

## Model Architectures

```
CharLSTM (baseline)                    CharGRU (SVRFL, paper-faithful)
─────────────────                      ────────────────────────────────
Embedding(67, 64)                      Embedding(67, 64)
LSTM(64, 128, layers=2)                GRU(64, 128, layers=2)
Dropout(0.3)                           Dropout(0.3)
Linear(128, 67)                        Linear(128, 67)
~244K params                           ~186K params
```

## Outputs

### FedAvg Baseline → `results/`

| File | Description |
|:---|:---|
| `metrics.json` | Per-round loss, perplexity, accuracy |
| `loss_vs_rounds.png` | Loss curve |
| `perplexity_vs_rounds.png` | Perplexity curve |
| `accuracy_vs_rounds.png` | Accuracy curve |
| `combined_metrics.png` | Side-by-side loss & perplexity |
| `global_model.pt` | Final model checkpoint |

### SVRFL Experiments → `results/<defense>_<attack>_r<rounds>/`

All of the above, plus:

| File | Description |
|:---|:---|
| `round_logs.json` | Per-round Shapley values, reputations, detections, utility scores |
| `config.json` | Full experiment configuration |
| `reputation_trajectories.png` | Benign vs. malicious client reputations over rounds |
| `freerider_detection.png` | Detection precision and FRDR over rounds |
| `utility_trajectories.png` | Client utility score evolution |

## Algorithms

### FedAvg — [McMahan et al., 2017](https://arxiv.org/abs/1602.05629)

```
for round r = 1 .. R:
    S ← sample K clients
    for client k ∈ S:
        w_k ← LocalTrain(w_global, data_k)
    w_global ← Σ (n_k / n) · w_k
```

### SVRFL — [Paper](https://doi.org/...)

```
for round r = 1 .. R:
    I_r ← sample K clients (reputation ≥ 0)
    g_i = w_global - w_i for each i ∈ I_r           # client updates
    sv_i ← MonteCarlo Shapley(g, D_val)             # Shapley estimation
    d_i = |sv_i| / (L_cos_i² + ε)                   # feature values
    FR ← KMeans(d_i, k=2) if ratio > h              # free-rider detection
    u_i = α·u_i + (1-α)·sv_i                        # utility update
    w_global -= η · mean{g_i : u_i > 0}             # binary-filtered aggregation
```

## Metrics

| Metric | Formula | Notes |
|:---|:---|:---|
| **Cross-Entropy Loss** | $-\frac{1}{N}\sum \log p(c_t)$ | Standard NLL loss |
| **Perplexity** | $e^{\text{loss}}$ | Lower = better |
| **Accuracy** | correct / total | Character-level prediction |
| **FRDR** | TP / (TP + FN) | Free-rider detection rate |
| **Precision** | TP / (TP + FP) | Detection precision |

## References

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017
- *Robust and Fair Federated Learning Based on Model-Agnostic Shapley Value*

## License

This project is released under the [MIT License](LICENSE).
