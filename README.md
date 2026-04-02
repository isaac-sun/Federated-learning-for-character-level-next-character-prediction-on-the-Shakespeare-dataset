<div align="center">

# 🎭 Federated Shakespeare

**Federated Learning for Character-Level Language Modeling on Shakespeare**

Built with PyTorch · Flower · FedAvg

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.6%2B-green.svg)](https://flower.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[中文文档](README_CN.md)

</div>

---

## Overview

A complete federated learning pipeline for **next-character prediction** on Shakespeare's plays. Each Shakespeare character (ROMEO, JULIET, HAMLET, etc.) acts as an independent FL client, training a local **LSTM** model on their own dialogue. The server aggregates updates via **Federated Averaging (FedAvg)**.

### Highlights

- 🔁 **FedAvg** — weighted aggregation of model parameters across 500+ clients
- 🧠 **CharLSTM** — multi-layer LSTM with character embeddings
- 📦 **Auto Dataset** — downloads [TinyShakespeare](https://github.com/karpathy/char-rnn), parses by character, with synthetic fallback
- 🌸 **Flower Compatible** — implements `NumPyClient` interface
- 📈 **Metrics & Plots** — loss, perplexity, accuracy curves generated automatically
- ⚙️ **Fully Configurable** — all hyperparameters tunable via CLI
- 💾 **Checkpointing** — saves global model with full metadata

## Project Structure

```
federated_shakespeare/
├── data/
│   └── shakespeare_loader.py    # Dataset download, parsing, and client partitioning
├── models/
│   └── lstm_model.py            # CharLSTM language model
├── federated/
│   ├── client.py                # Flower NumPyClient implementation
│   └── server.py                # FedAvg aggregation and training loop
├── utils/
│   ├── preprocessing.py         # Character vocabulary and sequence generation
│   └── metrics.py               # Metrics computation and plotting
├── experiments/
│   └── train.py                 # Main entry point
├── requirements.txt
└── .gitignore
```

## Getting Started

### Prerequisites

- Python 3.10+
- (Optional) CUDA / Apple MPS for GPU acceleration

### Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/federated-shakespeare.git
cd federated-shakespeare/federated_shakespeare

# Create environment (pick one)
conda create -n fl_env python=3.10 && conda activate fl_env
# or
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Default: 25 rounds, 5 clients/round
python experiments/train.py
```

```bash
# Custom run
python experiments/train.py \
    --num-rounds 50 \
    --clients-per-round 10 \
    --local-epochs 2 \
    --hidden-dim 256 \
    --embed-dim 128 \
    --lr 0.0008
```

## Configuration

All hyperparameters are configurable via command-line arguments:

| Category | Flag | Default | Description |
|:---|:---|:---:|:---|
| **FL** | `--num-rounds` | `25` | Federated communication rounds |
| | `--clients-per-round` | `5` | Clients sampled per round |
| | `--local-epochs` | `1` | Local training epochs per round |
| **Model** | `--embed-dim` | `64` | Character embedding dimension |
| | `--hidden-dim` | `128` | LSTM hidden dimension |
| | `--num-layers` | `2` | Number of LSTM layers |
| | `--dropout` | `0.3` | Dropout rate |
| **Training** | `--batch-size` | `16` | Mini-batch size |
| | `--lr` | `0.001` | Local learning rate (Adam) |
| | `--seq-length` | `50` | Input sequence length |
| **Other** | `--seed` | `42` | Random seed |
| | `--device` | `auto` | `cpu` / `cuda` / `mps` / `auto` |

Run `python experiments/train.py --help` for full details.

## Model Architecture

```
CharLSTM(
  Embedding(vocab_size, embed_dim)      # char index → dense vector
  LSTM(embed_dim, hidden_dim, layers=2) # sequence modeling
  Dropout(0.3)                          # regularization
  Linear(hidden_dim, vocab_size)        # next-char logits
)
```

Default config: **~244K parameters**.

## Algorithm

**Federated Averaging (FedAvg)** — [McMahan et al., 2017](https://arxiv.org/abs/1602.05629)

```
for each round r = 1, 2, ..., R:
    S ← random subset of K clients
    for each client k ∈ S (in parallel):
        w_k ← ClientUpdate(w_global, local_data_k)
    w_global ← Σ (n_k / n_total) · w_k
```

Each client trains with **Adam** optimizer and **gradient clipping** (max norm = 5.0) to stabilize LSTM training.

## Outputs

After training completes, `results/` contains:

| File | Description |
|:---|:---|
| `metrics.json` | Per-round loss, perplexity, accuracy |
| `loss_vs_rounds.png` | Loss curve |
| `perplexity_vs_rounds.png` | Perplexity curve |
| `accuracy_vs_rounds.png` | Accuracy curve |
| `combined_metrics.png` | Side-by-side loss & perplexity |
| `global_model.pt` | Final model checkpoint with metadata |

## Metrics

| Metric | Formula | Notes |
|:---|:---|:---|
| **Cross-Entropy Loss** | $-\frac{1}{N}\sum \log p(c_t)$ | Standard classification loss |
| **Perplexity** | $e^{\text{loss}}$ | Lower = better; measures model "surprise" |
| **Accuracy** | $\frac{\text{correct predictions}}{\text{total predictions}}$ | Character-level next-char accuracy |

## License

This project is released under the [MIT License](LICENSE).
