<div align="center">

# 🎭 Federated Shakespeare

**Federated Learning for Character-Level Language Modeling on Shakespeare**

Built with PyTorch · Flower · FedAvg

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.6%2B-green.svg)](https://flower.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](#overview) · [中文](#概述)

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

---

## 概述

本项目实现了一个完整的**联邦学习**流水线，用于莎士比亚戏剧的**字符级下一字符预测**。每个莎士比亚角色（如 ROMEO、JULIET、HAMLET）作为独立的联邦客户端，在自己的对话文本上训练本地 LSTM 模型，服务器通过**联邦平均（FedAvg）**聚合更新。

### 功能特性

- 🔁 **FedAvg** — 跨 500+ 客户端的模型参数加权聚合
- 🧠 **CharLSTM** — 多层 LSTM + 字符嵌入
- 📦 **自动数据集** — 自动下载 TinyShakespeare 并按角色解析，带合成数据回退
- 🌸 **Flower 兼容** — 实现 `NumPyClient` 接口
- 📈 **度量与图表** — 自动生成损失、困惑度、准确率曲线
- ⚙️ **完全可配置** — 所有超参数均可通过命令行调整
- 💾 **模型保存** — 保存包含完整元数据的全局模型

### 快速开始

```bash
cd federated_shakespeare
pip install -r requirements.txt
python experiments/train.py                         # 默认配置
python experiments/train.py --num-rounds 50         # 自定义轮次
python experiments/train.py --help                  # 查看所有参数
```

### 评估度量

| 度量 | 公式 | 说明 |
|:---|:---|:---|
| **交叉熵损失** | $-\frac{1}{N}\sum \log p(c_t)$ | 标准分类损失 |
| **困惑度** | $e^{\text{loss}}$ | 越低越好，衡量模型"惊讶程度" |
| **准确率** | 正确预测数 / 总预测数 | 字符级下一字符准确率 |

---

## License

This project is released under the [MIT License](LICENSE).

本项目基于 [MIT 许可证](LICENSE) 发布。
