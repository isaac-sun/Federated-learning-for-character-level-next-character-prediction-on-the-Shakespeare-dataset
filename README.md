# 🎭 Federated Shakespeare Language Model

> Federated learning for character-level next-character prediction on the Shakespeare dataset, built with **PyTorch** and **Flower (flwr)**.

---

## 📖 Table of Contents / 目录

- [English](#english)
- [中文](#中文)

---

<a name="english"></a>
## 🇬🇧 English

### Overview

This project implements a **federated learning** system for character-level language modeling on Shakespeare's plays. Each Shakespeare character (e.g., ROMEO, JULIET, HAMLET) acts as an independent federated client, training a local LSTM model on their own dialogue text. The server aggregates model updates using the **Federated Averaging (FedAvg)** algorithm.

### Features

- 🔁 **Federated Averaging (FedAvg)**: Weighted aggregation of model parameters across clients
- 🧠 **LSTM Language Model**: Multi-layer LSTM with character embeddings for next-character prediction
- 📦 **Automatic Dataset Handling**: Auto-downloads TinyShakespeare and parses it by character, with synthetic fallback
- 🌸 **Flower (flwr) Integration**: Implements `NumPyClient` interface for standard federated learning workflows
- 📈 **Metrics & Visualization**: Tracks loss, perplexity, and accuracy per round with automatic plot generation
- ⚙️ **Configurable Hyperparameters**: All training, model, and FL parameters adjustable via command-line arguments
- 💾 **Model Checkpointing**: Saves the final global model with full metadata for inference

### Project Structure

```
federated_shakespeare/
│
├── data/
│   └── shakespeare_loader.py   # Dataset download, parsing, and loading
│
├── models/
│   └── lstm_model.py           # CharLSTM language model
│
├── federated/
│   ├── client.py               # Flower NumPyClient implementation
│   └── server.py               # FedAvg server and simulation loop
│
├── utils/
│   ├── preprocessing.py        # Character vocabulary and sequence generation
│   └── metrics.py              # Loss, perplexity, accuracy, plotting
│
├── experiments/
│   └── train.py                # Main training orchestration script
│
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore                  # Git ignore rules
```

### Installation

**Option 1: Using conda (recommended)**

```bash
conda create -n fl_env python=3.10
conda activate fl_env
cd federated_shakespeare
pip install -r requirements.txt
```

**Option 2: Using venv**

```bash
python3 -m venv .venv
source .venv/bin/activate
cd federated_shakespeare
pip install -r requirements.txt
```

### Quick Start

**Run with default settings (25 rounds, 5 clients/round):**

```bash
cd federated_shakespeare
python experiments/train.py
```

**Custom configuration:**

```bash
python experiments/train.py \
    --num-rounds 50 \
    --clients-per-round 10 \
    --local-epochs 2 \
    --hidden-dim 256 \
    --lr 0.0005 \
    --batch-size 32
```

**See all options:**

```bash
python experiments/train.py --help
```

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `--num-rounds` | 25 | Number of federated communication rounds |
| `--clients-per-round` | 5 | Clients sampled per round |
| `--local-epochs` | 1 | Local training epochs per round |
| `--embed-dim` | 64 | Character embedding dimension |
| `--hidden-dim` | 128 | LSTM hidden state dimension |
| `--num-layers` | 2 | Number of LSTM layers |
| `--dropout` | 0.3 | Dropout rate |
| `--batch-size` | 16 | Mini-batch size |
| `--lr` | 0.001 | Local learning rate |
| `--seq-length` | 50 | Input sequence length |
| `--seed` | 42 | Random seed |
| `--device` | auto | Compute device (auto/cpu/cuda/mps) |

### Outputs

After training, the `results/` directory will contain:

- `metrics.json` — Per-round loss, perplexity, and accuracy
- `loss_vs_rounds.png` — Loss curve over federated rounds
- `perplexity_vs_rounds.png` — Perplexity curve over federated rounds
- `combined_metrics.png` — Side-by-side loss and perplexity plots
- `global_model.pt` — Final global model checkpoint

### Algorithm

**Federated Averaging (FedAvg)** — McMahan et al., 2017

Each round:
1. **Server** sends global model parameters to a random subset of clients
2. Each selected **client** trains the model on its local data for `E` epochs
3. Clients return updated parameters and sample counts to the server
4. **Server** computes a weighted average of client parameters (weighted by sample count)
5. The aggregated model becomes the new global model

### Model Architecture

```
Input (char indices) → Embedding(vocab_size, 64)
                     → LSTM(64, 128, num_layers=2, dropout=0.3)
                     → Dropout(0.3)
                     → Linear(128, vocab_size)
                     → Output (logits)
```

### Metrics

- **Cross-Entropy Loss**: Standard loss for classification at each character position
- **Perplexity**: `exp(loss)` — measures how "surprised" the model is; lower is better
- **Accuracy**: Fraction of correctly predicted next characters

---

<a name="中文"></a>
## 🇨🇳 中文

### 概述

本项目实现了一个基于莎士比亚戏剧的**联邦学习**字符级语言建模系统。每个莎士比亚角色（如ROMEO、JULIET、HAMLET）作为独立的联邦学习客户端，在自己的对话文本上训练本地LSTM模型。服务器使用**联邦平均（FedAvg）**算法聚合模型更新。

### 功能特性

- 🔁 **联邦平均（FedAvg）**：跨客户端的模型参数加权聚合
- 🧠 **LSTM语言模型**：多层LSTM配合字符嵌入，用于下一个字符预测
- 📦 **自动数据集处理**：自动下载TinyShakespeare并按角色解析，带合成数据回退
- 🌸 **Flower (flwr) 集成**：实现`NumPyClient`接口，支持标准联邦学习工作流
- 📈 **度量与可视化**：跟踪每轮的损失、困惑度和准确率，自动生成图表
- ⚙️ **可配置超参数**：所有训练、模型和FL参数均可通过命令行调整
- 💾 **模型检查点**：保存包含完整元数据的最终全局模型

### 项目结构

```
federated_shakespeare/
│
├── data/
│   └── shakespeare_loader.py   # 数据集下载、解析和加载
│
├── models/
│   └── lstm_model.py           # CharLSTM语言模型
│
├── federated/
│   ├── client.py               # Flower NumPyClient实现
│   └── server.py               # FedAvg服务器和模拟循环
│
├── utils/
│   ├── preprocessing.py        # 字符词汇表和序列生成
│   └── metrics.py              # 损失、困惑度、准确率、绘图
│
├── experiments/
│   └── train.py                # 主训练编排脚本
│
├── requirements.txt            # Python依赖
├── README.md                   # 本文件
└── .gitignore                  # Git忽略规则
```

### 安装

**方式一：使用conda（推荐）**

```bash
conda create -n fl_env python=3.10
conda activate fl_env
cd federated_shakespeare
pip install -r requirements.txt
```

**方式二：使用venv**

```bash
python3 -m venv .venv
source .venv/bin/activate
cd federated_shakespeare
pip install -r requirements.txt
```

### 快速开始

**使用默认设置运行（25轮，每轮5个客户端）：**

```bash
cd federated_shakespeare
python experiments/train.py
```

**自定义配置：**

```bash
python experiments/train.py \
    --num-rounds 50 \
    --clients-per-round 10 \
    --local-epochs 2 \
    --hidden-dim 256 \
    --lr 0.0005 \
    --batch-size 32
```

**查看所有选项：**

```bash
python experiments/train.py --help
```

### 配置参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--num-rounds` | 25 | 联邦通信轮数 |
| `--clients-per-round` | 5 | 每轮采样的客户端数 |
| `--local-epochs` | 1 | 每轮本地训练epoch数 |
| `--embed-dim` | 64 | 字符嵌入维度 |
| `--hidden-dim` | 128 | LSTM隐藏状态维度 |
| `--num-layers` | 2 | LSTM层数 |
| `--dropout` | 0.3 | Dropout率 |
| `--batch-size` | 16 | Mini-batch大小 |
| `--lr` | 0.001 | 本地学习率 |
| `--seq-length` | 50 | 输入序列长度 |
| `--seed` | 42 | 随机种子 |
| `--device` | auto | 计算设备（auto/cpu/cuda/mps） |

### 输出结果

训练完成后，`results/` 目录将包含：

- `metrics.json` — 每轮的损失、困惑度和准确率
- `loss_vs_rounds.png` — 联邦轮次的损失曲线
- `perplexity_vs_rounds.png` — 联邦轮次的困惑度曲线
- `combined_metrics.png` — 并排的损失和困惑度图表
- `global_model.pt` — 最终全局模型检查点

### 算法

**联邦平均（FedAvg）** — McMahan等人，2017

每轮：
1. **服务器**将全局模型参数发送给随机选择的客户端子集
2. 每个被选中的**客户端**在本地数据上训练模型E个epoch
3. 客户端将更新后的参数和样本数量返回给服务器
4. **服务器**计算客户端参数的加权平均（按样本数量加权）
5. 聚合后的模型成为新的全局模型

### 模型架构

```
输入（字符索引） → Embedding(vocab_size, 64)
               → LSTM(64, 128, num_layers=2, dropout=0.3)
               → Dropout(0.3)
               → Linear(128, vocab_size)
               → 输出（logits）
```

### 评估度量

- **交叉熵损失**：每个字符位置的标准分类损失
- **困惑度**：`exp(损失)` — 衡量模型的"惊讶程度"；越低越好
- **准确率**：正确预测的下一个字符的比例

---

## 📄 License / 许可证

This project is released under the MIT License.
本项目基于MIT许可证发布。
