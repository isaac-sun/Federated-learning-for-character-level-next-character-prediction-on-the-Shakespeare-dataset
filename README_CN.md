<div align="center">

# 🎭 联邦莎士比亚

**基于莎士比亚数据集的联邦学习字符级语言建模**

PyTorch · Flower · FedAvg

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.6%2B-green.svg)](https://flower.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README.md)

</div>

---

## 概述

一个完整的联邦学习流水线，用于莎士比亚戏剧的**字符级下一字符预测**。每个莎士比亚角色（如 ROMEO、JULIET、HAMLET）作为独立的联邦客户端，在自己的对话文本上训练本地 **LSTM** 模型，服务器通过**联邦平均（FedAvg）**聚合更新。

### 特性

- 🔁 **FedAvg** — 跨 500+ 客户端的模型参数加权聚合
- 🧠 **CharLSTM** — 多层 LSTM + 字符嵌入
- 📦 **自动数据集** — 自动下载 [TinyShakespeare](https://github.com/karpathy/char-rnn) 并按角色解析，带合成数据回退
- 🌸 **Flower 兼容** — 实现 `NumPyClient` 接口
- 📈 **度量与图表** — 自动生成损失、困惑度、准确率曲线
- ⚙️ **完全可配置** — 所有超参数均可通过命令行调整
- 💾 **模型保存** — 保存包含完整元数据的全局模型检查点

## 项目结构

```
federated_shakespeare/
├── data/
│   └── shakespeare_loader.py    # 数据集下载、解析和客户端分区
├── models/
│   └── lstm_model.py            # CharLSTM 语言模型
├── federated/
│   ├── client.py                # Flower NumPyClient 实现
│   └── server.py                # FedAvg 聚合与训练循环
├── utils/
│   ├── preprocessing.py         # 字符词汇表与序列生成
│   └── metrics.py               # 度量计算与绘图
├── experiments/
│   └── train.py                 # 主入口
├── requirements.txt
└── .gitignore
```

## 快速开始

### 环境要求

- Python 3.10+
- （可选）CUDA / Apple MPS 用于 GPU 加速

### 安装

```bash
# 克隆仓库
git clone https://github.com/<your-username>/federated-shakespeare.git
cd federated-shakespeare/federated_shakespeare

# 创建环境（二选一）
conda create -n fl_env python=3.10 && conda activate fl_env
# 或
python3 -m venv .venv && source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 运行

```bash
# 默认配置：25 轮，每轮 5 个客户端
python experiments/train.py
```

```bash
# 自定义配置
python experiments/train.py \
    --num-rounds 50 \
    --clients-per-round 10 \
    --local-epochs 2 \
    --hidden-dim 256 \
    --embed-dim 128 \
    --lr 0.0008
```

## 配置参数

所有超参数均可通过命令行调整：

| 类别 | 参数 | 默认值 | 说明 |
|:---|:---|:---:|:---|
| **联邦学习** | `--num-rounds` | `25` | 联邦通信轮数 |
| | `--clients-per-round` | `5` | 每轮采样的客户端数 |
| | `--local-epochs` | `1` | 每轮本地训练 epoch 数 |
| **模型** | `--embed-dim` | `64` | 字符嵌入维度 |
| | `--hidden-dim` | `128` | LSTM 隐藏维度 |
| | `--num-layers` | `2` | LSTM 层数 |
| | `--dropout` | `0.3` | Dropout 率 |
| **训练** | `--batch-size` | `16` | Mini-batch 大小 |
| | `--lr` | `0.001` | 本地学习率 (Adam) |
| | `--seq-length` | `50` | 输入序列长度 |
| **其他** | `--seed` | `42` | 随机种子 |
| | `--device` | `auto` | `cpu` / `cuda` / `mps` / `auto` |

运行 `python experiments/train.py --help` 查看完整参数列表。

## 模型架构

```
CharLSTM(
  Embedding(vocab_size, embed_dim)      # 字符索引 → 稠密向量
  LSTM(embed_dim, hidden_dim, layers=2) # 序列建模
  Dropout(0.3)                          # 正则化
  Linear(hidden_dim, vocab_size)        # 下一字符 logits
)
```

默认配置：**~244K 参数**。

## 算法

**联邦平均（FedAvg）** — [McMahan et al., 2017](https://arxiv.org/abs/1602.05629)

```
对于每轮 r = 1, 2, ..., R:
    S ← 随机采样 K 个客户端
    对于每个客户端 k ∈ S（并行）:
        w_k ← 本地训练(w_global, local_data_k)
    w_global ← Σ (n_k / n_total) · w_k
```

每个客户端使用 **Adam** 优化器和**梯度裁剪**（最大范数 = 5.0）来稳定 LSTM 训练。

## 输出文件

训练完成后，`results/` 目录包含：

| 文件 | 说明 |
|:---|:---|
| `metrics.json` | 每轮损失、困惑度、准确率 |
| `loss_vs_rounds.png` | 损失曲线 |
| `perplexity_vs_rounds.png` | 困惑度曲线 |
| `accuracy_vs_rounds.png` | 准确率曲线 |
| `combined_metrics.png` | 损失与困惑度并排图表 |
| `global_model.pt` | 最终模型检查点（含元数据） |

## 评估度量

| 度量 | 公式 | 说明 |
|:---|:---|:---|
| **交叉熵损失** | $-\frac{1}{N}\sum \log p(c_t)$ | 标准分类损失 |
| **困惑度** | $e^{\text{loss}}$ | 越低越好，衡量模型"惊讶程度" |
| **准确率** | 正确预测数 / 总预测数 | 字符级下一字符准确率 |

## 许可证

本项目基于 [MIT 许可证](LICENSE) 发布。
