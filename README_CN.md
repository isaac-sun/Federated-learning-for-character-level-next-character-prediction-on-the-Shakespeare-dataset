<div align="center">

# 🎭 FedBard

**基于 Shapley 值防御的鲁棒联邦学习 · 莎士比亚数据集**

PyTorch · Flower · FedAvg · SVRFL

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.6%2B-green.svg)](https://flower.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README.md)

</div>

---

## 概述

FedBard 是一个面向莎士比亚**字符级下一字符预测**的联邦学习研究平台，忠实复现了论文 **SVRFL**（基于模型无关 Shapley 值的鲁棒公平联邦学习）中的防御机制：

> *"Robust and Fair Federated Learning Based on Model-Agnostic Shapley Value"*

每个莎士比亚角色作为独立的联邦客户端。系统同时支持 **FedAvg 基线** 和完整的 **SVRFL 防御流水线** ——包括蒙特卡罗 Shapley 估计、搭便车检测、恶意更新缓解和信誉管理——并支持多种对抗攻击场景。

### 核心特性

| | 特性 | 说明 |
|:---|:---|:---|
| 🔁 | **FedAvg 基线** | 跨 500+ 客户端（LSTM）或 20 个选定客户端（GRU）的加权参数聚合 |
| 🛡️ | **SVRFL 防御** | 基于 Shapley 值的搭便车检测 + 二值效用过滤的投毒缓解 |
| ⚔️ | **攻击套件** | DFR、SDFR、AFR（搭便车）和 SF（符号翻转投毒），及并发攻击模式 |
| 🧠 | **双模型** | CharLSTM（基线）和 CharGRU（SVRFL，与论文一致） |
| 📦 | **自动数据集** | 自动下载 [TinyShakespeare](https://github.com/karpathy/char-rnn) 并按角色解析，带合成数据回退 |
| 🌸 | **Flower 兼容** | 实现 `NumPyClient` 接口 |
| 📈 | **丰富可视化** | 损失、困惑度、准确率、信誉轨迹、检测指标、效用分数 |
| ⚙️ | **完全可配置** | 所有超参数均可通过命令行调整 |

## 项目结构

```
federated_shakespeare/
├── attacks/
│   ├── __init__.py              # 攻击类型常量
│   ├── free_rider.py            # DFR、SDFR、AFR 攻击
│   └── poisoning.py             # SF（符号翻转）攻击
├── data/
│   └── shakespeare_loader.py    # 数据集下载、解析、客户端分区、
│                                # 服务器验证集构建
├── models/
│   ├── lstm_model.py            # CharLSTM（基线）
│   └── gru_model.py             # CharGRU（SVRFL，忠于论文）
├── federated/
│   ├── client.py                # Flower NumPyClient
│   ├── server.py                # FedAvg 聚合循环
│   └── svrfl_server.py          # SVRFL 实验运行器
├── utils/
│   ├── preprocessing.py         # 字符词汇表与序列生成
│   ├── metrics.py               # 度量计算与绘图
│   ├── shapley.py               # 蒙特卡罗 Shapley 值估计
│   └── svrfl.py                 # SVRFL 防御逻辑（检测 + 缓解）
├── experiments/
│   ├── train.py                 # FedAvg 基线入口
│   └── train_svrfl.py           # SVRFL 实验入口
├── requirements.txt
└── .gitignore
```

## 快速开始

### 环境要求

- Python 3.10+
- （可选）CUDA / Apple MPS 用于 GPU 加速

### 安装

```bash
git clone https://github.com/<your-username>/FedBard.git
cd FedBard/federated_shakespeare

# 创建环境（二选一）
conda create -n fedbard python=3.10 && conda activate fedbard
# 或
python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

### 运行 FedAvg 基线

```bash
python experiments/train.py                           # 默认：25 轮，每轮 5 个客户端，LSTM
python experiments/train.py --num-rounds 50 --clients-per-round 10
```

### 运行 SVRFL 防御实验

```bash
# SVRFL 防御 DFR 搭便车攻击
python experiments/train_svrfl.py --defense svrfl --attack dfr

# SVRFL 防御符号翻转投毒
python experiments/train_svrfl.py --defense svrfl --attack sf

# SVRFL 防御并发攻击（3 个搭便车者 + 2 个投毒者）
python experiments/train_svrfl.py --defense svrfl --attack concurrent

# 无防御的 FedAvg 受攻击（对比基线）
python experiments/train_svrfl.py --defense fedavg --attack dfr

# 无攻击基线
python experiments/train_svrfl.py --defense fedavg --attack none
```

## SVRFL 防御机制

SVRFL 流水线在每轮联邦训练中执行两个防御分支：

### A. 搭便车检测

1. 通过蒙特卡罗排列估计每个客户端的 **Shapley 值**
2. 计算**特征值**：$d_i = |\text{sv}_i| \;/\; (L_{\cos,i}^2 + \varepsilon)$，其中 $L_{\cos,i} = 1 - \cos(w_i, w_g)$
3. 对特征值运行 **KMeans(k=2)**；若质心比超过阈值 $h$ 则标记高质心聚类
4. 惩罚检测到的搭便车者：降低信誉，将其 Shapley 值置零

### B. 恶意更新缓解

1. 维护**指数移动平均效用**分数：$u_i = \alpha \cdot u_i + (1-\alpha) \cdot \text{sv}_i$
2. **二值过滤**：仅聚合 $u_i > 0$ 的客户端的更新
3. 基于聚合参与情况更新**信誉**

### 已实现的攻击

| 攻击 | 类型 | 说明 |
|:---|:---|:---|
| **DFR** | 搭便车 | 衰减高斯噪声：$g = \sigma \cdot t^{-\gamma} \cdot \varepsilon$ |
| **SDFR** | 搭便车 | 模拟先前全局更新方向和规模 |
| **AFR** | 搭便车 | SDFR + 在随机 10% 坐标上添加高斯噪声 |
| **SF** | 投毒 | 符号翻转：$g_{\text{attack}} = -g_{\text{honest}}$ |
| **并发** | 混合 | 1 DFR + 1 SDFR + 1 AFR + 2 SF |

## 配置参数

### FedAvg 基线 (`train.py`)

| 类别 | 参数 | 默认值 | 说明 |
|:---|:---|:---:|:---|
| **联邦学习** | `--num-rounds` | `25` | 通信轮数 |
| | `--clients-per-round` | `5` | 每轮采样客户端数 |
| | `--local-epochs` | `1` | 本地训练 epoch 数 |
| **模型** | `--embed-dim` | `64` | 嵌入维度 |
| | `--hidden-dim` | `128` | LSTM 隐藏维度 |
| | `--num-layers` | `2` | LSTM 层数 |
| **训练** | `--batch-size` | `16` | 批次大小 |
| | `--lr` | `0.001` | 学习率 (Adam) |
| | `--seq-length` | `50` | 序列长度 |

### SVRFL 实验 (`train_svrfl.py`)

包含上述所有基线参数，额外支持：

| 类别 | 参数 | 默认值 | 说明 |
|:---|:---|:---:|:---|
| **实验** | `--defense` | `svrfl` | `fedavg` 或 `svrfl` |
| | `--attack` | `none` | `none` / `dfr` / `sdfr` / `afr` / `sf` / `concurrent` |
| **SVRFL** | `--shapley-mc-samples` | `50` | Shapley 估计的蒙特卡罗排列数 |
| | `--alpha` | `0.9` | 效用分数的 EMA 平滑因子 |
| | `--threshold-h` | `200` | 搭便车检测 KMeans 阈值 |
| | `--eta-server` | `1.0` | 服务器端学习率 |
| **规模** | `--num-clients` | `20` | 总客户端池大小 |
| | `--clients-per-round` | `10` | 每轮采样客户端数 |
| | `--val-samples` | `1000` | 服务器端验证集大小 |

运行 `python experiments/train_svrfl.py --help` 查看完整参数列表。

## 模型架构

```
CharLSTM（基线）                        CharGRU（SVRFL，忠于论文）
─────────────────                      ────────────────────────────────
Embedding(67, 64)                      Embedding(67, 64)
LSTM(64, 128, layers=2)                GRU(64, 128, layers=2)
Dropout(0.3)                           Dropout(0.3)
Linear(128, 67)                        Linear(128, 67)
~244K 参数                              ~186K 参数
```

## 输出文件

### FedAvg 基线 → `results/`

| 文件 | 说明 |
|:---|:---|
| `metrics.json` | 每轮损失、困惑度、准确率 |
| `loss_vs_rounds.png` | 损失曲线 |
| `perplexity_vs_rounds.png` | 困惑度曲线 |
| `accuracy_vs_rounds.png` | 准确率曲线 |
| `combined_metrics.png` | 损失与困惑度并排图表 |
| `global_model.pt` | 最终模型检查点 |

### SVRFL 实验 → `results/<defense>_<attack>_r<rounds>/`

包含上述所有文件，额外生成：

| 文件 | 说明 |
|:---|:---|
| `round_logs.json` | 每轮 Shapley 值、信誉、检测结果、效用分数 |
| `config.json` | 完整实验配置 |
| `reputation_trajectories.png` | 良性 vs 恶意客户端信誉轨迹 |
| `freerider_detection.png` | 检测精度和 FRDR |
| `utility_trajectories.png` | 客户端效用分数演化 |

## 算法

### FedAvg — [McMahan et al., 2017](https://arxiv.org/abs/1602.05629)

```
对于每轮 r = 1 .. R:
    S ← 采样 K 个客户端
    对于客户端 k ∈ S:
        w_k ← 本地训练(w_global, data_k)
    w_global ← Σ (n_k / n) · w_k
```

### SVRFL — [论文](https://doi.org/...)

```
对于每轮 r = 1 .. R:
    I_r ← 采样 K 个客户端（信誉 ≥ 0）
    g_i = w_global - w_i，对每个 i ∈ I_r          # 客户端更新
    sv_i ← 蒙特卡罗 Shapley(g, D_val)             # Shapley 估计
    d_i = |sv_i| / (L_cos_i² + ε)                 # 特征值
    FR ← KMeans(d_i, k=2) 若比值 > h              # 搭便车检测
    u_i = α·u_i + (1-α)·sv_i                      # 效用更新
    w_global -= η · mean{g_i : u_i > 0}           # 二值过滤聚合
```

## 评估度量

| 度量 | 公式 | 说明 |
|:---|:---|:---|
| **交叉熵损失** | $-\frac{1}{N}\sum \log p(c_t)$ | 标准 NLL 损失 |
| **困惑度** | $e^{\text{loss}}$ | 越低越好 |
| **准确率** | 正确数 / 总数 | 字符级预测 |
| **FRDR** | TP / (TP + FN) | 搭便车检测率 |
| **精度** | TP / (TP + FP) | 检测精度 |

## 参考文献

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017
- *Robust and Fair Federated Learning Based on Model-Agnostic Shapley Value*

## 许可证

本项目基于 [MIT 许可证](LICENSE) 发布。
