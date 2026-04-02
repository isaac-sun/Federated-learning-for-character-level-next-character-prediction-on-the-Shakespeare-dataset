<div align="center">

# 🎭 FedBard

**基于 Shapley 值防御的鲁棒联邦学习 · 莎士比亚数据集**

PyTorch · Flower · FedAvg · SVRFL

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.6%2B-green.svg)](https://flower.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey.svg)]()

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
| 🖥️ | **跨平台 GPU** | 自动检测 CUDA（NVIDIA）· MPS（Apple Silicon）· CPU；支持 `cuda:N` 多卡选择 |
| 🌸 | **Flower 兼容** | 实现 `NumPyClient` 接口 |
| 📈 | **丰富可视化** | 损失、困惑度、准确率、信誉轨迹、检测指标、效用分数 |
| ⚙️ | **完全可配置** | 所有超参数均可通过命令行调整 |

---

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
│   ├── svrfl.py                 # SVRFL 防御逻辑（检测 + 缓解）
│   └── device.py                # 跨平台设备解析与 GPU 信息
├── experiments/
│   ├── train.py                 # FedAvg 基线入口
│   ├── train_svrfl.py           # SVRFL 实验入口
│   └── run_all.py               # 一键运行全部实验套件
├── requirements.txt
└── .gitignore
```

---

## 环境配置

> **快速上手：** 使用默认的 `--device auto`，FedBard 会自动选择最优设备。

### 系统要求

| 组件 | 最低要求 | 推荐配置 |
|:---|:---|:---|
| Python | 3.10 | 3.11 |
| 内存 | 8 GB | 16 GB |
| 存储 | 500 MB | 1 GB |
| GPU | — | NVIDIA（CUDA 12+）或 Apple Silicon |

---

### 🍎 macOS — Apple Silicon（M1/M2/M3/M4）

Apple Silicon Mac 使用 **MPS**（Metal Performance Shaders）进行 GPU 加速。

```bash
# 1. 创建环境
conda create -n fedbard python=3.10
conda activate fedbard

# 2. 安装 PyTorch（macOS 标准版已内置 MPS 支持）
pip install torch torchvision

# 3. 安装其余依赖
pip install -r requirements.txt

# 4. 运行（自动选择 MPS）
python experiments/train_svrfl.py --defense svrfl --attack dfr
# 或显式指定 MPS：
python experiments/train_svrfl.py --device mps --defense svrfl --attack dfr
```

> **注意：** MPS 支持需要 macOS 12.3+。验证方式：`python -c "import torch; print(torch.backends.mps.is_available())"`

---

### 🍎 macOS — Intel Mac

Intel Mac 没有 MPS，设备自动回退到 CPU。

```bash
conda create -n fedbard python=3.10 && conda activate fedbard
pip install torch torchvision
pip install -r requirements.txt

python experiments/train_svrfl.py --device cpu --defense svrfl --attack dfr
```

---

### 🪟 Windows — NVIDIA GPU（推荐）

Windows 上需要安装**支持 CUDA 的 PyTorch 版本**。在 Windows 上直接 `pip install torch` 安装的是仅 CPU 版本。

**第一步 — 查看 CUDA 版本：**
```powershell
nvidia-smi
# 查看右上角 "CUDA Version: XX.X"
```

**第二步 — 安装支持 CUDA 的 PyTorch：**

| CUDA 版本 | 安装命令 |
|:---:|:---|
| 12.4 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| 12.1 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| 11.8 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |

> 完整 wheel 列表请访问 [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)。

**第三步 — 安装其余依赖：**
```powershell
conda create -n fedbard python=3.10
conda activate fedbard
# （先按上方安装 CUDA 版 PyTorch）
pip install -r requirements.txt
```

**第四步 — 运行：**
```powershell
# 自动检测 NVIDIA GPU
python experiments/train_svrfl.py --defense svrfl --attack dfr

# 显式使用 GPU 0
python experiments/train_svrfl.py --device cuda:0 --defense svrfl --attack dfr

# 多卡环境下指定特定显卡
python experiments/train_svrfl.py --device cuda:1 --defense svrfl --attack sf
```

**验证 CUDA 是否可用：**
```python
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 预期输出：True  NVIDIA GeForce RTX 5070 Ti
```

---

### 🐧 Linux — NVIDIA GPU

```bash
conda create -n fedbard python=3.10 && conda activate fedbard

# 安装 CUDA 版 PyTorch（根据驱动版本调整 cu124）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python experiments/train_svrfl.py --defense svrfl --attack dfr
```

---

### 设备选择参考

所有入口脚本（`train.py`、`train_svrfl.py`、`run_all.py`）均支持 `--device` 参数。

| 值 | 硬件 | 说明 |
|:---:|:---|:---|
| `auto` | 最优可用设备 | **默认值。** CUDA → MPS → CPU |
| `cuda` | NVIDIA GPU 0 | 需要 CUDA 版 PyTorch |
| `cuda:0` | NVIDIA GPU 0 | 显式指定（与 `cuda` 等效） |
| `cuda:1` | NVIDIA GPU 1 | 多卡环境下指定特定显卡 |
| `mps` | Apple Silicon GPU | 仅 macOS 12.3+ |
| `cpu` | 仅 CPU | 始终可用 |

启动时 FedBard 会打印设备信息块：
```
  🖥️  Device / 计算设备
     Type    : cuda
     GPU     : NVIDIA GeForce RTX 5070 Ti
     VRAM    : 16.0 GB
     CUDA    : 12.4
```

---

## 快速开始

### ▶ 推荐：一键运行全部实验

一条命令运行全部 12 种实验组合（两种防御 × 全部攻击）：

```bash
cd federated_shakespeare

# 完整套件——保存所有结果 + summary.csv（GPU 约需 6–12 小时）
python experiments/run_all.py

# 快速冒烟测试——减少轮次，适合验证环境配置
python experiments/run_all.py --quick

# 指定运行设备
python experiments/run_all.py --device cuda        # NVIDIA GPU
python experiments/run_all.py --device mps         # Apple Silicon
python experiments/run_all.py --device cpu         # 仅 CPU

# 只运行部分攻击
python experiments/run_all.py --attacks dfr sf concurrent
```

结果保存在 `results/` 目录下，包含对比所有配置的 `summary.csv`。

---

### 单独运行实验

#### FedAvg 基线

```bash
python experiments/train.py                                    # 25 轮，每轮 5 个客户端
python experiments/train.py --num-rounds 50 --clients-per-round 10
python experiments/train.py --device cuda --num-rounds 50
```

#### SVRFL 防御实验

```bash
# 无攻击基线
python experiments/train_svrfl.py --defense fedavg --attack none

# SVRFL 防御各类攻击
python experiments/train_svrfl.py --defense svrfl --attack dfr
python experiments/train_svrfl.py --defense svrfl --attack sdfr
python experiments/train_svrfl.py --defense svrfl --attack afr
python experiments/train_svrfl.py --defense svrfl --attack sf

# SVRFL 防御并发攻击（3 个搭便车者 + 2 个投毒者）
python experiments/train_svrfl.py --defense svrfl --attack concurrent

# 无防御的 FedAvg 受攻击（消融对比）
python experiments/train_svrfl.py --defense fedavg --attack sf
```

---

## SVRFL 防御机制

SVRFL 流水线在每轮联邦训练中执行两个防御分支：

### A. 搭便车检测

1. 通过蒙特卡罗排列在服务器验证集上估计每个客户端的 **Shapley 值**
2. 计算**特征值**：$d_i = |\text{sv}_i| \;/\; (L_{\cos,i}^2 + \varepsilon)$，其中 $L_{\cos,i} = 1 - \cos(w_i, w_g)$
3. 对特征值运行 **KMeans(k=2)**；若质心比超过阈值 $h$ 则标记高质心聚类
4. 惩罚检测到的搭便车者：降低信誉，将本轮 Shapley 值置零

### B. 恶意更新缓解

1. 维护**指数移动平均效用**分数：$u_i = \alpha \cdot u_i + (1-\alpha) \cdot \text{sv}_i$
2. **二值过滤**：仅聚合 $u_i > 0$ 的客户端的更新
3. 基于聚合参与情况更新**信誉**分数

### 已实现的攻击

| 攻击 | 类型 | 说明 |
|:---|:---|:---|
| **DFR** | 搭便车 | 衰减高斯噪声：$g = \sigma \cdot t^{-\gamma} \cdot \varepsilon$ |
| **SDFR** | 搭便车 | 模拟先前全局更新方向和规模 |
| **AFR** | 搭便车 | SDFR + 在随机 10% 坐标上添加高斯噪声 |
| **SF** | 投毒 | 符号翻转：$g_{\text{attack}} = -g_{\text{honest}}$ |
| **并发** | 混合 | 1 DFR + 1 SDFR + 1 AFR + 2 SF 同时攻击 |

---

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
| **系统** | `--device` | `auto` | `auto` / `cuda` / `cuda:N` / `mps` / `cpu` |

### SVRFL 实验 (`train_svrfl.py`)

包含上述所有基线参数，额外支持：

| 类别 | 参数 | 默认值 | 说明 |
|:---|:---|:---:|:---|
| **实验** | `--defense` | `svrfl` | `fedavg` 或 `svrfl` |
| | `--attack` | `none` | `none` / `dfr` / `sdfr` / `afr` / `sf` / `concurrent` |
| **SVRFL** | `--shapley-mc-samples` | `50` | Shapley 估计的蒙特卡罗排列数 |
| | `--alpha` | `0.9` | 效用分数的 EMA 平滑因子 |
| | `--threshold-h` | `200` | 搭便车检测 KMeans 质心比阈值 |
| | `--eta-server` | `1.0` | 服务器端聚合学习率 |
| **规模** | `--num-clients` | `20` | 总客户端池大小 |
| | `--clients-per-round` | `10` | 每轮采样客户端数 |
| | `--val-samples` | `1000` | 服务器端验证集大小 |
| **系统** | `--device` | `auto` | `auto` / `cuda` / `cuda:N` / `mps` / `cpu` |

运行 `python experiments/train_svrfl.py --help` 查看完整参数列表。

---

## 模型架构

```
CharLSTM（基线）                            CharGRU（SVRFL，忠于论文）
──────────────────────────────             ──────────────────────────────────
Embedding(vocab=67, dim=64)                Embedding(vocab=67, dim=64)
LSTM(input=64, hidden=128, layers=2)       GRU(input=64, hidden=128, layers=2)
Dropout(p=0.3)                             Dropout(p=0.3)
Linear(128 → 67)                           Linear(128 → 67)
≈ 244K 参数                                 ≈ 186K 参数
```

词汇表大小（67）由数据集自动确定。两个模型均在长度为 50 的字符级序列上运行。

---

## 输出文件

### FedAvg 基线 → `results/`

| 文件 | 说明 |
|:---|:---|
| `metrics.json` | 每轮损失、困惑度、准确率 |
| `loss_vs_rounds.png` | 训练损失曲线 |
| `perplexity_vs_rounds.png` | 困惑度曲线 |
| `accuracy_vs_rounds.png` | 字符级准确率曲线 |
| `combined_metrics.png` | 损失与困惑度并排图表 |
| `global_model.pt` | 最终全局模型检查点 |

### SVRFL 实验 → `results/<defense>_<attack>_r<rounds>/`

包含上述所有文件，额外生成：

| 文件 | 说明 |
|:---|:---|
| `round_logs.json` | 每轮 Shapley 值、信誉、检测结果、效用分数 |
| `config.json` | 完整实验配置快照 |
| `reputation_trajectories.png` | 良性 vs 恶意客户端信誉轨迹 |
| `freerider_detection.png` | 每轮检测精度和 FRDR |
| `utility_trajectories.png` | 客户端效用分数演化 |

---

## 算法

### FedAvg — [McMahan et al., 2017](https://arxiv.org/abs/1602.05629)

```
对于每轮 r = 1 .. R:
    S ← 采样 K 个客户端
    对于客户端 k ∈ S:
        w_k ← 本地训练(w_global, data_k)
    w_global ← Σ (n_k / n) · w_k
```

### SVRFL

```
对于每轮 r = 1 .. R:
    I_r ← 采样 K 个信誉 ≥ 0 的客户端
    g_i = w_global - w_i，对每个 i ∈ I_r           # 计算更新量
    sv_i ← 蒙特卡罗 Shapley(g, D_val)              # Shapley 估计
    d_i = |sv_i| / (L_cos_i² + ε)                  # 特征值
    FR ← KMeans(d_i, k=2) 若质心比 > h             # 搭便车检测
    u_i = α·u_i + (1-α)·sv_i                       # 效用 EMA 更新
    w_global -= η · mean{ g_i : u_i > 0 }          # 二值过滤聚合
    r_i ← 更新每个 i ∈ I_r 的信誉
```

---

## 评估指标

| 指标 | 公式 | 说明 |
|:---|:---|:---|
| **交叉熵损失** | $-\frac{1}{N}\sum \log p(c_t)$ | 在服务器验证集上评估 |
| **困惑度** | $e^{\text{loss}}$ | 越低越好 |
| **准确率** | 正确预测数 / 总数 | 字符级下一字符预测 |
| **FRDR** | TP / (TP + FN) | 搭便车检测召回率 |
| **精度** | TP / (TP + FP) | 搭便车检测精度 |

---

## 性能参考

实验运行时间因硬件差异较大。以下为**标准完整运行**（25 轮、每轮 10 个客户端、R=50 Shapley 采样）的基准数据：

| 硬件 | 每轮约耗时 | 完整 25 轮 |
|:---|:---:|:---:|
| NVIDIA RTX 4090 / 5070 Ti | 约 1–2 分钟 | 约 30–50 分钟 |
| Apple M2 Pro（MPS） | 约 3–4 分钟 | 约 80–100 分钟 |
| 仅 CPU（8 核） | 约 8–12 分钟 | 约 3–5 小时 |

> **提示：** 开发调试时可使用 `--shapley-mc-samples 20` 加速迭代；最终实验建议设置为 `50`–`100`。

---

## 参考文献

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017. [[arXiv]](https://arxiv.org/abs/1602.05629)
- *Robust and Fair Federated Learning Based on Model-Agnostic Shapley Value*

---

## 许可证

本项目基于 [MIT 许可证](LICENSE) 发布。
