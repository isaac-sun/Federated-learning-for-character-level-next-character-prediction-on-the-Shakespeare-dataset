"""
Metrics computation, logging, and visualization utilities for federated learning.
联邦学习的度量计算、日志记录和可视化工具。

Provides:
  - Cross-entropy loss computation
  - Perplexity calculation (exp(loss))
  - Character-level accuracy
  - Metrics logging to JSON
  - Training curve plotting

提供：
  - 交叉熵损失计算
  - 困惑度计算 (exp(loss))
  - 字符级准确率
  - 度量记录到JSON
  - 训练曲线绘制
"""

import json
import math
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments / 服务器环境的非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def compute_loss(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Compute average cross-entropy loss over a dataset.
    计算数据集上的平均交叉熵损失。

    Args:
        model: The language model to evaluate.
               要评估的语言模型。
        data_loader: DataLoader yielding (input, target) batches.
                     产出(输入, 目标)批次的DataLoader。
        device: Torch compute device (cpu/cuda/mps).
                Torch计算设备（cpu/cuda/mps）。

    Returns:
        Average cross-entropy loss as a float.
        平均交叉熵损失（浮点数）。
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            # Reshape: (batch*seq_len, vocab_size) vs (batch*seq_len,)
            # 重塑：(batch*seq_len, vocab_size) 对 (batch*seq_len,)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            batch_tokens = targets.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    return total_loss / max(total_tokens, 1)


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.
    从交叉熵损失计算困惑度。

    Perplexity = exp(loss). Lower perplexity indicates better prediction.
    困惑度 = exp(损失)。困惑度越低表示预测越好。

    Args:
        loss: Cross-entropy loss value. 交叉熵损失值。

    Returns:
        Perplexity value. 困惑度值。
    """
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def compute_accuracy(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Compute character-level prediction accuracy.
    计算字符级预测准确率。

    For each position in the sequence, checks if the predicted character
    (argmax of logits) matches the target character.
    对序列中的每个位置，检查预测字符（logits的argmax）是否与目标字符匹配。

    Args:
        model: The language model. 语言模型。
        data_loader: DataLoader yielding (input, target) batches. 数据加载器。
        device: Torch compute device. Torch计算设备。

    Returns:
        Accuracy as a float in [0, 1]. 准确率，范围[0, 1]。
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            predictions = logits.argmax(dim=-1)
            correct += (predictions == targets).sum().item()
            total += targets.numel()

    return correct / max(total, 1)


class MetricsLogger:
    """
    Logger for tracking federated learning metrics across rounds.
    用于跟踪联邦学习各轮次度量的日志记录器。

    Stores per-round loss, perplexity, and accuracy, and supports
    serialization to JSON and loading from file.
    存储每轮的损失、困惑度和准确率，支持序列化为JSON和从文件加载。
    """

    def __init__(self):
        """
        Initialize empty metrics storage.
        初始化空的度量存储。
        """
        self.history: Dict[str, List[float]] = {
            "round": [],
            "loss": [],
            "perplexity": [],
            "accuracy": [],
        }

    def log(
        self,
        round_num: int,
        loss: float,
        perplexity: float,
        accuracy: float = 0.0,
    ) -> None:
        """
        Log metrics for a single federated round.
        记录单轮联邦训练的度量。

        Args:
            round_num: Current round number. 当前轮次编号。
            loss: Average cross-entropy loss. 平均交叉熵损失。
            perplexity: Perplexity value. 困惑度。
            accuracy: Character-level accuracy. 字符级准确率。
        """
        self.history["round"].append(round_num)
        self.history["loss"].append(loss)
        self.history["perplexity"].append(perplexity)
        self.history["accuracy"].append(accuracy)

    def save(self, filepath: str) -> None:
        """
        Save metrics history to a JSON file.
        将度量历史保存到JSON文件。

        Args:
            filepath: Path to the output JSON file.
                      输出JSON文件的路径。
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=2)

    def load(self, filepath: str) -> None:
        """
        Load metrics history from a JSON file.
        从JSON文件加载度量历史。

        Args:
            filepath: Path to the JSON file. JSON文件的路径。
        """
        with open(filepath, "r") as f:
            self.history = json.load(f)

    def get_latest(self) -> Dict[str, float]:
        """
        Get the most recent round's metrics.
        获取最近一轮的度量。

        Returns:
            Dict with the latest values for each metric.
            包含每个度量最新值的字典。
        """
        if not self.history["round"]:
            return {}
        return {k: v[-1] for k, v in self.history.items()}


def plot_metrics(
    metrics: MetricsLogger,
    save_dir: str = "results",
    show: bool = False,
) -> None:
    """
    Plot training curves: loss and perplexity vs. federated rounds.
    绘制训练曲线：损失和困惑度随联邦轮次的变化。

    Generates three plots:
      1. Loss vs. Rounds (individual)
      2. Perplexity vs. Rounds (individual)
      3. Combined side-by-side plot

    生成三张图表：
      1. 损失随轮次变化（单独）
      2. 困惑度随轮次变化（单独）
      3. 组合并排图表

    Args:
        metrics: MetricsLogger with recorded history.
                 记录了历史数据的MetricsLogger。
        save_dir: Directory to save plot images. 保存图片的目录。
        show: Whether to display plots interactively. 是否交互式显示图表。
    """
    os.makedirs(save_dir, exist_ok=True)
    rounds = metrics.history["round"]

    if not rounds:
        print("No metrics to plot. / 没有可绘制的度量数据。")
        return

    # ---- Plot 1: Loss vs Rounds ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, metrics.history["loss"], "b-o", markersize=4, label="Loss")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Federated Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = os.path.join(save_dir, "loss_vs_rounds.png")
    fig.savefig(loss_path, dpi=150)
    print(f"  Loss plot saved to {loss_path}")
    if show:
        plt.show()
    plt.close(fig)

    # ---- Plot 2: Perplexity vs Rounds ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        rounds, metrics.history["perplexity"],
        "r-s", markersize=4, label="Perplexity",
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Perplexity")
    ax.set_title("Federated Training Perplexity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    ppl_path = os.path.join(save_dir, "perplexity_vs_rounds.png")
    fig.savefig(ppl_path, dpi=150)
    print(f"  Perplexity plot saved to {ppl_path}")
    if show:
        plt.show()
    plt.close(fig)

    # ---- Plot 3: Combined side-by-side ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(rounds, metrics.history["loss"], "b-o", markersize=4)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Loss")
    ax1.set_title("Cross-Entropy Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(rounds, metrics.history["perplexity"], "r-s", markersize=4)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Perplexity")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Federated Learning Metrics",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    combined_path = os.path.join(save_dir, "combined_metrics.png")
    fig.savefig(combined_path, dpi=150)
    print(f"  Combined plot saved to {combined_path}")
    if show:
        plt.show()
    plt.close(fig)

    # ---- Plot 4: Accuracy vs Rounds ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        rounds, metrics.history["accuracy"],
        "g-^", markersize=4, label="Accuracy",
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.set_title("Federated Training Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    acc_path = os.path.join(save_dir, "accuracy_vs_rounds.png")
    fig.savefig(acc_path, dpi=150)
    print(f"  Accuracy plot saved to {acc_path}")
    if show:
        plt.show()
    plt.close(fig)
