"""
Federated learning server with FedAvg aggregation strategy.
使用FedAvg聚合策略的联邦学习服务器。

Provides:
  - FedAvg parameter aggregation (weighted by client sample counts)
  - Standalone federated simulation loop (no Ray dependency)
  - Model parameter get/set utilities
  - Flower-compatible strategy creation

提供：
  - FedAvg参数聚合（按客户端样本数加权）
  - 独立联邦模拟循环（无Ray依赖）
  - 模型参数获取/设置工具
  - Flower兼容的策略创建
"""

import logging
import math
import random
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def fedavg_aggregate(
    results: List[Tuple[List[np.ndarray], int]],
) -> List[np.ndarray]:
    """
    Federated Averaging: compute weighted average of model parameters.
    联邦平均：计算模型参数的加权平均。

    Each client's parameters are weighted by the number of training
    samples it used, following the original FedAvg algorithm by
    McMahan et al. (2017).
    每个客户端的参数按其使用的训练样本数加权，
    遵循McMahan等人(2017)的原始FedAvg算法。

    Args:
        results: List of (parameters, num_samples) tuples from each client.
                 每个客户端的(参数, 样本数)元组列表。

    Returns:
        Aggregated parameters as a list of numpy arrays.
        聚合后的参数，作为numpy数组列表。
    """
    total_samples = sum(num for _, num in results)
    if total_samples == 0:
        logger.warning("Total samples is 0; returning first client's parameters")
        return results[0][0]

    # Initialize aggregated parameters with zeros
    # 用零初始化聚合参数
    aggregated = [np.zeros_like(p) for p in results[0][0]]

    # Weighted sum / 加权求和
    for params, num_samples in results:
        weight = num_samples / total_samples
        for i, p in enumerate(params):
            aggregated[i] += p.astype(np.float64) * weight

    # Cast back to float32 / 转换回float32
    aggregated = [p.astype(np.float32) for p in aggregated]

    return aggregated


def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """
    Extract model parameters as a list of numpy arrays.
    将模型参数提取为numpy数组列表。

    Args:
        model: PyTorch model. PyTorch模型。

    Returns:
        List of numpy arrays, one per parameter tensor in the state_dict.
        numpy数组列表，state_dict中每个参数张量一个。
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Load parameters from numpy arrays into a PyTorch model.
    将numpy数组中的参数加载到PyTorch模型中。

    Args:
        model: Target PyTorch model. 目标PyTorch模型。
        parameters: List of numpy arrays matching the state_dict order.
                    与state_dict顺序匹配的numpy数组列表。
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(np.copy(v)) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)


def run_federated_simulation(
    client_fn: Callable,
    client_ids: List[str],
    global_model: nn.Module,
    num_rounds: int = 25,
    clients_per_round: int = 5,
    seed: int = 42,
) -> Tuple[List[np.ndarray], Dict[str, List]]:
    """
    Run a federated learning simulation with FedAvg aggregation.
    运行使用FedAvg聚合的联邦学习模拟。

    This is a standalone simulation loop that orchestrates:
      1. Client sampling (random subset each round)
      2. Local training (each client trains on its data)
      3. FedAvg aggregation (weighted parameter averaging)
      4. Distributed evaluation (each client evaluates locally)
      5. Metrics logging (loss, perplexity, accuracy per round)

    这是一个独立的模拟循环，协调以下步骤：
      1. 客户端采样（每轮随机子集）
      2. 本地训练（每个客户端在其数据上训练）
      3. FedAvg聚合（加权参数平均）
      4. 分布式评估（每个客户端本地评估）
      5. 度量记录（每轮的损失、困惑度、准确率）

    Args:
        client_fn: Factory function: client_id → ShakespeareClient instance.
                   工厂函数：client_id → ShakespeareClient实例。
        client_ids: List of all available client identifiers.
                    所有可用客户端标识符的列表。
        global_model: Initialized global model (used for initial parameters).
                      初始化的全局模型（用于获取初始参数）。
        num_rounds: Total number of federated communication rounds (default: 25).
                    联邦通信总轮数（默认：25）。
        clients_per_round: Number of clients sampled per round (default: 5).
                           每轮采样的客户端数（默认：5）。
        seed: Random seed for reproducible client sampling (default: 42).
              用于可重复客户端采样的随机种子（默认：42）。

    Returns:
        Tuple of:
          - final_parameters: Aggregated model parameters after all rounds
          - history: Dict with keys "round", "loss", "perplexity", "accuracy"
        返回元组：
          - final_parameters：所有轮次后的聚合模型参数
          - history：包含键"round"、"loss"、"perplexity"、"accuracy"的字典
    """
    rng = random.Random(seed)

    # Initialize global parameters from the model
    # 从模型初始化全局参数
    global_params = get_model_parameters(global_model)

    # Metrics history / 度量历史
    history: Dict[str, List] = {
        "round": [],
        "loss": [],
        "perplexity": [],
        "accuracy": [],
    }

    # Print simulation configuration / 打印模拟配置
    print("=" * 60)
    print("  🌐 Federated Training (FedAvg) / 联邦训练 (FedAvg)")
    print("=" * 60)
    print(f"  Total clients / 总客户端数:        {len(client_ids)}")
    print(f"  Rounds / 轮次:                     {num_rounds}")
    print(f"  Clients per round / 每轮客户端数:  {clients_per_round}")
    print("=" * 60)

    for round_num in range(1, num_rounds + 1):
        # ---- Step 1: Client Sampling / 步骤1：客户端采样 ----
        k = min(clients_per_round, len(client_ids))
        selected_ids = rng.sample(client_ids, k)

        # ---- Step 2: Local Training / 步骤2：本地训练 ----
        fit_results = []
        for cid in selected_ids:
            client = client_fn(cid)
            updated_params, num_samples, train_metrics = client.fit(
                global_params, config={}
            )
            fit_results.append((updated_params, num_samples))

        # ---- Step 3: FedAvg Aggregation / 步骤3：FedAvg聚合 ----
        global_params = fedavg_aggregate(fit_results)

        # ---- Step 4: Distributed Evaluation / 步骤4：分布式评估 ----
        eval_losses = []
        eval_accuracies = []
        eval_total_samples = 0

        for cid in selected_ids:
            client = client_fn(cid)
            loss, num_samples, eval_metrics = client.evaluate(
                global_params, config={}
            )
            eval_losses.append((loss, num_samples))
            eval_accuracies.append(
                (eval_metrics.get("accuracy", 0.0), num_samples)
            )
            eval_total_samples += num_samples

        # Weighted average of evaluation metrics
        # 评估度量的加权平均
        if eval_total_samples > 0:
            avg_loss = sum(l * n for l, n in eval_losses) / eval_total_samples
            avg_accuracy = (
                sum(a * n for a, n in eval_accuracies) / eval_total_samples
            )
        else:
            avg_loss = float("inf")
            avg_accuracy = 0.0

        # Compute perplexity with overflow protection
        # 计算困惑度（带溢出保护）
        clamped_loss = min(avg_loss, 50.0)
        avg_perplexity = math.exp(clamped_loss)

        # ---- Step 5: Record Metrics / 步骤5：记录度量 ----
        history["round"].append(round_num)
        history["loss"].append(avg_loss)
        history["perplexity"].append(avg_perplexity)
        history["accuracy"].append(avg_accuracy)

        # Print round summary / 打印轮次总结
        client_names = [c[:15] for c in selected_ids]
        print(
            f"  Round {round_num:3d}/{num_rounds} │ "
            f"Loss: {avg_loss:.4f} │ "
            f"PPL: {avg_perplexity:8.2f} │ "
            f"Acc: {avg_accuracy:.4f} │ "
            f"Clients: {client_names}"
        )

    print("=" * 60)
    print("  ✅ Training complete / 训练完成")
    print("=" * 60)

    return global_params, history
