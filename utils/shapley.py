"""
Monte Carlo Shapley value estimation for federated learning.
联邦学习中基于蒙特卡罗的Shapley值估计。

Implements the permutation-based Shapley value estimator used in SVRFL:
  - For each random permutation of selected clients, incrementally build
    coalitions and measure each client's marginal contribution.
  - Value function: v(S, D_v) = F(w_g, D_v) - F(w_S, D_v)
    where F is cross-entropy loss on server validation set D_v.
  - Positive Shapley value means the client helped reduce loss.

实现SVRFL中使用的基于排列的Shapley值估计器：
  - 对于选定客户端的每个随机排列，递增构建联盟并测量每个客户端的边际贡献。
  - 价值函数：v(S, D_v) = F(w_g, D_v) - F(w_S, D_v)
  - 正的Shapley值表示客户端帮助降低了损失。
"""

import logging
import random
from collections import OrderedDict
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def set_model_params_fast(model: nn.Module, params_list: List[np.ndarray]) -> None:
    """
    Load numpy parameter arrays into a PyTorch model's state_dict.
    将numpy参数数组加载到PyTorch模型的state_dict中。
    """
    keys = list(model.state_dict().keys())
    state_dict = OrderedDict()
    for k, v in zip(keys, params_list):
        state_dict[k] = torch.tensor(np.copy(v))
    model.load_state_dict(state_dict, strict=True)


@torch.no_grad()
def evaluate_model_loss(
    model: nn.Module,
    params_list: List[np.ndarray],
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Load parameters into model and compute cross-entropy loss on validation data.
    将参数加载到模型中并在验证数据上计算交叉熵损失。

    Args:
        model: PyTorch model instance (will be mutated). 模型实例（会被修改）。
        params_list: Parameters to evaluate. 要评估的参数。
        val_loader: Validation data loader. 验证数据加载器。
        device: Torch device. 计算设备。

    Returns:
        Average cross-entropy loss on validation set.
        验证集上的平均交叉熵损失。
    """
    set_model_params_fast(model, params_list)
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0

    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits, _ = model(inputs)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()

    return total_loss / max(total_tokens, 1)


def estimate_shapley_monte_carlo(
    global_params: List[np.ndarray],
    updates: Dict[str, List[np.ndarray]],
    selected_ids: List[str],
    val_loader: torch.utils.data.DataLoader,
    model_fn: Callable[[], nn.Module],
    device: torch.device,
    eta_server: float = 1.0,
    num_samples: int = 50,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Estimate per-client Shapley values using Monte Carlo permutations.
    使用蒙特卡罗排列估计每个客户端的Shapley值。

    For R random permutations of the k selected clients, incrementally
    build coalitions and compute marginal contributions:
      SV_i = (1/R) * sum_perms [ v(S_j ∪ {i}) - v(S_j) ]

    Value function: v(S) = F(w_g, D_v) - F(w_S, D_v)
    价值函数：v(S) = F(w_g, D_v) - F(w_S, D_v)
    where: w_S = w_g - (eta_server / |S|) * sum_{i in S} g_i
    其中：w_S = w_g - (eta_server / |S|) * sum_{i in S} g_i

    Positive SV means the client helped reduce validation loss.
    正SV表示客户端帮助降低了验证损失。

    Args:
        global_params: Current global model parameters w_g^t.
                       当前全局模型参数。
        updates: Dict mapping cid → update g_i = w_g - w_i.
                 客户端ID → 更新的映射。
        selected_ids: Client IDs selected this round.
                      本轮选定的客户端ID。
        val_loader: Server-side validation data loader.
                    服务器端验证数据加载器。
        model_fn: Factory that creates a fresh model instance.
                  创建新模型实例的工厂函数。
        device: Torch compute device. 计算设备。
        eta_server: Server learning rate (default 1.0). 服务器学习率。
        num_samples: Number of Monte Carlo permutations R (default 50).
                     蒙特卡罗排列数R。
        seed: Random seed for reproducibility. 随机种子。

    Returns:
        Dict mapping cid → estimated Shapley value.
        客户端ID → 估计的Shapley值的映射。
    """
    rng = random.Random(seed)
    k = len(selected_ids)

    if k == 0:
        return {}

    marginal_sums = {cid: 0.0 for cid in selected_ids}

    # Create a single model instance reused for all evaluations
    # 创建单个模型实例用于所有评估
    model = model_fn()
    model.to(device)
    model.eval()

    # Pre-compute global model validation loss: F(w_g, D_v)
    # 预计算全局模型验证损失
    global_loss = evaluate_model_loss(model, global_params, val_loader, device)

    for sample_idx in range(num_samples):
        perm = list(selected_ids)
        rng.shuffle(perm)

        # Incrementally build coalition and accumulate updates
        # 递增构建联盟并累积更新
        sum_updates = [np.zeros_like(p, dtype=np.float64) for p in global_params]
        prev_value = 0.0  # v(∅) = 0

        for j, cid in enumerate(perm):
            # Add this client's update to running sum
            # 将该客户端的更新添加到累积和
            for i in range(len(sum_updates)):
                sum_updates[i] += updates[cid][i].astype(np.float64)

            # Coalition model: w_S = w_g - (eta / |S|) * sum_updates
            # 联盟模型：w_S = w_g - (eta / |S|) * sum_updates
            coalition_size = j + 1
            coalition_params = [
                (g.astype(np.float64) - (eta_server / coalition_size) * s).astype(
                    np.float32
                )
                for g, s in zip(global_params, sum_updates)
            ]

            # Evaluate: F(w_S, D_v)
            coalition_loss = evaluate_model_loss(
                model, coalition_params, val_loader, device
            )

            # v(S) = F(w_g) - F(w_S);  positive = coalition reduced loss
            curr_value = global_loss - coalition_loss

            # Marginal contribution of client cid
            # 客户端cid的边际贡献
            marginal = curr_value - prev_value
            marginal_sums[cid] += marginal

            prev_value = curr_value

    # Average over all permutations / 对所有排列取平均
    shapley_values = {
        cid: marginal_sums[cid] / num_samples for cid in selected_ids
    }

    return shapley_values
