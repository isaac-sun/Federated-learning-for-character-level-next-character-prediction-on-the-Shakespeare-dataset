"""
SVRFL defense logic: free-rider detection and poisonous update mitigation.
SVRFL防御逻辑：搭便车检测和恶意更新缓解。

Implements both defense branches from the paper:
  A) Free-rider detection: feature value d_i + KMeans clustering
  B) Poisonous update mitigation: exponential utility scores + binary filtering

实现论文中的两个防御分支：
  A) 搭便车检测：特征值d_i + KMeans聚类
  B) 恶意更新缓解：指数效用分数 + 二值过滤
"""

import logging
from typing import Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# A. Free-rider detection / 搭便车检测
# ============================================================


def compute_cosine_distance(
    local_params: List[np.ndarray],
    global_params: List[np.ndarray],
) -> float:
    """
    Compute cosine distance between local and global model parameters.
    计算本地模型和全局模型参数之间的余弦距离。

    L_cosine_i = 1 - cosine_similarity(w_i, w_g)

    Free riders have L_cosine ≈ 0 (their model barely changed).
    搭便车者的L_cosine ≈ 0（模型几乎没有变化）。

    Args:
        local_params: Local model parameters w_i. 本地模型参数。
        global_params: Global model parameters w_g. 全局模型参数。

    Returns:
        Cosine distance in [0, 2]. 余弦距离，范围[0, 2]。
    """
    flat_local = np.concatenate(
        [p.flatten().astype(np.float64) for p in local_params]
    )
    flat_global = np.concatenate(
        [p.flatten().astype(np.float64) for p in global_params]
    )

    dot = np.dot(flat_local, flat_global)
    norm_l = np.linalg.norm(flat_local)
    norm_g = np.linalg.norm(flat_global)

    if norm_l < 1e-12 or norm_g < 1e-12:
        return 1.0

    cos_sim = np.clip(dot / (norm_l * norm_g), -1.0, 1.0)
    return float(1.0 - cos_sim)


def compute_feature_values(
    shapley_values: Dict[str, float],
    cosine_distances: Dict[str, float],
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute feature values for free-rider detection.
    计算用于搭便车检测的特征值。

    d_i = |sv_i| / (L_cosine_i^2 + eps)

    Free riders have disproportionately high d_i because their update
    barely changes the model (low L_cosine) yet their |sv_i| is nonzero
    due to noise.
    搭便车者的d_i不成比例地高，因为他们的更新几乎不改变模型
    （低L_cosine），但由于噪声其|sv_i|不为零。

    Args:
        shapley_values: Per-client Shapley values. 每个客户端的Shapley值。
        cosine_distances: Per-client cosine distances. 每个客户端的余弦距离。
        eps: Stability constant (default 1e-8). 稳定性常数。

    Returns:
        Dict mapping cid → feature value d_i.
    """
    feature_values = {}
    for cid in shapley_values:
        sv_abs = abs(shapley_values[cid])
        L = cosine_distances.get(cid, 0.0)
        feature_values[cid] = sv_abs / (L ** 2 + eps)
    return feature_values


def kmeans_2_clusters(
    values: np.ndarray, max_iter: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-Means clustering with k=2 on 1D values (no sklearn dependency).
    对一维值进行k=2的K-Means聚类（无sklearn依赖）。

    Args:
        values: 1D array of feature values. 一维特征值数组。
        max_iter: Maximum iterations. 最大迭代次数。

    Returns:
        (labels, centroids): labels[i] ∈ {0,1}, centroids shape (2,).
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 2:
        return np.zeros(n, dtype=int), np.array([np.mean(values), np.mean(values)])

    # Initialize centroids to min and max
    c0, c1 = float(np.min(values)), float(np.max(values))
    if abs(c0 - c1) < 1e-12:
        return np.zeros(n, dtype=int), np.array([c0, c1])

    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        d0 = np.abs(values - c0)
        d1 = np.abs(values - c1)
        new_labels = (d1 < d0).astype(int)
        # label 0 → closer to c0, label 1 → closer to c1

        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        mask0 = labels == 0
        mask1 = labels == 1
        c0 = float(values[mask0].mean()) if mask0.any() else c0
        c1 = float(values[mask1].mean()) if mask1.any() else c1

    centroids = np.array([c0, c1])
    return labels, centroids


def detect_free_riders(
    feature_values: Dict[str, float],
    threshold_h: float = 200.0,
) -> Set[str]:
    """
    Detect free riders using KMeans(k=2) on feature values d_i.
    使用KMeans(k=2)在特征值d_i上检测搭便车者。

    Algorithm:
      1. Run KMeans(k=2) on d_i values
      2. Compute cluster centroids Pc1, Pc2
      3. If max(Pc1,Pc2) > h * min(Pc1,Pc2):
           flag clients in the cluster with the LARGER centroid
      4. Else: no free riders detected in this round

    算法：
      1. 对d_i值运行KMeans(k=2)
      2. 计算聚类质心Pc1, Pc2
      3. 如果max(Pc1,Pc2) > h * min(Pc1,Pc2)：
           标记具有较大质心的聚类中的客户端
      4. 否则：本轮未检测到搭便车者

    Args:
        feature_values: Per-client feature values d_i. 每个客户端的特征值。
        threshold_h: Detection threshold (default 200). 检测阈值。

    Returns:
        Set of detected free-rider client IDs. 检测到的搭便车者客户端ID集合。
    """
    if len(feature_values) < 2:
        return set()

    cids = sorted(feature_values.keys())
    vals = np.array([feature_values[cid] for cid in cids])

    labels, centroids = kmeans_2_clusters(vals)

    c_min = float(np.min(centroids))
    c_max = float(np.max(centroids))

    # Only flag if the separation is large enough
    # 仅在分离足够大时标记
    if c_min <= 0 or c_max <= threshold_h * c_min:
        return set()

    # Flag clients in the cluster with the larger centroid
    # 标记具有较大质心的聚类中的客户端
    larger_label = int(np.argmax(centroids))
    detected = {cids[i] for i in range(len(cids)) if labels[i] == larger_label}

    return detected


# ============================================================
# B. Poisonous update mitigation / 恶意更新缓解
# ============================================================


def update_utility_scores(
    utility_scores: Dict[str, float],
    shapley_values: Dict[str, float],
    alpha: float = 0.9,
) -> Dict[str, float]:
    """
    Update exponential moving average utility scores.
    更新指数移动平均效用分数。

    u_i = alpha * u_i + (1 - alpha) * sv_i^t

    Clients with persistently positive Shapley values accumulate
    positive utility; those with negative values trend toward zero.
    具有持续正Shapley值的客户端积累正效用。

    Args:
        utility_scores: Current utility scores (mutated in place).
                        当前效用分数（就地修改）。
        shapley_values: This round's Shapley values.
                        本轮的Shapley值。
        alpha: EMA smoothing factor (default 0.9). EMA平滑因子。

    Returns:
        Updated utility_scores dict. 更新后的效用分数字典。
    """
    for cid, sv in shapley_values.items():
        old = utility_scores.get(cid, 0.0)
        utility_scores[cid] = alpha * old + (1.0 - alpha) * sv
    return utility_scores


def svrfl_aggregate(
    global_params: List[np.ndarray],
    updates: Dict[str, List[np.ndarray]],
    selected_ids: List[str],
    utility_scores: Dict[str, float],
    eta_server: float = 1.0,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    SVRFL aggregation with binary positive-utility filtering.
    使用二值正效用过滤的SVRFL聚合。

    w_g^{t+1} = w_g - eta_server * [sum s(u_i)*g_i] / [sum s(u_i)]
    where s(x) = 1 if x > 0 else 0.

    This is NOT continuous Shapley-weighted FedAvg. It is binary
    selection: only updates from clients with positive utility
    are included in the aggregate, each weighted equally.
    这不是连续的Shapley加权FedAvg。这是二值选择：
    只有具有正效用的客户端的更新被包含在聚合中，每个权重相等。

    Fallback: if no client has positive utility, use standard FedAvg
    over all selected clients to avoid stalling.
    回退：如果没有客户端具有正效用，使用所有选定客户端的标准FedAvg。

    Args:
        global_params: Current global parameters w_g. 当前全局参数。
        updates: Dict cid → update g_i. 客户端更新映射。
        selected_ids: Selected client IDs this round. 本轮选定的客户端ID。
        utility_scores: Current utility scores. 当前效用分数。
        eta_server: Server learning rate (default 1.0). 服务器学习率。

    Returns:
        (new_global_params, aggregated_ids): new parameters and list
        of clients whose updates were actually used.
        (新全局参数, 聚合使用的客户端ID列表)。
    """
    positive_ids = [
        cid for cid in selected_ids if utility_scores.get(cid, 0.0) > 0
    ]

    if len(positive_ids) == 0:
        # Fallback: standard FedAvg over all selected clients
        # 回退：所有选定客户端的标准FedAvg
        logger.warning(
            "No client with positive utility; falling back to FedAvg. "
            "没有正效用的客户端；回退到FedAvg。"
        )
        aggregated_ids = list(selected_ids)
    else:
        aggregated_ids = positive_ids

    m = len(aggregated_ids)

    # Average update: (1/m) * sum g_i for aggregated clients
    # 平均更新：聚合客户端的(1/m) * sum g_i
    sum_updates = [np.zeros_like(p, dtype=np.float64) for p in global_params]
    for cid in aggregated_ids:
        for i, u in enumerate(updates[cid]):
            sum_updates[i] += u.astype(np.float64)

    avg_updates = [(s / m) for s in sum_updates]

    # w_g^{t+1} = w_g - eta_server * avg_update
    new_params = [
        (g.astype(np.float64) - eta_server * u).astype(np.float32)
        for g, u in zip(global_params, avg_updates)
    ]

    return new_params, aggregated_ids


def update_reputations_freerider(
    reputations: Dict[str, float],
    detected_ids: Set[str],
    round_num: int,
    n_total: int,
) -> Dict[str, float]:
    """
    Decrease reputation for detected free riders.
    降低检测到的搭便车者的信誉。

    r_i = r_i - t / n  for each detected free rider.
    对每个检测到的搭便车者：r_i = r_i - t / n。

    Args:
        reputations: Persistent reputation dict (mutated). 持久化信誉字典。
        detected_ids: IDs flagged as free riders. 被标记的搭便车者ID。
        round_num: Current round number t. 当前轮次t。
        n_total: Total number of clients n. 客户端总数n。

    Returns:
        Updated reputations. 更新后的信誉。
    """
    penalty = round_num / n_total
    for cid in detected_ids:
        reputations[cid] = reputations.get(cid, 0.0) - penalty
    return reputations


def update_reputations_aggregation(
    reputations: Dict[str, float],
    selected_ids: List[str],
    utility_scores: Dict[str, float],
) -> Dict[str, float]:
    """
    Update reputation based on aggregation participation.
    基于聚合参与更新信誉。

    r_i = r_i + s(u_i) / sum_j s(u_j)  for selected clients.
    对选定客户端：r_i = r_i + s(u_i) / sum_j s(u_j)。

    Clients with positive utility gain reputation.
    具有正效用的客户端获得信誉。

    Args:
        reputations: Persistent reputation dict (mutated). 信誉字典。
        selected_ids: Selected client IDs. 选定的客户端ID。
        utility_scores: Current utility scores. 当前效用分数。

    Returns:
        Updated reputations. 更新后的信誉。
    """
    positive_count = sum(
        1 for cid in selected_ids if utility_scores.get(cid, 0.0) > 0
    )
    if positive_count == 0:
        return reputations

    reward = 1.0 / positive_count
    for cid in selected_ids:
        if utility_scores.get(cid, 0.0) > 0:
            reputations[cid] = reputations.get(cid, 0.0) + reward

    return reputations
