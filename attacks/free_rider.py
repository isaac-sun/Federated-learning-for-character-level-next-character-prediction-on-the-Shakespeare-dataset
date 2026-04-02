"""
Free-rider attack implementations for federated learning.
联邦学习中的搭便车攻击实现。

Implements three free-rider attacks from the SVRFL paper:
  - DFR (Disguised Free Rider): decaying Gaussian noise
  - SDFR (Stochastic Disguised Free Rider): mimics previous global update
  - AFR (Advanced Free Rider): SDFR + random coordinate noise

实现SVRFL论文中的三种搭便车攻击：
  - DFR（伪装搭便车者）：衰减高斯噪声
  - SDFR（随机伪装搭便车者）：模拟先前全局更新
  - AFR（高级搭便车者）：SDFR + 随机坐标噪声

Convention: g_i = w_g - w_i (update = global - local).
约定：g_i = w_g - w_i（更新 = 全局 - 本地）。
Attack functions return fake local params w_fake such that
the server computes g_fake = w_g - w_fake.
攻击函数返回伪造的本地参数w_fake，服务器据此计算g_fake = w_g - w_fake。
"""

import numpy as np
from typing import List, Optional


def dfr_attack(
    global_params: List[np.ndarray],
    round_num: int,
    sigma: float = 0.5,
    gamma: float = 1.0,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Disguised Free Rider (DFR) attack.
    伪装搭便车（DFR）攻击。

    Generates a fake update g_fake = phi(t) * epsilon,
    where phi(t) = sigma * t^(-gamma) and epsilon ~ N(0, I).
    Returns w_fake = w_g - g_fake so the server sees g_fake.
    生成伪造更新g_fake = phi(t) * epsilon，
    其中phi(t) = sigma * t^(-gamma)，epsilon ~ N(0, I)。
    返回w_fake = w_g - g_fake，使服务器看到g_fake。

    The decaying factor phi(t) makes the fake update look like a
    model that is gradually converging over rounds.
    衰减因子phi(t)使伪造更新看起来像模型在逐轮收敛。

    Args:
        global_params: Current global model parameters w_g^t.
                       当前全局模型参数。
        round_num: Current round number (1-indexed). 当前轮次。
        sigma: Noise scale factor (default 0.5). 噪声缩放因子。
               Should be calibrated to match the scale of honest updates.
               应校准为与诚实更新的规模匹配。
        gamma: Decay exponent (default 1.0). 衰减指数。
        seed: Random seed. 随机种子。

    Returns:
        Fake local model parameters w_fake. 伪造的本地模型参数。
    """
    rng = np.random.RandomState(seed)
    phi_t = sigma * (round_num ** (-gamma))

    fake_local_params = []
    for p in global_params:
        epsilon = rng.randn(*p.shape).astype(np.float32)
        # w_fake = w_g - g_fake = w_g - phi_t * epsilon
        fake_local_params.append(p - phi_t * epsilon)

    return fake_local_params


def sdfr_attack(
    global_params_current: List[np.ndarray],
    global_params_prev: List[np.ndarray],
    global_params_prev_prev: List[np.ndarray],
    eps: float = 1e-8,
) -> List[np.ndarray]:
    """
    Stochastic Disguised Free Rider (SDFR) attack.
    随机伪装搭便车（SDFR）攻击。

    Generates a fake update that mimics the previous global model change:
      delta_t     = w_g^t - w_g^{t-1}
      delta_{t-1} = w_g^{t-1} - w_g^{t-2}
      g_fake = (||delta_t|| / (||delta_{t-1}|| + eps)) * delta_t

    The update is scaled so its norm is proportional to the ratio of
    consecutive global update norms, making it harder to detect.
    更新被缩放使其范数与连续全局更新范数的比率成正比。

    Args:
        global_params_current: w_g^t. 当前全局参数。
        global_params_prev: w_g^{t-1}. 上一轮全局参数。
        global_params_prev_prev: w_g^{t-2}. 两轮前全局参数。
        eps: Numerical stability. 数值稳定性。

    Returns:
        Fake local model parameters w_fake. 伪造的本地模型参数。
    """
    delta_t = [
        (a.astype(np.float64) - b.astype(np.float64))
        for a, b in zip(global_params_current, global_params_prev)
    ]
    delta_t_minus_1 = [
        (a.astype(np.float64) - b.astype(np.float64))
        for a, b in zip(global_params_prev, global_params_prev_prev)
    ]

    norm_dt = np.sqrt(sum(np.sum(d ** 2) for d in delta_t))
    norm_dt1 = np.sqrt(sum(np.sum(d ** 2) for d in delta_t_minus_1))

    scale = norm_dt / (norm_dt1 + eps)

    # g_fake = scale * delta_t;  w_fake = w_g - g_fake
    fake_local_params = [
        (g.astype(np.float64) - scale * d).astype(np.float32)
        for g, d in zip(global_params_current, delta_t)
    ]
    return fake_local_params


def afr_attack(
    global_params_current: List[np.ndarray],
    global_params_prev: List[np.ndarray],
    global_params_prev_prev: List[np.ndarray],
    noise_fraction: float = 0.1,
    noise_scale: float = 0.01,
    eps: float = 1e-8,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Advanced Free Rider (AFR) attack.
    高级搭便车（AFR）攻击。

    Starts from SDFR, then adds Gaussian noise to a randomly selected
    subset of parameter coordinates to evade detection while remaining
    a free rider.
    从SDFR开始，然后向随机选择的参数坐标子集添加高斯噪声，
    以在保持搭便车的同时逃避检测。

    Args:
        global_params_current: w_g^t. 当前全局参数。
        global_params_prev: w_g^{t-1}. 上一轮全局参数。
        global_params_prev_prev: w_g^{t-2}. 两轮前全局参数。
        noise_fraction: Fraction of coordinates to perturb (default 0.1).
                        要扰动的坐标比例。
        noise_scale: Std of added noise (default 0.01). 添加噪声的标准差。
        eps: Numerical stability. 数值稳定性。
        seed: Random seed. 随机种子。

    Returns:
        Fake local model parameters w_fake. 伪造的本地模型参数。
    """
    # Start from SDFR to get the base fake update
    # 从SDFR开始获得基本伪造更新
    sdfr_params = sdfr_attack(
        global_params_current, global_params_prev, global_params_prev_prev, eps
    )

    rng = np.random.RandomState(seed)

    # Compute the fake update from SDFR: g_sdfr = w_g - w_sdfr
    # 计算SDFR的伪造更新
    fake_update = [
        g.astype(np.float64) - w.astype(np.float64)
        for g, w in zip(global_params_current, sdfr_params)
    ]

    # Add noise to random coordinates / 向随机坐标添加噪声
    noisy_update = []
    for u in fake_update:
        flat = u.flatten().copy()
        n_perturb = max(1, int(len(flat) * noise_fraction))
        indices = rng.choice(len(flat), size=n_perturb, replace=False)
        flat[indices] += rng.randn(n_perturb) * noise_scale
        noisy_update.append(flat.reshape(u.shape))

    # w_fake = w_g - noisy_update
    fake_local_params = [
        (g.astype(np.float64) - u).astype(np.float32)
        for g, u in zip(global_params_current, noisy_update)
    ]
    return fake_local_params
