"""
Poisoning attack implementations for federated learning.
联邦学习中的投毒攻击实现。

Implements the Sign-Flipping (SF) attack from the SVRFL paper.
实现SVRFL论文中的符号翻转（SF）攻击。

Convention: g_i = w_g - w_i (update = global - local).
约定：g_i = w_g - w_i（更新 = 全局 - 本地）。
"""

import numpy as np
from typing import List


def sf_attack(
    global_params: List[np.ndarray],
    honest_local_params: List[np.ndarray],
    multiplier: float = -1.0,
) -> List[np.ndarray]:
    """
    Sign-Flipping (SF) poisoning attack.
    符号翻转（SF）投毒攻击。

    The malicious client trains honestly to obtain g_honest = w_g - w_i,
    then flips the sign: g_attack = multiplier * g_honest.
    Returns w_fake = w_g - g_attack.

    恶意客户端诚实训练得到g_honest = w_g - w_i，
    然后翻转符号：g_attack = multiplier * g_honest。
    返回w_fake = w_g - g_attack。

    With multiplier = -1:
      g_attack = -(w_g - w_i) = w_i - w_g
      w_fake   = w_g - (w_i - w_g) = 2*w_g - w_i
    This pushes the global model in the opposite direction.
    这将全局模型推向相反方向。

    Note: LF (label-flipping) is NOT implemented for Shakespeare
    since it is a next-character prediction task, not classification.
    注意：LF（标签翻转）未实现，因为莎士比亚是下一字符预测任务。

    Args:
        global_params: Current global model parameters w_g.
                       当前全局模型参数。
        honest_local_params: Parameters from honest local training w_i.
                             诚实本地训练的参数。
        multiplier: Sign-flip multiplier (default -1.0). 符号翻转乘数。

    Returns:
        Fake local model parameters w_fake. 伪造的本地模型参数。
    """
    fake_local_params = []
    for g, w in zip(global_params, honest_local_params):
        g_honest = g.astype(np.float64) - w.astype(np.float64)
        g_attack = multiplier * g_honest
        # w_fake = w_g - g_attack
        fake_local_params.append(
            (g.astype(np.float64) - g_attack).astype(np.float32)
        )
    return fake_local_params
