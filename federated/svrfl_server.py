"""
SVRFL experiment runner: federated training with Shapley-based defense.
SVRFL实验运行器：基于Shapley值的联邦训练与防御。

Implements a unified experiment loop supporting:
  - Baseline FedAvg (no defense, no attack)
  - FedAvg under attack
  - SVRFL defense under various attacks

Maintains persistent server-side state across rounds:
  - Client reputations
  - Client utility scores
  - Global model parameter history (for SDFR/AFR attacks)

实现统一的实验循环，支持：
  - 基线FedAvg（无防御，无攻击）
  - 受攻击下的FedAvg
  - 各种攻击下的SVRFL防御

在各轮之间维护持久的服务器端状态：
  - 客户端信誉
  - 客户端效用分数
  - 全局模型参数历史（用于SDFR/AFR攻击）
"""

import json
import logging
import math
import os
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from federated.server import (
    fedavg_aggregate,
    get_model_parameters,
    set_model_parameters,
)
from utils.shapley import estimate_shapley_monte_carlo, evaluate_model_loss
from utils.svrfl import (
    compute_cosine_distance,
    compute_feature_values,
    detect_free_riders,
    svrfl_aggregate,
    update_reputations_aggregation,
    update_reputations_freerider,
    update_utility_scores,
)
from attacks import HONEST, DFR, SDFR, AFR, SF
from attacks.free_rider import dfr_attack, sdfr_attack, afr_attack
from attacks.poisoning import sf_attack

logger = logging.getLogger(__name__)


def run_experiment(
    defense: str,
    attack_assignment: Dict[str, str],
    client_fn: Callable,
    client_ids: List[str],
    global_model: nn.Module,
    model_fn: Callable[[], nn.Module],
    val_loader: DataLoader,
    device: torch.device,
    num_rounds: int = 25,
    clients_per_round: int = 10,
    local_epochs: int = 1,
    eta_server: float = 1.0,
    alpha: float = 0.9,
    threshold_h: float = 200.0,
    shapley_mc_samples: int = 50,
    eps: float = 1e-8,
    seed: int = 42,
) -> Tuple[List[np.ndarray], Dict[str, List], List[Dict[str, Any]]]:
    """
    Run a federated learning experiment with optional SVRFL defense and attacks.
    运行联邦学习实验，可选SVRFL防御和攻击。

    This is the main experiment loop. Each round:
      1. Filter clients by reputation (SVRFL only)
      2. Sample k clients from eligible pool
      3. Honest clients train; attackers generate fake updates
      4. Compute updates g_i = w_g - w_i
      5. (SVRFL) Estimate Shapley values via MC permutations
      6. (SVRFL) Detect free riders via feature value + KMeans
      7. (SVRFL) Update utility scores; aggregate via binary filtering
         (FedAvg) Standard weighted average aggregation
      8. Evaluate global model on validation set
      9. Log all round-level details

    这是主实验循环。每轮：
      1. 按信誉过滤客户端（仅SVRFL）
      2. 从合格池中采样k个客户端
      3. 诚实客户端训练；攻击者生成伪造更新
      4. 计算更新g_i = w_g - w_i
      5. (SVRFL)通过MC排列估计Shapley值
      6. (SVRFL)通过特征值+KMeans检测搭便车者
      7. (SVRFL)更新效用分数；通过二值过滤聚合
         (FedAvg)标准加权平均聚合
      8. 在验证集上评估全局模型
      9. 记录所有轮次级别的详细信息

    Args:
        defense: Defense type, "fedavg" or "svrfl". 防御类型。
        attack_assignment: Dict mapping cid → attack type. 攻击分配。
        client_fn: Factory: cid → ShakespeareClient. 客户端工厂。
        client_ids: All client IDs. 所有客户端ID。
        global_model: Initialized global model. 初始化的全局模型。
        model_fn: Factory creating a fresh model. 创建新模型的工厂。
        val_loader: Server validation DataLoader. 服务器验证数据加载器。
        device: Torch device. 计算设备。
        num_rounds: Total FL rounds. 总轮数。
        clients_per_round: k clients per round. 每轮客户端数。
        local_epochs: Local training epochs. 本地训练epoch数。
        eta_server: Server learning rate. 服务器学习率。
        alpha: EMA factor for utility scores. 效用分数的EMA因子。
        threshold_h: Free-rider detection threshold. 搭便车检测阈值。
        shapley_mc_samples: MC permutations for Shapley. Shapley的MC排列数。
        eps: Numerical stability. 数值稳定性。
        seed: Random seed. 随机种子。

    Returns:
        (final_params, history, round_logs):
          - final_params: Final global model parameters. 最终全局模型参数。
          - history: Dict with "round", "loss", "perplexity", "accuracy". 历史。
          - round_logs: List of per-round detail dicts. 每轮详细日志列表。
    """
    rng = random.Random(seed)
    use_svrfl = defense.lower() == "svrfl"

    # Determine benign and malicious client sets
    # 确定良性和恶意客户端集合
    malicious_ids = {
        cid for cid, atk in attack_assignment.items() if atk != HONEST
    }
    benign_ids = set(client_ids) - malicious_ids

    # Initialize global parameters / 初始化全局参数
    global_params = get_model_parameters(global_model)

    # Parameter history for SDFR/AFR (need t-1 and t-2)
    # SDFR/AFR需要的参数历史
    params_history: List[List[np.ndarray]] = [global_params]

    # Persistent server state / 持久化服务器状态
    reputations: Dict[str, float] = {cid: 0.0 for cid in client_ids}
    utility_scores: Dict[str, float] = {cid: 0.0 for cid in client_ids}

    # Metrics history / 度量历史
    history: Dict[str, List] = {
        "round": [], "loss": [], "perplexity": [], "accuracy": [],
    }
    round_logs: List[Dict[str, Any]] = []

    # Print configuration / 打印配置
    print("=" * 70)
    defense_label = "SVRFL" if use_svrfl else "FedAvg"
    n_malicious = len(malicious_ids)
    print(f"  🌐 {defense_label} Experiment / {defense_label}实验")
    print("=" * 70)
    print(f"  Total clients / 总客户端数:        {len(client_ids)}")
    print(f"  Rounds / 轮次:                     {num_rounds}")
    print(f"  Clients per round / 每轮客户端数:  {clients_per_round}")
    print(f"  Malicious / 恶意:                  {n_malicious}")
    if malicious_ids:
        for mid in sorted(malicious_ids):
            print(f"     {mid}: {attack_assignment[mid]}")
    if use_svrfl:
        print(f"  Shapley MC samples / Shapley采样:  {shapley_mc_samples}")
        print(f"  Alpha / Alpha:                     {alpha}")
        print(f"  Threshold h / 阈值h:               {threshold_h}")
        print(f"  eta_server:                        {eta_server}")
    print("=" * 70)

    import time as _time
    round_times: list[float] = []  # track elapsed time per round / 记录每轮耗时

    for round_num in range(1, num_rounds + 1):
        round_start = _time.time()
        round_log: Dict[str, Any] = {"round": round_num}

        print(f"\n  ── Round {round_num}/{num_rounds} ", "─" * 45)

        # ==== Step 1: Client sampling with reputation filtering ====
        # ==== 步骤1：基于信誉过滤的客户端采样 ====
        if use_svrfl:
            eligible = [
                cid for cid in client_ids if reputations.get(cid, 0.0) >= 0
            ]
        else:
            eligible = list(client_ids)

        k = min(clients_per_round, len(eligible))
        if k == 0:
            logger.warning(f"Round {round_num}: no eligible clients; skipping.")
            continue
        selected_ids = rng.sample(eligible, k)
        round_log["selected_ids"] = selected_ids

        # ==== Step 2: Local training and attack generation ====
        # ==== 步骤2：本地训练和攻击生成 ====
        local_params: Dict[str, List[np.ndarray]] = {}
        updates: Dict[str, List[np.ndarray]] = {}
        sample_counts: Dict[str, int] = {}

        n_honest = sum(1 for c in selected_ids if attack_assignment.get(c, HONEST) in (HONEST, SF))
        n_attack = len(selected_ids) - n_honest + sum(1 for c in selected_ids if attack_assignment.get(c, HONEST) == SF)
        print(f"     📡 Training {len(selected_ids)} clients...", end="", flush=True)
        train_t0 = _time.time()

        for idx, cid in enumerate(selected_ids):
            atk = attack_assignment.get(cid, HONEST)

            if atk == HONEST or atk == SF:
                # Honest training (SF also trains honestly first)
                # 诚实训练（SF也先诚实训练）
                client = client_fn(cid)
                trained_params, n_samples, _ = client.fit(
                    global_params, config={}
                )
                sample_counts[cid] = n_samples

                if atk == SF:
                    fake_params = sf_attack(global_params, trained_params)
                    local_params[cid] = fake_params
                else:
                    local_params[cid] = trained_params

            elif atk == DFR:
                local_params[cid] = dfr_attack(
                    global_params, round_num, seed=seed + round_num
                )
                sample_counts[cid] = 1

            elif atk == SDFR:
                if len(params_history) >= 3:
                    local_params[cid] = sdfr_attack(
                        params_history[-1],
                        params_history[-2],
                        params_history[-3],
                    )
                else:
                    # Not enough history; fall back to honest training
                    # 历史不足；回退到诚实训练
                    client = client_fn(cid)
                    trained_params, n_samples, _ = client.fit(
                        global_params, config={}
                    )
                    local_params[cid] = trained_params
                    sample_counts[cid] = n_samples
                if cid not in sample_counts:
                    sample_counts[cid] = 1

            elif atk == AFR:
                if len(params_history) >= 3:
                    local_params[cid] = afr_attack(
                        params_history[-1],
                        params_history[-2],
                        params_history[-3],
                        seed=seed + round_num,
                    )
                else:
                    client = client_fn(cid)
                    trained_params, n_samples, _ = client.fit(
                        global_params, config={}
                    )
                    local_params[cid] = trained_params
                    sample_counts[cid] = n_samples
                if cid not in sample_counts:
                    sample_counts[cid] = 1

        train_elapsed = _time.time() - train_t0
        print(f" done ({train_elapsed:.1f}s)")

        # Compute updates: g_i = w_g - w_i
        # 计算更新：g_i = w_g - w_i
        for cid in selected_ids:
            updates[cid] = [
                (g.astype(np.float64) - w.astype(np.float64)).astype(np.float32)
                for g, w in zip(global_params, local_params[cid])
            ]

        # ==== Step 3: Aggregation ====
        # ==== 步骤3：聚合 ====
        if use_svrfl:
            # ---- SVRFL defense pipeline / SVRFL防御流程 ----

            # A. Shapley value estimation / Shapley值估计
            sv_t0 = _time.time()
            sv = estimate_shapley_monte_carlo(
                global_params=global_params,
                updates=updates,
                selected_ids=selected_ids,
                val_loader=val_loader,
                model_fn=model_fn,
                device=device,
                eta_server=eta_server,
                num_samples=shapley_mc_samples,
                seed=seed + round_num * 1000,
            )
            sv_elapsed = _time.time() - sv_t0
            print(f"\r     🎲 Shapley estimation (R={shapley_mc_samples}, k={k})... "
                  f"done ({sv_elapsed:.1f}s)      ")
            round_log["shapley_values"] = {
                cid: float(v) for cid, v in sv.items()
            }

            # B. Free-rider detection / 搭便车检测
            cosine_dists: Dict[str, float] = {}
            for cid in selected_ids:
                cosine_dists[cid] = compute_cosine_distance(
                    local_params[cid], global_params
                )
            round_log["cosine_distances"] = cosine_dists

            feature_vals = compute_feature_values(sv, cosine_dists, eps=eps)
            round_log["feature_values"] = {
                cid: float(v) for cid, v in feature_vals.items()
            }

            detected_fr = detect_free_riders(feature_vals, threshold_h)
            round_log["detected_freeriders"] = sorted(detected_fr)

            # Penalize detected free riders / 惩罚检测到的搭便车者
            if detected_fr:
                reputations = update_reputations_freerider(
                    reputations, detected_fr, round_num, len(client_ids)
                )
                # Zero out their Shapley values for this round
                # 将其本轮Shapley值置零
                for cid in detected_fr:
                    sv[cid] = 0.0

            # C. Update utility scores / 更新效用分数
            utility_scores = update_utility_scores(utility_scores, sv, alpha)
            # Log the FULL persistent utility state for all clients, not just
            # selected ones. Utility persists across rounds, so every client
            # always has a valid current value.
            # 记录所有客户端的完整持久效用状态，而非仅记录本轮选中的客户端。
            round_log["utility_scores"] = {
                cid: float(utility_scores[cid]) for cid in client_ids
            }

            # D. Binary positive-utility aggregation
            # D. 二值正效用聚合
            new_global_params, aggregated_ids = svrfl_aggregate(
                global_params, updates, selected_ids, utility_scores, eta_server
            )
            round_log["aggregated_ids"] = aggregated_ids

            # E. Update reputations based on aggregation
            # E. 基于聚合更新信誉
            reputations = update_reputations_aggregation(
                reputations, selected_ids, utility_scores
            )

            global_params = new_global_params

        else:
            # ---- Standard FedAvg / 标准FedAvg ----
            # Reconstruct local params as (params, num_samples) tuples
            # 重构为(参数, 样本数)元组
            results = [
                (local_params[cid], sample_counts.get(cid, 1))
                for cid in selected_ids
            ]
            global_params = fedavg_aggregate(results)

        # Record parameter history (keep last 3 for SDFR/AFR)
        # 记录参数历史（保留最近3个用于SDFR/AFR）
        params_history.append(global_params)
        if len(params_history) > 3:
            params_history.pop(0)

        # ==== Step 4: Evaluate global model ====
        # ==== 步骤4：评估全局模型 ====
        eval_model = model_fn()
        val_loss = evaluate_model_loss(
            eval_model, global_params, val_loader, device
        )

        # Also compute accuracy on validation set
        # 同时在验证集上计算准确率
        set_model_parameters(eval_model, global_params)
        eval_model.to(device)
        eval_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _ = eval_model(inputs)
                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.numel()

        accuracy = correct / max(total, 1)
        clamped_loss = min(val_loss, 50.0)
        perplexity = math.exp(clamped_loss)

        # Record metrics / 记录度量
        history["round"].append(round_num)
        history["loss"].append(val_loss)
        history["perplexity"].append(perplexity)
        history["accuracy"].append(accuracy)

        # Record reputations for this round / 记录本轮信誉
        round_log["reputations"] = {
            cid: float(reputations.get(cid, 0.0)) for cid in client_ids
        }

        round_logs.append(round_log)

        # Print round summary / 打印轮次总结
        round_elapsed = _time.time() - round_start
        round_times.append(round_elapsed)
        avg_round = sum(round_times) / len(round_times)
        remaining = avg_round * (num_rounds - round_num)
        remaining_str = f"{int(remaining//60)}m{int(remaining%60):02d}s"

        fr_str = ""
        if use_svrfl and detected_fr:
            fr_str = f" │ FR: {sorted(detected_fr)}"
        agg_str = ""
        if use_svrfl:
            n_agg = len(round_log.get("aggregated_ids", selected_ids))
            agg_str = f" │ Agg: {n_agg}/{k}"
        print(
            f"     ✅ Loss: {val_loss:.4f} │ "
            f"PPL: {perplexity:8.2f} │ "
            f"Acc: {accuracy:.4f} │ "
            f"{round_elapsed:.0f}s │ "
            f"ETA: {remaining_str}"
            f"{agg_str}{fr_str}"
        )

    print("=" * 70)
    print(f"  ✅ {defense_label} experiment complete / {defense_label}实验完成")
    print("=" * 70)

    return global_params, history, round_logs


def compute_freerider_metrics(
    round_logs: List[Dict[str, Any]],
    true_freerider_ids: Set[str],
) -> Dict[str, List[float]]:
    """
    Compute free-rider detection precision and FRDR across rounds.
    计算各轮次的搭便车检测精度和FRDR。

    Precision = TP / (TP + FP)
    FRDR (Free Rider Detection Rate) = TP / (TP + FN)

    Args:
        round_logs: List of per-round log dicts. 每轮日志。
        true_freerider_ids: Ground-truth free-rider IDs. 真实搭便车者ID。

    Returns:
        Dict with "precision" and "frdr" lists, one per round.
    """
    precisions = []
    frdrs = []
    rounds = []

    for log in round_logs:
        detected = set(log.get("detected_freeriders", []))
        # Only consider selected clients this round
        selected = set(log.get("selected_ids", []))
        true_in_round = true_freerider_ids & selected

        if not detected and not true_in_round:
            # No free riders present or detected → perfect score
            precisions.append(1.0)
            frdrs.append(1.0)
        elif not detected and true_in_round:
            precisions.append(1.0)  # No false positives
            frdrs.append(0.0)      # Missed all free riders
        else:
            tp = len(detected & true_in_round)
            fp = len(detected - true_in_round)
            fn = len(true_in_round - detected)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            frdr = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            precisions.append(precision)
            frdrs.append(frdr)

        rounds.append(log["round"])

    return {"round": rounds, "precision": precisions, "frdr": frdrs}
