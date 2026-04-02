"""
SVRFL experiment orchestration script.
SVRFL实验编排脚本。

Runs SVRFL defense experiments on the Shakespeare dataset including:
  - Baseline FedAvg (no attack)
  - FedAvg under various attacks
  - SVRFL defense under various attacks
  - Concurrent attack scenarios

用法 / Usage:
    cd federated_shakespeare
    python experiments/train_svrfl.py --defense svrfl --attack dfr
    python experiments/train_svrfl.py --defense fedavg --attack none
    python experiments/train_svrfl.py --defense svrfl --attack concurrent
    python experiments/train_svrfl.py --help
"""

import argparse
import json
import logging
import os
import sys
import time

# Add project root to Python path / 将项目根目录添加到Python路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.shakespeare_loader import get_client_datasets, build_server_validation_set
from models.gru_model import CharGRU
from federated.client import ShakespeareClient
from federated.server import set_model_parameters
from federated.svrfl_server import run_experiment, compute_freerider_metrics
from utils.metrics import MetricsLogger, plot_metrics, plot_svrfl_metrics
from utils.device import get_device, print_device_info
from attacks import HONEST, DFR, SDFR, AFR, SF


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for SVRFL experiments.
    解析SVRFL实验的命令行参数。
    """
    parser = argparse.ArgumentParser(
        description=(
            "SVRFL Defense Experiment for Shakespeare Task\n"
            "莎士比亚任务的SVRFL防御实验"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Defense / Attack ----
    exp_group = parser.add_argument_group("Experiment / 实验")
    exp_group.add_argument(
        "--defense", type=str, default="svrfl",
        choices=["fedavg", "svrfl"],
        help="Defense method (default: svrfl). 防御方法。",
    )
    exp_group.add_argument(
        "--attack", type=str, default="none",
        choices=["none", "dfr", "sdfr", "afr", "sf", "concurrent"],
        help="Attack scenario (default: none). 攻击场景。",
    )
    exp_group.add_argument(
        "--num-freeriders", type=int, default=3,
        help="Number of free-rider attackers (default: 3). 搭便车攻击者数量。",
    )
    exp_group.add_argument(
        "--num-poisoners", type=int, default=2,
        help="Number of SF poisoners (default: 2). SF投毒者数量。",
    )

    # ---- Federated Learning ----
    fl_group = parser.add_argument_group("Federated Learning / 联邦学习")
    fl_group.add_argument(
        "--num-rounds", type=int, default=25,
        help="Number of FL rounds (default: 25). 联邦学习轮数。",
    )
    fl_group.add_argument(
        "--clients-per-round", type=int, default=10,
        help="Clients sampled per round (default: 10). 每轮采样客户端数。",
    )
    fl_group.add_argument(
        "--num-clients", type=int, default=20,
        help="Total number of FL clients (default: 20). 总客户端数。",
    )
    fl_group.add_argument(
        "--local-epochs", type=int, default=1,
        help="Local training epochs (default: 1). 本地训练epoch数。",
    )

    # ---- SVRFL Parameters ----
    svrfl_group = parser.add_argument_group("SVRFL Parameters / SVRFL参数")
    svrfl_group.add_argument(
        "--shapley-mc-samples", type=int, default=50,
        help="Monte Carlo permutations for Shapley (default: 50). "
             "Shapley的蒙特卡罗排列数。",
    )
    svrfl_group.add_argument(
        "--alpha", type=float, default=0.9,
        help="EMA factor for utility scores (default: 0.9). 效用分数EMA因子。",
    )
    svrfl_group.add_argument(
        "--threshold-h", type=float, default=200.0,
        help="Free-rider detection threshold (default: 200). 搭便车检测阈值。",
    )
    svrfl_group.add_argument(
        "--eta-server", type=float, default=1.0,
        help="Server learning rate (default: 1.0). 服务器学习率。",
    )

    # ---- Model ----
    model_group = parser.add_argument_group("Model / 模型")
    model_group.add_argument(
        "--embed-dim", type=int, default=64,
        help="Embedding dimension (default: 64). 嵌入维度。",
    )
    model_group.add_argument(
        "--hidden-dim", type=int, default=128,
        help="GRU hidden dimension (default: 128). GRU隐藏维度。",
    )
    model_group.add_argument(
        "--num-layers", type=int, default=2,
        help="Number of GRU layers (default: 2). GRU层数。",
    )
    model_group.add_argument(
        "--dropout", type=float, default=0.3,
        help="Dropout rate (default: 0.3). Dropout率。",
    )

    # ---- Training ----
    train_group = parser.add_argument_group("Training / 训练")
    train_group.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size (default: 64). 批次大小。",
    )
    train_group.add_argument(
        "--lr", type=float, default=0.001,
        help="Local learning rate (default: 0.001). 本地学习率。",
    )
    train_group.add_argument(
        "--seq-length", type=int, default=50,
        help="Sequence length (default: 50). 序列长度。",
    )
    train_group.add_argument(
        "--val-samples", type=int, default=1000,
        help="Server validation set size (default: 1000). 服务器验证集大小。",
    )

    # ---- Output ----
    out_group = parser.add_argument_group("Output / 输出")
    out_group.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (auto-generated if None). 输出目录。",
    )
    out_group.add_argument(
        "--save-model", action="store_true", default=True,
        help="Save final global model. 保存最终全局模型。",
    )

    # ---- Misc ----
    misc_group = parser.add_argument_group("Misc / 其他")
    misc_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42). 随机种子。",
    )
    misc_group.add_argument(
        "--device", type=str, default="auto",
        help=(
            "Compute device: auto / cpu / cuda / cuda:N / mps  (default: auto)\n"
            "  auto   – 自动选择（CUDA → MPS → CPU）\n"
            "  cuda   – NVIDIA GPU（等同于 cuda:0）\n"
            "  cuda:N – 指定第 N 块 NVIDIA GPU（如 cuda:0、cuda:1）\n"
            "  mps    – Apple Silicon GPU（仅 macOS）\n"
            "  cpu    – 仅使用 CPU\n"
            "计算设备（默认：auto 自动选择）。"
        ),
    )
    misc_group.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. 日志级别。",
    )

    return parser.parse_args()


def build_attack_assignment(
    client_ids: list,
    attack: str,
    num_freeriders: int = 3,
    num_poisoners: int = 2,
    seed: int = 42,
) -> dict:
    """
    Build the attack assignment mapping: cid → attack type.
    构建攻击分配映射：cid → 攻击类型。

    Malicious clients are assigned from the END of the sorted client
    list (so the first clients are always benign, matching paper convention).
    恶意客户端从排序客户端列表的末尾分配。

    Args:
        client_ids: Sorted list of client IDs. 排序后的客户端ID列表。
        attack: Attack scenario name. 攻击场景名称。
        num_freeriders: Number of free-rider attackers. 搭便车攻击者数量。
        num_poisoners: Number of poisoners. 投毒者数量。
        seed: Random seed. 随机种子。

    Returns:
        Dict mapping each cid → attack type string.
    """
    assignment = {cid: HONEST for cid in client_ids}

    if attack == "none":
        return assignment

    if attack in ("dfr", "sdfr", "afr"):
        # Single type of free-rider attack
        n = min(num_freeriders, len(client_ids))
        for i in range(n):
            cid = client_ids[-(i + 1)]
            assignment[cid] = attack

    elif attack == "sf":
        n = min(num_poisoners, len(client_ids))
        for i in range(n):
            cid = client_ids[-(i + 1)]
            assignment[cid] = SF

    elif attack == "concurrent":
        # 3 free riders (1 DFR + 1 SDFR + 1 AFR) + 2 SF poisoners
        # 3个搭便车者(各1个DFR/SDFR/AFR) + 2个SF投毒者
        fr_types = [DFR, SDFR, AFR]
        idx = len(client_ids) - 1
        for atk_type in fr_types:
            if idx >= 0:
                assignment[client_ids[idx]] = atk_type
                idx -= 1
        for _ in range(min(num_poisoners, idx + 1)):
            if idx >= 0:
                assignment[client_ids[idx]] = SF
                idx -= 1

    return assignment


def main():
    """
    Main entry point for SVRFL experiments.
    SVRFL实验的主入口。
    """
    args = parse_args()

    # ---- Logging / 日志 ----
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("train_svrfl")

    # ---- Reproducibility / 可重复性 ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.device)
    log.info(f"Using device: {device}")
    print_device_info(device)

    # Auto-generate output directory / 自动生成输出目录
    if args.output_dir is None:
        output_dir = os.path.join(
            PROJECT_ROOT, "results",
            f"{args.defense}_{args.attack}_r{args.num_rounds}"
        )
    else:
        output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ================================================================
    # Step 1: Load Data / 步骤1：加载数据
    # ================================================================
    print("\n📚 Loading Shakespeare dataset...")
    start = time.time()

    client_datasets, vocab = get_client_datasets(
        seq_length=args.seq_length,
        train_ratio=0.8,
    )

    # Select top-N clients by training data volume (paper: first 20)
    # 按训练数据量选择前N个客户端（论文：前20个）
    all_cids = sorted(
        client_datasets.keys(),
        key=lambda c: len(client_datasets[c]["train"]),
        reverse=True,
    )
    selected_pool = all_cids[: args.num_clients]
    selected_pool.sort()  # Sort alphabetically for consistent ordering

    # Filter clients with fewer than 2 training sequences (paper requirement)
    # 过滤训练序列少于2的客户端（论文要求）
    selected_pool = [
        cid for cid in selected_pool
        if len(client_datasets[cid]["train"]) >= 2
    ]

    load_time = time.time() - start
    print(f"   ✓ Loaded {len(client_datasets)} total clients in {load_time:.1f}s")
    print(f"   ✓ Selected {len(selected_pool)} clients for experiment")
    print(f"   ✓ Vocabulary size: {vocab.vocab_size}")
    for cid in selected_pool[:5]:
        n_train = len(client_datasets[cid]["train"])
        n_test = len(client_datasets[cid]["test"])
        print(f"     {cid:20s}  train={n_train:5d}  test={n_test:5d}")
    if len(selected_pool) > 5:
        print(f"     ... and {len(selected_pool) - 5} more")

    # ================================================================
    # Step 2: Build server validation set / 步骤2：构建服务器验证集
    # ================================================================
    print(f"\n📋 Building server validation set ({args.val_samples} samples)...")
    val_dataset = build_server_validation_set(
        client_datasets, num_samples=args.val_samples, seed=args.seed
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 4,
        shuffle=False, drop_last=False,
    )
    print(f"   ✓ Validation set: {len(val_dataset)} samples")

    # ================================================================
    # Step 3: Initialize model / 步骤3：初始化模型
    # ================================================================
    print(f"\n🧠 Creating CharGRU model...")

    def model_fn():
        """
        Factory for creating fresh model instances.
        创建新模型实例的工厂函数。
        """
        return CharGRU(
            vocab_size=vocab.vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )

    global_model = model_fn()
    total_params = sum(p.numel() for p in global_model.parameters())
    print(f"   ✓ Parameters: {total_params:,}")
    print(
        f"   ✓ Architecture: Embed({vocab.vocab_size}, {args.embed_dim}) → "
        f"GRU({args.embed_dim}, {args.hidden_dim}, layers={args.num_layers}) → "
        f"Linear({args.hidden_dim}, {vocab.vocab_size})"
    )

    # ================================================================
    # Step 4: Setup attack assignment / 步骤4：设置攻击分配
    # ================================================================
    attack_assignment = build_attack_assignment(
        selected_pool, args.attack,
        num_freeriders=args.num_freeriders,
        num_poisoners=args.num_poisoners,
        seed=args.seed,
    )

    malicious_ids = {
        cid for cid, atk in attack_assignment.items() if atk != HONEST
    }
    freerider_ids = {
        cid for cid, atk in attack_assignment.items()
        if atk in (DFR, SDFR, AFR)
    }

    print(f"\n⚔️  Attack scenario: {args.attack}")
    for cid in sorted(attack_assignment.keys()):
        atk = attack_assignment[cid]
        if atk != HONEST:
            print(f"     {cid}: {atk}")
    if not malicious_ids:
        print("     (no attackers)")

    # ================================================================
    # Step 5: Define client factory / 步骤5：定义客户端工厂
    # ================================================================
    def client_fn(cid: str) -> ShakespeareClient:
        """
        Create a Flower client for the given ID.
        为给定ID创建Flower客户端。
        """
        model = model_fn()
        return ShakespeareClient(
            cid=cid,
            model=model,
            train_dataset=client_datasets[cid]["train"],
            test_dataset=client_datasets[cid]["test"],
            device=device,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )

    # ================================================================
    # Step 6: Run experiment / 步骤6：运行实验
    # ================================================================
    print(f"\n🔁 Starting {args.defense.upper()} experiment...")
    train_start = time.time()

    final_params, history, round_logs = run_experiment(
        defense=args.defense,
        attack_assignment=attack_assignment,
        client_fn=client_fn,
        client_ids=selected_pool,
        global_model=global_model,
        model_fn=model_fn,
        val_loader=val_loader,
        device=device,
        num_rounds=args.num_rounds,
        clients_per_round=args.clients_per_round,
        local_epochs=args.local_epochs,
        eta_server=args.eta_server,
        alpha=args.alpha,
        threshold_h=args.threshold_h,
        shapley_mc_samples=args.shapley_mc_samples,
        seed=args.seed,
    )

    train_time = time.time() - train_start
    print(f"\n⏱  Experiment completed in {train_time:.1f}s")

    # ================================================================
    # Step 7: Save results / 步骤7：保存结果
    # ================================================================

    # Save metrics JSON / 保存度量JSON
    metrics_logger = MetricsLogger()
    for i in range(len(history["round"])):
        metrics_logger.log(
            round_num=history["round"][i],
            loss=history["loss"][i],
            perplexity=history["perplexity"][i],
            accuracy=history["accuracy"][i],
        )
    metrics_path = os.path.join(output_dir, "metrics.json")
    metrics_logger.save(metrics_path)
    print(f"\n💾 Metrics saved to {metrics_path}")

    # Save round logs / 保存轮次日志
    logs_path = os.path.join(output_dir, "round_logs.json")
    with open(logs_path, "w") as f:
        json.dump(round_logs, f, indent=2, default=str)
    print(f"💾 Round logs saved to {logs_path}")

    # Save experiment config / 保存实验配置
    config = {
        "defense": args.defense,
        "attack": args.attack,
        "num_clients": len(selected_pool),
        "clients_per_round": args.clients_per_round,
        "num_rounds": args.num_rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "eta_server": args.eta_server,
        "alpha": args.alpha,
        "threshold_h": args.threshold_h,
        "shapley_mc_samples": args.shapley_mc_samples,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "seq_length": args.seq_length,
        "val_samples": args.val_samples,
        "seed": args.seed,
        "malicious_ids": sorted(malicious_ids),
        "freerider_ids": sorted(freerider_ids),
        "attack_assignment": attack_assignment,
        "selected_clients": selected_pool,
        "train_time_seconds": train_time,
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"💾 Config saved to {config_path}")

    # ================================================================
    # Step 8: Generate plots / 步骤8：生成图表
    # ================================================================
    print("\n📈 Generating plots...")

    # Standard training curve plots / 标准训练曲线
    plot_metrics(metrics_logger, save_dir=output_dir)

    # SVRFL-specific plots / SVRFL特有图表
    if args.defense == "svrfl":
        detection_metrics = None
        if freerider_ids:
            detection_metrics = compute_freerider_metrics(
                round_logs, freerider_ids
            )
        plot_svrfl_metrics(
            round_logs,
            save_dir=output_dir,
            malicious_ids=malicious_ids,
            detection_metrics=detection_metrics,
        )

    # Save final model / 保存最终模型
    if args.save_model:
        set_model_parameters(global_model, final_params)
        model_path = os.path.join(output_dir, "global_model.pt")
        torch.save(
            {
                "model_state_dict": global_model.state_dict(),
                "vocab_size": vocab.vocab_size,
                "config": config,
            },
            model_path,
        )
        print(f"💾 Global model saved to {model_path}")

    # ================================================================
    # Final Summary / 最终总结
    # ================================================================
    final = metrics_logger.get_latest()
    print()
    print("=" * 70)
    print(f"  📊 {args.defense.upper()} + {args.attack} Results")
    print("=" * 70)
    print(f"  Final Loss:       {final.get('loss', 0):.4f}")
    print(f"  Final Perplexity: {final.get('perplexity', 0):.2f}")
    print(f"  Final Accuracy:   {final.get('accuracy', 0):.4f}")
    print(f"  Total Time:       {train_time:.1f}s")
    print(f"  Output:           {output_dir}")
    print("=" * 70)
    print("\n✅ Done!\n")


if __name__ == "__main__":
    main()
