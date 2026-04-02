#!/usr/bin/env python3
"""
One-click experiment runner for FedBard.
一键运行所有 FedBard 实验。

Runs the full experiment matrix:
  1. FedAvg baseline (no attack)
  2. FedAvg under each attack (no defense, for comparison)
  3. SVRFL defense under each attack

Results are saved to results/<defense>_<attack>_r<rounds>/.
A final comparison summary is printed and saved as results/summary.csv.

Usage:
    cd federated_shakespeare
    python experiments/run_all.py                    # full experiments
    python experiments/run_all.py --quick            # quick smoke test
    python experiments/run_all.py --attacks dfr sf   # specific attacks only
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# Experiment matrix / 实验矩阵
# ============================================================

# (defense, attack) pairs to run / 要运行的(防御, 攻击)组合
FULL_MATRIX = [
    # Baselines / 基线
    ("fedavg", "none"),
    # FedAvg under attack (no defense) / FedAvg 受攻击（无防御）
    ("fedavg", "dfr"),
    ("fedavg", "sdfr"),
    ("fedavg", "afr"),
    ("fedavg", "sf"),
    ("fedavg", "concurrent"),
    # SVRFL defense / SVRFL 防御
    ("svrfl", "none"),
    ("svrfl", "dfr"),
    ("svrfl", "sdfr"),
    ("svrfl", "afr"),
    ("svrfl", "sf"),
    ("svrfl", "concurrent"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all FedBard experiments / 运行所有 FedBard 实验",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Experiment filtering / 实验筛选 ----
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 5 rounds, 10 MC samples (for testing). "
             "快速模式：5轮，10次MC采样（用于测试）。",
    )
    parser.add_argument(
        "--attacks", nargs="+", default=None,
        choices=["none", "dfr", "sdfr", "afr", "sf", "concurrent"],
        help="Run only specific attacks. 仅运行指定攻击。",
    )
    parser.add_argument(
        "--defenses", nargs="+", default=None,
        choices=["fedavg", "svrfl"],
        help="Run only specific defenses. 仅运行指定防御。",
    )

    # ---- FL training / 联邦训练参数 ----
    parser.add_argument(
        "--num-rounds", type=int, default=None,
        help="Communication rounds (default: 5 in quick mode, 25 otherwise). 通信轮数。",
    )
    parser.add_argument(
        "--clients-per-round", type=int, default=None,
        help="Clients sampled per round (default: train_svrfl.py default=10). 每轮采样客户端数。",
    )
    parser.add_argument(
        "--num-clients", type=int, default=None,
        help="Total FL client pool size (default: train_svrfl.py default=20). 总客户端数。",
    )
    parser.add_argument(
        "--local-epochs", type=int, default=None,
        help="Local training epochs per round (default: train_svrfl.py default=1). 每轮本地训练轮次。",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Training batch size (default: train_svrfl.py default=64). 训练批次大小。",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Local learning rate (default: train_svrfl.py default=0.001). 本地学习率。",
    )

    # ---- SVRFL hyperparameters / SVRFL超参数 ----
    parser.add_argument(
        "--shapley-mc-samples", type=int, default=None,
        help="Monte Carlo permutations for Shapley (default: 10 quick, 50 full). MC排列采样数。",
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="EMA smoothing factor for utility scores (default: train_svrfl.py default=0.9). 效用EMA系数。",
    )
    parser.add_argument(
        "--threshold-h", type=float, default=None,
        help="KMeans free-rider detection threshold (default: train_svrfl.py default=200). 搭便车检测阈值。",
    )
    parser.add_argument(
        "--eta-server", type=float, default=None,
        help="Server-side aggregation learning rate (default: train_svrfl.py default=1.0). 服务端学习率。",
    )
    parser.add_argument(
        "--val-samples", type=int, default=None,
        help="Server validation set size (default: train_svrfl.py default=1000). 服务端验证集大小。",
    )

    # ---- Attack settings / 攻击设置 ----
    parser.add_argument(
        "--num-freeriders", type=int, default=None,
        help="Number of free-rider attackers (default: train_svrfl.py default=3). 搭便车攻击者数量。",
    )
    parser.add_argument(
        "--num-poisoners", type=int, default=None,
        help="Number of SF poisoners (default: train_svrfl.py default=2). 投毒者数量。",
    )

    # ---- System / 系统 ----
    parser.add_argument(
        "--device", type=str, default="auto",
        help=(
            "Compute device: auto / cpu / cuda / cuda:N / mps  (default: auto). "
            "计算设备（默认：auto 自动选择）。"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42). 随机种子。",
    )
    return parser.parse_args()


def build_experiment_list(args) -> list:
    """
    Build the list of (defense, attack) experiments to run.
    构建要运行的(防御, 攻击)实验列表。
    """
    matrix = FULL_MATRIX

    if args.attacks:
        matrix = [(d, a) for d, a in matrix if a in args.attacks]
    if args.defenses:
        matrix = [(d, a) for d, a in matrix if d in args.defenses]

    return matrix


def run_single_experiment(
    defense: str,
    attack: str,
    num_rounds: int,
    mc_samples: int,
    device: str,
    seed: int,
    extra_args: list[str] | None = None,
) -> dict:
    """
    Run a single experiment as a subprocess.
    以子进程方式运行单个实验。
    """
    cmd = [
        sys.executable, os.path.join(PROJECT_ROOT, "experiments", "train_svrfl.py"),
        "--defense", defense,
        "--attack", attack,
        "--num-rounds", str(num_rounds),
        "--shapley-mc-samples", str(mc_samples),
        "--device", device,
        "--seed", str(seed),
    ]
    if extra_args:
        cmd.extend(extra_args)

    output_dir = os.path.join(
        PROJECT_ROOT, "results", f"{defense}_{attack}_r{num_rounds}"
    )

    start = time.time()
    # Stream output in real-time instead of buffering
    # 实时输出而非缓冲，避免长时间无显示
    result = subprocess.run(
        cmd, cwd=PROJECT_ROOT,
        stdout=None, stderr=None,  # inherit parent's stdout/stderr
    )
    elapsed = time.time() - start

    success = result.returncode == 0

    # Parse final metrics from output dir if available
    # 如果可用，从输出目录解析最终度量
    metrics = {}
    metrics_path = os.path.join(output_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data = json.load(f)
        if data.get("loss"):
            metrics["final_loss"] = data["loss"][-1]
            metrics["final_perplexity"] = data["perplexity"][-1]
            metrics["final_accuracy"] = data["accuracy"][-1]

    return {
        "defense": defense,
        "attack": attack,
        "success": success,
        "elapsed": elapsed,
        "output_dir": output_dir,
        **metrics,
        "stderr": "",
    }


def print_summary(results: list, summary_path: str):
    """
    Print and save a comparison summary table.
    打印并保存比较汇总表。
    """
    print("\n" + "=" * 90)
    print("  📊 Experiment Summary / 实验汇总")
    print("=" * 90)
    print(
        f"  {'Defense':<8} {'Attack':<12} {'Status':<8} "
        f"{'Loss':>8} {'PPL':>8} {'Acc':>8} {'Time':>10}"
    )
    print("-" * 90)

    rows = []
    for r in results:
        status = "✅" if r["success"] else "❌"
        loss = f"{r.get('final_loss', 0):.4f}" if r.get("final_loss") else "  N/A"
        ppl = f"{r.get('final_perplexity', 0):.2f}" if r.get("final_perplexity") else "   N/A"
        acc = f"{r.get('final_accuracy', 0):.4f}" if r.get("final_accuracy") else "  N/A"
        t = str(timedelta(seconds=int(r["elapsed"])))

        print(
            f"  {r['defense']:<8} {r['attack']:<12} {status:<8} "
            f"{loss:>8} {ppl:>8} {acc:>8} {t:>10}"
        )

        rows.append({
            "defense": r["defense"],
            "attack": r["attack"],
            "success": r["success"],
            "final_loss": r.get("final_loss", ""),
            "final_perplexity": r.get("final_perplexity", ""),
            "final_accuracy": r.get("final_accuracy", ""),
            "time_seconds": round(r["elapsed"], 1),
        })

    print("=" * 90)

    # Save CSV / 保存CSV
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n💾 Summary saved to {summary_path}")


def main():
    args = parse_args()
    experiments = build_experiment_list(args)

    if not experiments:
        print("No experiments to run. Check --attacks / --defenses filters.")
        return

    # Determine settings / 确定设置
    if args.quick:
        num_rounds = args.num_rounds or 5
        mc_samples = args.shapley_mc_samples or 10
        mode_label = "Quick"
    else:
        num_rounds = args.num_rounds or 25
        mc_samples = args.shapley_mc_samples or 50
        mode_label = "Full"

    # Build optional pass-through args / 构建透传参数列表
    extra_args: list[str] = []
    for flag, val in [
        ("--clients-per-round", args.clients_per_round),
        ("--num-clients",       args.num_clients),
        ("--local-epochs",      args.local_epochs),
        ("--batch-size",        args.batch_size),
        ("--lr",                args.lr),
        ("--alpha",             args.alpha),
        ("--threshold-h",       args.threshold_h),
        ("--eta-server",        args.eta_server),
        ("--val-samples",       args.val_samples),
        ("--num-freeriders",    args.num_freeriders),
        ("--num-poisoners",     args.num_poisoners),
    ]:
        if val is not None:
            extra_args += [flag, str(val)]

    print("=" * 70)
    print(f"  🎭 FedBard — {mode_label} Experiment Suite")
    print("=" * 70)
    print(f"  Experiments to run:  {len(experiments)}")
    print(f"  Rounds per run:      {num_rounds}")
    print(f"  Shapley MC samples:  {mc_samples}")
    print(f"  Device:              {args.device}")
    print(f"  Seed:                {args.seed}")
    if extra_args:
        print(f"  Extra args:          {' '.join(extra_args)}")
    print("=" * 70)

    for i, (d, a) in enumerate(experiments, 1):
        print(f"  [{i}/{len(experiments)}] {d} + {a}")
    print()

    total_start = time.time()
    results = []

    for i, (defense, attack) in enumerate(experiments, 1):
        label = f"{defense} + {attack}"
        print(f"\n{'─' * 70}")
        print(f"  🔬 [{i}/{len(experiments)}] Running: {label}")
        print(f"{'─' * 70}")

        r = run_single_experiment(
            defense=defense,
            attack=attack,
            num_rounds=num_rounds,
            mc_samples=mc_samples,
            device=args.device,
            seed=args.seed,
            extra_args=extra_args,
        )
        results.append(r)

        status = "✅ Done" if r["success"] else "❌ Failed"
        t = str(timedelta(seconds=int(r["elapsed"])))
        acc_str = f", Acc={r['final_accuracy']:.4f}" if r.get("final_accuracy") else ""
        print(f"  {status} ({t}{acc_str})")

        if not r["success"]:
            print(f"  Check logs in {r['output_dir']}")

    total_time = time.time() - total_start

    # Summary / 汇总
    summary_path = os.path.join(PROJECT_ROOT, "results", "summary.csv")
    print_summary(results, summary_path)

    passed = sum(1 for r in results if r["success"])
    print(f"\n  {passed}/{len(results)} experiments completed successfully")
    print(f"  Total time: {timedelta(seconds=int(total_time))}")
    print("\n✅ All done!\n")


if __name__ == "__main__":
    main()
