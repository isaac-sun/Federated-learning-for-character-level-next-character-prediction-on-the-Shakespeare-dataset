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
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 5 rounds, 10 MC samples (for testing). "
             "快速模式：5轮，10次MC采样（用于测试）。",
    )
    parser.add_argument(
        "--num-rounds", type=int, default=None,
        help="Override number of rounds. 覆盖轮数。",
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
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Compute device (default: auto). 计算设备。",
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
        mc_samples = 10
        mode_label = "Quick"
    else:
        num_rounds = args.num_rounds or 25
        mc_samples = 50
        mode_label = "Full"

    print("=" * 70)
    print(f"  🎭 FedBard — {mode_label} Experiment Suite")
    print("=" * 70)
    print(f"  Experiments to run:  {len(experiments)}")
    print(f"  Rounds per run:     {num_rounds}")
    print(f"  Shapley MC samples: {mc_samples}")
    print(f"  Device:             {args.device}")
    print(f"  Seed:               {args.seed}")
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
