"""
Main training script for federated Shakespeare language model.
联邦莎士比亚语言模型的主训练脚本。

Orchestrates the full federated learning pipeline:
  1. Download and preprocess the Shakespeare dataset
  2. Create per-client train/test datasets
  3. Initialize the CharLSTM global model
  4. Run federated simulation with FedAvg
  5. Save metrics, plots, and the final global model

协调完整的联邦学习流程：
  1. 下载并预处理莎士比亚数据集
  2. 创建每个客户端的训练/测试数据集
  3. 初始化CharLSTM全局模型
  4. 运行使用FedAvg的联邦模拟
  5. 保存度量、图表和最终的全局模型

Usage / 用法:
    cd federated_shakespeare
    python experiments/train.py
    python experiments/train.py --num-rounds 50 --clients-per-round 10
    python experiments/train.py --help
"""

import argparse
import logging
import os
import sys
import time

# Add project root to Python path so all package imports work
# 将项目根目录添加到Python路径，使所有包导入正常工作
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from data.shakespeare_loader import get_client_datasets
from models.lstm_model import CharLSTM
from federated.client import ShakespeareClient
from federated.server import run_federated_simulation, set_model_parameters
from utils.metrics import MetricsLogger, plot_metrics
from utils.device import get_device, print_device_info


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for federated training.
    解析联邦训练的命令行参数。

    Returns:
        Parsed arguments namespace. 解析后的参数命名空间。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Federated Shakespeare Language Model Training\n"
            "联邦莎士比亚语言模型训练"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Federated Learning Parameters / 联邦学习参数 ----
    fl_group = parser.add_argument_group(
        "Federated Learning / 联邦学习"
    )
    fl_group.add_argument(
        "--num-rounds", type=int, default=25,
        help="Number of federated communication rounds (default: 25). "
             "联邦通信轮数（默认：25）。",
    )
    fl_group.add_argument(
        "--clients-per-round", type=int, default=5,
        help="Number of clients sampled per round (default: 5). "
             "每轮采样的客户端数（默认：5）。",
    )
    fl_group.add_argument(
        "--local-epochs", type=int, default=1,
        help="Local training epochs per round (default: 1). "
             "每轮本地训练epoch数（默认：1）。",
    )

    # ---- Model Architecture Parameters / 模型架构参数 ----
    model_group = parser.add_argument_group("Model / 模型")
    model_group.add_argument(
        "--embed-dim", type=int, default=64,
        help="Character embedding dimension (default: 64). "
             "字符嵌入维度（默认：64）。",
    )
    model_group.add_argument(
        "--hidden-dim", type=int, default=128,
        help="LSTM hidden state dimension (default: 128). "
             "LSTM隐藏状态维度（默认：128）。",
    )
    model_group.add_argument(
        "--num-layers", type=int, default=2,
        help="Number of LSTM layers (default: 2). LSTM层数（默认：2）。",
    )
    model_group.add_argument(
        "--dropout", type=float, default=0.3,
        help="Dropout rate (default: 0.3). Dropout率（默认：0.3）。",
    )

    # ---- Training Parameters / 训练参数 ----
    train_group = parser.add_argument_group("Training / 训练")
    train_group.add_argument(
        "--batch-size", type=int, default=16,
        help="Mini-batch size (default: 16). Mini-batch大小（默认：16）。",
    )
    train_group.add_argument(
        "--lr", type=float, default=0.001,
        help="Local learning rate (default: 0.001). 本地学习率（默认：0.001）。",
    )
    train_group.add_argument(
        "--seq-length", type=int, default=50,
        help="Input sequence length (default: 50). 输入序列长度（默认：50）。",
    )

    # ---- Data Parameters / 数据参数 ----
    data_group = parser.add_argument_group("Data / 数据")
    data_group.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory for data caching. 数据缓存目录。",
    )
    data_group.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="Train/test split ratio (default: 0.8). "
             "训练/测试划分比例（默认：0.8）。",
    )

    # ---- Output Parameters / 输出参数 ----
    out_group = parser.add_argument_group("Output / 输出")
    out_group.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for results output (default: results). "
             "结果输出目录（默认：results）。",
    )
    out_group.add_argument(
        "--save-model", action="store_true", default=True,
        help="Save the final global model checkpoint. "
             "保存最终的全局模型检查点。",
    )
    out_group.add_argument(
        "--no-save-model", action="store_false", dest="save_model",
        help="Do not save the final model. 不保存最终模型。",
    )

    # ---- Other Parameters / 其他参数 ----
    misc_group = parser.add_argument_group("Misc / 其他")
    misc_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42). "
             "随机种子，用于可重复性（默认：42）。",
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
        help="Logging verbosity (default: INFO). 日志详细程度（默认：INFO）。",
    )

    return parser.parse_args()


def main():
    """
    Main entry point: orchestrate the full federated training pipeline.
    主入口：协调完整的联邦训练流程。
    """
    args = parse_args()

    # ---- Logging Setup / 日志设置 ----
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("train")

    # ---- Reproducibility / 可重复性 ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    print_device_info(device)

    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ================================================================
    # Step 1: Load and Preprocess Data / 步骤1：加载和预处理数据
    # ================================================================
    print("\n📚 Loading Shakespeare dataset... / 加载莎士比亚数据集...")
    start_time = time.time()

    client_datasets, vocab = get_client_datasets(
        seq_length=args.seq_length,
        train_ratio=args.train_ratio,
        data_dir=args.data_dir,
    )

    load_time = time.time() - start_time
    client_ids = sorted(client_datasets.keys())

    print(f"   ✓ Loaded {len(client_ids)} clients in {load_time:.1f}s")
    print(f"   ✓ Vocabulary size: {vocab.vocab_size}")
    print(f"   ✓ Clients: {client_ids[:8]}{'...' if len(client_ids) > 8 else ''}")
    print()

    # Print per-client statistics / 打印每个客户端的统计信息
    for cid in client_ids[:5]:
        n_train = len(client_datasets[cid]["train"])
        n_test = len(client_datasets[cid]["test"])
        print(f"     {cid:20s}  train={n_train:5d}  test={n_test:5d}")
    if len(client_ids) > 5:
        print(f"     ... and {len(client_ids) - 5} more clients")

    # ================================================================
    # Step 2: Initialize Global Model / 步骤2：初始化全局模型
    # ================================================================
    print(f"\n🧠 Creating CharLSTM model... / 创建CharLSTM模型...")

    global_model = CharLSTM(
        vocab_size=vocab.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    total_params = sum(p.numel() for p in global_model.parameters())
    trainable_params = sum(
        p.numel() for p in global_model.parameters() if p.requires_grad
    )

    print(f"   ✓ Total parameters:     {total_params:,}")
    print(f"   ✓ Trainable parameters: {trainable_params:,}")
    print(
        f"   ✓ Architecture: "
        f"Embed({vocab.vocab_size}, {args.embed_dim}) → "
        f"LSTM({args.embed_dim}, {args.hidden_dim}, layers={args.num_layers}) → "
        f"Linear({args.hidden_dim}, {vocab.vocab_size})"
    )

    # ================================================================
    # Step 3: Define Client Factory / 步骤3：定义客户端工厂
    # ================================================================
    def client_fn(cid: str) -> ShakespeareClient:
        """
        Create a fresh Flower client for the given client ID.
        为给定的客户端ID创建一个新的Flower客户端。
        """
        # Each call creates a new model instance to avoid state leakage
        # 每次调用创建新的模型实例以避免状态泄漏
        model = CharLSTM(
            vocab_size=vocab.vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )

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
    # Step 4: Run Federated Simulation / 步骤4：运行联邦模拟
    # ================================================================
    print(f"\n🔁 Starting Federated Training (FedAvg)... / 开始联邦训练 (FedAvg)...")
    print(f"   Rounds / 轮次:                    {args.num_rounds}")
    print(f"   Clients per round / 每轮客户端数: {args.clients_per_round}")
    print(f"   Local epochs / 本地epoch数:       {args.local_epochs}")
    print(f"   Batch size / 批次大小:            {args.batch_size}")
    print(f"   Learning rate / 学习率:           {args.lr}")
    print()

    train_start = time.time()

    final_params, history = run_federated_simulation(
        client_fn=client_fn,
        client_ids=client_ids,
        global_model=global_model,
        num_rounds=args.num_rounds,
        clients_per_round=args.clients_per_round,
        seed=args.seed,
    )

    train_time = time.time() - train_start
    print(f"\n⏱  Training completed in {train_time:.1f}s / 训练完成")

    # ================================================================
    # Step 5: Save Results / 步骤5：保存结果
    # ================================================================

    # Save metrics to JSON / 保存度量到JSON
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

    # Generate training curve plots / 生成训练曲线图
    print("\n📈 Generating plots... / 生成图表...")
    plot_metrics(metrics_logger, save_dir=output_dir)

    # Save global model checkpoint / 保存全局模型检查点
    if args.save_model:
        set_model_parameters(global_model, final_params)

        model_path = os.path.join(output_dir, "global_model.pt")
        torch.save(
            {
                "model_state_dict": global_model.state_dict(),
                "vocab_size": vocab.vocab_size,
                "embed_dim": args.embed_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "char2idx": vocab.char2idx,
                "idx2char": {str(k): v for k, v in vocab.idx2char.items()},
                "training_config": {
                    "num_rounds": args.num_rounds,
                    "clients_per_round": args.clients_per_round,
                    "local_epochs": args.local_epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "seq_length": args.seq_length,
                    "seed": args.seed,
                },
            },
            model_path,
        )
        print(f"💾 Global model saved to {model_path}")

    # ================================================================
    # Final Summary / 最终总结
    # ================================================================
    final = metrics_logger.get_latest()
    print()
    print("=" * 60)
    print("  📊 Final Results / 最终结果")
    print("=" * 60)
    print(f"  Final Loss / 最终损失:         {final.get('loss', 0):.4f}")
    print(f"  Final Perplexity / 最终困惑度: {final.get('perplexity', 0):.2f}")
    print(f"  Final Accuracy / 最终准确率:   {final.get('accuracy', 0):.4f}")
    print(f"  Total Time / 总时间:           {train_time:.1f}s")
    print(f"  Output Directory / 输出目录:   {output_dir}")
    print("=" * 60)
    print("\n✅ Done! / 完成!\n")


if __name__ == "__main__":
    main()
