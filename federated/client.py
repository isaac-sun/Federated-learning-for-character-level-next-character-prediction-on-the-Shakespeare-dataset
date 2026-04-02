"""
Flower NumPyClient implementation for federated Shakespeare language modeling.
用于联邦莎士比亚语言建模的Flower NumPyClient实现。

Each client represents a single Shakespeare character and trains
a local LSTM model on that character's dialogue text.
每个客户端代表一个莎士比亚角色，在该角色的对话文本上训练本地LSTM模型。
"""

import logging
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class ShakespeareClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for character-level Shakespeare language modeling.
    用于字符级莎士比亚语言建模的Flower NumPyClient。

    Implements the standard Flower client interface:
      - get_parameters(): return model weights as numpy arrays
      - set_parameters(): load model weights from numpy arrays
      - fit(): train locally and return updated weights
      - evaluate(): evaluate on local test data

    实现标准的Flower客户端接口：
      - get_parameters(): 将模型权重作为numpy数组返回
      - set_parameters(): 从numpy数组加载模型权重
      - fit(): 本地训练并返回更新后的权重
      - evaluate(): 在本地测试数据上评估
    """

    def __init__(
        self,
        cid: str,
        model: nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
        device: torch.device,
        local_epochs: int = 1,
        batch_size: int = 16,
        learning_rate: float = 0.001,
    ):
        """
        Initialize a federated Shakespeare client.
        初始化联邦莎士比亚客户端。

        Args:
            cid: Client identifier (character name, e.g., "ROMEO").
                 客户端标识符（角色名，如"ROMEO"）。
            model: CharLSTM model instance (freshly initialized).
                   CharLSTM模型实例（新初始化的）。
            train_dataset: Local training dataset. 本地训练数据集。
            test_dataset: Local test dataset. 本地测试数据集。
            device: Torch compute device. Torch计算设备。
            local_epochs: Number of local training epochs per FL round.
                          每轮联邦训练的本地epoch数。
            batch_size: Mini-batch size for local training. 本地训练的mini-batch大小。
            learning_rate: Learning rate for local SGD/Adam optimizer.
                           本地SGD/Adam优化器的学习率。
        """
        super().__init__()
        self.cid = cid
        self.model = model
        self.device = device
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Create data loaders / 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config: Dict = None) -> List[np.ndarray]:
        """
        Return current model parameters as a list of numpy arrays.
        将当前模型参数作为numpy数组列表返回。

        This is called by the Flower framework to retrieve model weights
        for aggregation on the server.
        Flower框架调用此方法获取模型权重以在服务器端聚合。

        Args:
            config: Optional configuration dict (unused).
                    可选配置字典（未使用）。

        Returns:
            List of numpy arrays, one per model parameter tensor.
            numpy数组列表，每个模型参数张量一个。
        """
        return [
            val.cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from a list of numpy arrays.
        从numpy数组列表设置模型参数。

        This loads the global model weights received from the server
        into the local model before training or evaluation.
        在训练或评估前将从服务器接收的全局模型权重加载到本地模型中。

        Args:
            parameters: List of numpy arrays matching the model's state_dict order.
                        与模型state_dict顺序匹配的numpy数组列表。
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(np.copy(v)) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model on local data for one or more epochs.
        在本地数据上训练模型一个或多个epoch。

        Workflow:
          1. Load global parameters into local model
          2. Train on local data for `local_epochs` epochs
          3. Return updated parameters, sample count, and metrics

        工作流程：
          1. 将全局参数加载到本地模型
          2. 在本地数据上训练local_epochs个epoch
          3. 返回更新后的参数、样本数和度量

        Args:
            parameters: Global model parameters from server.
                        来自服务器的全局模型参数。
            config: Training configuration from server (e.g., epochs, lr).
                    来自服务器的训练配置。

        Returns:
            Tuple of (updated_parameters, num_training_samples, metrics_dict).
            返回元组(更新后的参数, 训练样本数, 度量字典)。
        """
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        total_loss = 0.0
        total_tokens = 0

        for epoch in range(self.local_epochs):
            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                logits, _ = self.model(inputs)

                # Flatten for cross-entropy: (N*seq_len, vocab) vs (N*seq_len,)
                # 展平用于交叉熵计算
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )
                loss.backward()

                # Gradient clipping to prevent exploding gradients in LSTM
                # 梯度裁剪，防止LSTM中的梯度爆炸
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=5.0
                )
                optimizer.step()

                batch_tokens = targets.numel()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        num_samples = len(self.train_loader.dataset)

        return (
            self.get_parameters(config={}),
            num_samples,
            {"train_loss": float(avg_loss)},
        )

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on local test data.
        在本地测试数据上评估模型。

        Computes cross-entropy loss, perplexity, and accuracy on the
        client's held-out test set.
        在客户端的保留测试集上计算交叉熵损失、困惑度和准确率。

        Args:
            parameters: Global model parameters to evaluate.
                        要评估的全局模型参数。
            config: Evaluation configuration (unused).
                    评估配置（未使用）。

        Returns:
            Tuple of (loss, num_test_samples, metrics_dict).
            返回元组(损失, 测试样本数, 度量字典)。
        """
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0
        correct = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                logits, _ = self.model(inputs)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )

                batch_tokens = targets.numel()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

                # Character-level accuracy / 字符级准确率
                predictions = logits.argmax(dim=-1)
                correct += (predictions == targets).sum().item()

        avg_loss = total_loss / max(total_tokens, 1)
        accuracy = correct / max(total_tokens, 1)
        num_samples = len(self.test_loader.dataset)

        return (
            float(avg_loss),
            num_samples,
            {
                "accuracy": float(accuracy),
                "perplexity": float(np.exp(min(avg_loss, 50))),
            },
        )
