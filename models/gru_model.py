"""
GRU-based character-level language model for next-character prediction.
基于GRU的字符级语言模型，用于下一个字符预测。

The paper "Robust and Fair Federated Learning Based on Model-Agnostic
Shapley Value" uses GRU (not LSTM) for the Shakespeare task.
论文使用GRU（而非LSTM）用于莎士比亚任务。

Architecture: Embedding → GRU → Dropout → Linear
架构：嵌入层 → GRU → Dropout → 全连接层
"""

import torch
import torch.nn as nn


class CharGRU(nn.Module):
    """
    Character-level GRU language model for next-character prediction.
    用于下一个字符预测的字符级GRU语言模型。

    Same interface as CharLSTM: forward(x, hidden) -> (logits, hidden).
    GRU hidden state is a single tensor h (not (h, c) like LSTM).
    与CharLSTM接口相同：forward(x, hidden) -> (logits, hidden)。
    GRU隐藏状态是单个张量h（不像LSTM的(h, c)）。
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        """
        Initialize the CharGRU model.
        初始化CharGRU模型。

        Args:
            vocab_size: Size of the character vocabulary. 字符词汇表大小。
            embed_dim: Embedding dimension (default: 64). 嵌入维度。
            hidden_dim: GRU hidden dimension (default: 128). GRU隐藏维度。
            num_layers: Number of GRU layers (default: 2). GRU层数。
            dropout: Dropout probability (default: 0.3). Dropout概率。
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None) -> tuple:
        """
        Forward pass. Returns (logits, hidden_state).
        前向传播。返回(logits, hidden_state)。

        Args:
            x: Input indices, shape (batch, seq_len). 输入索引。
            hidden: Optional GRU hidden state, shape (num_layers, batch, hidden_dim).
                    可选的GRU隐藏状态。

        Returns:
            (logits, hidden): logits shape (batch, seq_len, vocab_size).
        """
        embeds = self.embedding(x)
        if hidden is not None:
            gru_out, hidden = self.gru(embeds, hidden)
        else:
            gru_out, hidden = self.gru(embeds)
        out = self.dropout(gru_out)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize GRU hidden state with zeros.
        用零初始化GRU隐藏状态。
        """
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, device=device
        )
