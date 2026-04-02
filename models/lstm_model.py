"""
LSTM-based character-level language model for next-character prediction.
基于LSTM的字符级语言模型，用于下一个字符预测。

Architecture: Embedding → LSTM → Dropout → Linear
架构：嵌入层 → LSTM → Dropout → 全连接层

The model takes a sequence of character indices as input and predicts
the next character at each position in the sequence.
模型接收字符索引序列作为输入，预测序列中每个位置的下一个字符。
"""

import torch
import torch.nn as nn


class CharLSTM(nn.Module):
    """
    Character-level LSTM language model for next-character prediction.
    用于下一个字符预测的字符级LSTM语言模型。

    Layers:
      1. Embedding: maps character indices to dense vectors
      2. LSTM: processes the embedded sequence
      3. Dropout: regularization before the output layer
      4. Linear: projects LSTM output to vocabulary logits

    层结构：
      1. 嵌入层：将字符索引映射为稠密向量
      2. LSTM层：处理嵌入后的序列
      3. Dropout层：输出层前的正则化
      4. 全连接层：将LSTM输出投影到词汇表logits
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
        Initialize the CharLSTM model.
        初始化CharLSTM模型。

        Args:
            vocab_size: Size of the character vocabulary (including special tokens).
                        字符词汇表大小（包括特殊标记）。
            embed_dim: Dimension of character embeddings (default: 64).
                       字符嵌入维度（默认：64）。
            hidden_dim: LSTM hidden state dimension (default: 128).
                        LSTM隐藏状态维度（默认：128）。
            num_layers: Number of stacked LSTM layers (default: 2).
                        堆叠的LSTM层数（默认：2）。
            dropout: Dropout probability (default: 0.3).
                     Dropout概率（默认：0.3）。
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer: character index → dense vector
        # 嵌入层：字符索引 → 稠密向量
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,  # PAD token at index 0 / PAD标记在索引0
        )

        # Stacked LSTM layers
        # 堆叠LSTM层
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Dropout for regularization before the output projection
        # 输出投影前的Dropout正则化
        self.dropout = nn.Dropout(dropout)

        # Output projection: hidden state → vocabulary logits
        # 输出投影：隐藏状态 → 词汇表logits
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple = None,
    ) -> tuple:
        """
        Forward pass through the model.
        模型的前向传播。

        Args:
            x: Input tensor of shape (batch_size, seq_length) containing
               character indices.
               形状为(batch_size, seq_length)的输入张量，包含字符索引。
            hidden: Optional tuple (h_0, c_0) for initial LSTM hidden state.
                    If None, LSTM uses zeros.
                    可选的(h_0, c_0)元组作为LSTM初始隐藏状态。
                    如果为None，LSTM使用零值。

        Returns:
            Tuple of (logits, hidden_state):
              - logits: shape (batch_size, seq_length, vocab_size)
              - hidden_state: (h_n, c_n) tuple for the final hidden state
            返回元组(logits, hidden_state)：
              - logits：形状为(batch_size, seq_length, vocab_size)
              - hidden_state：最终隐藏状态的(h_n, c_n)元组
        """
        # Embed character indices → (batch, seq_len, embed_dim)
        # 嵌入字符索引 → (batch, seq_len, embed_dim)
        embeds = self.embedding(x)

        # LSTM processes the sequence → (batch, seq_len, hidden_dim)
        # LSTM处理序列 → (batch, seq_len, hidden_dim)
        if hidden is not None:
            lstm_out, hidden = self.lstm(embeds, hidden)
        else:
            lstm_out, hidden = self.lstm(embeds)

        # Apply dropout and project to vocabulary logits
        # 应用Dropout并投影到词汇表logits
        out = self.dropout(lstm_out)
        logits = self.fc(out)  # (batch, seq_len, vocab_size)

        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> tuple:
        """
        Initialize LSTM hidden state with zeros.
        用零值初始化LSTM隐藏状态。

        Args:
            batch_size: Number of sequences in the batch. 批次中的序列数。
            device: Torch device for the tensors. 张量的Torch设备。

        Returns:
            Tuple (h_0, c_0) of zero tensors, each of shape
            (num_layers, batch_size, hidden_dim).
            零张量的元组(h_0, c_0)，每个形状为
            (num_layers, batch_size, hidden_dim)。
        """
        h0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, device=device
        )
        c0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, device=device
        )
        return (h0, c0)
