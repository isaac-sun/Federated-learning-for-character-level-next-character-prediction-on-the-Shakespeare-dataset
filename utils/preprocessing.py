"""
Character-level text preprocessing utilities for Shakespeare dataset.
莎士比亚数据集的字符级文本预处理工具。

Provides vocabulary building, character encoding/decoding, and
sequence generation for next-character prediction tasks.
提供词汇表构建、字符编码/解码以及用于下一个字符预测任务的序列生成功能。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# Default character set covering printable ASCII
# 默认字符集，涵盖可打印的ASCII字符
ALL_CHARS = (
    "\n !\"#$%&'()*+,-./0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
)


class CharVocab:
    """
    Character-level vocabulary with encode/decode capabilities.
    具有编码/解码功能的字符级词汇表。

    Builds a mapping from characters to integer indices and vice versa,
    with special PAD and UNK tokens at indices 0 and 1.
    构建字符到整数索引及其反向映射，PAD和UNK特殊标记分别在索引0和1。
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, chars: Optional[str] = None):
        """
        Build vocabulary from a set of characters.
        从字符集构建词汇表。

        Args:
            chars: Characters to include in the vocabulary.
                   If None, uses the default printable ASCII set.
                   要包含在词汇表中的字符。
                   如果为None，使用默认的可打印ASCII字符集。
        """
        if chars is None:
            chars = ALL_CHARS

        # Deduplicate and sort characters for deterministic ordering
        # 去重并排序字符以确保确定性顺序
        self.chars = sorted(set(chars))

        # Build char → index mapping with special tokens
        # 构建字符→索引映射（含特殊标记）
        self.char2idx: Dict[str, int] = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        for i, ch in enumerate(self.chars):
            self.char2idx[ch] = i + 2

        # Build reverse mapping: index → char
        # 构建反向映射：索引→字符
        self.idx2char: Dict[int, str] = {v: k for k, v in self.char2idx.items()}

        self.vocab_size: int = len(self.char2idx)

    def encode(self, text: str) -> List[int]:
        """
        Encode a string into a list of integer indices.
        将字符串编码为整数索引列表。

        Args:
            text: Input text to encode. 要编码的输入文本。

        Returns:
            List of integer indices. 整数索引列表。
        """
        unk = self.char2idx[self.UNK_TOKEN]
        return [self.char2idx.get(ch, unk) for ch in text]

    def decode(self, indices: List[int]) -> str:
        """
        Decode a list of integer indices back to a string.
        将整数索引列表解码回字符串。

        Args:
            indices: List of integer indices. 整数索引列表。

        Returns:
            Decoded string. 解码后的字符串。
        """
        return "".join(self.idx2char.get(i, self.UNK_TOKEN) for i in indices)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"CharVocab(vocab_size={self.vocab_size})"


def create_sequences(
    text: str,
    vocab: CharVocab,
    seq_length: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate input-target sequence pairs for next-character prediction.
    为下一个字符预测生成输入-目标序列对。

    Uses a sliding window approach: for each position i, the input is
    text[i:i+seq_length] and the target is text[i+1:i+seq_length+1].
    使用滑动窗口方法：对于每个位置i，输入为text[i:i+seq_length]，
    目标为text[i+1:i+seq_length+1]。

    Args:
        text: Raw text string to generate sequences from.
              用于生成序列的原始文本字符串。
        vocab: Character vocabulary for encoding.
               用于编码的字符词汇表。
        seq_length: Length of each input/target sequence (default: 50).
                    每个输入/目标序列的长度（默认：50）。

    Returns:
        Tuple of (inputs, targets) as numpy arrays of shape (N, seq_length).
        返回形状为(N, seq_length)的numpy数组元组(输入, 目标)。
    """
    encoded = vocab.encode(text)

    # Need at least seq_length+1 characters to form one pair
    # 至少需要seq_length+1个字符才能形成一对
    if len(encoded) <= seq_length:
        return (
            np.array([], dtype=np.int64).reshape(0, seq_length),
            np.array([], dtype=np.int64).reshape(0, seq_length),
        )

    num_sequences = len(encoded) - seq_length
    inputs = np.zeros((num_sequences, seq_length), dtype=np.int64)
    targets = np.zeros((num_sequences, seq_length), dtype=np.int64)

    for i in range(num_sequences):
        inputs[i] = encoded[i : i + seq_length]
        targets[i] = encoded[i + 1 : i + seq_length + 1]

    return inputs, targets
