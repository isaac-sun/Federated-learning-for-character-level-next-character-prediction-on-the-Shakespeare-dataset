"""
Shakespeare dataset downloader and loader for federated learning.
用于联邦学习的莎士比亚数据集下载器和加载器。

Data loading strategy (with automatic fallback):
  1. Check for cached TinyShakespeare file on disk
  2. Download TinyShakespeare from GitHub and parse by character
  3. Fall back to synthetic Shakespeare-like data

数据加载策略（自动回退）：
  1. 检查磁盘上缓存的TinyShakespeare文件
  2. 从GitHub下载TinyShakespeare并按角色解析
  3. 回退到合成的类莎士比亚数据

Each Shakespeare character (e.g., ROMEO, JULIET) becomes one federated client.
每个莎士比亚角色（如ROMEO、JULIET）成为一个联邦学习客户端。
"""

import json
import logging
import os
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# TinyShakespeare: a widely-used Shakespeare corpus from Karpathy's char-rnn
# TinyShakespeare：来自Karpathy的char-rnn的广泛使用的莎士比亚语料库
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

# Default cache directory (inside the data/ folder)
# 默认缓存目录（在data/文件夹内）
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")


class ShakespeareDataset(Dataset):
    """
    PyTorch Dataset for character-level Shakespeare text sequences.
    用于字符级莎士比亚文本序列的PyTorch数据集。

    Each sample is an (input_sequence, target_sequence) pair where
    the target is the input shifted by one character.
    每个样本是一个(输入序列, 目标序列)对，目标是输入向右移动一个字符。
    """

    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        """
        Args:
            inputs: Integer-encoded input sequences, shape (N, seq_len).
                    整数编码的输入序列，形状为(N, seq_len)。
            targets: Integer-encoded target sequences, shape (N, seq_len).
                     整数编码的目标序列，形状为(N, seq_len)。
        """
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def download_file(url: str, dest: str, timeout: int = 120) -> bool:
    """
    Download a file from a URL to a local path.
    从URL下载文件到本地路径。

    Args:
        url: Source URL to download from. 要下载的源URL。
        dest: Destination file path. 目标文件路径。
        timeout: HTTP request timeout in seconds. HTTP请求超时（秒）。

    Returns:
        True if download succeeded, False otherwise.
        下载成功返回True，否则返回False。
    """
    try:
        logger.info(f"Downloading from {url} ...")
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        file_size = os.path.getsize(dest)
        logger.info(f"Downloaded successfully: {dest} ({file_size:,} bytes)")
        return True

    except Exception as e:
        logger.warning(f"Download failed: {e}")
        return False


def parse_shakespeare_by_character(text: str) -> Dict[str, str]:
    """
    Parse Shakespeare play text into per-character dialogue.
    将莎士比亚戏剧文本按角色解析为对话。

    Detects character names as lines matching the pattern "NAME:"
    (capitalized words followed by a colon) and assigns subsequent
    lines as that character's dialogue.
    检测格式为"名字:"的行（大写单词后跟冒号）作为角色名，
    后续行作为该角色的对话。

    Args:
        text: Full Shakespeare play text. 完整的莎士比亚戏剧文本。

    Returns:
        Dict mapping character names to their concatenated dialogue text.
        将角色名映射到其拼接对话文本的字典。
    """
    # Character header pattern: e.g., "ROMEO:", "First Citizen:", "KING HENRY:"
    # 角色标题模式：如 "ROMEO:", "First Citizen:", "KING HENRY:"
    character_pattern = re.compile(r"^([A-Z][A-Za-z ]+):\s*$")

    character_data: Dict[str, List[str]] = {}
    current_character: Optional[str] = None

    for line in text.split("\n"):
        stripped = line.strip()
        match = character_pattern.match(stripped)

        if match:
            current_character = match.group(1).strip()
            if current_character not in character_data:
                character_data[current_character] = []
        elif current_character is not None and stripped:
            character_data[current_character].append(stripped)

    # Concatenate each character's lines and filter out those with too little text
    # 拼接每个角色的台词并过滤掉文本过少的角色
    MIN_TEXT_LENGTH = 100  # Minimum characters to be a viable client / 作为可行客户端的最少字符数
    result = {}
    for name, lines in character_data.items():
        joined = "\n".join(lines)
        if len(joined) >= MIN_TEXT_LENGTH:
            result[name] = joined

    logger.info(
        f"Parsed {len(result)} characters with sufficient text "
        f"(filtered from {len(character_data)} total)"
    )
    return result


def generate_fallback_data(num_clients: int = 20) -> Dict[str, str]:
    """
    Generate synthetic Shakespeare-style data as a final fallback.
    生成合成的莎士比亚风格数据作为最终备选。

    Used when real data cannot be downloaded. Produces enough text
    per client for meaningful training sequences.
    当无法下载真实数据时使用。为每个客户端生成足够的训练序列文本。

    Args:
        num_clients: Number of synthetic clients to generate.
                     要生成的合成客户端数量。

    Returns:
        Dict mapping character names to synthetic dialogue text.
        将角色名映射到合成对话文本的字典。
    """
    logger.warning("Using synthetic fallback data / 使用合成备选数据")

    passages = [
        (
            "To be, or not to be, that is the question:\n"
            "Whether 'tis nobler in the mind to suffer\n"
            "The slings and arrows of outrageous fortune,\n"
            "Or to take arms against a sea of troubles,\n"
            "And by opposing end them. To die, to sleep;\n"
            "No more; and by a sleep to say we end\n"
            "The heart-ache and the thousand natural shocks\n"
            "That flesh is heir to: 'tis a consummation\n"
            "Devoutly to be wish'd. To die, to sleep;\n"
            "To sleep, perchance to dream.\n"
        ),
        (
            "All the world's a stage,\n"
            "And all the men and women merely players;\n"
            "They have their exits and their entrances,\n"
            "And one man in his time plays many parts,\n"
            "His acts being seven ages. At first the infant,\n"
            "Mewling and puking in the nurse's arms.\n"
        ),
        (
            "If music be the food of love, play on,\n"
            "Give me excess of it; that surfeiting,\n"
            "The appetite may sicken, and so die.\n"
            "That strain again, it had a dying fall;\n"
            "O, it came o'er my ear like the sweet sound\n"
            "That breathes upon a bank of violets,\n"
            "Stealing and giving odour. Enough, no more;\n"
        ),
        (
            "But soft, what light through yonder window breaks?\n"
            "It is the east, and Juliet is the sun.\n"
            "Arise, fair sun, and kill the envious moon,\n"
            "Who is already sick and pale with grief\n"
            "That thou, her maid, art far more fair than she.\n"
            "Be not her maid since she is envious;\n"
            "Her vestal livery is but sick and green.\n"
        ),
        (
            "Friends, Romans, countrymen, lend me your ears;\n"
            "I come to bury Caesar, not to praise him.\n"
            "The evil that men do lives after them;\n"
            "The good is oft interred with their bones.\n"
            "So let it be with Caesar. The noble Brutus\n"
            "Hath told you Caesar was ambitious.\n"
        ),
        (
            "Now is the winter of our discontent\n"
            "Made glorious summer by this sun of York;\n"
            "And all the clouds that lour'd upon our house\n"
            "In the deep bosom of the ocean buried.\n"
            "Now are our brows bound with victorious wreaths;\n"
        ),
        (
            "O Romeo, Romeo, wherefore art thou Romeo?\n"
            "Deny thy father and refuse thy name;\n"
            "Or if thou wilt not, be but sworn my love,\n"
            "And I'll no longer be a Capulet.\n"
            "'Tis but thy name that is my enemy;\n"
            "Thou art thyself, though not a Montague.\n"
        ),
        (
            "The quality of mercy is not strain'd;\n"
            "It droppeth as the gentle rain from heaven\n"
            "Upon the place beneath. It is twice blest:\n"
            "It blesseth him that gives and him that takes.\n"
            "'Tis mightiest in the mightiest; it becomes\n"
            "The throned monarch better than his crown.\n"
        ),
    ]

    names = [
        "HAMLET", "OTHELLO", "MACBETH", "ROMEO", "JULIET",
        "PROSPERO", "LEAR", "PORTIA", "VIOLA", "BEATRICE",
        "BENEDICK", "OBERON", "TITANIA", "PUCK", "ARIEL",
        "MIRANDA", "CORDELIA", "DESDEMONA", "OPHELIA", "HORATIO",
    ]

    rng = random.Random(42)
    result = {}
    for i in range(min(num_clients, len(names))):
        # Each client gets multiple repeated passages for sufficient text
        # 每个客户端获得多个重复段落以确保有足够的文本
        num_repeats = rng.randint(5, 12)
        text = "\n".join(rng.choice(passages) for _ in range(num_repeats))
        result[names[i]] = text

    return result


def load_shakespeare_data(
    data_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Load Shakespeare data with automatic download and fallback.
    加载莎士比亚数据，支持自动下载和回退。

    Strategy:
      1. Check for cached TinyShakespeare file on disk
      2. Download TinyShakespeare and parse by character
      3. Fall back to synthetic Shakespeare-style data

    策略：
      1. 检查磁盘上缓存的TinyShakespeare文件
      2. 下载TinyShakespeare并按角色解析
      3. 回退到合成的莎士比亚风格数据

    Args:
        data_dir: Directory to store/cache data files.
                  存储/缓存数据文件的目录。

    Returns:
        Dict mapping character names (client IDs) to their text data.
        将角色名（客户端ID）映射到其文本数据的字典。
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    os.makedirs(data_dir, exist_ok=True)
    text_path = os.path.join(data_dir, "tinyshakespeare.txt")

    # Step 1: Try loading from cache / 步骤1：尝试从缓存加载
    if os.path.exists(text_path):
        logger.info(f"Loading cached Shakespeare data from {text_path}")
        with open(text_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        client_data = parse_shakespeare_by_character(raw_text)
        if client_data:
            return client_data

    # Step 2: Download TinyShakespeare / 步骤2：下载TinyShakespeare
    logger.info("Attempting to download TinyShakespeare dataset...")
    if download_file(TINY_SHAKESPEARE_URL, text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        client_data = parse_shakespeare_by_character(raw_text)
        if client_data:
            return client_data

    # Step 3: Fallback to synthetic data / 步骤3：回退到合成数据
    logger.warning("All download attempts failed. Using synthetic data.")
    return generate_fallback_data()


def get_client_datasets(
    seq_length: int = 50,
    train_ratio: float = 0.8,
    data_dir: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, ShakespeareDataset]], "CharVocab"]:
    """
    Load Shakespeare data and create per-client PyTorch datasets.
    加载莎士比亚数据并创建每个客户端的PyTorch数据集。

    This is the main entry point for data loading. It:
      1. Downloads/loads the Shakespeare text
      2. Parses it into per-character data
      3. Builds a shared vocabulary
      4. Creates train/test datasets for each client

    这是数据加载的主入口。它：
      1. 下载/加载莎士比亚文本
      2. 按角色解析数据
      3. 构建共享词汇表
      4. 为每个客户端创建训练/测试数据集

    Args:
        seq_length: Length of input/target sequences (default: 50).
                    输入/目标序列的长度（默认：50）。
        train_ratio: Fraction of each client's data used for training (default: 0.8).
                     每个客户端用于训练的数据比例（默认：0.8）。
        data_dir: Directory for caching downloaded data.
                  缓存下载数据的目录。

    Returns:
        Tuple of:
          - Dict mapping client_id → {"train": ShakespeareDataset, "test": ShakespeareDataset}
          - CharVocab instance used for character encoding
        返回元组：
          - 将客户端ID映射到{"train": ShakespeareDataset, "test": ShakespeareDataset}的字典
          - 用于字符编码的CharVocab实例
    """
    # Import preprocessing utilities (train.py sets up sys.path)
    # 导入预处理工具（train.py已设置sys.path）
    from utils.preprocessing import CharVocab, create_sequences

    # Load raw per-character text data / 加载每个角色的原始文本数据
    character_texts = load_shakespeare_data(data_dir=data_dir)

    # Build a shared vocabulary from all available text
    # 从所有可用文本构建共享词汇表
    all_text = "\n".join(character_texts.values())
    vocab = CharVocab(chars=all_text)
    logger.info(f"Vocabulary size: {vocab.vocab_size}")

    # Create per-client train/test datasets
    # 创建每个客户端的训练/测试数据集
    client_datasets: Dict[str, Dict[str, ShakespeareDataset]] = {}

    for name, text in character_texts.items():
        inputs, targets = create_sequences(text, vocab, seq_length)

        if len(inputs) < 2:
            logger.debug(
                f"Skipping client '{name}': insufficient sequences ({len(inputs)})"
            )
            continue

        # Split into train and test sets / 划分为训练集和测试集
        n = len(inputs)
        split_idx = max(1, int(n * train_ratio))

        train_ds = ShakespeareDataset(inputs[:split_idx], targets[:split_idx])
        test_ds = ShakespeareDataset(inputs[split_idx:], targets[split_idx:])

        client_datasets[name] = {"train": train_ds, "test": test_ds}

    logger.info(f"Created datasets for {len(client_datasets)} clients")
    return client_datasets, vocab
