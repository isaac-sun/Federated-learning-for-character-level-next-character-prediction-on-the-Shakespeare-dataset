"""
Centralized device resolution and GPU info utilities.
统一的设备解析与 GPU 信息工具模块。

Supports: auto / cpu / cuda / cuda:N / mps
支持：auto / cpu / cuda / cuda:N / mps
"""

import torch


def get_device(device_str: str) -> torch.device:
    """
    Resolve a device string to a torch.device.
    将设备字符串解析为 torch.device。

    Supported values / 支持的值:
        "auto"    – CUDA → MPS → CPU (recommended default / 推荐默认)
        "cuda"    – first NVIDIA GPU (equivalent to cuda:0)
        "cuda:N"  – specific NVIDIA GPU index N (e.g. "cuda:0", "cuda:1")
        "mps"     – Apple Silicon GPU (macOS only)
        "cpu"     – CPU only

    Args:
        device_str: Device identifier string. 设备标识字符串。

    Returns:
        torch.device: Resolved device. 解析后的计算设备。

    Raises:
        ValueError: If the requested device is not available.
                    如果请求的设备不可用则抛出。
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # Handle cuda:N explicitly / 处理 cuda:N 格式
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA is not available on this machine. "
                "请检查 NVIDIA 驱动和 PyTorch CUDA 版本是否匹配。"
            )
        return torch.device(device_str)

    if device_str == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError(
                "MPS is not available. Requires macOS 12.3+ with Apple Silicon. "
                "MPS 不可用，需要 macOS 12.3+ 和 Apple Silicon。"
            )
        return torch.device("mps")

    if device_str == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unknown device '{device_str}'. "
        f"Use: auto / cpu / cuda / cuda:N / mps"
    )


def get_device_info(device: torch.device) -> dict:
    """
    Return a dictionary of device information for logging.
    返回用于日志记录的设备信息字典。
    """
    info: dict = {"device": str(device)}

    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        info["gpu_name"] = props.name
        info["vram_gb"] = round(props.total_memory / 1024 ** 3, 1)
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
    elif device.type == "mps":
        info["gpu_name"] = "Apple Silicon (MPS)"
        info["platform"] = "macOS"
    else:
        info["gpu_name"] = "N/A (CPU only)"

    return info


def print_device_info(device: torch.device) -> None:
    """
    Print a formatted device info block at startup.
    在启动时打印格式化的设备信息块。
    """
    info = get_device_info(device)
    print("\n  🖥️  Device / 计算设备")
    print(f"     Type    : {info['device']}")
    print(f"     GPU     : {info.get('gpu_name', 'N/A')}")
    if "vram_gb" in info:
        print(f"     VRAM    : {info['vram_gb']} GB")
    if "cuda_version" in info:
        print(f"     CUDA    : {info['cuda_version']}")
    if "gpu_count" in info and info["gpu_count"] > 1:
        print(f"     GPUs    : {info['gpu_count']} available (using {info['device']})")
    print()
