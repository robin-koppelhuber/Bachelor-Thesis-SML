"""Device management utilities"""

import logging

import torch

logger = logging.getLogger(__name__)


def get_device(device: str = "auto") -> torch.device:
    """
    Get PyTorch device

    Args:
        device: Device string ("auto", "cpu", "cuda", "xpu")

    Returns:
        PyTorch device object
    """
    if device == "auto":
        if torch.cuda.is_available():
            device_obj = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device_obj = torch.device("xpu")
            logger.info("Using Intel XPU device")
        else:
            device_obj = torch.device("cpu")
            logger.info("Using CPU device")
    else:
        device_obj = torch.device(device)
        logger.info(f"Using specified device: {device}")

    return device_obj


def get_device_info() -> dict:
    """
    Get information about available devices

    Returns:
        Dictionary with device information
    """
    info = {
        "cpu": True,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "xpu_available": hasattr(torch, "xpu") and torch.xpu.is_available(),
    }

    if info["cuda_available"]:
        info["cuda_devices"] = [
            torch.cuda.get_device_name(i) for i in range(info["cuda_device_count"])
        ]

    return info
