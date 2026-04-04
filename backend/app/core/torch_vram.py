"""
CUDA VRAM helpers for post-inference logging.

Log current and peak allocated VRAM on CUDA after an inference step.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from loguru import Logger


def reset_cuda_peak_stats(device: "torch.device") -> None:
    if device.type == "cuda":
        import torch as _torch

        _torch.cuda.reset_peak_memory_stats(device)


def log_cuda_inference_vram(logger: "Logger", model_label: str, device: "torch.device") -> None:
    if device.type != "cuda":
        return
    import torch as _torch

    current_gb = _torch.cuda.memory_allocated(device) / (1024**3)
    peak_gb = _torch.cuda.max_memory_allocated(device) / (1024**3)
    logger.info(
        "[{}] Inference complete. Current VRAM: {:.2f}GB, Peak VRAM: {:.2f}GB",
        model_label,
        current_gb,
        peak_gb,
    )
