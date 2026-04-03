"""
Training configurations for all modules.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Tuple

_IS_WINDOWS = sys.platform == 'win32'


def _optimal_workers() -> int:
    """
    Return a sensible default num_workers based on available CPU cores.
    """

    try:
        cpus = os.cpu_count() or 4
        cap = 4 if _IS_WINDOWS else 16
        return min(cpus, cap)
    except Exception:
        return 4

@dataclass
class PerformanceConfig:
    """
    Hardware-aware performance knobs applied to every training run.
    """

    num_workers: int = field(default_factory=_optimal_workers)
    pin_memory: bool = True
    persistent_workers: bool = not _IS_WINDOWS
    prefetch_factor: int = 2 if _IS_WINDOWS else 3
    cudnn_benchmark: bool = True
    use_amp: bool = True
    compile_model: bool = False
    compile_backend: str = 'inductor'


# Label convention (shared across all modules)
FAKE_LABEL = 0
REAL_LABEL = 1
DECISION_THRESHOLD = 0.5


@dataclass
class CLIPStreamConfig:
    """
    Hyperparameters for the Spatial Stream in the Image Detection Module.
    """

    clip_model_name: str = 'openai/clip-vit-large-patch14'
    num_classes: int = 2
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 8e-5
    weight_decay: float = 0.0
    scheduler: str = 'cosine'
    scheduler_t_max: int = 50
    scheduler_eta_min: float = 5e-5
    early_stopping_patience: int = 10
    early_stopping_mode: str = 'max'
    num_workers: int = field(default_factory=_optimal_workers)
    crop_size: int = 224
    perf: PerformanceConfig = field(default_factory=PerformanceConfig)


@dataclass
class NoiseStreamConfig:
    """
    Hyperparameters for the Noise Stream in the Image Detection Module.
    """

    epochs: int = 50
    batch_size: int = 64
    early_stopping_patience: int = 10
    early_stopping_mode: str = 'max'
    label_smoothing: float = 0.05
    weight_decay: float = 0.05
    num_workers: int = field(default_factory=_optimal_workers)
    noise_truncation: Tuple[float, float] = (-3.0, 3.0)
    lr_early_features: float = 1e-4
    lr_late_features: float = 5e-5
    lr_classifier: float = 1e-4
    scheduler: str = 'cosine_warm_restarts'
    scheduler_t0: int = 10
    scheduler_t_mult: int = 2
    scheduler_eta_min: float = 1e-6
    perf: PerformanceConfig = field(default_factory=PerformanceConfig)


@dataclass
class ImageFusionConfig:
    """
    Hyperparameters for the two stream fusion in the Image Detection Module.
    """

    initial_clip_weight: float = 0.55
    freeze_streams: bool = True


@dataclass
class AudioTrainConfig:
    """
    Hyperparameters for the Audio Detection Module.
    """

    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    scheduler: str = 'reduce_on_plateau'
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    early_stopping_patience: int = 5
    early_stopping_mode: str = 'max'
    num_workers: int = field(default_factory=_optimal_workers)
    target_sr: int = 16000
    target_duration: float = 4.0
    n_mels: int = 128
    n_lfcc: int = 128
    n_fft: int = 1024
    hop_length: int = 512
    perf: PerformanceConfig = field(default_factory=PerformanceConfig)


@dataclass
class VideoTrainConfig:
    """
    Hyperparameters for the Video Detection Module.
    """

    epochs: int = 30
    batch_size: int = 4
    weight_decay: float = 1e-2
    scheduler: str = 'cosine'
    scheduler_t_max: int = 30
    scheduler_eta_min: float = 1e-6
    early_stopping_patience: int = 10
    early_stopping_mode: str = 'max'
    grad_clip_max_norm: float = 1.0
    num_workers: int = field(default_factory=_optimal_workers)
    num_frames: int = 15
    lr_sync_stream: float = 5e-5
    lr_fusion: float = 1e-4
    lr_classifier: float = 1e-4
    perf: PerformanceConfig = field(default_factory=PerformanceConfig)
