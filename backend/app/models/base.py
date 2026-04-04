"""
Abstract base class and result dataclass for all deepfake detectors (Image / Video / Audio).
"""

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from loguru import logger


def load_trusted_checkpoint(path: Union[str, Path], map_location: Any) -> Any:
    """
    Load a local .pth checkpoint.
    """

    p = Path(path)
    try:
        return torch.load(p, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError:
        logger.debug("weights_only load failed for {}; retrying with weights_only=False", p)
        return torch.load(p, map_location=map_location, weights_only=False)


@dataclass
class DetectionResult:
    """
    Detection result dataclass for Image / Video / Audio detectors.
    """

    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "processing_time": round(self.processing_time, 3),
            "metadata": self.metadata or {},
        }

    def __repr__(self) -> str:
        return f"DetectionResult(prediction={self.prediction}, confidence={self.confidence:.2%})"


class DeepfakeDetector(ABC):
    """
    Interface that all detector implementations (Image / Video / Audio) must follow.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.model_path = model_path
        self.device = torch.device(device)
        self.model: Optional[torch.nn.Module] = None
        self.detector_type: str = self.__class__.__name__.replace("Detector", "")
        self.is_loaded: bool = False
        logger.info(
            "[MODEL:{}] Initialising detector on device={}",
            self.detector_type.upper(),
            self.device,
        )

    @abstractmethod
    def load_model(self, model_path: Optional[str] = None) -> None: ...

    @abstractmethod
    def preprocess(self, input_data: Any) -> Any: ...

    @abstractmethod
    def predict(self, input_tensor: Any) -> DetectionResult: ...

    def detect(self, input_data: Any) -> DetectionResult:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        tag = self.detector_type.upper()
        logger.debug("[MODEL:{}] Starting preprocessing...", tag)
        input_tensor = self.preprocess(input_data)
        logger.debug("[MODEL:{}] Preprocessing complete, running inference...", tag)
        result = self.predict(input_tensor)

        logger.info(
            "[MODEL:{}] Detection result: prediction={}, confidence={:.2%}, "
            "probabilities=[Real={:.4f}, Deepfake={:.4f}], inference_time={:.3f}s",
            tag,
            result.prediction,
            result.confidence,
            result.probabilities.get("real", 0.0),
            result.probabilities.get("deepfake", 0.0),
            result.processing_time,
        )
        return result

    def get_model_info(self) -> Dict[str, Any]:
        if not self.is_loaded or self.model is None:
            return {"status": "not_loaded", "detector_type": self.detector_type}

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "status": "loaded",
            "detector_type": self.detector_type,
            "model_path": str(self.model_path),
            "device": str(self.device),
            "total_parameters": total,
            "trainable_parameters": trainable,
            "model_size_mb": total * 4 / (1024 * 1024),
        }

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            logger.info("[MODEL:{}] Model unloaded and memory released", self.detector_type.upper())

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return f"{self.__class__.__name__}(device={self.device}, status={status})"
