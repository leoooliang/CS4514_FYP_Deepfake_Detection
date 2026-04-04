"""
Image Detection Module with a dual-stream CLIP + SRM/EfficientNet architecture (Spatial Stream + Noise Stream).

Stream A: CLIP ViT-L/14 -> 512-D
Stream B: SRM/EfficientNet -> 512-D
Fusion:   weighted score-level fusion -> 1 logit (sigmoid)
"""

import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from loguru import logger

from app.config import settings
from app.core.exceptions import NoFaceDetectedError
from app.core.torch_vram import log_cuda_inference_vram, reset_cuda_peak_stats
from app.models.base import DeepfakeDetector, DetectionResult, load_trusted_checkpoint
from app.preprocessing_pipelines.image_preprocessor import ImagePreprocessor

try:
    from transformers import CLIPVisionModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("transformers library not available")


class CLIPClassifier(nn.Module):
    """
    Spatial Stream for Image Deepfake Detection with CLIP ViT-L/14 and LN-Tuning.
    """

    def __init__(self, clip_model_name="openai/clip-vit-large-patch14", num_classes=2):
        super().__init__()
        try:
            self.clip_vision = CLIPVisionModel.from_pretrained(
                clip_model_name, use_safetensors=True, attn_implementation="eager",
            )
        except TypeError:
            self.clip_vision = CLIPVisionModel.from_pretrained(clip_model_name, use_safetensors=True)
            warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

        for p in self.clip_vision.parameters():
            p.requires_grad = False
        for name, p in self.clip_vision.named_parameters():
            if "norm" in name.lower():
                p.requires_grad = True

        self.classifier = nn.Linear(self.clip_vision.config.hidden_size, num_classes)

    def forward(self, pixel_values, return_normalized_features=False):
        cls_token = self.clip_vision(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        normed = F.normalize(cls_token, p=2, dim=1)
        return normed if return_normalized_features else self.classifier(normed)


class SRMConv2d(nn.Module):
    """
    Fixed 5×5 SRM filter for noise residual extraction from images.
    """

    def __init__(self, in_channels=3):
        super().__init__()
        kernel = np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8,-12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1],
        ], dtype=np.float32) / 12.0
        kernels = np.tile(kernel[np.newaxis, np.newaxis, :, :], (in_channels, 1, 1, 1))
        self.conv = nn.Conv2d(in_channels, in_channels, 5, padding=2, bias=False, groups=in_channels)
        self.conv.weight.data = torch.from_numpy(kernels).float()
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)


class NoiseEfficientNet(nn.Module):
    """
    Noise Stream for Image Deepfake Detection with SRM and EfficientNetV2-S.
    """

    def __init__(self):
        super().__init__()
        self.efficientnet = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        self.efficientnet.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, 1))

    def forward(self, x):
        return self.efficientnet(x)


class ImageDetector(DeepfakeDetector):
    """
    Image detector wrapping CLIP + SRM/EfficientNet models.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        spatial_model_path: Optional[str] = None,
        noise_model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__(model_path, device)

        spatial = spatial_model_path if spatial_model_path is not None else settings.IMAGE_SPATIAL_MODEL_PATH
        noise = noise_model_path if noise_model_path is not None else settings.IMAGE_NOISE_MODEL_PATH
        self.clip_model_path = Path(spatial) if spatial else Path("models/best_clip.pth")
        self.noise_model_path = Path(noise) if noise else Path("models/best_noise_efficientnet.pth")
        self.model_path = f"{self.clip_model_path};{self.noise_model_path}"

        self.fusion_weight = 0.6
        self.preprocessor = ImagePreprocessor(device=str(device), target_size=(224, 224))
        self.srm_filter = SRMConv2d(in_channels=3)
        self.clip_model = None
        self.noise_model = None

    def load_model(self, model_path: Optional[str] = None) -> None:
        if not CLIP_AVAILABLE:
            raise RuntimeError("transformers library required -- pip install transformers")

        logger.info("[MODEL:IMAGE] Loading CLIP spatial stream...")
        self.clip_model = CLIPClassifier()
        if self.clip_model_path.exists():
            ckpt = load_trusted_checkpoint(self.clip_model_path, self.device)
            sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            self.clip_model.load_state_dict(sd, strict=True)
            logger.info("[MODEL:IMAGE] CLIP spatial stream loaded from {}", self.clip_model_path)
        else:
            logger.warning("[MODEL:IMAGE] CLIP checkpoint not found at {}, using default weights", self.clip_model_path)
        self.clip_model.eval().to(self.device)

        logger.info("[MODEL:IMAGE] Loading SRM/EfficientNet noise stream...")
        self.noise_model = NoiseEfficientNet()
        self.srm_filter.to(self.device)
        if self.noise_model_path.exists():
            ckpt = load_trusted_checkpoint(self.noise_model_path, self.device)
            sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            self.noise_model.load_state_dict(sd, strict=True)
            logger.info("[MODEL:IMAGE] Noise stream loaded from {}", self.noise_model_path)
        else:
            logger.warning("[MODEL:IMAGE] Noise checkpoint not found at {}, using default weights", self.noise_model_path)
        self.noise_model.eval().to(self.device)

        self.is_loaded = True
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        logger.info("[MODEL:IMAGE] Dual-stream model fully loaded (fusion_weight={})", self.fusion_weight)

    def get_model_info(self) -> Dict[str, Any]:
        if not self.is_loaded or self.clip_model is None or self.noise_model is None:
            return {
                "status": "not_loaded",
                "detector_type": self.detector_type,
                "spatial_model_path": str(self.clip_model_path),
                "noise_model_path": str(self.noise_model_path),
            }
        clip_params = sum(p.numel() for p in self.clip_model.parameters())
        noise_params = sum(p.numel() for p in self.noise_model.parameters())
        total = clip_params + noise_params
        trainable = (
            sum(p.numel() for p in self.clip_model.parameters() if p.requires_grad)
            + sum(p.numel() for p in self.noise_model.parameters() if p.requires_grad)
        )
        return {
            "status": "loaded",
            "detector_type": self.detector_type,
            "model_path": self.model_path,
            "spatial_model_path": str(self.clip_model_path),
            "noise_model_path": str(self.noise_model_path),
            "device": str(self.device),
            "total_parameters": total,
            "trainable_parameters": trainable,
            "model_size_mb": total * 4 / (1024 * 1024),
        }

    def unload_model(self) -> None:
        if self.clip_model is not None:
            del self.clip_model
            self.clip_model = None
        if self.noise_model is not None:
            del self.noise_model
            self.noise_model = None
        self.is_loaded = False
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("[MODEL:{}] Model unloaded and memory released", self.detector_type.upper())

    def preprocess(self, input_data: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            clip_t, noise_base = self.preprocessor.process(input_data)
            clip_t = clip_t.to(self.device)
            noise_base = noise_base.to(self.device)

            with torch.no_grad():
                noise_t = torch.clamp(self.srm_filter(noise_base), -3.0, 3.0) / 3.0
            return clip_t, noise_t
        except NoFaceDetectedError:
            raise
        except Exception as e:
            raise ValueError(f"Image preprocessing error: {e}") from e

    @torch.no_grad()
    def predict(self, input_tensor: Tuple[torch.Tensor, torch.Tensor]) -> DetectionResult:
        if not self.is_loaded:
            raise RuntimeError("Models not loaded")

        reset_cuda_peak_stats(self.device)
        start = time.time()
        clip_t, noise_t = input_tensor

        clip_probs = torch.softmax(self.clip_model(clip_t), dim=1)[0]
        clip_deepfake = clip_probs[0].item()

        noise_real = torch.sigmoid(self.noise_model(noise_t))[0].item()
        noise_deepfake = 1.0 - noise_real

        w = self.fusion_weight
        fused_deepfake = w * clip_deepfake + (1 - w) * noise_deepfake

        clip_real = clip_probs[1].item()
        logger.info(
            "[MODEL:IMAGE] Spatial stream (CLIP): Deepfake={:.4f}, Real={:.4f}",
            clip_deepfake,
            clip_real,
        )
        logger.info(
            "[MODEL:IMAGE] Noise stream (SRM/EfficientNet): Deepfake={:.4f}, Real={:.4f}",
            noise_deepfake,
            noise_real,
        )
        logger.info("[MODEL:IMAGE] Fused (w_clip={}): Deepfake={:.4f}", w, fused_deepfake)

        log_cuda_inference_vram(logger, "MODEL:IMAGE", self.device)

        return DetectionResult(
            prediction="Deepfake" if fused_deepfake > 0.5 else "Real",
            confidence=fused_deepfake,
            probabilities={"real": 1.0 - fused_deepfake, "deepfake": fused_deepfake},
            processing_time=time.time() - start,
            metadata={
                "model_type": "DualStreamFusion",
                "fusion_weight": {"clip": w, "noise": 1 - w},
                "stream_predictions": {
                    "clip": {"real": clip_real, "deepfake": clip_deepfake},
                    "noise": {"real": noise_real, "deepfake": noise_deepfake},
                },
                "device": str(self.device),
            },
        )
