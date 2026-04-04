"""
Image preprocessing pipeline for the Image Detection Module (CLIP + SRM Noise).

Pipeline:
    1. Load input -> RGB PIL Image
    2. MTCNN face detection -> crop with 15% margin (fail-fast if no face)
    3. Transform -> (CLIP tensor, SRM Noise tensor scaled to [0, 255])
"""

import io
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
from loguru import logger
from PIL import Image

from app.core.exceptions import NoFaceDetectedError

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    logger.warning("facenet-pytorch not available, face detection disabled")


class ImagePreprocessor:

    def __init__(
        self,
        device: str = "cpu",
        target_size: Tuple[int, int] = (224, 224),
        face_margin: float = 0.15,
        center_crop_ratio: float = 0.50,
    ):
        self.device = torch.device(device)
        self.target_size = target_size
        self.face_margin = face_margin
        self.center_crop_ratio = center_crop_ratio

        self.mtcnn = (
            MTCNN(keep_all=False, post_process=False, device=self.device)
            if MTCNN_AVAILABLE
            else None
        )

        self.clip_transform = T.Compose([
            T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        self.noise_base_transform = T.Compose([
            T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])

    def process(self, input_data: Union[str, bytes, np.ndarray, Image.Image]) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            image = self._load_image(input_data)
            face_crop = self._detect_and_crop_face(image)

            clip_tensor = self.clip_transform(face_crop).unsqueeze(0)
            noise_base_tensor = self.noise_base_transform(face_crop).unsqueeze(0) * 255.0

            return clip_tensor, noise_base_tensor
        except NoFaceDetectedError:
            raise
        except Exception as e:
            raise ValueError(f"Preprocessing error: {e}") from e

    def _load_image(self, data: Union[str, bytes, np.ndarray, Image.Image]) -> Image.Image:
        if isinstance(data, str):
            p = Path(data)
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {data}")
            return Image.open(p).convert("RGB")
        if isinstance(data, bytes):
            return Image.open(io.BytesIO(data)).convert("RGB")
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                data = np.stack([data] * 3, axis=-1)
            elif data.ndim == 3 and data.shape[2] == 4:
                data = data[:, :, :3]
            if data.dtype != np.uint8:
                data = (data * 255).astype(np.uint8) if data.max() <= 1.0 else data.astype(np.uint8)
            return Image.fromarray(data)
        if isinstance(data, Image.Image):
            return data.convert("RGB")
        raise ValueError(f"Unsupported input type: {type(data)}")

    def _detect_and_crop_face(self, image: Image.Image) -> Image.Image:
        if self.mtcnn is None:
            raise NoFaceDetectedError(
                user_message="Unable to process image.",
                technical_details="MTCNN not installed.",
            )
        try:
            boxes, probs = self.mtcnn.detect(np.array(image))
        except Exception as e:
            raise NoFaceDetectedError(
                user_message="Unable to process image.",
                technical_details=f"Face detection failed: {e}",
            ) from e

        if boxes is None or len(boxes) == 0:
            raise NoFaceDetectedError()

        idx = int(np.argmax(probs)) if len(boxes) > 1 else 0
        x1, y1, x2, y2 = boxes[idx]
        w, h = x2 - x1, y2 - y1
        iw, ih = image.size
        crop_box = (
            max(0, int(x1 - w * self.face_margin)),
            max(0, int(y1 - h * self.face_margin)),
            min(iw, int(x2 + w * self.face_margin)),
            min(ih, int(y2 + h * self.face_margin)),
        )
        return image.crop(crop_box)
