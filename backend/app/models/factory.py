"""
Detector Factory for creating and caching detector instances (Image / Video / Audio).
"""

from typing import Dict, Optional

from loguru import logger

from app.config import DEVICE, settings
from app.models.base import DeepfakeDetector


class DetectorFactory:

    def __init__(self, device: Optional[str] = None):
        self._device = device or DEVICE
        self._detectors: Dict[str, DeepfakeDetector] = {}
        logger.info("DetectorFactory initialised on device={}", self._device)

    def get_detector(self, detector_type: str) -> DeepfakeDetector:
        detector_type = detector_type.lower()

        if detector_type in self._detectors:
            return self._detectors[detector_type]

        logger.info("[MODEL:{}] Creating detector instance...", detector_type.upper())
        try:
            detector = self._create_detector(detector_type)
            self._detectors[detector_type] = detector
            model_info = detector.get_model_info()
            params = model_info.get("total_parameters", "N/A")
            size_mb = model_info.get("model_size_mb", "N/A")
            if isinstance(params, int):
                logger.info(
                    "[MODEL:{}] Detector ready (parameters={:,}, size={:.1f} MB)",
                    detector_type.upper(),
                    params,
                    size_mb,
                )
            else:
                logger.info("[MODEL:{}] Detector ready", detector_type.upper())
            return detector
        except Exception as e:
            logger.error("[MODEL:{}] Failed to create detector: {}", detector_type.upper(), e)
            raise RuntimeError(f"Could not initialise {detector_type} detector: {e}") from e

    def reload_detector(
        self, detector_type: str, model_path: Optional[str] = None
    ) -> DeepfakeDetector:
        detector_type = detector_type.lower()
        if detector_type in self._detectors:
            self._detectors[detector_type].unload_model()
            del self._detectors[detector_type]
        return self.get_detector(detector_type)

    def unload_all(self) -> None:
        for dtype, det in self._detectors.items():
            try:
                det.unload_model()
            except Exception as e:
                logger.error("[MODEL:{}] Error during unload: {}", dtype.upper(), e)
        self._detectors.clear()

    def get_loaded_detectors(self) -> Dict[str, dict]:
        return {dt: det.get_model_info() for dt, det in self._detectors.items()}

    def _create_detector(self, detector_type: str) -> DeepfakeDetector:
        from app.models.audio_detector import AudioDetector
        from app.models.image_detector import ImageDetector
        from app.models.video_detector import VideoDetector

        registry = {
            "video": (VideoDetector, settings.VIDEO_MODEL_PATH),
            "audio": (AudioDetector, settings.AUDIO_MODEL_PATH),
        }

        if detector_type == "image":
            detector = ImageDetector(
                spatial_model_path=settings.IMAGE_SPATIAL_MODEL_PATH,
                noise_model_path=settings.IMAGE_NOISE_MODEL_PATH,
                device=self._device,
            )
            model_hint = f"spatial={settings.IMAGE_SPATIAL_MODEL_PATH}, noise={settings.IMAGE_NOISE_MODEL_PATH}"
        elif detector_type in registry:
            cls, model_path = registry[detector_type]
            detector = cls(model_path=model_path, device=self._device)
            model_hint = model_path
        else:
            raise ValueError(
                f"Unknown detector type '{detector_type}'. "
                f"Valid types: image, {', '.join(registry)}"
            )

        try:
            detector.load_model()
        except FileNotFoundError:
            logger.warning(
                "[MODEL:{}] Model file not found ({}), using initialised weights",
                detector_type.upper(),
                model_hint,
            )

        return detector

    def __repr__(self) -> str:
        loaded = list(self._detectors.keys())
        return f"DetectorFactory(device={self._device}, loaded={loaded})"
