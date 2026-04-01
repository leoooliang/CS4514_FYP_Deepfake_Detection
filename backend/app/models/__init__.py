"""
ML Models Package - Deepfake Detection Models

This package contains:
    - Base abstract class for all detectors
    - Factory pattern for detector creation
    - Image, video, and audio detector implementations
"""

from app.models.base import DeepfakeDetector, DetectionResult
from app.models.factory import DetectorFactory, get_factory, create_detector
from app.models.image_detector import ImageDetector, detect_image_deepfake
from app.models.video_detector import VideoDetector, detect_video_deepfake
from app.models.audio_detector import AudioDetector, detect_audio_deepfake

__all__ = [
    # Base classes
    "DeepfakeDetector",
    "DetectionResult",
    
    # Factory
    "DetectorFactory",
    "get_factory",
    "create_detector",
    
    # Detector implementations
    "ImageDetector",
    "VideoDetector",
    "AudioDetector",
    
    # Convenience functions
    "detect_image_deepfake",
    "detect_video_deepfake",
    "detect_audio_deepfake",
]
