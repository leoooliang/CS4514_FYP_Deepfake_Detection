"""
ML models package - detector ABC, factory, and concrete implementations.
"""

from app.models.base import DeepfakeDetector, DetectionResult
from app.models.factory import DetectorFactory

__all__ = ["DeepfakeDetector", "DetectionResult", "DetectorFactory"]
