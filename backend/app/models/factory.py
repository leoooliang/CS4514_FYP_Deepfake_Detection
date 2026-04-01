"""
============================================================================
Detector Factory - Factory Pattern Implementation
============================================================================
This module implements the Factory Pattern for creating detector instances.

Benefits:
    - Centralized detector creation logic
    - Easy to add new detector types
    - Lazy loading for memory efficiency
    - Singleton pattern for detector instances
    - Configuration management

Design Pattern: Factory + Singleton
    - Factory: Creates appropriate detector based on type
    - Singleton: Ensures only one instance per detector type

Usage:
    factory = DetectorFactory()
    image_detector = factory.get_detector("image")
    result = image_detector.detect(my_image)

Author: Senior AI Systems Architect
Date: 2026-01-28
============================================================================
"""

from typing import Dict, Optional
from loguru import logger

from app.models.base import DeepfakeDetector
from app.models.image_detector import ImageDetector
from app.models.video_detector import VideoDetector
from app.models.audio_detector import AudioDetector
from app.config import settings, DEVICE


class DetectorFactory:
    """
    Factory class for creating and managing deepfake detector instances.
    
    This factory:
        - Creates detector instances on-demand (lazy loading)
        - Caches detector instances to avoid reloading models
        - Provides a clean API for detector access
        - Handles configuration and device management
    
    Attributes:
        _detectors: Cache of instantiated detector objects
        _device: Computing device for all detectors
    """
    
    # Class-level cache for detector instances (Singleton pattern)
    _detectors: Dict[str, DeepfakeDetector] = {}
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the factory with configuration.
        
        Args:
            device: Computing device (cpu, cuda, mps). If None, uses auto-detected device.
        """
        self._device = device or DEVICE
        logger.info(f"Detector factory initialized with device: {self._device}")
    
    def get_detector(self, detector_type: str) -> DeepfakeDetector:
        """
        Get a detector instance by type, creating it if necessary.
        
        This method implements lazy loading - detectors are only created and
        loaded when first requested. Subsequent calls return the cached instance.
        
        Args:
            detector_type: Type of detector - "image", "video", or "audio"
        
        Returns:
            DeepfakeDetector: Configured and loaded detector instance
        
        Raises:
            ValueError: If detector_type is not recognized
            RuntimeError: If detector creation or loading fails
        
        Example:
            factory = DetectorFactory()
            
            # First call: creates and loads the detector
            image_det = factory.get_detector("image")
            
            # Second call: returns cached instance (fast)
            same_det = factory.get_detector("image")
            assert image_det is same_det  # True - same object
        """
        detector_type = detector_type.lower()
        
        # Check if detector is already created and cached
        if detector_type in self._detectors:
            logger.debug(f"Returning cached {detector_type} detector")
            return self._detectors[detector_type]
        
        # Create new detector instance
        logger.info(f"Creating new {detector_type} detector...")
        
        try:
            detector = self._create_detector(detector_type)
            
            # Cache the detector for future use
            self._detectors[detector_type] = detector
            
            logger.success(f"✓ {detector_type} detector ready")
            return detector
            
        except Exception as e:
            logger.error(f"Failed to create {detector_type} detector: {str(e)}")
            raise RuntimeError(
                f"Could not initialize {detector_type} detector: {str(e)}"
            ) from e
    
    def _create_detector(self, detector_type: str) -> DeepfakeDetector:
        """
        Internal method to create and load a detector instance.
        
        This method:
            1. Selects the appropriate detector class
            2. Gets the model path from configuration
            3. Instantiates the detector
            4. Loads the model weights
        
        Args:
            detector_type: Type of detector to create
        
        Returns:
            DeepfakeDetector: Loaded detector instance
        
        Raises:
            ValueError: If detector type is invalid
        """
        # Map detector types to classes and model paths
        detector_config = {
            "image": {
                "class": ImageDetector,
                "model_path": settings.IMAGE_MODEL_PATH,
                "description": "Image deepfake detection"
            },
            "video": {
                "class": VideoDetector,
                "model_path": settings.VIDEO_MODEL_PATH,
                "description": "Video deepfake detection"
            },
            "audio": {
                "class": AudioDetector,
                "model_path": settings.AUDIO_MODEL_PATH,
                "description": "Audio deepfake detection"
            }
        }
        
        # Validate detector type
        if detector_type not in detector_config:
            valid_types = ", ".join(detector_config.keys())
            raise ValueError(
                f"Invalid detector type: '{detector_type}'. "
                f"Valid types are: {valid_types}"
            )
        
        config = detector_config[detector_type]
        
        # Instantiate the detector
        detector_class = config["class"]
        model_path = config["model_path"]
        
        logger.debug(
            f"Creating {config['description']} "
            f"(model: {model_path or 'placeholder'})"
        )
        
        # Create detector instance
        detector = detector_class(
            model_path=model_path,
            device=self._device
        )
        
        # Load the model
        try:
            detector.load_model()
        except FileNotFoundError:
            # Model file doesn't exist - this is expected for placeholder setup
            logger.warning(
                f"Model file not found: {model_path}. "
                f"Using placeholder model for {detector_type} detection."
            )
            # The detector will use its default placeholder model
        
        return detector
    
    def reload_detector(self, detector_type: str, model_path: Optional[str] = None) -> DeepfakeDetector:
        """
        Reload a detector with a new model.
        
        This is useful for:
            - Hot-swapping models without restarting the server
            - A/B testing different model versions
            - Updating to newly trained models
        
        Args:
            detector_type: Type of detector to reload
            model_path: Optional new model path. If None, uses config path.
        
        Returns:
            DeepfakeDetector: Reloaded detector instance
        
        Example:
            factory = DetectorFactory()
            
            # Load initial model
            detector = factory.get_detector("image")
            
            # Later: swap to a new model
            detector = factory.reload_detector(
                "image",
                model_path="models/new_image_model_v2.pth"
            )
        """
        logger.info(f"Reloading {detector_type} detector...")
        
        # Unload existing detector if cached
        if detector_type in self._detectors:
            self._detectors[detector_type].unload_model()
            del self._detectors[detector_type]
        
        # Update model path if provided
        if model_path:
            if detector_type == "image":
                settings.IMAGE_MODEL_PATH = model_path
            elif detector_type == "video":
                settings.VIDEO_MODEL_PATH = model_path
            elif detector_type == "audio":
                settings.AUDIO_MODEL_PATH = model_path
        
        # Create new detector instance
        return self.get_detector(detector_type)
    
    def unload_all(self) -> None:
        """
        Unload all detectors from memory.
        
        Useful for:
            - Freeing GPU/CPU memory
            - Graceful shutdown
            - Testing memory cleanup
        """
        logger.info("Unloading all detectors...")
        
        for detector_type, detector in self._detectors.items():
            try:
                detector.unload_model()
                logger.debug(f"✓ Unloaded {detector_type} detector")
            except Exception as e:
                logger.error(f"Error unloading {detector_type}: {str(e)}")
        
        self._detectors.clear()
        logger.success("All detectors unloaded")
    
    def get_loaded_detectors(self) -> Dict[str, Dict[str, any]]:
        """
        Get information about all loaded detectors.
        
        Returns:
            dict: Status and metadata for each loaded detector
        
        Example:
            {
                "image": {
                    "status": "loaded",
                    "model_path": "models/image_detector.pth",
                    "device": "cuda",
                    "parameters": 25000000
                },
                ...
            }
        """
        return {
            detector_type: detector.get_model_info()
            for detector_type, detector in self._detectors.items()
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        loaded = list(self._detectors.keys())
        return f"DetectorFactory(device={self._device}, loaded={loaded})"


# =============================================================================
# Convenience Functions
# =============================================================================

# Global factory instance for easy import
_global_factory: Optional[DetectorFactory] = None


def get_factory() -> DetectorFactory:
    """
    Get the global detector factory instance (Singleton).
    
    Returns:
        DetectorFactory: Global factory instance
    
    Usage:
        from app.models.factory import get_factory
        
        factory = get_factory()
        detector = factory.get_detector("image")
    """
    global _global_factory
    
    if _global_factory is None:
        _global_factory = DetectorFactory()
    
    return _global_factory


def create_detector(detector_type: str, device: Optional[str] = None) -> DeepfakeDetector:
    """
    Convenience function to create a detector directly.
    
    Args:
        detector_type: Type of detector ("image", "video", "audio")
        device: Optional device override
    
    Returns:
        DeepfakeDetector: Loaded detector instance
    
    Usage:
        from app.models.factory import create_detector
        
        # Quick way to get a detector
        detector = create_detector("image")
        result = detector.detect(my_image)
    """
    factory = DetectorFactory(device=device)
    return factory.get_detector(detector_type)


# =============================================================================
# Usage Examples
# =============================================================================
"""
Example 1: Basic Usage
----------------------
from app.models.factory import DetectorFactory

factory = DetectorFactory()
image_detector = factory.get_detector("image")
result = image_detector.detect(my_image)


Example 2: Hot-Swapping Models
-------------------------------
factory = DetectorFactory()

# Use initial model
detector = factory.get_detector("image")
result1 = detector.detect(image1)

# Swap to a new model
detector = factory.reload_detector("image", "models/new_model.pth")
result2 = detector.detect(image2)


Example 3: Multiple Detectors
------------------------------
factory = DetectorFactory()

# Load all detector types
image_det = factory.get_detector("image")
video_det = factory.get_detector("video")
audio_det = factory.get_detector("audio")

# Process different media types
image_result = image_det.detect(my_image)
video_result = video_det.detect(my_video_path)
audio_result = audio_det.detect(my_audio_array)


Example 4: Memory Management
-----------------------------
factory = DetectorFactory()

# Load detectors
factory.get_detector("image")
factory.get_detector("video")

# Check status
info = factory.get_loaded_detectors()
logger.info(f"Loaded detectors: {info}")

# Clean up
factory.unload_all()  # Free memory
"""
