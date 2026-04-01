"""
============================================================================
Abstract Base Class for Deepfake Detectors
============================================================================
This module defines the contract that all detector implementations must follow.

Design Pattern: Abstract Base Class (ABC)
Benefits:
    - Enforces consistent interface across all detector types
    - Makes it easy to swap detector implementations
    - Enables polymorphism for cleaner code
    - Provides type safety and IDE autocomplete

Usage:
    class MyNewDetector(DeepfakeDetector):
        def load_model(self, model_path: str) -> None:
            # Your implementation
            
        def preprocess(self, input_data: Any) -> torch.Tensor:
            # Your implementation
            
        def predict(self, input_tensor: torch.Tensor) -> DetectionResult:
            # Your implementation

Author: Senior AI Systems Architect
Date: 2026-01-28
============================================================================
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import torch
from loguru import logger


# =============================================================================
# Data Classes for Structured Results
# =============================================================================

@dataclass
class DetectionResult:
    """
    Structured result from a deepfake detection prediction.
    
    This class ensures consistent response format across all detector types,
    making it easier for the API layer to handle results uniformly.
    
    Attributes:
        prediction: Binary prediction - "real" or "deepfake"
        confidence: Confidence score (0-1) for the prediction
        probabilities: Dictionary with probabilities for each class
        processing_time: Time taken for inference in seconds
        metadata: Additional information specific to the detector type
    """
    prediction: str  # "real" or "deepfake"
    confidence: float  # 0.0 to 1.0
    probabilities: Dict[str, float]  # {"real": 0.15, "deepfake": 0.85}
    processing_time: float  # seconds
    metadata: Optional[Dict[str, Any]] = None  # Detector-specific data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "processing_time": round(self.processing_time, 3),
            "metadata": self.metadata or {}
        }
    
    def __repr__(self) -> str:
        """Human-readable representation."""
        return f"DetectionResult(prediction={self.prediction}, confidence={self.confidence:.2%})"


# =============================================================================
# Abstract Base Class - Deepfake Detector Interface
# =============================================================================

class DeepfakeDetector(ABC):
    """
    Abstract base class defining the interface for all deepfake detectors.
    
    All detector implementations (Image, Video, Audio) MUST inherit from this class
    and implement all abstract methods. This ensures consistency and makes it
    easy to add new detector types in the future.
    
    Lifecycle:
        1. Instantiation: __init__() - Initialize basic properties
        2. Model Loading: load_model() - Load pretrained weights
        3. Inference: detect() - Main entry point for prediction
            3a. Preprocessing: preprocess() - Prepare input data
            3b. Forward Pass: predict() - Run model inference
            3c. Postprocessing: (included in predict)
    
    Attributes:
        model_path: Path to the model weights file
        device: Computing device (cpu, cuda, mps)
        model: The loaded PyTorch model
        detector_type: Type identifier (image, video, audio)
        is_loaded: Flag indicating if model is loaded
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Initialize the detector with basic configuration.
        
        Args:
            model_path: Path to pretrained model weights (optional)
            device: Device for computation - "cpu", "cuda", or "mps"
        """
        self.model_path = model_path
        self.device = torch.device(device)
        self.model: Optional[torch.nn.Module] = None
        self.detector_type: str = self.__class__.__name__.lower()
        self.is_loaded: bool = False
        
        logger.info(f"Initializing {self.detector_type} detector on device: {self.device}")
    
    # =========================================================================
    # Abstract Methods - MUST be implemented by subclasses
    # =========================================================================
    
    @abstractmethod
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the pretrained model into memory.
        
        This method is responsible for:
            1. Loading model architecture
            2. Loading pretrained weights
            3. Setting model to evaluation mode
            4. Moving model to the specified device
        
        Args:
            model_path: Path to model weights. If None, use self.model_path
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        
        Example Implementation:
            def load_model(self, model_path: Optional[str] = None) -> None:
                path = model_path or self.model_path
                if not Path(path).exists():
                    raise FileNotFoundError(f"Model not found: {path}")
                
                self.model = YourModelClass()
                self.model.load_state_dict(torch.load(path, map_location=self.device))
                self.model.eval()
                self.model.to(self.device)
                self.is_loaded = True
        """
        pass
    
    @abstractmethod
    def preprocess(self, input_data: Any) -> torch.Tensor:
        """
        Preprocess raw input data into model-ready tensor format.
        
        This method handles all data transformation required before inference:
            - Image: Resize, normalize, convert to tensor
            - Video: Extract frames, resize, stack into batch
            - Audio: Resample, extract features (spectrogram/MFCC), normalize
        
        Args:
            input_data: Raw input data (PIL Image, video path, audio array, etc.)
        
        Returns:
            torch.Tensor: Preprocessed tensor ready for model input
        
        Raises:
            ValueError: If input data is invalid or corrupted
        
        Example Implementation:
            def preprocess(self, input_data: PIL.Image) -> torch.Tensor:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
                tensor = transform(input_data).unsqueeze(0)  # Add batch dim
                return tensor.to(self.device)
        """
        pass
    
    @abstractmethod
    def predict(self, input_tensor: torch.Tensor) -> DetectionResult:
        """
        Perform inference and return structured prediction results.
        
        This method:
            1. Runs forward pass through the model
            2. Applies activation functions (softmax/sigmoid)
            3. Extracts probabilities and confidence
            4. Creates DetectionResult object
        
        Args:
            input_tensor: Preprocessed tensor from preprocess()
        
        Returns:
            DetectionResult: Structured prediction with confidence and metadata
        
        Raises:
            RuntimeError: If model is not loaded or inference fails
        
        Example Implementation:
            @torch.no_grad()
            def predict(self, input_tensor: torch.Tensor) -> DetectionResult:
                if not self.is_loaded:
                    raise RuntimeError("Model not loaded")
                
                start_time = time.time()
                
                # Forward pass
                output = self.model(input_tensor)
                probs = torch.softmax(output, dim=1)[0]
                
                # Extract results (Class 0 = Fake, Class 1 = Real)
                deepfake_prob = probs[0].item()
                real_prob = probs[1].item()
                
                prediction = "deepfake" if deepfake_prob > 0.5 else "real"
                confidence = deepfake_prob
                
                processing_time = time.time() - start_time
                
                return DetectionResult(
                    prediction=prediction,
                    confidence=confidence,
                    probabilities={
                        "real": real_prob,
                        "deepfake": deepfake_prob
                    },
                    processing_time=processing_time
                )
        """
        pass
    
    # =========================================================================
    # Concrete Methods - Default implementation, can be overridden
    # =========================================================================
    
    def detect(self, input_data: Any) -> DetectionResult:
        """
        Main entry point for detection - combines preprocess and predict.
        
        This is the method that API endpoints will call. It orchestrates
        the full inference pipeline.
        
        Args:
            input_data: Raw input (image, video path, audio array, etc.)
        
        Returns:
            DetectionResult: Complete prediction results
        
        Raises:
            RuntimeError: If model is not loaded
            ValueError: If input data is invalid
        """
        if not self.is_loaded:
            logger.error(f"{self.detector_type} model not loaded")
            raise RuntimeError(
                f"Model not loaded. Call load_model() first."
            )
        
        logger.debug(f"Starting detection with {self.detector_type} detector")
        
        # Step 1: Preprocess input
        input_tensor = self.preprocess(input_data)
        # Some detectors (e.g., dual-stream image) return multiple tensors from preprocess().
        # Support tuple/list outputs to avoid runtime errors when logging.
        if hasattr(input_tensor, "shape"):
            logger.debug(f"Preprocessed input shape: {input_tensor.shape}")
        elif isinstance(input_tensor, (tuple, list)):
            shapes = []
            for item in input_tensor:
                shapes.append(getattr(item, "shape", type(item).__name__))
            logger.debug(f"Preprocessed input shape(s): {shapes}")
        else:
            logger.debug(
                "Preprocessed input shape: "
                f"{type(input_tensor).__name__} (no .shape attribute)"
            )
        
        # Step 2: Run inference
        result = self.predict(input_tensor)
        logger.info(
            f"{self.detector_type} detection complete: "
            f"{result.prediction} ({result.confidence:.2%} confidence)"
        )
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model metadata including parameters, device, etc.
        """
        if not self.is_loaded or self.model is None:
            return {
                "status": "not_loaded",
                "detector_type": self.detector_type
            }
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        return {
            "status": "loaded",
            "detector_type": self.detector_type,
            "model_path": str(self.model_path),
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def unload_model(self) -> None:
        """
        Unload model from memory to free resources.
        
        Useful for:
            - Switching between models
            - Freeing GPU memory
            - Graceful shutdown
        """
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False
            
            # Clear CUDA cache if using GPU
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            logger.info(f"{self.detector_type} model unloaded from memory")
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "loaded" if self.is_loaded else "not loaded"
        return f"{self.__class__.__name__}(device={self.device}, status={status})"
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.unload_model()


# =============================================================================
# Usage Example (for documentation)
# =============================================================================
"""
Example of implementing a concrete detector:

from app.models.base import DeepfakeDetector, DetectionResult
import torch
import torch.nn as nn

class ImageDetector(DeepfakeDetector):
    def load_model(self, model_path: Optional[str] = None) -> None:
        # Load your custom model
        self.model = MyCustomCNN()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.to(self.device)
        self.is_loaded = True
    
    def preprocess(self, image: PIL.Image) -> torch.Tensor:
        # Transform image to tensor
        # ... your preprocessing logic
        return tensor
    
    def predict(self, input_tensor: torch.Tensor) -> DetectionResult:
        # Run inference
        # ... your prediction logic
        return DetectionResult(...)

# Usage:
detector = ImageDetector(model_path="model.pth", device="cuda")
detector.load_model()
result = detector.detect(my_image)
logger.info(f"Result: {result.prediction}, Confidence: {result.confidence}")
"""
