"""
============================================================================
Image Preprocessing Pipeline for Dual-Stream Deepfake Detection
============================================================================
This module provides a robust, reusable image preprocessing pipeline for the
dual-stream (CLIP + Noise) deepfake detection architecture.

Key Features:
    1. Face Detection & Cropping (MTCNN from facenet-pytorch)
       - Detects and crops faces with 20% margin expansion
       - Falls back to 50% center crop if no face detected
       
    2. Dual-Stream Transformation Pipeline:
       - Stream A (CLIP): CLIP-normalized tensor for semantic analysis
       - Stream B (Noise): Base tensor scaled to [0, 255] for SRM filter
       
    3. Robust Input Handling:
       - Supports: str (file path), bytes, np.ndarray, PIL.Image
       - Safe conversion to RGB PIL Image
       
    4. Error Handling & Logging:
       - Comprehensive error handling for production use
       - Detailed Loguru logging for debugging

Technical Details:
    - MTCNN Configuration: keep_all=False, post_process=False
    - Face Selection: Highest probability face if multiple detected
    - Bounding Box Expansion: 20% margin (clamped to image boundaries)
    - Fallback Strategy: 50% center crop + warning log
    - CLIP Normalization: mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275]
    - Noise Base Tensor: ToTensor() [0, 1] → multiply by 255.0

Author: Senior ML Engineer
Date: 2026-03-24
============================================================================
"""

import io
from typing import Tuple, Union, Optional
from pathlib import Path

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from loguru import logger

from app.core.exceptions import NoFaceDetectedError


# Try to import facenet-pytorch MTCNN - graceful fallback if not available
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    logger.warning(
        "facenet-pytorch not available. Face detection disabled. "
        "Install with: pip install facenet-pytorch"
    )


class ImagePreprocessor:
    """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║           Image Preprocessing Pipeline for Dual-Stream               ║
    ║                    MTCNN Face Detection + Transforms                 ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    Robust face extraction and tensor formatting for dual-stream deepfake
    detection (CLIP + Noise streams).
    
    Pipeline Overview:
        1. Input Handling → RGB PIL Image
        2. MTCNN Face Detection → Crop with 20% margin
        3. Fallback → 50% center crop if no face detected
        4. Dual Transforms → (CLIP tensor, Noise base tensor)
        
    Stream A (CLIP):
        - Resize to 224x224
        - ImageNet normalization (CLIP statistics)
        - ToTensor() → [0, 1] → normalize
        
    Stream B (Noise):
        - Resize to 224x224
        - ToTensor() → [0, 1]
        - Multiply by 255.0 (prepares for SRM filter in detector)
        
    Example:
        >>> preprocessor = ImagePreprocessor()
        >>> clip_tensor, noise_base_tensor = preprocessor.process("face.jpg")
        >>> print(clip_tensor.shape, noise_base_tensor.shape)
        torch.Size([1, 3, 224, 224]) torch.Size([1, 3, 224, 224])
    """
    
    def __init__(
        self,
        device: str = "cpu",
        target_size: Tuple[int, int] = (224, 224),
        face_margin: float = 0.20,
        center_crop_ratio: float = 0.50
    ):
        """
        Initialize ImagePreprocessor with MTCNN face detector and transforms.
        
        Args:
            device: Computing device for MTCNN (cpu, cuda, mps)
            target_size: Target image size (height, width)
            face_margin: Margin expansion ratio for face bounding box (default: 0.20 = 20%)
            center_crop_ratio: Center crop ratio for fallback (default: 0.50 = 50%)
        """
        self.device = torch.device(device)
        self.target_size = target_size
        self.face_margin = face_margin
        self.center_crop_ratio = center_crop_ratio
        
        # =====================================================================
        # Initialize MTCNN Face Detector
        # =====================================================================
        if MTCNN_AVAILABLE:
            self.mtcnn = MTCNN(
                keep_all=False,        # Return only the best face
                post_process=False,    # Don't apply post-processing
                device=self.device
            )
            logger.debug(f"MTCNN face detector initialized on {self.device}")
        else:
            self.mtcnn = None
            logger.warning(
                "MTCNN not available - will use center crop fallback for all images"
            )
        
        # =====================================================================
        # Stream A: CLIP Transformation Pipeline
        # =====================================================================
        # CLIP uses specific ImageNet normalization statistics
        self.clip_transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),  # Converts to [0, 1]
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],   # CLIP mean
                std=[0.26862954, 0.26130258, 0.27577711]     # CLIP std
            )
        ])
        
        # =====================================================================
        # Stream B: Noise Base Transformation Pipeline
        # =====================================================================
        # This prepares the base tensor for SRM filtering
        # We scale to [0, 255] here (detector will apply SRM filter)
        self.noise_base_transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()  # Converts to [0, 1]
            # Note: We multiply by 255.0 in the process() method
        ])
        
        logger.info(
            f"ImagePreprocessor initialized:\n"
            f"  Target size: {target_size}\n"
            f"  Face margin: {face_margin * 100}%\n"
            f"  Center crop fallback: {center_crop_ratio * 100}%\n"
            f"  MTCNN available: {MTCNN_AVAILABLE}"
        )
    
    def _load_image(self, input_data: Union[str, bytes, np.ndarray, Image.Image]) -> Image.Image:
        """
        Convert various input types to RGB PIL Image.
        
        Args:
            input_data: Input in supported format
            
        Returns:
            PIL.Image: RGB image
            
        Raises:
            ValueError: If input type is unsupported
            FileNotFoundError: If file path doesn't exist
        """
        try:
            # Case 1: File path (string)
            if isinstance(input_data, str):
                file_path = Path(input_data)
                if not file_path.exists():
                    raise FileNotFoundError(f"Image file not found: {input_data}")
                image = Image.open(file_path).convert("RGB")
                logger.debug(f"Loaded image from path: {input_data} (size: {image.size})")
                return image
            
            # Case 2: Bytes
            elif isinstance(input_data, bytes):
                image = Image.open(io.BytesIO(input_data)).convert("RGB")
                logger.debug(f"Loaded image from bytes (size: {image.size})")
                return image
            
            # Case 3: NumPy array
            elif isinstance(input_data, np.ndarray):
                # Validate shape
                if input_data.ndim == 2:
                    # Grayscale - convert to RGB
                    input_data = np.stack([input_data] * 3, axis=-1)
                elif input_data.ndim == 3:
                    if input_data.shape[2] not in [3, 4]:
                        raise ValueError(
                            f"Expected RGB/RGBA array (H, W, 3/4), got shape {input_data.shape}"
                        )
                    # Convert RGBA to RGB if needed
                    if input_data.shape[2] == 4:
                        input_data = input_data[:, :, :3]
                else:
                    raise ValueError(
                        f"Expected 2D or 3D array, got {input_data.ndim}D array"
                    )
                
                # Ensure uint8 type
                if input_data.dtype != np.uint8:
                    # Normalize to [0, 255] if float
                    if input_data.dtype in [np.float32, np.float64]:
                        if input_data.max() <= 1.0:
                            input_data = (input_data * 255).astype(np.uint8)
                        else:
                            input_data = input_data.astype(np.uint8)
                    else:
                        input_data = input_data.astype(np.uint8)
                
                image = Image.fromarray(input_data)
                logger.debug(f"Converted NumPy array to PIL Image (size: {image.size})")
                return image
            
            # Case 4: PIL Image
            elif isinstance(input_data, Image.Image):
                image = input_data.convert("RGB")
                logger.debug(f"Using PIL Image (size: {image.size})")
                return image
            
            else:
                raise ValueError(
                    f"Unsupported input type: {type(input_data)}. "
                    "Expected: str (file path), bytes, np.ndarray, or PIL.Image"
                )
                
        except Exception as e:
            logger.error(f"Failed to load image: {str(e)}")
            raise
    
    def _detect_and_crop_face(self, image: Image.Image) -> Image.Image:
        """
        Detect face using MTCNN and crop with margin expansion.
        
        FAIL-FAST: Raises NoFaceDetectedError if no face is detected.
        
        Args:
            image: RGB PIL Image
            
        Returns:
            PIL.Image: Cropped face
            
        Raises:
            NoFaceDetectedError: If no face is detected in the image
        """
        if self.mtcnn is None:
            logger.error("MTCNN not available - cannot perform face detection")
            raise NoFaceDetectedError(
                user_message="Unable to process image. Please try again later.",
                technical_details="MTCNN is required but not installed."
            )
        
        try:
            # Convert PIL to NumPy for MTCNN
            img_array = np.array(image)
            
            # Detect faces - returns (boxes, probs)
            boxes, probs = self.mtcnn.detect(img_array)
            
            # FAIL-FAST: Reject images without faces
            if boxes is None or len(boxes) == 0:
                logger.warning("No face detected in the provided image")
                raise NoFaceDetectedError()
            
            # Select face with highest probability
            if len(boxes) > 1:
                best_idx = np.argmax(probs)
                box = boxes[best_idx]
                prob = probs[best_idx]
                logger.debug(
                    f"Multiple faces detected ({len(boxes)}), "
                    f"selected highest probability: {prob:.3f}"
                )
            else:
                box = boxes[0]
                prob = probs[0]
                logger.debug(f"Single face detected with probability: {prob:.3f}")
            
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box
            
            # Expand bounding box by margin
            width = x2 - x1
            height = y2 - y1
            margin_w = width * self.face_margin
            margin_h = height * self.face_margin
            
            # Apply margin (ensure within image boundaries)
            img_width, img_height = image.size
            x1_expanded = max(0, int(x1 - margin_w))
            y1_expanded = max(0, int(y1 - margin_h))
            x2_expanded = min(img_width, int(x2 + margin_w))
            y2_expanded = min(img_height, int(y2 + margin_h))
            
            # Crop face
            face_crop = image.crop((x1_expanded, y1_expanded, x2_expanded, y2_expanded))
            
            logger.debug(
                f"Face cropped with {self.face_margin * 100}% margin: "
                f"bbox=[{x1_expanded}, {y1_expanded}, {x2_expanded}, {y2_expanded}], "
                f"size={face_crop.size}"
            )
            
            return face_crop
            
        except NoFaceDetectedError:
            # Re-raise NoFaceDetectedError as-is
            raise
        except Exception as e:
            logger.error(f"Face detection failed with error: {str(e)}")
            raise NoFaceDetectedError(
                user_message="Unable to process image. Please try again later.",
                technical_details=f"Face detection failed: {str(e)}"
            ) from e
    
    def process(
        self,
        input_data: Union[str, bytes, np.ndarray, Image.Image]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete preprocessing pipeline: Load → Detect Face → Dual Transform.
        
        This is the main entry point for preprocessing. It performs:
            1. Load and convert input to RGB PIL Image
            2. Detect face with MTCNN (or center crop fallback)
            3. Apply dual-stream transformations
            
        Args:
            input_data: Image in supported format (str, bytes, np.ndarray, PIL.Image)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (clip_tensor, noise_base_tensor)
                - clip_tensor: (1, 3, 224, 224) - CLIP normalized, ready for model
                - noise_base_tensor: (1, 3, 224, 224) - Scaled to [0, 255], ready for SRM
                
        Raises:
            ValueError: If input processing fails
            FileNotFoundError: If file path doesn't exist
            
        Example:
            >>> preprocessor = ImagePreprocessor()
            >>> clip_t, noise_t = preprocessor.process("face.jpg")
            >>> print(f"CLIP: {clip_t.shape}, Noise: {noise_t.shape}")
            CLIP: torch.Size([1, 3, 224, 224]), Noise: torch.Size([1, 3, 224, 224])
        """
        try:
            logger.debug("Starting image preprocessing pipeline...")
            
            # Step 1: Load image
            image = self._load_image(input_data)
            
            # Step 2: Detect and crop face (or fallback to center crop)
            face_crop = self._detect_and_crop_face(image)
            
            # Step 3: Apply dual-stream transformations
            
            # Stream A: CLIP transformation
            clip_tensor = self.clip_transform(face_crop)
            clip_tensor = clip_tensor.unsqueeze(0)  # Add batch dimension
            
            # Stream B: Noise base transformation
            noise_base_tensor = self.noise_base_transform(face_crop)
            noise_base_tensor = noise_base_tensor.unsqueeze(0)  # Add batch dimension
            
            # CRITICAL FIX: Scale to [0, 255] for SRM filter
            # This fixes the truncation scale bug mentioned in the requirements
            noise_base_tensor = noise_base_tensor * 255.0
            
            logger.debug(
                f"Preprocessing complete:\n"
                f"  CLIP tensor: shape={clip_tensor.shape}, "
                f"range=[{clip_tensor.min():.3f}, {clip_tensor.max():.3f}]\n"
                f"  Noise base tensor: shape={noise_base_tensor.shape}, "
                f"range=[{noise_base_tensor.min():.1f}, {noise_base_tensor.max():.1f}]"
            )
            
            return clip_tensor, noise_base_tensor
            
        except NoFaceDetectedError:
            # Re-raise validation errors as-is (user-friendly messages)
            raise
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise ValueError(f"Preprocessing error: {str(e)}") from e


# =============================================================================
# Utility Functions
# =============================================================================

def preprocess_image_dual_stream(
    input_data: Union[str, bytes, np.ndarray, Image.Image],
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function for dual-stream image preprocessing.
    
    Args:
        input_data: Image in supported format
        device: Computing device for MTCNN
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (clip_tensor, noise_base_tensor)
        
    Example:
        >>> from app.preprocessing_pipelines.image_preprocessor import preprocess_image_dual_stream
        >>> clip_t, noise_t = preprocess_image_dual_stream("face.jpg")
    """
    preprocessor = ImagePreprocessor(device=device)
    return preprocessor.process(input_data)


# =============================================================================
# Testing / Validation
# =============================================================================

if __name__ == "__main__":
    """
    Test ImagePreprocessor with dummy inputs.
    """
    logger.info("=== Testing ImagePreprocessor ===")
    
    # Test 1: NumPy array input
    logger.info("\n[Test 1] NumPy Array Input")
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    preprocessor = ImagePreprocessor(device="cpu")
    clip_tensor, noise_tensor = preprocessor.process(dummy_img)
    
    logger.info(f"✓ CLIP tensor: shape={clip_tensor.shape}")
    logger.info(f"✓ Noise tensor: shape={noise_tensor.shape}")
    
    assert clip_tensor.shape == (1, 3, 224, 224), "CLIP tensor shape mismatch!"
    assert noise_tensor.shape == (1, 3, 224, 224), "Noise tensor shape mismatch!"
    assert noise_tensor.max() <= 255.0 and noise_tensor.min() >= 0.0, "Noise tensor range incorrect!"
    
    logger.success("\n=== All ImagePreprocessor Tests Passed! ===")
