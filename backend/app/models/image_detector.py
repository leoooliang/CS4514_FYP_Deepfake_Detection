"""
============================================================================
Image Deepfake Detector Implementation - DUAL-STREAM FUSION
============================================================================
TECHNICAL NOVELTY: Two-Stream Fusion Architecture

This implementation uses a dual-stream score-level fusion approach:
    Stream A (CLIP Stream): CLIP ViT-L/14 with LN-Tuning for semantic features
    Stream B (Noise Stream): SRM + EfficientNetV2-S for noise residual analysis

Key Innovation:
    - CLIP Stream: Captures high-level semantic manipulation patterns
    - Noise Stream: Extracts high-frequency noise artifacts via SRM filter [NOVELTY]
    - Score-Level Fusion: Weighted average of probabilities from both streams

Detects:
    - Face swaps (e.g., FaceSwap, DeepFaceLab)
    - GAN-generated faces (e.g., StyleGAN, ProGAN)
    - Face reenactment (e.g., Face2Face)
    - Facial attribute manipulation

Architecture:
    RGB Image (224×224) → CLIP ViT-L/14 → Softmax → [p_clip]
    RGB Image (224×224) → SRM Filter → Truncate → EfficientNetV2-S → Sigmoid → [p_noise]
    Fusion: p_final = (w * p_clip) + ((1 - w) * p_noise)

Training Details:
    - CLIP Model: best_clip.pth (CLIP ViT-L/14 with LN-Tuning)
    - Noise Model: best_noise_efficientnet.pth (SRM + EfficientNetV2-S)
    - Fusion: Score-level weighted average (optimal weight found via grid search)

Author: Senior ML Engineer & Backend Architect
Date: 2026-03-24
============================================================================
"""

import time
import warnings
from typing import Optional, Any, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from loguru import logger

from app.models.base import DeepfakeDetector, DetectionResult
from app.config import settings
from app.preprocessing_pipelines.image_preprocessor import ImagePreprocessor
from app.core.exceptions import NoFaceDetectedError

# Try to import CLIP and transformers - graceful fallback if not available
try:
    from transformers import CLIPVisionModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("transformers not available. Install with: pip install transformers")


# =============================================================================
# Stream A: CLIP Classifier (Semantic Stream)
# =============================================================================

class CLIPClassifier(nn.Module):
    """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                    ★ CLIP SPATIAL STREAM ★                          ║
    ║                       Stream A for Dual-Stream                       ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    CLIP Vision Encoder Stream for Deepfake Detection with LN-Tuning.
    
    This model uses CLIP's pre-trained vision encoder with parameter-efficient
    fine-tuning for deepfake detection.
    
    Key features:
    - CLIP ViT-L/14 vision encoder (pre-trained)
    - LN-Tuning: Only LayerNorm parameters are trainable (parameter-efficient)
    - L2 normalization: Projects CLS token onto hypersphere
    - Binary classification output
    
    Args:
        clip_model_name: HuggingFace model name (default: 'openai/clip-vit-large-patch14')
        num_classes: Number of output classes (default: 2 for binary classification)
        
    Input:
        CLIP-preprocessed images of shape (B, 3, 224, 224)
        
    Output:
        Logits of shape (B, num_classes)
    """
    
    def __init__(self, clip_model_name='openai/clip-vit-large-patch14', num_classes=2):
        super(CLIPClassifier, self).__init__()
        
        # Load pre-trained CLIP vision encoder.
        # Prefer eager attention to avoid PyTorch SDPA flash-attention warnings on builds
        # where flash attention is unavailable (common on Windows setups).
        try:
            self.clip_vision = CLIPVisionModel.from_pretrained(
                clip_model_name,
                use_safetensors=True,
                attn_implementation="eager",
            )
        except TypeError:
            # Backward compatibility with older transformers versions
            self.clip_vision = CLIPVisionModel.from_pretrained(
                clip_model_name,
                use_safetensors=True,
            )
            warnings.filterwarnings(
                "ignore",
                message=".*Torch was not compiled with flash attention.*",
                category=UserWarning,
            )
        
        # Freeze everything first
        for param in self.clip_vision.parameters():
            param.requires_grad = False
        
        # Unfreeze LayerNorms (LN-Tuning) - parameter-efficient fine-tuning
        for name, param in self.clip_vision.named_parameters():
            if 'norm' in name.lower():
                param.requires_grad = True
        
        # Get hidden size from CLIP config
        hidden_size = self.clip_vision.config.hidden_size
        
        # Classification head
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, pixel_values, return_normalized_features=False):
        """
        Forward pass through CLIP encoder and classifier.
        
        Args:
            pixel_values: CLIP-preprocessed images (B, 3, 224, 224)
            return_normalized_features: If True, return L2-normalized features instead of logits
            
        Returns:
            If return_normalized_features=False: Logits (B, num_classes)
            If return_normalized_features=True: L2-normalized CLS token (B, hidden_size)
        """
        # Extract CLS token from CLIP vision encoder
        outputs = self.clip_vision(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]  # (B, hidden_size)
        
        # L2 normalization - project onto hypersphere
        normalized_features = F.normalize(cls_token, p=2, dim=1)
        
        if return_normalized_features:
            return normalized_features
        
        # Classification
        logits = self.classifier(normalized_features)
        return logits


# =============================================================================
# Stream B: SRM + EfficientNet (Noise Stream)
# =============================================================================

class SRMConv2d(nn.Module):
    """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                        ★ TECHNICAL NOVELTY ★                         ║
    ║            SRM: Spatial Rich Model for Noise Extraction              ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    Spatial Rich Model (SRM) filter layer for noise residual extraction.
    
    The 5x5 SRM filter (from Warsaw University paper):
    [[ -1,  2, -2,  2, -1],
     [  2, -6,  8, -6,  2],
     [ -2,  8,-12,  8, -2],
     [  2, -6,  8, -6,  2],
     [ -1,  2, -2,  2, -1]] * (1/12)
    
    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
    """
    
    def __init__(self, in_channels=3):
        super(SRMConv2d, self).__init__()
        
        # Define the fixed 5x5 SRM kernel
        srm_kernel = np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8,-12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=np.float32) / 12.0
        
        # Create kernels for each input channel (depthwise convolution)
        srm_kernels = np.tile(srm_kernel[np.newaxis, np.newaxis, :, :], (in_channels, 1, 1, 1))
        
        # Create depthwise conv layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=5,
            padding=2,
            bias=False,
            groups=in_channels
        )
        
        # Load fixed SRM weights (non-trainable)
        self.conv.weight.data = torch.from_numpy(srm_kernels).float()
        self.conv.weight.requires_grad = False
        
    def forward(self, x):
        """Apply SRM filter to extract noise residuals."""
        return self.conv(x)


class NoiseEfficientNet(nn.Module):
    """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║           ★ NOISE RESIDUAL STREAM NETWORK ★                          ║
    ║                    Stream B for Dual-Stream                          ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    Noise Residual Stream Architecture with EfficientNetV2.
    
    RGB → SRM Filter (3-channel) → Truncate → EfficientNetV2 → Binary Classification
    
    Key features:
    - Processes 3-channel SRM noise residuals from RGB images
    - Pre-trained EfficientNetV2-S backbone
    - Dropout(p=0.5) for regularization
    - Binary classification output (sigmoid)
    """
    
    def __init__(self):
        super(NoiseEfficientNet, self).__init__()
        
        # Load pre-trained EfficientNetV2-S
        self.efficientnet = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        
        # Replace final classifier for binary classification
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1280, 1)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: SRM noise tensor (B, 3, H, W)
            
        Returns:
            Logits (B, 1) for binary classification
        """
        return self.efficientnet(x)


# =============================================================================
# Dual-Stream Image Detector
# =============================================================================

class ImageDetector(DeepfakeDetector):
    """
    Image deepfake detector with DUAL-STREAM FUSION architecture.
    
    ╔══════════════════════════════════════════════════════════════════════╗
    ║           Dual-Stream Fusion Deepfake Detector                       ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    Two-Stream Score-Level Fusion:
        - Stream A (CLIP): Semantic-level features for high-level manipulation
        - Stream B (Noise): SRM noise residuals for low-level artifacts
        - Fusion: Weighted average of probabilities
    
    Technical Approach:
        1. CLIP stream analyzes semantic manipulation patterns
        2. Noise stream detects high-frequency artifacts via SRM [NOVELTY]
        3. Score-level fusion combines both predictions
    
    Input formats:
        - PIL Image
        - NumPy array  
        - File path (string)
    
    Output:
        - Binary prediction (real/deepfake)
        - Confidence score
        - Individual stream probabilities
        - Fusion weight used
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Initialize dual-stream image detector.
        
        Args:
            model_path: Not used for dual-stream (loads both models separately)
            device: Computing device (cpu, cuda, mps)
        """
        super().__init__(model_path, device)
        
        # Paths to both stream models (from config)
        # For dual-stream, we use separate CLIP and Noise models
        # Config provides IMAGE_MODEL_PATH which points to the noise model by default
        # We derive CLIP model path by convention (best_clip.pth in same directory)
        base_model_path = Path(settings.IMAGE_MODEL_PATH) if settings.IMAGE_MODEL_PATH else Path("models/best_noise_efficientnet.pth")
        model_dir = base_model_path.parent
        
        self.clip_model_path = model_dir / "best_clip.pth"
        self.noise_model_path = base_model_path
        
        # Fusion weight (can be tuned; 0.5 = equal weight)
        # Optimal weight should be determined via validation set
        self.fusion_weight = 0.8  # w for CLIP stream, (1-w) for Noise stream
        
        # =====================================================================
        # Initialize ImagePreprocessor for face detection and dual transforms
        # =====================================================================
        self.preprocessor = ImagePreprocessor(
            device=str(device),
            target_size=(224, 224),
            face_margin=0.20,
            center_crop_ratio=0.50
        )
        
        # =====================================================================
        # SRM filter for Stream B noise processing
        # =====================================================================
        self.srm_filter = SRMConv2d(in_channels=3)
        
        # Model placeholders
        self.clip_model = None
        self.noise_model = None
        
        logger.debug(
            "Dual-stream image detector configured:\n"
            "  Preprocessing: MTCNN face detection + dual transforms\n"
            "  Stream A: CLIP ViT-L/14 (semantic features)\n"
            "  Stream B: SRM + EfficientNetV2-S (noise residuals) [NOVELTY]\n"
            f"  Fusion weight: {self.fusion_weight} (CLIP) / {1-self.fusion_weight} (Noise)"
        )
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load both dual-stream models.
        
        Model Architecture:
            Stream A: CLIP ViT-L/14 with LN-Tuning
            Stream B: SRM + EfficientNetV2-S
            Fusion: Weighted average of probabilities
        """
        logger.info("Loading dual-stream image detection models...")
        
        if not CLIP_AVAILABLE:
            raise RuntimeError(
                "CLIP (transformers) not available. Install with: pip install transformers"
            )
        
        try:
            # =====================================================================
            # Load Stream A: CLIP Model
            # =====================================================================
            logger.info("Loading CLIP stream (semantic features)...")
            self.clip_model = CLIPClassifier(
                clip_model_name='openai/clip-vit-large-patch14',
                num_classes=2
            )
            
            if self.clip_model_path.exists():
                logger.info(f"Loading CLIP weights from: {self.clip_model_path}")
                checkpoint = torch.load(self.clip_model_path, map_location=self.device, weights_only=True)
                
                # Extract model weights from checkpoint (handles both formats)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    clip_state = checkpoint["model_state_dict"]
                    logger.debug(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    clip_state = checkpoint
                
                self.clip_model.load_state_dict(clip_state, strict=True)
                logger.success("✓ CLIP model loaded successfully")
            else:
                logger.warning(
                    f"CLIP model not found at: {self.clip_model_path}\n"
                    f"Using initialized CLIP (pretrained but not fine-tuned for deepfakes)"
                )
            
            self.clip_model.eval()
            self.clip_model.to(self.device)
            
            # =====================================================================
            # Load Stream B: Noise Model
            # =====================================================================
            logger.info("Loading Noise stream (SRM + EfficientNetV2-S)...")
            self.noise_model = NoiseEfficientNet()
            
            # Move SRM filter to device
            self.srm_filter.to(self.device)
            
            if self.noise_model_path.exists():
                logger.info(f"Loading Noise model weights from: {self.noise_model_path}")
                checkpoint = torch.load(self.noise_model_path, map_location=self.device, weights_only=True)
                
                # Extract model weights from checkpoint (handles both formats)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    noise_state = checkpoint["model_state_dict"]
                    logger.debug(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    noise_state = checkpoint
                
                self.noise_model.load_state_dict(noise_state, strict=True)
                logger.success("✓ Noise model loaded successfully")
            else:
                logger.warning(
                    f"Noise model not found at: {self.noise_model_path}\n"
                    f"Using initialized model (EfficientNetV2-S pretrained on ImageNet)"
                )
            
            self.noise_model.eval()
            self.noise_model.to(self.device)
            
            # =====================================================================
            # Mark as loaded
            # =====================================================================
            self.is_loaded = True
            
            # Enable CUDA optimizations
            if self.device.type == "cuda":
                torch.backends.cudnn.benchmark = True
                logger.info("✓ CUDA optimizations enabled")
            
            logger.success(
                f"✓ Dual-stream image detector ready on {self.device}\n"
                f"  Stream A: CLIP ViT-L/14 (semantic)\n"
                f"  Stream B: SRM + EfficientNetV2-S (noise)\n"
                f"  Fusion: Score-level weighted average"
            )
            
        except Exception as e:
            logger.error(f"Failed to load dual-stream models: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}") from e
    
    def preprocess(self, input_data: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess image data for DUAL-STREAM model input.
        
        ★ GENERATES TWO REPRESENTATIONS:
            1. CLIP representation (RGB with CLIP normalization)
            2. Noise representation (SRM filtered + truncated + normalized)
        
        Args:
            input_data: Image in various formats (str, bytes, np.ndarray, PIL.Image)
        
        Returns:
            Tuple of (clip_tensor, noise_tensor):
                - clip_tensor: (1, 3, 224, 224) - CLIP preprocessed, ready for model
                - noise_tensor: (1, 3, 224, 224) - SRM noise residuals, normalized for EfficientNetV2
        """
        try:
            # =====================================================================
            # Step 1: ImagePreprocessor handles face detection and base transforms
            # =====================================================================
            # This returns:
            #   - clip_tensor: Already CLIP-normalized, ready for model
            #   - noise_base_tensor: Scaled to [0, 255], ready for SRM filter
            clip_tensor, noise_base_tensor = self.preprocessor.process(input_data)
            
            # Move tensors to device
            clip_tensor = clip_tensor.to(self.device)
            noise_base_tensor = noise_base_tensor.to(self.device)
            
            # =====================================================================
            # Step 2: Complete Stream B noise processing
            # =====================================================================
            # Apply SRM filter to extract noise residuals
            with torch.no_grad():
                noise_residual = self.srm_filter(noise_base_tensor)
            
            # Truncate to [-3, 3] range (standard SRM practice)
            noise_residual = torch.clamp(noise_residual, -3.0, 3.0)
            
            # Normalize for EfficientNetV2 (which expects ImageNet-like input)
            # Divide by 3.0 to bring [-3, 3] range to approximately [-1, 1]
            noise_tensor = noise_residual / 3.0
            
            logger.debug(
                f"Preprocessed dual-stream tensors:\n"
                f"  CLIP: shape={clip_tensor.shape}, "
                f"range=[{clip_tensor.min():.3f}, {clip_tensor.max():.3f}]\n"
                f"  Noise: shape={noise_tensor.shape}, "
                f"range=[{noise_tensor.min():.3f}, {noise_tensor.max():.3f}]"
            )
            
            return clip_tensor, noise_tensor
            
        except NoFaceDetectedError:
            # Re-raise validation errors as-is (user-friendly messages)
            raise
        except Exception as e:
            logger.error(f"Dual-stream preprocessing failed: {str(e)}")
            raise ValueError(f"Image preprocessing error: {str(e)}") from e
    
    @torch.no_grad()
    def predict(self, input_tensor: Tuple[torch.Tensor, torch.Tensor]) -> DetectionResult:
        """
        Perform deepfake detection using DUAL-STREAM FUSION.
        
        ★ SCORE-LEVEL FUSION: Combines probabilities from both streams
        
        Args:
            input_tensor: Tuple of (clip_tensor, noise_tensor)
                - clip_tensor: (1, 3, 224, 224) - CLIP preprocessed
                - noise_tensor: (1, 3, 224, 224) - SRM noise residuals
        
        Returns:
            DetectionResult: Prediction with individual stream contributions
        
        Fusion Strategy:
            p_final = (w * p_clip) + ((1 - w) * p_noise)
            where w is the fusion weight (default 0.5)
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_model() first.")
        
        logger.debug("Running dual-stream image deepfake detection...")
        start_time = time.time()
        
        try:
            # Unpack dual-stream inputs
            clip_tensor, noise_tensor = input_tensor
            
            # =====================================================================
            # Stream A: CLIP Forward Pass
            # =====================================================================
            clip_logits = self.clip_model(clip_tensor)  # Shape: (1, 2)
            clip_probs = torch.softmax(clip_logits, dim=1)[0]  # Shape: (2,)
            
            # IMPORTANT: Training convention is Class 0 = Fake, Class 1 = Real
            clip_deepfake_prob = clip_probs[0].item()   # Class 0 = Fake
            clip_real_prob = clip_probs[1].item()  # Class 1 = Real
            
            # =====================================================================
            # Stream B: Noise Forward Pass
            # =====================================================================
            noise_logits = self.noise_model(noise_tensor)  # Shape: (1, 1)
            # IMPORTANT: Training uses BCEWithLogitsLoss with labels (Fake=0, Real=1)
            # So sigmoid(logit) represents P(Real), not P(Fake)
            noise_prob_real = torch.sigmoid(noise_logits)[0].item()  # Probability of being REAL
            
            noise_real_prob = noise_prob_real
            noise_deepfake_prob = 1.0 - noise_prob_real
            
            # =====================================================================
            # Score-Level Fusion - ★ NOVELTY ★
            # =====================================================================
            # Weighted average of deepfake probabilities
            w = self.fusion_weight
            fused_deepfake_prob = (w * clip_deepfake_prob) + ((1 - w) * noise_deepfake_prob)
            fused_real_prob = 1.0 - fused_deepfake_prob
            
            # Determine prediction
            prediction = "deepfake" if fused_deepfake_prob > 0.5 else "real"
            confidence = fused_deepfake_prob
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare metadata with both stream contributions
            metadata = {
                "model_type": "DualStreamFusion",
                "architecture": {
                    "stream_a": "CLIP ViT-L/14 (semantic features)",
                    "stream_b": "SRM + EfficientNetV2-S (noise residuals) [NOVELTY]",
                    "fusion": "Score-level weighted average"
                },
                "fusion_weight": {
                    "clip_weight": w,
                    "noise_weight": 1 - w
                },
                "stream_predictions": {
                    "clip": {
                        "real": clip_real_prob,
                        "deepfake": clip_deepfake_prob
                    },
                    "noise": {
                        "real": noise_real_prob,
                        "deepfake": noise_deepfake_prob
                    }
                },
                "device": str(self.device),
                "novelty_contribution": {
                    "dual_stream_fusion": "Combines semantic and noise-based analysis",
                    "srm_filter": "Extracts high-frequency manipulation artifacts",
                    "advantage": "Robust to both semantic and low-level manipulations"
                }
            }
            
            logger.info(
                f"Dual-stream prediction: {prediction} "
                f"(confidence: {confidence:.2%}, "
                f"CLIP: {clip_deepfake_prob:.2%}, Noise: {noise_deepfake_prob:.2%}, "
                f"time: {processing_time:.3f}s)"
            )
            
            return DetectionResult(
                prediction=prediction,
                confidence=confidence,
                probabilities={
                    "real": fused_real_prob,
                    "deepfake": fused_deepfake_prob
                },
                processing_time=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Dual-stream prediction failed: {str(e)}")
            raise RuntimeError(f"Image detection error: {str(e)}") from e


# =============================================================================
# Utility Functions
# =============================================================================

def detect_image_deepfake(
    image: Any,
    model_path: Optional[str] = None,
    device: str = "cpu"
) -> DetectionResult:
    """
    Convenience function for dual-stream image deepfake detection.
    
    Uses dual-stream fusion architecture:
        - CLIP stream (semantic features)
        - Noise stream (SRM + EfficientNetV2-S) [NOVELTY]
    
    Args:
        image: Image in any supported format (PIL, NumPy, path)
        model_path: Not used (loads both models automatically)
        device: Computing device
    
    Returns:
        DetectionResult: Detection results with dual-stream analysis
    
    Example:
        from app.models.image_detector import detect_image_deepfake
        
        result = detect_image_deepfake("photo.jpg")
        print(f"Prediction: {result.prediction}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"CLIP: {result.metadata['stream_predictions']['clip']}")
        print(f"Noise: {result.metadata['stream_predictions']['noise']}")
    """
    detector = ImageDetector(model_path=model_path, device=device)
    detector.load_model()
    return detector.detect(image)
