"""
============================================================================
Audio Deepfake Detector Implementation - DUAL-STREAM CNN-GRU
============================================================================
TECHNICAL NOVELTY: Dual-Feature Audio Analysis with ResNet18 + GRU

This implementation uses a dual-stream approach that processes two complementary
audio features to detect deepfake audio:
    Stream A (Mel-Spectrogram): Captures low-frequency human phonetic structures
    Stream B (LFCC): Captures high-frequency synthetic vocoder artifacts [NOVELTY]

Key Innovation:
    - Mel-Spectrogram stream captures natural human speech patterns
    - LFCC stream detects synthetic vocoder artifacts invisible to Mel analysis
    - Attention-based fusion dynamically weights both streams
    - Bidirectional GRU for temporal consistency checking

Detects:
    - Voice cloning (e.g., Real-Time Voice Cloning, Lyrebird)
    - Speech synthesis (e.g., WaveNet, Tacotron, FastSpeech)
    - Voice conversion attacks (e.g., StarGAN-VC, AutoVC)
    - Audio splicing and manipulation

Architecture:
    Waveform (4s @ 16kHz) → Mel-Spec (128 bins) → ResNet18 → [512-dim]
    Waveform (4s @ 16kHz) → LFCC (128 bins) → ResNet18 → [512-dim]
    Attention Fusion → GRU (Bidirectional, 2 layers) → Binary Classification

Training Details:
    - Model: DualFeature_CNN_GRU (from model_training/utils/models.py)
    - Trained on: For-2-Seconds & In-The-Wild datasets
    - Best checkpoint: best_audio_cnn_gru.pth

Author: Senior ML Engineer & Backend Architect
Date: 2026-03-24
============================================================================
"""

import time
from typing import Optional, Any, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from loguru import logger

from app.models.base import DeepfakeDetector, DetectionResult
from app.config import settings
from app.preprocessing_pipelines.audio_preprocessor import AudioPreprocessor
from app.core.exceptions import NoVoiceDetectedError


# =============================================================================
# Dual-Stream Audio Model Architecture
# =============================================================================

class DualFeature_CNN_GRU(nn.Module):
    """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║           ★ DUAL-STREAM AUDIO NETWORK ★                              ║
    ║                    Technical Novelty for FYP                         ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    Dual-Stream Audio Deepfake Detection Model using ResNet18 Backbones + Attention Fusion.
    
    This model processes two complementary audio features in parallel:
    1. Mel-Spectrogram Stream: Captures low-frequency human phonetic structures
    2. LFCC Stream: Captures high-frequency synthetic vocoder artifacts [NOVELTY]
    
    Architecture:
    - Two ResNet18 feature extractors (one for Mel, one for LFCC)
    - Modified stride patterns to preserve temporal dimension
    - Attention-based fusion: Dynamically weighted combination of streams
    - Bidirectional GRU for temporal modeling
    - Classification head for binary deepfake detection
    
    Input:
        - mel_x: Log-Mel Spectrogram (Batch, 1, 128, TimeFrames)
        - lfcc_x: LFCC features (Batch, 1, 128, TimeFrames)
    
    Output:
        - Logits for binary classification (Batch, 2)
    """
    
    def __init__(self):
        super(DualFeature_CNN_GRU, self).__init__()
        
        # =====================================================================
        # Mel-Spectrogram ResNet18 Stream
        # =====================================================================
        # Load ResNet18 with pre-trained ImageNet weights
        self.mel_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Adapt conv1 from 3 channels (RGB) to 1 channel (audio)
        weight = self.mel_resnet.conv1.weight.clone()
        new_weight = weight.sum(dim=1, keepdim=True)
        self.mel_resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.mel_resnet.conv1.weight = nn.Parameter(new_weight)
        
        # Remove avgpool and fc layers to preserve time dimension
        self.mel_resnet.avgpool = nn.Identity()
        self.mel_resnet.fc = nn.Identity()
        
        # =====================================================================
        # LFCC ResNet18 Stream
        # =====================================================================
        # Load ResNet18 with pre-trained ImageNet weights
        self.lfcc_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Adapt conv1 from 3 channels (RGB) to 1 channel (audio)
        weight = self.lfcc_resnet.conv1.weight.clone()
        new_weight = weight.sum(dim=1, keepdim=True)
        self.lfcc_resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.lfcc_resnet.conv1.weight = nn.Parameter(new_weight)
        
        # Remove avgpool and fc layers to preserve time dimension
        self.lfcc_resnet.avgpool = nn.Identity()
        self.lfcc_resnet.fc = nn.Identity()
        
        # =====================================================================
        # Fix ResNet Temporal Decimation
        # =====================================================================
        # Modify stride patterns to preserve time dimension (only downsample frequency)
        
        # Fix Mel ResNet
        self.mel_resnet.conv1.stride = (2, 1)
        self.mel_resnet.maxpool.stride = (2, 1)
        
        for layer in [self.mel_resnet.layer2, self.mel_resnet.layer3, self.mel_resnet.layer4]:
            layer[0].conv1.stride = (2, 1)
            if layer[0].downsample is not None:
                layer[0].downsample[0].stride = (2, 1)
        
        # Fix LFCC ResNet
        self.lfcc_resnet.conv1.stride = (2, 1)
        self.lfcc_resnet.maxpool.stride = (2, 1)
        
        for layer in [self.lfcc_resnet.layer2, self.lfcc_resnet.layer3, self.lfcc_resnet.layer4]:
            layer[0].conv1.stride = (2, 1)
            if layer[0].downsample is not None:
                layer[0].downsample[0].stride = (2, 1)
        
        # Adaptive pooling to reduce frequency dimension only
        self.mel_freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.lfcc_freq_pool = nn.AdaptiveAvgPool2d((1, None))
        
        # =====================================================================
        # Attention-Based Fusion - ★ NOVELTY ★
        # =====================================================================
        self.attention_fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.Tanh(),
            nn.Linear(256, 2),
            nn.Softmax(dim=-1)
        )
        
        # =====================================================================
        # Fusion Normalization
        # =====================================================================
        self.fusion_norm = nn.LayerNorm(512)
        
        # =====================================================================
        # Temporal Dropout
        # =====================================================================
        self.temporal_dropout = nn.Dropout(0.2)
        
        # =====================================================================
        # GRU Sequence Modeling
        # =====================================================================
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # =====================================================================
        # Classification Head
        # =====================================================================
        # Bidirectional GRU outputs 2 * hidden_size = 512
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
        
        logger.info(
            f"✓ Dual-Stream Audio Network initialized\n"
            f"  Mel-Spectrogram Stream: ResNet18 → 512D\n"
            f"  LFCC Stream: ResNet18 → 512D [NOVELTY]\n"
            f"  Fusion: Attention-based → GRU → Binary Classification"
        )
    
    def forward(self, mel_x, lfcc_x):
        """
        Forward pass with dual-stream input and attention-based fusion.
        
        Args:
            mel_x: Log-Mel Spectrogram tensor of shape (Batch, 1, 128, TimeFrames)
            lfcc_x: LFCC tensor of shape (Batch, 1, 128, TimeFrames)
        
        Returns:
            Logits of shape (Batch, 2)
        """
        # =====================================================================
        # Mel-Spectrogram ResNet18 Stream
        # =====================================================================
        mel_x = self.mel_resnet.conv1(mel_x)
        mel_x = self.mel_resnet.bn1(mel_x)
        mel_x = self.mel_resnet.relu(mel_x)
        mel_x = self.mel_resnet.maxpool(mel_x)
        mel_x = self.mel_resnet.layer1(mel_x)
        mel_x = self.mel_resnet.layer2(mel_x)
        mel_x = self.mel_resnet.layer3(mel_x)
        mel_x = self.mel_resnet.layer4(mel_x)
        
        # Pool frequency dimension only, preserve time
        mel_x = self.mel_freq_pool(mel_x)
        mel_x = mel_x.squeeze(2)
        mel_x = mel_x.permute(0, 2, 1)
        
        # =====================================================================
        # LFCC ResNet18 Stream
        # =====================================================================
        lfcc_x = self.lfcc_resnet.conv1(lfcc_x)
        lfcc_x = self.lfcc_resnet.bn1(lfcc_x)
        lfcc_x = self.lfcc_resnet.relu(lfcc_x)
        lfcc_x = self.lfcc_resnet.maxpool(lfcc_x)
        lfcc_x = self.lfcc_resnet.layer1(lfcc_x)
        lfcc_x = self.lfcc_resnet.layer2(lfcc_x)
        lfcc_x = self.lfcc_resnet.layer3(lfcc_x)
        lfcc_x = self.lfcc_resnet.layer4(lfcc_x)
        
        # Pool frequency dimension only, preserve time
        lfcc_x = self.lfcc_freq_pool(lfcc_x)
        lfcc_x = lfcc_x.squeeze(2)
        lfcc_x = lfcc_x.permute(0, 2, 1)
        
        # =====================================================================
        # Attention-Based Fusion - ★ NOVELTY ★
        # =====================================================================
        stacked = torch.cat([mel_x, lfcc_x], dim=2)
        attention_weights = self.attention_fc(stacked)
        attention_weights = attention_weights.unsqueeze(-1)
        dual_features = torch.stack([mel_x, lfcc_x], dim=2)
        fused_features = (dual_features * attention_weights).sum(dim=2)
        fused_features = self.fusion_norm(fused_features)
        
        # =====================================================================
        # GRU Sequence Modeling
        # =====================================================================
        gru_out, hidden = self.gru(fused_features)
        final_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        # =====================================================================
        # Classification
        # =====================================================================
        logits = self.classifier(final_hidden)
        
        return logits


# Alias for backward compatibility
AudioDeepfakeNet = DualFeature_CNN_GRU


# =============================================================================
# Audio Detector Implementation
# =============================================================================

class AudioDetector(DeepfakeDetector):
    """
    Audio deepfake detector with DUAL-STREAM CNN-GRU architecture.
    
    ╔══════════════════════════════════════════════════════════════════════╗
    ║        Dual-Stream Audio Deepfake Detector                           ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    Supports:
        - Voice cloning detection (Real-Time Voice Cloning, Lyrebird)
        - Speech synthesis detection (WaveNet, Tacotron, FastSpeech)
        - Voice conversion detection (StarGAN-VC, AutoVC)
        - Audio splicing detection
    
    Technical Approach:
        1. Extract Mel-Spectrogram (128 bins) for phonetic patterns
        2. Extract LFCC (128 bins) for vocoder artifacts [NOVELTY]
        3. Process both streams through ResNet18 backbones
        4. Attention-based fusion for optimal weighting
        5. Bidirectional GRU for temporal modeling
        6. Binary classification (real vs deepfake)
    
    Input formats:
        - Audio file path (mp3, wav, flac, etc.)
        - NumPy array (audio waveform)
    
    Features:
        - Dual-stream processing (Mel + LFCC)
        - Segment-based analysis (4 seconds per segment)
        - Temporal consistency checking via GRU
        - Attention-weighted feature fusion
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Initialize dual-stream audio detector.
        
        Args:
            model_path: Path to pretrained model weights
            device: Computing device (cpu, cuda, mps)
        """
        super().__init__(model_path, device)
        
        # =====================================================================
        # Initialize AudioPreprocessor for dual-stream feature extraction
        # =====================================================================
        # Use config settings where available, with model-specific defaults as fallback
        self.preprocessor = AudioPreprocessor(
            sample_rate=settings.AUDIO_SAMPLE_RATE,
            segment_duration=float(settings.AUDIO_SEGMENT_DURATION),
            n_mels=128,  # Model architecture parameter
            n_lfcc=128,  # Model architecture parameter
            n_fft=1024,  # Model architecture parameter
            hop_length=512,  # Model architecture parameter
            f_min=50.0,  # Model architecture parameter
            f_max=8000.0  # Model architecture parameter
        )
        
        # Store parameters for reference
        self.sample_rate = self.preprocessor.sample_rate
        self.segment_duration = self.preprocessor.segment_duration
        self.segment_samples = self.preprocessor.segment_samples
        
        logger.debug(
            f"Dual-stream audio detector configured:\n"
            f"  Preprocessing: AudioPreprocessor with dual-stream extraction\n"
            f"  Sample rate: {self.sample_rate}Hz\n"
            f"  Segment duration: {self.segment_duration}s ({self.segment_samples} samples)\n"
            f"  Stream A: Mel-Spectrogram ({self.preprocessor.n_mels} bins)\n"
            f"  Stream B: LFCC ({self.preprocessor.n_lfcc} coefficients) [NOVELTY]"
        )
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the dual-stream audio deepfake detection model.
        
        Args:
            model_path: Path to model checkpoint. If None, uses self.model_path.
        
        Model Architecture:
            Mel-Spectrogram Stream: ResNet18 (pretrained)
            LFCC Stream: ResNet18 (pretrained) [NOVELTY]
            Fusion: Attention-based
            Temporal: Bidirectional GRU (2 layers)
            Classification: FC layers with dropout
        
        Model Loading:
            1. Looks for trained model at specified path
            2. If not found, uses initialized model (with pretrained ResNet18 backbones)
            3. Loads model to device (CUDA GPU if available, else CPU)
        
        Training:
            See: model_training/notebooks/audio_CNN_GRU.ipynb
        """
        path = model_path or self.model_path
        
        logger.info("Loading dual-stream audio detection model...")
        
        try:
            # Initialize dual-stream architecture
            self.model = DualFeature_CNN_GRU()
            
            # Load trained weights if available
            if path and Path(path).exists():
                logger.info(f"Loading trained weights from: {path}")
                try:
                    checkpoint = torch.load(path, map_location=self.device, weights_only=True)
                    
                    # Extract model weights from checkpoint (handles both formats)
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                        logger.debug(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                    else:
                        state_dict = checkpoint
                    
                    self.model.load_state_dict(state_dict, strict=True)
                    logger.success("✓ Trained dual-stream audio model loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load weights: {e}. Using initialized model.")
            else:
                logger.warning(
                    f"No trained model found at: {path}\n"
                    f"Using initialized model (ResNet18 backbones pretrained on ImageNet).\n"
                    f"For best results, train: model_training/notebooks/audio_CNN_GRU.ipynb"
                )
            
            # Set to evaluation mode and move to device
            self.model.eval()
            self.model.to(self.device)
            self.is_loaded = True
            
            # Enable CUDA optimizations if using GPU
            if self.device.type == "cuda":
                torch.backends.cudnn.benchmark = True
                logger.info("✓ CUDA optimizations enabled")
            
            logger.success(
                f"✓ Dual-stream audio detector ready on {self.device}\n"
                f"  Architecture: Mel-Spec (ResNet18) + LFCC (ResNet18) + Attention + GRU"
            )
            
        except Exception as e:
            logger.error(f"Failed to load audio detector: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}") from e
    
    def preprocess(self, input_data: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess audio data for DUAL-STREAM model input.
        
        ★ NOVELTY: Extracts BOTH Mel-Spectrogram and LFCC representations
        
        Args:
            input_data: Audio file path, bytes, or waveform array
        
        Returns:
            Tuple of (mel_batch, lfcc_batch):
                - mel_batch: (num_segments, 1, 128, time_frames) - Log-Mel Spectrograms
                - lfcc_batch: (num_segments, 1, 128, time_frames) - LFCC features
        
        Processing Pipeline:
            1. AudioPreprocessor handles loading, segmentation, and feature extraction
            2. Move tensors to device for model inference
        """
        try:
            # =====================================================================
            # AudioPreprocessor handles complete preprocessing pipeline
            # =====================================================================
            mel_batch, lfcc_batch = self.preprocessor.process(input_data, segment=True)
            
            # Move tensors to device
            mel_batch = mel_batch.to(self.device)
            lfcc_batch = lfcc_batch.to(self.device)
            
            logger.debug(
                f"Preprocessed dual-stream audio tensors:\n"
                f"  Mel-Spectrogram: {mel_batch.shape}\n"
                f"  LFCC: {lfcc_batch.shape}"
            )
            
            return mel_batch, lfcc_batch
            
        except NoVoiceDetectedError:
            # Re-raise validation errors as-is (user-friendly messages)
            raise
        except Exception as e:
            logger.error(f"Dual-stream audio preprocessing failed: {str(e)}")
            raise ValueError(f"Audio preprocessing error: {str(e)}") from e
    
    @torch.no_grad()
    def predict(self, input_tensor: Tuple[torch.Tensor, torch.Tensor]) -> DetectionResult:
        """
        Perform deepfake detection using DUAL-STREAM CNN-GRU architecture.
        
        ★ NOVELTY: Analyzes both Mel-Spectrogram and LFCC features simultaneously
        
        Args:
            input_tensor: Tuple of (mel_batch, lfcc_batch)
                - mel_batch: (num_segments, 1, 128, time_frames) - Mel-Spectrograms
                - lfcc_batch: (num_segments, 1, 128, time_frames) - LFCC features
        
        Returns:
            DetectionResult: Audio-level prediction with segment analysis
        
        Aggregation Strategy:
            - Analyze each segment independently with both streams
            - Average probabilities across segments
            - Generate final audio-level prediction
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Unpack dual-stream inputs
        mel_batch, lfcc_batch = input_tensor
        num_segments = mel_batch.shape[0]
        
        logger.debug(f"Running dual-stream audio deepfake detection on {num_segments} segments...")
        start_time = time.time()
        
        try:
            # Process segments in batches
            batch_size = 8
            segment_predictions = []
            
            for i in range(0, num_segments, batch_size):
                mel_segment = mel_batch[i:i + batch_size]
                lfcc_segment = lfcc_batch[i:i + batch_size]
                
                # =====================================================================
                # Dual-stream forward pass
                # Stream A: Mel-Spectrogram (ResNet18)
                # Stream B: LFCC (ResNet18) [NOVELTY]
                # Attention Fusion + GRU
                # =====================================================================
                logits = self.model(mel_segment, lfcc_segment)
                probs = torch.softmax(logits, dim=1)
                
                segment_predictions.append(probs.cpu())
            
            # Concatenate all predictions
            all_probs = torch.cat(segment_predictions, dim=0)  # (num_segments, 2)
            
            # Aggregate segment-level predictions
            avg_probs = all_probs.mean(dim=0)  # (2,)
            
            # Extract probabilities
            # Class 0: Fake, Class 1: Real (as per training labels)
            fake_prob = avg_probs[0].item()
            real_prob = avg_probs[1].item()
            
            # Convert to our naming convention
            deepfake_prob = fake_prob
            real_prob_score = real_prob
            
            # Determine prediction
            prediction = "deepfake" if deepfake_prob > 0.5 else "real"
            confidence = deepfake_prob
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Metadata with dual-stream architecture information
            metadata = {
                "model_type": "DualFeature_CNN_GRU",
                "architecture": {
                    "stream_a": "Mel-Spectrogram → ResNet18",
                    "stream_b": "LFCC → ResNet18 [NOVELTY]",
                    "fusion": "Attention-based",
                    "temporal": "Bidirectional GRU (2 layers)"
                },
                "num_segments_analyzed": num_segments,
                "segment_duration": self.segment_duration,
                "sample_rate": self.sample_rate,
                "device": str(self.device),
                "novelty_contribution": {
                    "lfcc_stream": "Captures high-frequency synthetic vocoder artifacts",
                    "attention_fusion": "Dynamically weights Mel and LFCC streams",
                    "advantage": "Detects vocoder artifacts invisible to Mel-Spectrogram analysis"
                }
            }
            
            logger.info(
                f"Dual-stream audio prediction: {prediction} "
                f"(confidence: {confidence:.2%}, "
                f"segments: {num_segments}, time: {processing_time:.3f}s)"
            )
            
            return DetectionResult(
                prediction=prediction,
                confidence=confidence,
                probabilities={
                    "real": real_prob_score,
                    "deepfake": deepfake_prob
                },
                processing_time=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Dual-stream audio prediction failed: {str(e)}")
            raise RuntimeError(f"Audio detection error: {str(e)}") from e


# =============================================================================
# Utility Functions
# =============================================================================

def detect_audio_deepfake(
    audio_input: Any,
    model_path: Optional[str] = None,
    device: str = "cpu"
) -> DetectionResult:
    """
    Convenience function for dual-stream audio deepfake detection.
    
    Uses novel dual-stream architecture:
        - Mel-Spectrogram stream (ResNet18) for phonetic patterns
        - LFCC stream (ResNet18) for vocoder artifacts [NOVELTY]
    
    Args:
        audio_input: Audio file path or waveform array
        model_path: Optional path to model weights
        device: Computing device
    
    Returns:
        DetectionResult: Detection results with dual-stream analysis
    
    Example:
        from app.models.audio_detector import detect_audio_deepfake
        
        result = detect_audio_deepfake("audio.wav")
        print(f"Prediction: {result.prediction}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Architecture: {result.metadata['architecture']}")
    """
    detector = AudioDetector(model_path=model_path, device=device)
    detector.load_model()
    return detector.detect(audio_input)
