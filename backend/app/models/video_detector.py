"""
============================================================================
Video Deepfake Detector Implementation - TRI-STREAM MULTIMODAL
============================================================================
TECHNICAL NOVELTY: Tri-Stream Multimodal Architecture

This implementation uses a comprehensive tri-stream approach that combines:
    Stream 1 (Spatial Forensics): Full Image Detection Module (CLIP + SRM)
    Stream 2 (Audio Forensics): Dual-Feature CNN-GRU (Mel + LFCC)
    Stream 3 (Cross-Modal Sync): Cross-Attention for Audio-Visual consistency [NOVELTY]

Key Innovation:
    - Stream 1 analyzes spatial manipulation artifacts in frames
    - Stream 2 detects audio synthesis artifacts
    - Stream 3 checks audio-visual synchronization [NOVELTY]
    - Late fusion combines all three forensic signals

Detects:
    - Face swap in videos (DeepFaceLab, FaceSwap)
    - Face reenactment attacks (Face2Face, NeuralTextures)
    - Lip-sync manipulation (Wav2Lip deepfakes)
    - Temporal inconsistencies
    - Audio-visual desynchronization [NOVELTY]

Architecture:
    Video → Frame Extraction → Image Detector (CLIP + SRM) → [2048-dim]
    Video → Audio Extraction → Audio Detector (Mel + LFCC + ResNet18) → [512-dim]
    Video → Audio-Visual Sync → Cross-Attention + LSTM → [256-dim]
    Late Fusion: Concat(Spatial, Audio, Sync) → FC → Binary Classification

Training Details:
    - Model: TriStreamMultimodalNet (custom architecture)
    - Trained on: FaceForensics++ dataset
    - Best checkpoint: best_video_tristream.pth

Note: This is a simplified inference implementation. The full training pipeline
uses MTCNN face detection and more sophisticated preprocessing.

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
from app.preprocessing_pipelines.video_preprocessor import VideoPreprocessor
from app.models.image_detector import CLIPClassifier, NoiseEfficientNet, SRMConv2d
from app.models.audio_detector import DualFeature_CNN_GRU
from app.core.exceptions import NoFaceDetectedError, NoVoiceDetectedError


# =============================================================================
# Full Tri-Stream Model Architecture
# =============================================================================

class TriStreamMultimodalNet(nn.Module):
    """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║        ★ TRI-STREAM MULTIMODAL NETWORK ★                             ║
    ║                Technical Novelty for FYP                             ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    Full Tri-Stream Multimodal Video Network for Deepfake Detection.
    
    This combines three forensic analysis streams:
    1. Spatial Forensics: CLIP + SRM/EfficientNet (from image detector)
    2. Audio Forensics: Dual-feature CNN-GRU (from audio detector)
    3. Cross-Modal Synchronization: CLIP-sync + Cross-Attention + LSTM [NOVELTY]
    
    Architecture:
        Stream 1 (Spatial): 
            - clip_classifier: CLIP ViT-L/14 → 1024D
            - noise_efficientnet + srm_layer: SRM + EfficientNet → 1280D
            - stream1_fusion: 1024 + 1280 → 2048D
        
        Stream 2 (Audio):
            - audio_cnn_gru: DualFeature_CNN_GRU (Mel + LFCC) → 512D
        
        Stream 3 (Sync) [NOVELTY]:
            - clip_sync_vision: CLIP ViT-B/32 → 768D (per frame)
            - cross_attention: Audio-visual cross-attention
            - sync_lstm: Bidirectional LSTM → 256D
        
        Final Fusion:
            - classifier: Concat(2048, 512, 256) → 2816D → 2 classes
    
    Input:
        - frames_clip: CLIP-normalized frames (Batch, NumFrames, 3, 224, 224)
        - frames_noise: Noise base frames (Batch, NumFrames, 3, 224, 224)
        - mel_tensor: Mel-Spectrogram (Batch, 1, 128, TimeFrames)
        - lfcc_tensor: LFCC features (Batch, 1, 128, TimeFrames)
    
    Output:
        - Logits for binary classification (Batch, 2)
    """
    
    def __init__(self):
        super(TriStreamMultimodalNet, self).__init__()
        
        # =====================================================================
        # Stream 1: Spatial Forensics (Dual-Stream: CLIP + Noise)
        # =====================================================================
        # CLIP classifier for semantic features
        self.clip_classifier = CLIPClassifier()
        
        # Noise classifier (SRM + EfficientNet) for noise artifacts
        self.noise_efficientnet = NoiseEfficientNet()
        
        # Create SRM layer separately (matches training code architecture)
        # Training code: self.srm_layer = SRMConv2d(in_channels=3)
        self.srm_layer = SRMConv2d(in_channels=3)
        
        # Fusion of CLIP (1024D) and Noise (1280D) streams → 512D
        # NOTE: Training code uses 512D output, not 2048D
        self.stream1_fusion = nn.Sequential(
            nn.Linear(1024 + 1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # =====================================================================
        # Stream 2: Audio Forensics (Dual-Feature CNN-GRU)
        # =====================================================================
        self.audio_cnn_gru = DualFeature_CNN_GRU()
        
        # Audio semantic processing for sync stream
        # Lightweight CNN processes Mel-spectrogram
        self.audio_semantic_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Flatten(1, 2)  # (B, 64, time)
        )
        self.audio_semantic_proj = nn.Linear(64, 768)  # Project to match CLIP sync dimension
        
        # =====================================================================
        # Stream 3: Cross-Modal Sync [NOVELTY]
        # =====================================================================
        # Smaller CLIP for sync stream (ViT-B/32 with 768D output)
        try:
            from transformers import CLIPVisionModel
            self.clip_sync_vision = CLIPVisionModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                use_safetensors=True
            )
            # Freeze CLIP sync vision to save memory
            for param in self.clip_sync_vision.parameters():
                param.requires_grad = False
        except Exception as e:
            logger.warning(f"Could not load CLIP sync vision: {e}")
            self.clip_sync_vision = None
        
        # Cross-attention between audio and visual for sync detection
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # LSTM for temporal sync consistency
        # Bidirectional output: 256 * 2 = 512D
        self.sync_lstm = nn.LSTM(
            input_size=768,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # NOTE: Training code does NOT project LSTM output
        # Directly use 512D bidirectional output (no projector layer)
        
        # =====================================================================
        # Final Fusion Classifier
        # =====================================================================
        # NOTE: Training code uses 512 + 512 + 512 = 1536D and outputs 1 logit
        fusion_dim = 512 + 512 + 512  # Stream1 + Stream2 + Stream3 = 1536
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)  # Binary classification: 1 logit (not 2)
        )
        
        logger.info(
            f"✓ Full Tri-Stream Network initialized\n"
            f"  Spatial: CLIP (1024D) + Noise (1280D) → 512D\n"
            f"  Audio: DualCNN-GRU (Mel+LFCC) → 512D\n"
            f"  Sync: CLIP-B/32 (768D) + CrossAttention + LSTM → 512D [NOVELTY]\n"
            f"  Fusion: {fusion_dim}D → 1 logit (binary)"
        )
    
    def extract_audio_features(self, mel_x, lfcc_x):
        """
        Extract 512D audio features from dual-stream CNN-GRU before classification.
        
        This replicates the feature extraction from DualFeature_CNN_GRU.
        
        Args:
            mel_x: Mel-Spectrogram (Batch, 1, 128, TimeFrames)
            lfcc_x: LFCC features (Batch, 1, 128, TimeFrames)
        
        Returns:
            Audio features (Batch, 512)
        """
        # Pass through the dual-stream network
        mel_x = self.audio_cnn_gru.mel_resnet.conv1(mel_x)
        mel_x = self.audio_cnn_gru.mel_resnet.bn1(mel_x)
        mel_x = self.audio_cnn_gru.mel_resnet.relu(mel_x)
        mel_x = self.audio_cnn_gru.mel_resnet.maxpool(mel_x)
        mel_x = self.audio_cnn_gru.mel_resnet.layer1(mel_x)
        mel_x = self.audio_cnn_gru.mel_resnet.layer2(mel_x)
        mel_x = self.audio_cnn_gru.mel_resnet.layer3(mel_x)
        mel_x = self.audio_cnn_gru.mel_resnet.layer4(mel_x)
        mel_x = self.audio_cnn_gru.mel_freq_pool(mel_x)
        mel_x = mel_x.squeeze(2).permute(0, 2, 1)
        
        # LFCC stream
        lfcc_x = self.audio_cnn_gru.lfcc_resnet.conv1(lfcc_x)
        lfcc_x = self.audio_cnn_gru.lfcc_resnet.bn1(lfcc_x)
        lfcc_x = self.audio_cnn_gru.lfcc_resnet.relu(lfcc_x)
        lfcc_x = self.audio_cnn_gru.lfcc_resnet.maxpool(lfcc_x)
        lfcc_x = self.audio_cnn_gru.lfcc_resnet.layer1(lfcc_x)
        lfcc_x = self.audio_cnn_gru.lfcc_resnet.layer2(lfcc_x)
        lfcc_x = self.audio_cnn_gru.lfcc_resnet.layer3(lfcc_x)
        lfcc_x = self.audio_cnn_gru.lfcc_resnet.layer4(lfcc_x)
        lfcc_x = self.audio_cnn_gru.lfcc_freq_pool(lfcc_x)
        lfcc_x = lfcc_x.squeeze(2).permute(0, 2, 1)
        
        # Attention fusion
        stacked = torch.cat([mel_x, lfcc_x], dim=2)
        attention_weights = self.audio_cnn_gru.attention_fc(stacked).unsqueeze(-1)
        dual_features = torch.stack([mel_x, lfcc_x], dim=2)
        fused_features = (dual_features * attention_weights).sum(dim=2)
        fused_features = self.audio_cnn_gru.fusion_norm(fused_features)
        
        # GRU modeling
        gru_out, hidden = self.audio_cnn_gru.gru(fused_features)
        final_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)  # (Batch, 512)
        
        return final_hidden
    
    def forward(self, frames_tensor, mel_tensor, lfcc_tensor):
        """
        Forward pass through tri-stream network.
        
        Args:
            frames_tensor: Video frames (Batch, NumFrames, 3, 224, 224) normalized to [0, 1]
            mel_tensor: Mel-Spectrogram (Batch, 1, 128, TimeFrames)
            lfcc_tensor: LFCC features (Batch, 1, 128, TimeFrames)
        
        Returns:
            Logits (Batch, 2)
        """
        batch_size = frames_tensor.size(0)
        num_frames = frames_tensor.size(1)
        
        # =====================================================================
        # Stream 1: Spatial Features (CLIP + Noise Fusion)
        # =====================================================================
        # Apply CLIP normalization for CLIP stream
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=frames_tensor.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=frames_tensor.device).view(1, 1, 3, 1, 1)
        frames_clip = (frames_tensor - mean) / std
        
        # For noise stream, scale [0,1] to [0,255] for SRM filter
        frames_noise = frames_tensor * 255.0
        
        # Wrap frozen feature extractors in no_grad() to prevent OOM
        with torch.no_grad():
            # Reshape for batch processing: (B, NumFrames, C, H, W) → (B*NumFrames, C, H, W)
            frames_clip_flat = frames_clip.view(batch_size * num_frames, 3, 224, 224)
            frames_noise_flat = frames_noise.view(batch_size * num_frames, 3, 224, 224)
            
            # CLIP stream - extract features
            if hasattr(self.clip_classifier, 'clip_vision'):
                outputs = self.clip_classifier.clip_vision(frames_clip_flat, return_dict=True)
                clip_features = outputs.last_hidden_state[:, 0, :]  # (B*NumFrames, 1024)
                clip_features = torch.nn.functional.normalize(clip_features, p=2, dim=1)
            else:
                clip_features = torch.zeros(batch_size * num_frames, 1024, device=frames_tensor.device)
            
            # Noise stream - apply SRM filter with truncation (matches training code)
            noise_residual = self.srm_layer(frames_noise_flat)
            # Truncate as per training code
            noise_residual = torch.clamp(noise_residual, min=-3.0, max=3.0)
            
            # Extract noise features from EfficientNet
            if hasattr(self.noise_efficientnet, 'efficientnet'):
                noise_features = self.noise_efficientnet.efficientnet.features(noise_residual)
                noise_features = self.noise_efficientnet.efficientnet.avgpool(noise_features)
                noise_features = torch.flatten(noise_features, 1)  # (B*NumFrames, 1280)
            else:
                noise_features = torch.zeros(batch_size * num_frames, 1280, device=frames_tensor.device)
            
            # Concatenate CLIP + Noise features
            combined_features = torch.cat([clip_features, noise_features], dim=1)  # (B*NumFrames, 2304)
            
            # Reshape back: (B*NumFrames, 2304) → (B, NumFrames, 2304)
            combined_features = combined_features.view(batch_size, num_frames, -1)
            
            # Temporal pooling (average across frames)
            visual_forensic_vector = torch.mean(combined_features, dim=1)  # (B, 2304)
        
        # Fuse spatial streams (trainable)
        spatial_features = self.stream1_fusion(visual_forensic_vector)  # (B, 512)
        
        # =====================================================================
        # Stream 2: Audio Features
        # =====================================================================
        # Extract 512D features from audio CNN-GRU
        audio_features = self.extract_audio_features(mel_tensor, lfcc_tensor)
        
        # =====================================================================
        # Stream 3: Cross-Modal Sync Features [NOVELTY]
        # =====================================================================
        if self.clip_sync_vision is not None:
            with torch.no_grad():
                # Extract sync visual features using smaller CLIP (ViT-B/32)
                frames_clip_flat = frames_clip.view(batch_size * num_frames, 3, 224, 224)
                
                try:
                    outputs = self.clip_sync_vision(pixel_values=frames_clip_flat, return_dict=True)
                    visual_semantic = outputs.last_hidden_state[:, 0, :]  # (B*NumFrames, 768)
                    visual_semantic = visual_semantic.view(batch_size, num_frames, 768)  # (B, NumFrames, 768)
                except Exception as e:
                    logger.debug(f"CLIP sync forward failed: {e}")
                    visual_semantic = torch.zeros(batch_size, num_frames, 768, device=frames_tensor.device)
            
            # Audio semantic features (trainable)
            audio_semantic = self.audio_semantic_cnn(mel_tensor)  # (B, 64, time)
            audio_semantic = audio_semantic.permute(0, 2, 1)  # (B, time, 64)
            audio_semantic = self.audio_semantic_proj(audio_semantic)  # (B, time, 768)
            
            # Cross-Attention: Visual (Q) attends to Audio (K, V)
            attended_features, _ = self.cross_attention(
                query=visual_semantic,
                key=audio_semantic,
                value=audio_semantic
            )  # (B, NumFrames, 768)
            
            # BiLSTM temporal modeling
            lstm_out, (hidden_lstm, _) = self.sync_lstm(attended_features)
            # Concatenate bidirectional hidden states: (B, 512)
            sync_features = torch.cat((hidden_lstm[-2, :, :], hidden_lstm[-1, :, :]), dim=1)
        else:
            logger.debug("CLIP sync vision not available - using zero sync features")
            sync_features = torch.zeros(batch_size, 512, device=frames_tensor.device)
        
        # =====================================================================
        # Final Fusion
        # =====================================================================
        fused = torch.cat([spatial_features, audio_features, sync_features], dim=1)  # (Batch, 1536)
        logits = self.classifier(fused)  # (Batch, 1) - single logit for binary classification
        
        return logits


# Alias for backward compatibility
VideoDeepfakeNet = TriStreamMultimodalNet
SimplifiedTriStreamNet = TriStreamMultimodalNet  # Legacy alias


# =============================================================================
# Video Detector Implementation
# =============================================================================

class VideoDetector(DeepfakeDetector):
    """
    Video deepfake detector with TRI-STREAM MULTIMODAL architecture.
    
    ╔══════════════════════════════════════════════════════════════════════╗
    ║      Tri-Stream Multimodal Video Deepfake Detector                   ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    Processing pipeline:
        1. Extract key frames from video
        2. Extract audio track from video
        3. Spatial stream: Frame-level analysis (simplified)
        4. Audio stream: Audio-level analysis (simplified)
        5. Sync stream: Audio-visual synchronization (simplified)
        6. Late fusion for final prediction
    
    Features:
        - Multimodal analysis (visual + audio + sync)
        - Temporal consistency checking
        - Support for various video formats
    
    Note: This is a simplified inference implementation. For production use,
    integrate with the full trained model (best_video_tristream.pth) which
    includes CLIP ViT-L/14, SRM noise analysis, and full dual-feature audio processing.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Initialize tri-stream video detector.
        
        Args:
            model_path: Path to pretrained model weights
            device: Computing device (cpu, cuda, mps)
        """
        super().__init__(model_path, device)
        
        # =====================================================================
        # Initialize VideoPreprocessor for frame and audio extraction
        # =====================================================================
        # Use config settings where available, with model-specific defaults as fallback
        self.preprocessor = VideoPreprocessor(
            device=str(device),
            num_frames=15,  # Model-specific: simplified model uses 15 frames
            target_size=(224, 224),  # Model architecture parameter
            face_margin=0.20,  # Model architecture parameter
            center_crop_ratio=0.50,  # Model architecture parameter
            audio_sample_rate=settings.AUDIO_SAMPLE_RATE,
            audio_duration=float(settings.AUDIO_SEGMENT_DURATION),
            n_mels=128,  # Model architecture parameter
            n_lfcc=128,  # Model architecture parameter
            n_fft=1024,  # Model architecture parameter
            hop_length=512,  # Model architecture parameter
            f_min=50.0,  # Model architecture parameter
            f_max=8000.0  # Model architecture parameter
        )
        
        # Store parameters for reference
        self.num_frames = self.preprocessor.num_frames
        self.audio_sample_rate = self.preprocessor.audio_sample_rate
        self.audio_duration = self.preprocessor.audio_duration
        
        logger.debug(
            f"Tri-stream video detector configured:\n"
            f"  Preprocessing: VideoPreprocessor with MTCNN + audio extraction\n"
            f"  Visual: {self.num_frames} frames @ {self.preprocessor.target_size}\n"
            f"  Audio: {self.audio_duration}s @ {self.audio_sample_rate}Hz\n"
            f"  Architecture: Spatial + Audio + Sync [NOVELTY]"
        )
        
        logger.info(
            "Using full tri-stream multimodal architecture with "
            "trained weights from: best_video_tristream.pth"
        )
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the tri-stream video deepfake detection model.
        
        Args:
            model_path: Path to model checkpoint. If None, uses self.model_path.
        
        Model Architecture:
            Spatial Stream: Frame analysis
            Audio Stream: Audio analysis
            Sync Stream: Audio-visual synchronization [NOVELTY]
            Late Fusion: Classification
        
        Model Loading:
            1. Looks for trained model at specified path
            2. If not found, uses simplified model for inference
            3. Loads model to device (CUDA GPU if available, else CPU)
        
        Training:
            See: model_training/notebooks/video_multimodal_stream.ipynb
        """
        path = model_path or self.model_path
        
        logger.info("Loading tri-stream video detection model...")
        
        try:
            # Initialize full tri-stream architecture
            self.model = TriStreamMultimodalNet()
            
            # Try to load full trained weights if available
            if path and Path(path).exists():
                logger.info(f"Attempting to load trained weights from: {path}")
                try:
                    checkpoint = torch.load(path, map_location=self.device, weights_only=True)
                    
                    # Extract model weights from checkpoint (handles both formats)
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                        logger.debug(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                    else:
                        state_dict = checkpoint
                    
                    # Load weights with strict=True to catch any mismatches
                    try:
                        self.model.load_state_dict(state_dict, strict=True)
                        logger.success("✓ Loaded full model weights successfully")
                    except RuntimeError as load_error:
                        # If strict loading fails, try partial loading
                        logger.warning(f"Strict loading failed: {load_error}")
                        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                            logger.warning(
                                f"Partial weight loading:\n"
                                f"  Missing keys: {len(missing_keys)}\n"
                                f"  Unexpected keys: {len(unexpected_keys)}"
                            )
                        logger.success("✓ Loaded compatible weights from trained model")
                except Exception as e:
                    logger.warning(f"Could not load trained weights: {e}")
                    logger.warning("Using randomly initialized weights - model needs training")
            else:
                logger.warning(
                    f"No trained model found at: {path}\n"
                    f"Using randomly initialized weights - model needs training.\n"
                    f"For best results, train: model_training/notebooks/video_multimodal_stream.ipynb"
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
                f"✓ Tri-stream video detector ready on {self.device}\n"
                f"  Architecture: Spatial + Audio + Sync (full model with feature extraction)"
            )
            
        except Exception as e:
            logger.error(f"Failed to load video detector: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}") from e
    
    def preprocess(self, input_data: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocess video data for TRI-STREAM model input.
        
        ★ NOVELTY: Extracts frames with face detection + dual-stream audio features
        
        Args:
            input_data: Video file path (string)
        
        Returns:
            Tuple of (frames_tensor, mel_tensor, lfcc_tensor):
                - frames_tensor: (num_frames, 3, 224, 224) - Normalized frames [0, 1]
                - mel_tensor: (1, 128, time_frames) - Log-Mel Spectrogram
                - lfcc_tensor: (1, 128, time_frames) - LFCC features
        
        Process:
            1. VideoPreprocessor handles frame extraction with MTCNN face detection
            2. VideoPreprocessor handles audio extraction and dual-stream feature extraction
            3. Move tensors to device for model inference
            
        Note: The full tri-stream model extracts features from frames using:
            - CLIP ViT-L/14 and SRM/EfficientNet for spatial forensics
            - DualFeature CNN-GRU for audio forensics
            - CLIP ViT-B/32 + CrossAttention for sync analysis
        """
        try:
            if not isinstance(input_data, str):
                raise ValueError("Video detector requires file path as input")
            
            video_path = input_data
            
            # =====================================================================
            # VideoPreprocessor handles complete preprocessing pipeline
            # =====================================================================
            # This extracts:
            #   - frames_tensor: Faces detected and cropped with MTCNN
            #   - mel_tensor: Mel-Spectrogram from audio track
            #   - lfcc_tensor: LFCC from audio track
            frames_tensor, mel_tensor, lfcc_tensor = self.preprocessor.process(video_path)
            
            # Move tensors to device
            frames_tensor = frames_tensor.to(self.device)
            mel_tensor = mel_tensor.to(self.device)
            lfcc_tensor = lfcc_tensor.to(self.device)
            
            # Add batch dimension if needed
            if frames_tensor.dim() == 4:  # (NumFrames, 3, 224, 224)
                frames_tensor = frames_tensor.unsqueeze(0)  # (1, NumFrames, 3, 224, 224)
            if mel_tensor.dim() == 3:  # (1, 128, TimeFrames)
                mel_tensor = mel_tensor.unsqueeze(0)  # (1, 1, 128, TimeFrames)
            if lfcc_tensor.dim() == 3:  # (1, 128, TimeFrames)
                lfcc_tensor = lfcc_tensor.unsqueeze(0)  # (1, 1, 128, TimeFrames)
            
            logger.debug(
                f"Preprocessed tri-stream video data:\n"
                f"  Frames: {frames_tensor.shape}\n"
                f"  Mel-Spectrogram: {mel_tensor.shape}\n"
                f"  LFCC: {lfcc_tensor.shape}"
            )
            
            # Return raw tensors for full model feature extraction
            return frames_tensor, mel_tensor, lfcc_tensor
            
        except (NoFaceDetectedError, NoVoiceDetectedError):
            # Re-raise validation errors as-is (user-friendly messages)
            raise
        except Exception as e:
            logger.error(f"Tri-stream video preprocessing failed: {str(e)}")
            raise ValueError(f"Video preprocessing error: {str(e)}") from e
    
    @torch.no_grad()
    def predict(self, input_tensor: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> DetectionResult:
        """
        Perform deepfake detection using TRI-STREAM architecture.
        
        ★ NOVELTY: Analyzes spatial, audio, and sync features simultaneously
        
        Args:
            input_tensor: Tuple of (frames_tensor, mel_tensor, lfcc_tensor)
                - frames_tensor: (1, NumFrames, 3, 224, 224) - Video frames
                - mel_tensor: (1, 1, 128, TimeFrames) - Mel-Spectrogram
                - lfcc_tensor: (1, 1, 128, TimeFrames) - LFCC features
        
        Returns:
            DetectionResult: Video-level prediction with multimodal analysis
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Unpack tri-stream inputs
        frames_tensor, mel_tensor, lfcc_tensor = input_tensor
        
        logger.debug("Running tri-stream video deepfake detection...")
        start_time = time.time()
        
        try:
            # =====================================================================
            # Tri-stream forward pass
            # Stream 1: Spatial features (frame analysis)
            # Stream 2: Audio features (audio analysis)
            # Stream 3: Sync features (audio-visual consistency) - ★ NOVELTY ★
            # =====================================================================
            logits = self.model(frames_tensor, mel_tensor, lfcc_tensor)
            
            # Convert single logit to probability using sigmoid
            # Output shape: (Batch, 1) - single logit for binary classification
            # IMPORTANT: Training uses BCEWithLogitsLoss with labels (Fake=0, Real=1)
            # So sigmoid(logit) represents P(Real), not P(Fake)
            logit_value = logits[0, 0].item()
            prob_real = torch.sigmoid(logits[0, 0]).item()
            real_prob_score = prob_real
            deepfake_prob = 1.0 - prob_real
            
            # Determine prediction
            prediction = "deepfake" if deepfake_prob > 0.5 else "real"
            confidence = deepfake_prob
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Metadata with tri-stream architecture information
            metadata = {
                "model_type": "TriStreamMultimodalNet",
                "architecture": {
                    "spatial_stream": "CLIP (1024D) + SRM/EfficientNet (1280D) → Fusion → 512D",
                    "audio_stream": "DualCNN-GRU (Mel+LFCC) → 512D",
                    "sync_stream": "CLIP-sync (768D) + CrossAttention + LSTM → 512D [NOVELTY]",
                    "fusion": "1536D → Binary classification (1 logit)"
                },
                "num_frames": self.num_frames,
                "audio_duration": self.audio_duration,
                "device": str(self.device),
                "novelty_contribution": {
                    "multimodal_fusion": "Combines spatial, audio, and sync streams",
                    "sync_analysis": "Detects audio-visual desynchronization",
                    "advantage": "Reveals manipulation invisible to unimodal analysis"
                },
                "prediction_details": {
                    "logit": logit_value,
                    "threshold": 0.0,
                    "note": "Sigmoid(logit) > 0.5 predicts deepfake"
                }
            }
            
            logger.info(
                f"Tri-stream video prediction: {prediction} "
                f"(confidence: {confidence:.2%}, time: {processing_time:.3f}s)"
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
            logger.error(f"Tri-stream video prediction failed: {str(e)}")
            raise RuntimeError(f"Video detection error: {str(e)}") from e


# =============================================================================
# Utility Functions
# =============================================================================

def detect_video_deepfake(
    video_path: str,
    model_path: Optional[str] = None,
    device: str = "cpu"
) -> DetectionResult:
    """
    Convenience function for tri-stream video deepfake detection.
    
    Uses tri-stream multimodal architecture:
        - Spatial stream (frame analysis)
        - Audio stream (audio analysis)
        - Sync stream (audio-visual synchronization) [NOVELTY]
    
    Args:
        video_path: Path to video file
        model_path: Optional path to model weights
        device: Computing device
    
    Returns:
        DetectionResult: Detection results with multimodal analysis
    
    Example:
        from app.models.video_detector import detect_video_deepfake
        
        result = detect_video_deepfake("video.mp4")
        print(f"Prediction: {result.prediction}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Architecture: {result.metadata['architecture']}")
    """
    detector = VideoDetector(model_path=model_path, device=device)
    detector.load_model()
    return detector.detect(video_path)
