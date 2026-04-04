"""
Video Detection Module with a tri-stream multimodal architecture (Image Stream + Audio Stream + Sync Stream).

Stream A: Image (CLIP + SRM Noise) -> 512-D
Stream B: Audio (Mel-spectrogram + LFCC) -> 512-D
Stream C: Sync (CLIP ViT-B/32 + Lightweight CNN + Cross-Attention + BiLSTM) -> 512-D
Fusion:   Concat + MLP -> 1 logit (sigmoid)
"""

import time
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger

from app.config import settings
from app.core.torch_vram import log_cuda_inference_vram, reset_cuda_peak_stats
from app.core.exceptions import NoFaceDetectedError, NoVoiceDetectedError
from app.models.base import DeepfakeDetector, DetectionResult, load_trusted_checkpoint
from app.models.audio_detector import DualFeature_CNN_GRU
from app.models.image_detector import CLIPClassifier, NoiseEfficientNet, SRMConv2d
from app.preprocessing_pipelines.video_preprocessor import VideoPreprocessor


class TriStreamMultimodalNet(nn.Module):
    """
    Tri-stream multimodal network combining Image (CLIP + SRM Noise) + Audio (Mel-spectrogram + LFCC) + Sync (CLIP + Lightweight CNN + Cross-Attention + BiLSTM).
    """

    def __init__(self):
        super().__init__()

        # Image Stream
        self.clip_classifier = CLIPClassifier()
        self.noise_efficientnet = NoiseEfficientNet()
        self.srm_layer = SRMConv2d(in_channels=3)
        self.stream1_fusion = nn.Sequential(
            nn.Linear(1024 + 1280, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
        )

        # Audio Stream
        self.audio_cnn_gru = DualFeature_CNN_GRU()

        # Sync Stream
        self.audio_semantic_cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)), nn.Flatten(1, 2),
        )
        self.audio_semantic_proj = nn.Linear(64, 768)

        try:
            from transformers import CLIPVisionModel
            self.clip_sync_vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
            for p in self.clip_sync_vision.parameters():
                p.requires_grad = False
        except Exception:
            self.clip_sync_vision = None

        self.cross_attention = nn.MultiheadAttention(768, 8, dropout=0.1, batch_first=True)
        self.sync_lstm = nn.LSTM(768, 256, 2, batch_first=True, bidirectional=True, dropout=0.3)

        # Final fusion -> 1 logit
        self.classifier = nn.Sequential(
            nn.Linear(512 + 512 + 512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 1),
        )

    def extract_audio_features(self, mel_x, lfcc_x):
        def _run(resnet, pool, x):
            x = resnet.conv1(x); x = resnet.bn1(x); x = resnet.relu(x); x = resnet.maxpool(x)
            x = resnet.layer1(x); x = resnet.layer2(x); x = resnet.layer3(x); x = resnet.layer4(x)
            return pool(x).squeeze(2).permute(0, 2, 1)

        mel_f = _run(self.audio_cnn_gru.mel_resnet, self.audio_cnn_gru.mel_freq_pool, mel_x)
        lfcc_f = _run(self.audio_cnn_gru.lfcc_resnet, self.audio_cnn_gru.lfcc_freq_pool, lfcc_x)

        stacked = torch.cat([mel_f, lfcc_f], dim=2)
        attn = self.audio_cnn_gru.attention_fc(stacked).unsqueeze(-1)
        dual = torch.stack([mel_f, lfcc_f], dim=2)
        fused = self.audio_cnn_gru.fusion_norm((dual * attn).sum(dim=2))

        _, hidden = self.audio_cnn_gru.gru(fused)
        return torch.cat((hidden[-2], hidden[-1]), dim=1)

    def forward(self, frames_tensor, mel_tensor, lfcc_tensor):
        B, N = frames_tensor.shape[:2]
        device = frames_tensor.device

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 1, 3, 1, 1)
        frames_clip = ((frames_tensor - mean) / std).view(B * N, 3, 224, 224)
        frames_noise = (frames_tensor * 255.0).view(B * N, 3, 224, 224)

        with torch.no_grad():
            clip_feat = torch.nn.functional.normalize(
                self.clip_classifier.clip_vision(frames_clip, return_dict=True).last_hidden_state[:, 0, :], p=2, dim=1,
            )
            noise_res = torch.clamp(self.srm_layer(frames_noise), -3, 3) / 3.0
            noise_feat = torch.flatten(
                self.noise_efficientnet.efficientnet.avgpool(
                    self.noise_efficientnet.efficientnet.features(noise_res)
                ), 1,
            )
            combined = torch.cat([clip_feat, noise_feat], dim=1).view(B, N, -1)
            spatial_pooled = combined.mean(dim=1)

        spatial = self.stream1_fusion(spatial_pooled)

        audio = self.extract_audio_features(mel_tensor, lfcc_tensor)

        if self.clip_sync_vision is not None:
            with torch.no_grad():
                vis_sem = self.clip_sync_vision(pixel_values=frames_clip, return_dict=True).last_hidden_state[:, 0, :]
                vis_sem = vis_sem.view(B, N, 768)
            aud_sem = self.audio_semantic_proj(self.audio_semantic_cnn(mel_tensor).permute(0, 2, 1))
            attended, _ = self.cross_attention(query=vis_sem, key=aud_sem, value=aud_sem)
            _, (h, _) = self.sync_lstm(attended)
            sync = torch.cat((h[-2], h[-1]), dim=1)
        else:
            sync = torch.zeros(B, 512, device=device)

        return self.classifier(torch.cat([spatial, audio, sync], dim=1))


VideoDeepfakeNet = TriStreamMultimodalNet


class VideoDetector(DeepfakeDetector):
    """
    Video detector wrapping TriStreamMultimodalNet model.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        super().__init__(model_path, device)
        self.preprocessor = VideoPreprocessor(
            device=str(device),
            num_frames=15,
            audio_sample_rate=settings.AUDIO_SAMPLE_RATE,
            audio_duration=float(settings.AUDIO_SEGMENT_DURATION),
        )
        self.num_frames = self.preprocessor.num_frames
        self.audio_duration = self.preprocessor.audio_duration

    def load_model(self, model_path: Optional[str] = None) -> None:
        path = model_path or self.model_path
        logger.info("[MODEL:VIDEO] Constructing TriStreamMultimodalNet...")
        self.model = TriStreamMultimodalNet()

        if path and Path(path).exists():
            logger.info("[MODEL:VIDEO] Loading checkpoint from {}", path)
            ckpt = load_trusted_checkpoint(path, self.device)
            sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            try:
                self.model.load_state_dict(sd, strict=True)
                logger.info("[MODEL:VIDEO] All weights loaded successfully (strict=True)")
            except RuntimeError:
                self.model.load_state_dict(sd, strict=False)
                logger.warning("[MODEL:VIDEO] Partial weight loading (strict=False), some layers use default weights")
        else:
            logger.warning("[MODEL:VIDEO] No trained model at '{}', using initialised weights", path)

        self.model.eval().to(self.device)
        self.is_loaded = True
        logger.info(
            "[MODEL:VIDEO] Tri-stream model ready (num_frames={}, audio_duration={}s)",
            self.num_frames,
            self.audio_duration,
        )

    def preprocess(self, input_data: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            if not isinstance(input_data, str):
                raise ValueError("Video detector requires a file path")
            frames, mel, lfcc = self.preprocessor.process(input_data)
            frames = frames.to(self.device)
            mel = mel.to(self.device)
            lfcc = lfcc.to(self.device)
            if frames.dim() == 4:
                frames = frames.unsqueeze(0)
            if mel.dim() == 3:
                mel = mel.unsqueeze(0)
            if lfcc.dim() == 3:
                lfcc = lfcc.unsqueeze(0)
            return frames, mel, lfcc
        except (NoFaceDetectedError, NoVoiceDetectedError):
            raise
        except Exception as e:
            raise ValueError(f"Video preprocessing error: {e}") from e

    @torch.no_grad()
    def predict(self, input_tensor: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> DetectionResult:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        frames, mel, lfcc = input_tensor
        reset_cuda_peak_stats(self.device)
        start = time.time()

        logits = self.model(frames, mel, lfcc)
        prob_real = torch.sigmoid(logits[0, 0]).item()
        deepfake_prob = 1.0 - prob_real

        log_cuda_inference_vram(logger, "MODEL:VIDEO", self.device)

        return DetectionResult(
            prediction="Deepfake" if deepfake_prob > 0.5 else "Real",
            confidence=deepfake_prob,
            probabilities={"real": prob_real, "deepfake": deepfake_prob},
            processing_time=time.time() - start,
            metadata={
                "model_type": "TriStreamMultimodalNet",
                "num_frames": self.num_frames,
                "audio_duration": self.audio_duration,
                "device": str(self.device),
            },
        )
