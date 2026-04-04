"""
Audio Detection Module with a dual-stream CNN-GRU architecture (Spectral Stream + Cepstral Stream).

Stream A: Mel-spectrogram -> ResNet-18
Stream B: LFCC -> ResNet-18
Fusion:   Attention-based -> BiGRU -> 2-class head (Deepfake, Real)
"""

import time
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models
from loguru import logger

from app.config import settings
from app.core.torch_vram import log_cuda_inference_vram, reset_cuda_peak_stats
from app.core.exceptions import NoVoiceDetectedError
from app.models.base import DeepfakeDetector, DetectionResult, load_trusted_checkpoint
from app.preprocessing_pipelines.audio_preprocessor import AudioPreprocessor


class DualFeature_CNN_GRU(nn.Module):
    """
    Dual-stream CNN-GRU architecture for audio deepfake detection.
    """

    def __init__(self):
        super().__init__()

        # Mel ResNet-18
        self.mel_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        w = self.mel_resnet.conv1.weight.clone().sum(dim=1, keepdim=True)
        self.mel_resnet.conv1 = nn.Conv2d(1, 64, 7, stride=(2, 2), padding=3, bias=False)
        self.mel_resnet.conv1.weight = nn.Parameter(w)
        self.mel_resnet.avgpool = nn.Identity()
        self.mel_resnet.fc = nn.Identity()

        # LFCC ResNet-18
        self.lfcc_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        w = self.lfcc_resnet.conv1.weight.clone().sum(dim=1, keepdim=True)
        self.lfcc_resnet.conv1 = nn.Conv2d(1, 64, 7, stride=(2, 2), padding=3, bias=False)
        self.lfcc_resnet.conv1.weight = nn.Parameter(w)
        self.lfcc_resnet.avgpool = nn.Identity()
        self.lfcc_resnet.fc = nn.Identity()

        # Downsample only on frequency axis to preserve temporal dimension
        for resnet in (self.mel_resnet, self.lfcc_resnet):
            resnet.conv1.stride = (2, 1)
            resnet.maxpool.stride = (2, 1)
            for layer in (resnet.layer2, resnet.layer3, resnet.layer4):
                layer[0].conv1.stride = (2, 1)
                if layer[0].downsample is not None:
                    layer[0].downsample[0].stride = (2, 1)

        self.mel_freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.lfcc_freq_pool = nn.AdaptiveAvgPool2d((1, None))

        # Attention fusion
        self.attention_fc = nn.Sequential(
            nn.Linear(512 * 2, 256), nn.Tanh(), nn.Linear(256, 2), nn.Softmax(dim=-1),
        )
        self.fusion_norm = nn.LayerNorm(512)
        self.temporal_dropout = nn.Dropout(0.2)

        # BiGRU
        self.gru = nn.GRU(512, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)

        # 2-class head (Deepfake, Real) — class index 0 = Deepfake, 1 = Real
        self.classifier = nn.Sequential(
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 2),
        )

    def forward(self, mel_x, lfcc_x):
        def _resnet_stream(resnet, pool, x):
            x = resnet.conv1(x); x = resnet.bn1(x); x = resnet.relu(x); x = resnet.maxpool(x)
            x = resnet.layer1(x); x = resnet.layer2(x); x = resnet.layer3(x); x = resnet.layer4(x)
            return pool(x).squeeze(2).permute(0, 2, 1)

        mel_feat = _resnet_stream(self.mel_resnet, self.mel_freq_pool, mel_x)
        lfcc_feat = _resnet_stream(self.lfcc_resnet, self.lfcc_freq_pool, lfcc_x)

        stacked = torch.cat([mel_feat, lfcc_feat], dim=2)
        attn = self.attention_fc(stacked).unsqueeze(-1)
        dual = torch.stack([mel_feat, lfcc_feat], dim=2)
        fused = self.fusion_norm((dual * attn).sum(dim=2))

        _, hidden = self.gru(fused)
        final = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(final)


AudioDeepfakeNet = DualFeature_CNN_GRU


class AudioDetector(DeepfakeDetector):
    """
    Audio detector wrapping DualFeature_CNN_GRU model.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        super().__init__(model_path, device)
        self.preprocessor = AudioPreprocessor(
            sample_rate=settings.AUDIO_SAMPLE_RATE,
            segment_duration=float(settings.AUDIO_SEGMENT_DURATION),
        )
        self.sample_rate = self.preprocessor.sample_rate
        self.segment_duration = self.preprocessor.segment_duration
        self.segment_samples = self.preprocessor.segment_samples

    def load_model(self, model_path: Optional[str] = None) -> None:
        path = model_path or self.model_path
        logger.info("[MODEL:AUDIO] Constructing DualFeature_CNN_GRU network...")
        self.model = DualFeature_CNN_GRU()
        if path and Path(path).exists():
            logger.info("[MODEL:AUDIO] Loading checkpoint from {}", path)
            ckpt = load_trusted_checkpoint(path, self.device)
            sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(sd, strict=True)
            logger.info("[MODEL:AUDIO] All weights loaded successfully")
        else:
            logger.warning("[MODEL:AUDIO] No trained model at '{}', using initialised weights", path)
        self.model.eval().to(self.device)
        self.is_loaded = True
        logger.info(
            "[MODEL:AUDIO] Dual-stream model ready (sample_rate={}, segment_duration={}s)",
            self.sample_rate,
            self.segment_duration,
        )

    def preprocess(self, input_data: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            mel, lfcc = self.preprocessor.process(input_data, segment=True)
            return mel.to(self.device), lfcc.to(self.device)
        except NoVoiceDetectedError:
            raise
        except Exception as e:
            raise ValueError(f"Audio preprocessing error: {e}") from e

    @torch.no_grad()
    def predict(self, input_tensor: Tuple[torch.Tensor, torch.Tensor]) -> DetectionResult:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        mel_batch, lfcc_batch = input_tensor
        n_seg = mel_batch.shape[0]
        reset_cuda_peak_stats(self.device)
        start = time.time()

        preds = []
        for i in range(0, n_seg, 8):
            logits = self.model(mel_batch[i : i + 8], lfcc_batch[i : i + 8])
            preds.append(torch.softmax(logits, dim=1).cpu())

        avg = torch.cat(preds).mean(dim=0)
        deepfake_prob, real_prob = avg[0].item(), avg[1].item()

        log_cuda_inference_vram(logger, "MODEL:AUDIO", self.device)

        return DetectionResult(
            prediction="Deepfake" if deepfake_prob > 0.5 else "Real",
            confidence=deepfake_prob,
            probabilities={"real": real_prob, "deepfake": deepfake_prob},
            processing_time=time.time() - start,
            metadata={
                "model_type": "DualFeature_CNN_GRU",
                "num_segments_analyzed": n_seg,
                "segment_duration": self.segment_duration,
                "device": str(self.device),
            },
        )
