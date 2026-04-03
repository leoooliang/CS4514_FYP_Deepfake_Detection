"""
Video deepfake detection model.

Contains:
- VideoTriStreamDetector: Video tri-stream deepfake detection model.
    Stream 1 (Spatial): CLIP ViT-L/14 + SRM Noise (EfficientNetV2-S) with temporal pooling
    Stream 2 (Audio):   Dual-Feature CNN-GRU (Mel + LFCC)
    Stream 3 (Sync):    CLIP ViT-B/32 + Lightweight CNN + Cross-Attention + BiLSTM
    Late fusion via Concat + MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel

from .image_detector import CLIPClassifier, NoiseEfficientNet, SRMConv2d
from .audio_detector import AudioDualStreamDetector


class VideoTriStreamDetector(nn.Module):
    """
    Video tri-stream deepfake detection model.
    """

    def __init__(
        self,
        clip_large_model='openai/clip-vit-large-patch14',
        clip_base_model='openai/clip-vit-base-patch32',
        load_image_models=True,
        image_clip_path=None,
        noise_efficientnet_path=None,
        audio_cnn_gru_path=None,
    ):
        super().__init__()

        self.clip_classifier = CLIPClassifier(clip_model_name=clip_large_model, num_classes=2)
        if load_image_models and image_clip_path is not None:
            ckpt = torch.load(image_clip_path, map_location='cpu')
            self.clip_classifier.load_state_dict(ckpt.get('model_state_dict', ckpt))

        self.srm_layer = SRMConv2d(in_channels=3)
        self.noise_efficientnet = NoiseEfficientNet()
        if load_image_models and noise_efficientnet_path is not None:
            ckpt = torch.load(noise_efficientnet_path, map_location='cpu')
            self.noise_efficientnet.load_state_dict(ckpt.get('model_state_dict', ckpt))

        for p in self.clip_classifier.parameters():
            p.requires_grad = False
        for p in self.noise_efficientnet.parameters():
            p.requires_grad = False

        self.clip_hidden_size = self.clip_classifier.clip_vision.config.hidden_size
        self.noise_feature_size = 1280

        self.stream1_fusion = nn.Sequential(
            nn.Linear(self.clip_hidden_size + self.noise_feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.audio_cnn_gru = AudioDualStreamDetector()
        if audio_cnn_gru_path is not None:
            ckpt = torch.load(audio_cnn_gru_path, map_location='cpu')
            self.audio_cnn_gru.load_state_dict(ckpt.get('model_state_dict', ckpt))

        for p in self.audio_cnn_gru.parameters():
            p.requires_grad = False

        self.audio_feature_size = 512

        self.clip_sync_vision = CLIPVisionModel.from_pretrained(clip_base_model, use_safetensors=True)
        for p in self.clip_sync_vision.parameters():
            p.requires_grad = False
        self.clip_sync_hidden = self.clip_sync_vision.config.hidden_size

        self.audio_semantic_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Flatten(1, 2),
        )
        self.audio_semantic_proj = nn.Linear(64, self.clip_sync_hidden)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.clip_sync_hidden,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )

        self.sync_lstm = nn.LSTM(
            input_size=self.clip_sync_hidden,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.sync_feature_size = 512

        total_dim = 512 + 512 + 512
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

    def _extract_audio_features(self, mel, lfcc):
        m = self.audio_cnn_gru
        mel_feat = m._run_resnet_stream(m.mel_resnet, m.mel_freq_pool, mel)
        lfcc_feat = m._run_resnet_stream(m.lfcc_resnet, m.lfcc_freq_pool, lfcc)

        stacked = torch.cat([mel_feat, lfcc_feat], dim=2)
        attn_w = m.attention_fc(stacked).unsqueeze(-1)
        dual = torch.stack([mel_feat, lfcc_feat], dim=2)
        fused = (dual * attn_w).sum(dim=2)
        fused = m.fusion_norm(fused)
        return fused

    def forward(self, visual, mel, lfcc):
        B, T = visual.shape[:2]
        flat = visual.view(B * T, 3, 224, 224)

        with torch.no_grad():
            clip_feat = self.clip_classifier.clip_vision(pixel_values=flat).last_hidden_state[:, 0, :]
            clip_feat = F.normalize(clip_feat, p=2, dim=1)

            noise_in = flat * 255.0
            noise_in = self.srm_layer(noise_in)
            noise_in = torch.clamp(noise_in, -3.0, 3.0) / 3.0
            noise_feat = self.noise_efficientnet.efficientnet.features(noise_in)
            noise_feat = self.noise_efficientnet.efficientnet.avgpool(noise_feat)
            noise_feat = torch.flatten(noise_feat, 1)

            combined = torch.cat([clip_feat, noise_feat], dim=1)
            combined = combined.view(B, T, -1)
            vis_vector = combined.mean(dim=1)

            audio_fused = self._extract_audio_features(mel, lfcc)
            _, hidden = self.audio_cnn_gru.gru(audio_fused)
            audio_vector = torch.cat((hidden[-2], hidden[-1]), dim=1)

            vis_sem = self.clip_sync_vision(pixel_values=flat).last_hidden_state[:, 0, :]
            vis_sem = vis_sem.view(B, T, self.clip_sync_hidden)

        vis_vector = self.stream1_fusion(vis_vector)

        audio_sem = self.audio_semantic_cnn(mel).permute(0, 2, 1)
        audio_sem = self.audio_semantic_proj(audio_sem)

        attended, _ = self.cross_attention(query=vis_sem, key=audio_sem, value=audio_sem)
        _, (h_lstm, _) = self.sync_lstm(attended)
        sync_vector = torch.cat((h_lstm[-2], h_lstm[-1]), dim=1)

        fused = torch.cat([vis_vector, audio_vector, sync_vector], dim=1)
        return self.classifier(fused)
