"""
Image deepfake detection model.

Contains:
- SRMConv2d: Spatial Rich Model filter layer for noise extraction
- NoiseEfficientNet: EfficientNetV2-S with SRM preprocessing for deepfake detection
- CLIPClassifier: CLIP ViT-L/14 with LN-Tuning for deepfake detection
- ImageDualStreamDetector: Dual-stream image deepfake detection model with weighted score-level fusion.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import CLIPVisionModel


class SRMConv2d(nn.Module):
    """
    Spatial Rich Model (SRM) filter layer for noise residual extraction.

    The 5x5 SRM kernel:
    [[ -1,  2, -2,  2, -1],
     [  2, -6,  8, -6,  2],
     [ -2,  8,-12,  8, -2],
     [  2, -6,  8, -6,  2],
     [ -1,  2, -2,  2, -1]] * (1/12)
    """

    def __init__(self, in_channels=3):
        super().__init__()

        srm_kernel = np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8,-12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=np.float32) / 12.0

        srm_kernels = np.tile(srm_kernel[np.newaxis, np.newaxis, :, :], (in_channels, 1, 1, 1))

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=5,
            padding=2,
            bias=False,
            groups=in_channels,
        )
        self.conv.weight.data = torch.from_numpy(srm_kernels).float()
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)


class NoiseEfficientNet(nn.Module):
    """
    EfficientNetV2-S with SRM preprocessing for deepfake detection.
    """

    def __init__(self):
        super().__init__()
        self.efficientnet = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1280, 1),
        )

    def forward(self, x):
        return self.efficientnet(x)


class CLIPClassifier(nn.Module):
    """
    CLIP Vision Encoder Stream for Deepfake Detection with LN-Tuning.
    """

    def __init__(self, clip_model_name='openai/clip-vit-large-patch14', num_classes=2):
        super().__init__()

        self.clip_vision = CLIPVisionModel.from_pretrained(clip_model_name, use_safetensors=True)

        for param in self.clip_vision.parameters():
            param.requires_grad = False

        for name, param in self.clip_vision.named_parameters():
            if 'norm' in name.lower():
                param.requires_grad = True

        hidden_size = self.clip_vision.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values, return_normalized_features=False):
        outputs = self.clip_vision(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        normalized_features = F.normalize(cls_token, p=2, dim=1)

        if return_normalized_features:
            return normalized_features

        return self.classifier(normalized_features)


class ImageDualStreamDetector(nn.Module):
    """
    Dual-stream image deepfake detection model with weighted score-level fusion.
    """

    def __init__(
        self,
        clip_model_name='openai/clip-vit-large-patch14',
        initial_clip_weight=0.5,
        clip_checkpoint_path=None,
        noise_checkpoint_path=None,
        freeze_streams=False,
    ):
        super().__init__()

        self.clip_stream = CLIPClassifier(clip_model_name=clip_model_name, num_classes=2)
        self.srm_layer = SRMConv2d(in_channels=3)
        self.noise_stream = NoiseEfficientNet()

        if clip_checkpoint_path is not None:
            ckpt = torch.load(clip_checkpoint_path, map_location='cpu')
            state = ckpt.get('model_state_dict', ckpt)
            self.clip_stream.load_state_dict(state)

        if noise_checkpoint_path is not None:
            ckpt = torch.load(noise_checkpoint_path, map_location='cpu')
            state = ckpt.get('model_state_dict', ckpt)
            self.noise_stream.load_state_dict(state)

        if freeze_streams:
            for param in self.clip_stream.parameters():
                param.requires_grad = False
            for param in self.noise_stream.parameters():
                param.requires_grad = False

        raw_w = torch.log(torch.tensor(initial_clip_weight / (1.0 - initial_clip_weight)))
        self._fusion_weight_logit = nn.Parameter(raw_w)

    @property
    def clip_weight(self):
        return torch.sigmoid(self._fusion_weight_logit)

    def forward(self, pixel_values):
        clip_logits = self.clip_stream(pixel_values)
        clip_score = clip_logits[:, 1:2] - clip_logits[:, 0:1]

        noise_input = pixel_values * 255.0
        noise_input = self.srm_layer(noise_input)
        noise_input = torch.clamp(noise_input, -3.0, 3.0) / 3.0
        noise_score = self.noise_stream(noise_input)

        w = self.clip_weight
        fused_logit = w * clip_score + (1.0 - w) * noise_score
        return fused_logit
