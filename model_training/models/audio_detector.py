"""
Audio deepfake detection model.

Contains:
- AudioDualStreamDetector: Dual-stream audio deepfake detection model using ResNet-18 for Log-Mel + ResNet-18 for LFCC, fused with attention mechanism and a 2-layer Bidirectional GRU.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class AudioDualStreamDetector(nn.Module):
    """
    Dual-stream audio deepfake detection model.
    """

    def __init__(self):
        super().__init__()

        self.mel_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self._adapt_resnet_for_audio(self.mel_resnet)

        self.lfcc_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self._adapt_resnet_for_audio(self.lfcc_resnet)

        self.mel_freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.lfcc_freq_pool = nn.AdaptiveAvgPool2d((1, None))

        self.attention_fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.Tanh(),
            nn.Linear(256, 2),
            nn.Softmax(dim=-1),
        )

        self.fusion_norm = nn.LayerNorm(512)
        self.temporal_dropout = nn.Dropout(0.2)

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    @staticmethod
    def _adapt_resnet_for_audio(resnet):
        """
        Convert a ResNet18 for single-channel audio with preserved time dimension.
        """

        weight = resnet.conv1.weight.clone()
        new_weight = weight.sum(dim=1, keepdim=True)
        resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 1), padding=(3, 3), bias=False
        )
        resnet.conv1.weight = nn.Parameter(new_weight)

        resnet.avgpool = nn.Identity()
        resnet.fc = nn.Identity()

        resnet.maxpool.stride = (2, 1)

        for layer in [resnet.layer2, resnet.layer3, resnet.layer4]:
            layer[0].conv1.stride = (2, 1)
            if layer[0].downsample is not None:
                layer[0].downsample[0].stride = (2, 1)

    def _run_resnet_stream(self, resnet, freq_pool, x):
        """
        Run a ResNet18 stream on the input audio features.
        """

        x = resnet.conv1(x)
        x = resnet.bn1(x)
        x = resnet.relu(x)
        x = resnet.maxpool(x)
        x = resnet.layer1(x)
        x = resnet.layer2(x)
        x = resnet.layer3(x)
        x = resnet.layer4(x)
        x = freq_pool(x).squeeze(2).permute(0, 2, 1)
        return x

    def forward(self, mel_x, lfcc_x):

        mel_feat = self._run_resnet_stream(self.mel_resnet, self.mel_freq_pool, mel_x)
        lfcc_feat = self._run_resnet_stream(self.lfcc_resnet, self.lfcc_freq_pool, lfcc_x)

        stacked = torch.cat([mel_feat, lfcc_feat], dim=2)
        attn_w = self.attention_fc(stacked).unsqueeze(-1)
        dual = torch.stack([mel_feat, lfcc_feat], dim=2)
        fused = (dual * attn_w).sum(dim=2)
        fused = self.fusion_norm(fused)

        gru_out, hidden = self.gru(fused)
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        return self.classifier(final_hidden)


DualFeature_CNN_GRU = AudioDualStreamDetector
