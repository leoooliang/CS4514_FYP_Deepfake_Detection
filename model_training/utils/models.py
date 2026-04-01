"""
Model architectures for deepfake detection.

This module contains:
- SRMConv2d: Spatial Rich Model filter layer for noise extraction
- NoiseResNet: ResNet-50 with SRM preprocessing for deepfake detection
- NoiseResNet50: ResNet-50 optimized for high-frequency noise detection
- NoiseEfficientNet: EfficientNetV2-S with SRM preprocessing for deepfake detection
- CLIPClassifier: CLIP ViT-L/14 with LN-Tuning for deepfake detection
- DualFeature_CNN_GRU: Dual-stream audio deepfake detection with ResNet18 + GRU
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
    
    This layer applies the fixed 5x5 SRM filter from the Warsaw University paper
    to extract manipulation artifacts from images. The filter is designed to capture
    high-frequency noise patterns that are indicative of image manipulations and deepfakes.
    
    The 5x5 SRM filter (from Warsaw University paper, Source 97/101):
    [[ -1,  2, -2,  2, -1],
     [  2, -6,  8, -6,  2],
     [ -2,  8,-12,  8, -2],
     [  2, -6,  8, -6,  2],
     [ -1,  2, -2,  2, -1]] * (1/12)
    
    This filter can be applied to either:
    - Grayscale images (1-channel input) producing 1-channel output
    - RGB images (3-channel input) producing 3-channel output (each channel processed independently)
    
    Reference: Warsaw University of Technology paper on SRM-based deepfake detection
    """
    
    def __init__(self, in_channels=3):
        """
        Initialize SRM filter.
        
        Args:
            in_channels: Number of input channels (default: 3 for RGB, can be 1 for grayscale)
        """
        super(SRMConv2d, self).__init__()
        
        # Define the fixed 5x5 SRM kernel from Warsaw University paper
        srm_kernel = np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8,-12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=np.float32) / 12.0
        
        # Create kernels for each input channel (depthwise convolution)
        # Shape: (in_channels, 1, 5, 5) for depthwise conv
        srm_kernels = np.tile(srm_kernel[np.newaxis, np.newaxis, :, :], (in_channels, 1, 1, 1))
        
        # Create depthwise conv layer to process each channel independently
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=5,
            padding=2,
            bias=False,
            groups=in_channels  # Depthwise: each input channel has its own filter
        )
        
        # Load fixed SRM weights (non-trainable)
        self.conv.weight.data = torch.from_numpy(srm_kernels).float()
        self.conv.weight.requires_grad = False
        
    def forward(self, x):
        """
        Apply SRM filter to extract noise residuals.
        
        Args:
            x: Image tensor (B, C, H, W) where C is 1 (grayscale) or 3 (RGB)
            
        Returns:
            Noise residual tensor (B, C, H, W) with same number of channels as input
        """
        # Apply SRM filter to each channel independently
        noise_residual = self.conv(x)
        
        return noise_residual


class NoiseResNet(nn.Module):
    """
    Noise Residual Stream Architecture for Deepfake Detection.
    
    This model uses ResNet-50 to process precomputed SRM (Spatial Rich Model) noise 
    residuals for deepfake detection. The architecture follows:
    
    Precomputed SRM Noise Tensor → ResNet-50 → Binary Classification
    
    Key features:
    - Accepts precomputed 3-channel SRM noise residuals as input
    - Pre-trained ResNet-50 backbone provides strong spatial feature extraction
    - Differential learning rates for different layers (configured in optimizer)
    - Dropout(p=0.5) for strong regularization on larger combined datasets
    - Binary classification output (real vs fake)
    - Optimized for larger combined datasets (FF++ and ArtiFact)
    
    Training Configuration:
    - Epochs: 50 (increased from 30 for larger dataset)
    - Early Stopping: patience=20 (compatible with CosineAnnealingWarmRestarts T_0=10)
    - Scheduler: CosineAnnealingWarmRestarts with T_0=10 avoids premature stopping
    
    Args:
        None
        
    Input:
        Precomputed SRM noise tensors of shape (B, 3, 224, 224)
        
    Output:
        Logits of shape (B, 1) for binary classification
    """
    
    def __init__(self):
        super(NoiseResNet, self).__init__()
        
        # Load pre-trained ResNet-50 with IMAGENET1K_V2 weights
        self.resnet = models.resnet50(weights="IMAGENET1K_V2")
        
        # Note: No layer freezing - differential learning rates are used instead
        # This allows all layers to learn, but at different rates:
        # - ResNet layers: lr=5e-5 (configured in optimizer)
        
        # Note: ResNet-50's first conv layer already accepts 3-channel input,
        # which matches our 3-channel SRM output
        
        # Replace final FC layer for binary classification with strong regularization
        # ResNet-50 has 2048 input features
        # Dropout(0.5) prevents overfitting on larger combined datasets (FF++ and ArtiFact)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1)
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Precomputed SRM noise tensor (B, 3, H, W)
            
        Returns:
            Logits (B, 1) for binary classification
        """
        # Pass precomputed noise residual through ResNet backbone
        out = self.resnet(x)
        
        return out


class NoiseResNet50(nn.Module):
    """
    Noise Residual Stream Architecture with ResNet50 for Deepfake Detection.
    
    This model uses ResNet50 to process SRM (Spatial Rich Model) noise residuals
    for deepfake detection. Unlike EfficientNetV2-S which uses depthwise separable
    convolutions that act as low-pass filters, ResNet50's standard convolutions
    preserve high-frequency noise patterns critical for deepfake detection.
    
    Architecture:
    RGB → SRM Filter (3-channel) → Truncate → ResNet50 → Binary Classification
    
    Key features:
    - Processes 3-channel SRM noise residuals from RGB images (preserves chrominance)
    - Pre-trained ResNet50 backbone with standard convolutions (better for high-frequency noise)
    - All layers trainable with 3-tier differential learning rates
    - Dropout(p=0.5) for strong regularization
    - Binary classification output (real vs fake)
    
    Training Configuration:
    - All Layers Unfrozen (entire model trainable)
    - 3-Tier Differential Learning Rates:
      * Early features (conv1, bn1, relu, maxpool, layer1): lr=1e-4 (fast adaptation to noise)
      * Late features (layer2, layer3, layer4): lr=5e-5 (moderate adaptation)
      * Classifier (fc): lr=1e-4 (task-specific head)
    
    Args:
        None
        
    Input:
        3-channel SRM noise tensors of shape (B, 3, H, W)
        
    Output:
        Logits of shape (B, 1) for binary classification
    """
    
    def __init__(self):
        super(NoiseResNet50, self).__init__()
        
        # Load pre-trained ResNet50 with IMAGENET1K_V2 weights
        self.resnet = models.resnet50(weights='IMAGENET1K_V2')
        
        # Replace final FC layer for binary classification with strong regularization
        # ResNet50 has 2048 input features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1)
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: SRM noise tensor (B, 3, H, W) from RGB input
            
        Returns:
            Logits (B, 1) for binary classification
        """
        # Input is already 3-channel SRM output from RGB
        # ResNet50's conv1 accepts 3-channel input directly
        out = self.resnet(x)
        
        return out


class NoiseEfficientNet(nn.Module):
    """
    Noise Residual Stream Architecture with EfficientNetV2 for Deepfake Detection.
    
    This model uses EfficientNetV2-B0/S to process SRM (Spatial Rich Model) noise 
    residuals for deepfake detection. The architecture follows:
    
    RGB → SRM Filter (3-channel) → Truncate → EfficientNetV2 → Binary Classification
    
    Key features:
    - Processes 3-channel SRM noise residuals from RGB images (preserves chrominance)
    - Pre-trained EfficientNetV2-S backbone provides efficient spatial feature extraction
    - All layers trainable with inverted differential learning rates
    - Dropout(p=0.5) for strong regularization
    - Binary classification output (real vs fake)
    
    Training Configuration:
    - All Layers Unfrozen (entire model trainable)
    - INVERTED Differential Learning Rates:
      * Early features (blocks 0-4): lr=1e-4 (fast adaptation to noise)
      * Late features (blocks 5+): lr=5e-5
      * Classifier: lr=1e-4
    
    Args:
        None
        
    Input:
        3-channel SRM noise tensors of shape (B, 3, H, W)
        
    Output:
        Logits of shape (B, 1) for binary classification
    """
    
    def __init__(self):
        super(NoiseEfficientNet, self).__init__()
        
        # Load pre-trained EfficientNetV2-S (similar to B0 architecture)
        # EfficientNetV2-S has 1280 output features
        self.efficientnet = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        
        # Replace final classifier for binary classification
        # EfficientNetV2-S has 1280 input features to the classifier
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1280, 1)
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: SRM noise tensor (B, 3, H, W) from RGB input
            
        Returns:
            Logits (B, 1) for binary classification
        """
        # Input is already 3-channel SRM output from RGB
        # No expansion needed - pass directly to EfficientNet
        out = self.efficientnet(x)
        
        return out


class CLIPClassifier(nn.Module):
    """
    CLIP Vision Encoder Stream for Deepfake Detection with LN-Tuning.
    
    This model uses CLIP's pre-trained vision encoder with parameter-efficient
    fine-tuning for deepfake detection. The architecture follows:
    
    CLIP Vision Encoder → L2 Normalization → Linear Classifier
    
    Key features:
    - CLIP ViT-L/14 vision encoder (pre-trained)
    - LN-Tuning: Only LayerNorm parameters are trainable (parameter-efficient)
    - L2 normalization: Projects CLS token onto hypersphere
    - Binary or multi-class classification support
    
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
        
        # Load pre-trained CLIP vision encoder
        self.clip_vision = CLIPVisionModel.from_pretrained(clip_model_name, use_safetensors=True)
        
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


class DualFeature_CNN_GRU(nn.Module):
    """
    Dual-Stream Audio Deepfake Detection Model using ResNet18 Backbones + Attention Fusion.
    
    This model processes two complementary audio features in parallel:
    1. Mel-Spectrogram Stream: Captures low-frequency human phonetic structures
    2. LFCC Stream: Captures high-frequency synthetic vocoder artifacts
    
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
        
        # === MEL-SPECTROGRAM RESNET18 STREAM ===
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
        
        # === LFCC RESNET18 STREAM ===
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
        
        # === FIX RESNET TEMPORAL DECIMATION ===
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
        
        # === ATTENTION-BASED FUSION ===
        self.attention_fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.Tanh(),
            nn.Linear(256, 2),
            nn.Softmax(dim=-1)
        )
        
        # === FUSION NORMALIZATION ===
        self.fusion_norm = nn.LayerNorm(512)
        
        # === TEMPORAL DROPOUT ===
        self.temporal_dropout = nn.Dropout(0.2)
        
        # === GRU SEQUENCE MODELING ===
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # === CLASSIFICATION HEAD ===
        # Bidirectional GRU outputs 2 * hidden_size = 512
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
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
        # === MEL-SPECTROGRAM RESNET18 STREAM ===
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
        
        # === LFCC RESNET18 STREAM ===
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
        
        # === ATTENTION-BASED FUSION ===
        stacked = torch.cat([mel_x, lfcc_x], dim=2)
        attention_weights = self.attention_fc(stacked)
        attention_weights = attention_weights.unsqueeze(-1)
        dual_features = torch.stack([mel_x, lfcc_x], dim=2)
        fused_features = (dual_features * attention_weights).sum(dim=2)
        fused_features = self.fusion_norm(fused_features)
        
        # === GRU SEQUENCE MODELING ===
        gru_out, hidden = self.gru(fused_features)
        final_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        # === CLASSIFICATION ===
        logits = self.classifier(final_hidden)
        
        return logits
