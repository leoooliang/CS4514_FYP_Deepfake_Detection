"""
Data augmentation utilities for deepfake detection.

This module contains:
- GPUAugmentation: GPU-based augmentation for image models
- JPEGCompression: GPU JPEG compression simulation
- create_clip_transforms: Create CLIP preprocessing transforms
- create_noise_augmentation: Create augmentation for noise residual models
"""

import torch
import torch.nn as nn
import kornia.augmentation as K


class GPUAugmentation(nn.Module):
    """
    GPU-based data augmentation for image deepfake detection.
    
    This module performs all augmentation operations on GPU for maximum efficiency.
    Commonly used for noise residual models that process raw RGB images.
    
    Args:
        apply_horizontal_flip: Apply random horizontal flip (default: True)
        apply_vertical_flip: Apply random vertical flip (default: True)
        apply_color_jitter: Apply color jitter (default: True)
        color_jitter_prob: Probability of applying color jitter (default: 0.3)
        apply_jpeg_compression: Apply JPEG compression simulation (default: True)
        jpeg_prob: Probability of applying JPEG compression (default: 0.3)
        jpeg_quality_range: JPEG quality range (default: (70, 100))
    """
    
    def __init__(self, apply_horizontal_flip=True, apply_vertical_flip=True,
                 apply_color_jitter=True, color_jitter_prob=0.3,
                 apply_jpeg_compression=True, jpeg_prob=0.3, 
                 jpeg_quality_range=(70, 100)):
        super(GPUAugmentation, self).__init__()
        
        augmentations = []
        
        # Geometric augmentations
        if apply_horizontal_flip:
            augmentations.append(K.RandomHorizontalFlip(p=0.5))
        if apply_vertical_flip:
            augmentations.append(K.RandomVerticalFlip(p=0.5))
        
        # Color augmentation (applied to both real and fake to prevent shortcut learning)
        if apply_color_jitter:
            augmentations.append(
                K.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05,
                    p=color_jitter_prob
                )
            )
        
        # JPEG compression simulation (applied to both real and fake)
        if apply_jpeg_compression:
            augmentations.append(
                K.RandomJPEG(
                    jpeg_quality=jpeg_quality_range,
                    p=jpeg_prob
                )
            )
        
        self.augmentation = nn.Sequential(*augmentations)
    
    def forward(self, x):
        """
        Apply augmentation to input images.
        
        Args:
            x: Input images (B, 3, H, W) in range [0, 255]
            
        Returns:
            Augmented images (B, 3, H, W) in range [0, 255]
        """
        return self.augmentation(x)


class CLIPValidationTransform(nn.Module):
    """
    GPU-based CLIP validation transform (no augmentation).
    
    This module only performs center crop for validation/testing.
    All images should already be CLIP-preprocessed (normalized with CLIP mean/std).
    
    Args:
        crop_size: Center crop size (default: 224)
    """
    
    def __init__(self, crop_size=224):
        super(CLIPValidationTransform, self).__init__()
        self.center_crop = K.CenterCrop(crop_size)
    
    def forward(self, x):
        """
        Apply center crop to CLIP-preprocessed images.
        
        Args:
            x: CLIP-preprocessed images (B, 3, H, W)
            
        Returns:
            Center-cropped images (B, 3, crop_size, crop_size)
        """
        return self.center_crop(x)


class CLIPTrainingTransform(nn.Module):
    """
    GPU-based CLIP training transform with augmentation.
    
    This module applies augmentation to CLIP-preprocessed images during training.
    
    Args:
        crop_size: Crop size (default: 224)
        apply_horizontal_flip: Apply random horizontal flip (default: True)
        apply_gaussian_blur: Apply Gaussian blur (default: True)
        blur_prob: Probability of applying Gaussian blur (default: 0.2)
        apply_color_jitter: Apply color jitter (default: True)
        color_jitter_prob: Probability of applying color jitter (default: 0.2)
        apply_jpeg_compression: Apply JPEG compression (default: True)
        jpeg_prob: Probability of applying JPEG compression (default: 0.2)
    """
    
    def __init__(self, crop_size=224, apply_horizontal_flip=True,
                 apply_gaussian_blur=True, blur_prob=0.2,
                 apply_color_jitter=True, color_jitter_prob=0.2,
                 apply_jpeg_compression=True, jpeg_prob=0.2):
        super(CLIPTrainingTransform, self).__init__()
        
        augmentations = []
        
        # Random crop
        augmentations.append(K.RandomCrop((crop_size, crop_size), p=1.0))
        
        # Horizontal flip
        if apply_horizontal_flip:
            augmentations.append(K.RandomHorizontalFlip(p=0.5))
        
        # Gaussian blur
        if apply_gaussian_blur:
            augmentations.append(K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=blur_prob))
        
        # Color jitter
        if apply_color_jitter:
            augmentations.append(
                K.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05,
                    p=color_jitter_prob
                )
            )
        
        # JPEG compression
        if apply_jpeg_compression:
            augmentations.append(K.RandomJPEG(jpeg_quality=(70, 100), p=jpeg_prob))
        
        self.augmentation = nn.Sequential(*augmentations)
    
    def forward(self, x):
        """
        Apply augmentation to CLIP-preprocessed images.
        
        Args:
            x: CLIP-preprocessed images (B, 3, H, W)
            
        Returns:
            Augmented images (B, 3, crop_size, crop_size)
        """
        return self.augmentation(x)


def create_noise_augmentation(apply_color_jitter=True, apply_jpeg=True, 
                              color_jitter_prob=0.3, jpeg_prob=0.3):
    """
    Create GPU augmentation for noise residual models.
    
    This augmentation is applied to raw RGB images before SRM filtering.
    Symmetric augmentation (color jitter + JPEG) prevents shortcut learning.
    
    Args:
        apply_color_jitter: Apply color jitter (default: True)
        apply_jpeg: Apply JPEG compression (default: True)
        color_jitter_prob: Probability of color jitter (default: 0.3)
        jpeg_prob: Probability of JPEG compression (default: 0.3)
        
    Returns:
        GPUAugmentation module
    """
    return GPUAugmentation(
        apply_horizontal_flip=True,
        apply_vertical_flip=True,
        apply_color_jitter=apply_color_jitter,
        color_jitter_prob=color_jitter_prob,
        apply_jpeg_compression=apply_jpeg,
        jpeg_prob=jpeg_prob,
        jpeg_quality_range=(70, 100)
    )


def create_clip_transforms(is_training=True, crop_size=224):
    """
    Create GPU transforms for CLIP models.
    
    Args:
        is_training: If True, create training transforms with augmentation
        crop_size: Crop size (default: 224)
        
    Returns:
        CLIPTrainingTransform or CLIPValidationTransform module
    """
    if is_training:
        return CLIPTrainingTransform(
            crop_size=crop_size,
            apply_horizontal_flip=True,
            apply_gaussian_blur=True,
            blur_prob=0.2,
            apply_color_jitter=True,
            color_jitter_prob=0.2,
            apply_jpeg_compression=True,
            jpeg_prob=0.2
        )
    else:
        return CLIPValidationTransform(crop_size=crop_size)
