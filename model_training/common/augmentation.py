"""
GPU-based data augmentation modules for image deepfake detection.
"""

import torch
import torch.nn as nn
import kornia.augmentation as K

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class GPUAugmentation(nn.Module):
    """
    GPU-based augmentation for raw-RGB image pipelines (noise stream).
    """

    def __init__(
        self,
        apply_horizontal_flip=True,
        apply_vertical_flip=True,
        apply_color_jitter=True,
        color_jitter_prob=0.3,
        apply_jpeg_compression=True,
        jpeg_prob=0.3,
        jpeg_quality_range=(70, 100),
        apply_gaussian_blur=True,
        blur_prob=0.2,
    ):
        super().__init__()
        augs = []
        if apply_horizontal_flip:
            augs.append(K.RandomHorizontalFlip(p=0.5))
        if apply_vertical_flip:
            augs.append(K.RandomVerticalFlip(p=0.5))
        if apply_color_jitter:
            augs.append(K.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1,
                p=color_jitter_prob,
            ))
        if apply_jpeg_compression:
            augs.append(K.RandomJPEG(jpeg_quality=jpeg_quality_range, p=jpeg_prob))
        if apply_gaussian_blur:
            augs.append(K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5), p=blur_prob))
        self.augmentation = nn.Sequential(*augs)

    @torch.no_grad()
    def forward(self, x):
        return self.augmentation(x)


class CLIPValidationTransform(nn.Module):
    """
    GPU center crop for CLIP-preprocessed images (validation / test).
    """

    def __init__(self, crop_size=224):
        super().__init__()
        self.center_crop = K.CenterCrop(crop_size)

    @torch.no_grad()
    def forward(self, x):
        return self.center_crop(x)


class CLIPTrainingTransform(nn.Module):
    """
    GPU augmentation for CLIP-preprocessed images (training).
    """

    def __init__(
        self,
        crop_size=224,
        apply_horizontal_flip=True,
        apply_gaussian_blur=True,
        blur_prob=0.2,
        apply_color_jitter=True,
        color_jitter_prob=0.2,
        apply_jpeg_compression=True,
        jpeg_prob=0.2,
    ):
        super().__init__()
        augs = [K.RandomCrop((crop_size, crop_size), p=1.0)]
        if apply_horizontal_flip:
            augs.append(K.RandomHorizontalFlip(p=0.5))
        if apply_gaussian_blur:
            augs.append(K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=blur_prob))
        if apply_color_jitter:
            augs.append(K.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05,
                p=color_jitter_prob,
            ))
        if apply_jpeg_compression:
            augs.append(K.RandomJPEG(jpeg_quality=(50, 100), p=jpeg_prob))
        self.augmentation = nn.Sequential(*augs)

    @torch.no_grad()
    def forward(self, x):
        return self.augmentation(x)


class CLIPNormalizeTransform(nn.Module):
    """
    CLIP mean/std normalization for validation and test (no augmentation).
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(CLIP_STD).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x):
        return (x - self.mean) / self.std


class AsymmetricCLIPTransform(nn.Module):
    """
    Label-aware GPU augmentation for CLIP-preprocessed images (training).
    """

    def __init__(self, asym_prob=0.2):
        super().__init__()
        self.base_transform = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.2),
        )
        self.asym_jitter = K.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0,
        )
        self.asym_jpeg = K.RandomJPEG(jpeg_quality=(50, 100))
        self.asym_prob = asym_prob
        self.register_buffer('mean', torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(CLIP_STD).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x, labels):
        x = self.base_transform(x)

        is_fake = labels == 0
        coin_flip = torch.rand(x.size(0), device=x.device) < self.asym_prob
        aug_indices = (is_fake & coin_flip).nonzero(as_tuple=True)[0]

        if aug_indices.numel() > 0:
            subset = self.asym_jitter(x[aug_indices])
            subset = self.asym_jpeg(subset)
            x = x.clone()
            x[aug_indices] = subset

        return (x - self.mean) / self.std


class AsymmetricNoiseTransform(nn.Module):
    """
    Label-aware asymmetric augmentation for the noise stream (training).
    """

    def __init__(self):
        super().__init__()
        self.base_transform = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
        )
        self.asym_augment = nn.Sequential(
            K.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5,
            ),
            K.RandomJPEG(jpeg_quality=(50, 90), p=0.5),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5), p=0.2),
        )

    @torch.no_grad()
    def forward(self, x, labels):
        x = self.base_transform(x)

        fake_mask = (labels == 0).view(-1, 1, 1, 1)
        aug_x = self.asym_augment(x)
        x = torch.where(fake_mask, aug_x, x)

        return x


def create_noise_augmentation(**kwargs):
    """
    Factory for noise-stream GPU augmentation with sensible defaults.
    """

    defaults = dict(
        apply_horizontal_flip=True,
        apply_vertical_flip=True,
        apply_color_jitter=True,
        color_jitter_prob=0.5,
        apply_jpeg_compression=True,
        jpeg_prob=0.5,
        jpeg_quality_range=(50, 90),
        apply_gaussian_blur=True,
        blur_prob=0.2,
    )
    defaults.update(kwargs)
    return GPUAugmentation(**defaults)


def create_asymmetric_noise_augmentation():
    """
    Factory for label-aware asymmetric noise-stream augmentation.
    """

    return AsymmetricNoiseTransform()


def create_clip_transforms(is_training=True, crop_size=224, asymmetric=False):
    """
    Return the appropriate CLIP GPU transform for training or validation.

    When *asymmetric* is True: use AsymmetricCLIPTransform 
    When *asymmetric* is False: use CLIPTrainingTransform
    """
    
    if is_training:
        if asymmetric:
            return AsymmetricCLIPTransform()
        return CLIPTrainingTransform(crop_size=crop_size)
    if asymmetric:
        return CLIPNormalizeTransform()
    return CLIPValidationTransform(crop_size=crop_size)
