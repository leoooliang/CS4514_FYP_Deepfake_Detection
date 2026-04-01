"""
Utility modules for deepfake detection model training.

This package contains:

Models:
- SRMConv2d: Spatial Rich Model filter layer
- NoiseResNet: ResNet-50 with SRM preprocessing
- NoiseEfficientNet: EfficientNetV2-S with SRM preprocessing
- CLIPClassifier: CLIP ViT-L/14 with LN-Tuning
- DualFeature_CNN_GRU: Dual-stream audio model with ResNet18 + GRU

Datasets:
- PrecomputedCLIPDataset: Dataset for CLIP-preprocessed images
- PrecomputedSRMDataset: Dataset for raw RGB tensors
- AudioDataset: Dataset for audio files with dual-stream features

Training:
- train_one_epoch: Generic training loop for single-stream models
- train_one_epoch_dual_stream: Training loop for dual-stream audio models
- train_one_epoch_noise: Training loop for noise residual models
- EarlyStopping: Early stopping callback

Evaluation:
- validate: Validation function for single-stream models
- validate_dual_stream: Validation function for dual-stream audio models
- validate_noise: Validation function for noise residual models
- compute_metrics: Compute comprehensive performance metrics
- evaluate_by_manipulation_type: Per-manipulation-type evaluation

Visualization:
- plot_confusion_matrix: Plot confusion matrix heatmap
- plot_roc_curve: Plot ROC curve
- plot_precision_recall_curve: Plot Precision-Recall curve
- plot_training_history: Plot training/validation metrics
- plot_manipulation_type_comparison: Compare performance across types
- plot_multiple_roc_curves: Compare multiple ROC curves
- plot_score_distribution: Plot score distributions

Augmentation:
- GPUAugmentation: GPU-based image augmentation
- CLIPTrainingTransform: CLIP training augmentation
- CLIPValidationTransform: CLIP validation transform
- create_noise_augmentation: Create noise model augmentation
- create_clip_transforms: Create CLIP transforms

Checkpoint:
- save_checkpoint: Save model checkpoint
- load_checkpoint: Load model checkpoint
- save_best_model: Save best model
- load_model_weights_only: Load only weights
- ModelCheckpoint: Automatic checkpointing callback
- get_latest_checkpoint: Get latest checkpoint path
"""

# Import all modules for easy access
from .models import (
    SRMConv2d,
    NoiseResNet,
    NoiseEfficientNet,
    CLIPClassifier,
    DualFeature_CNN_GRU
)

from .dataset import (
    PrecomputedCLIPDataset,
    PrecomputedSRMDataset,
    AudioDataset
)

from .training import (
    train_one_epoch,
    train_one_epoch_dual_stream,
    train_one_epoch_noise,
    EarlyStopping
)

from .evaluation import (
    validate,
    validate_dual_stream,
    validate_noise,
    compute_metrics,
    evaluate_by_manipulation_type
)

from .visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_training_history,
    plot_manipulation_type_comparison,
    plot_multiple_roc_curves,
    plot_score_distribution
)

from .augmentation import (
    GPUAugmentation,
    CLIPTrainingTransform,
    CLIPValidationTransform,
    create_noise_augmentation,
    create_clip_transforms
)

from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    save_best_model,
    load_model_weights_only,
    ModelCheckpoint,
    get_latest_checkpoint
)

__all__ = [
    # Models
    'SRMConv2d',
    'NoiseResNet',
    'NoiseEfficientNet',
    'CLIPClassifier',
    'DualFeature_CNN_GRU',
    # Datasets
    'PrecomputedCLIPDataset',
    'PrecomputedSRMDataset',
    'AudioDataset',
    # Training
    'train_one_epoch',
    'train_one_epoch_dual_stream',
    'train_one_epoch_noise',
    'EarlyStopping',
    # Evaluation
    'validate',
    'validate_dual_stream',
    'validate_noise',
    'compute_metrics',
    'evaluate_by_manipulation_type',
    # Visualization
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_training_history',
    'plot_manipulation_type_comparison',
    'plot_multiple_roc_curves',
    'plot_score_distribution',
    # Augmentation
    'GPUAugmentation',
    'CLIPTrainingTransform',
    'CLIPValidationTransform',
    'create_noise_augmentation',
    'create_clip_transforms',
    # Checkpoint
    'save_checkpoint',
    'load_checkpoint',
    'save_best_model',
    'load_model_weights_only',
    'ModelCheckpoint',
    'get_latest_checkpoint',
]
