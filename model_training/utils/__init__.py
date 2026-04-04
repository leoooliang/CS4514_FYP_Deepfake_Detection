"""
Legacy utils package — re-exports from new canonical locations.

New code should import directly from:
    models, data_loaders, engine, common, configs
"""

from ..models import (
    SRMConv2d,
    NoiseEfficientNet,
    CLIPClassifier,
    ImageDualStreamDetector,
    AudioDualStreamDetector,
    DualFeature_CNN_GRU,
    VideoTriStreamDetector,
)

from ..data_loaders import (
    PrecomputedCLIPDataset,
    PrecomputedSRMDataset,
    AudioDataset,
    TriStreamVideoDataset,
)

from ..engine import (
    train_one_epoch,
    EarlyStopping,
    evaluate,
    compute_metrics,
    evaluate_by_manipulation_type,
)

from ..common import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_training_history,
    plot_manipulation_type_comparison,
    plot_multiple_roc_curves,
    plot_score_distribution,
    GPUAugmentation,
    CLIPTrainingTransform,
    CLIPValidationTransform,
    create_noise_augmentation,
    create_clip_transforms,
    save_checkpoint,
    load_checkpoint,
    save_best_model,
    load_model_weights_only,
    ModelCheckpoint,
    get_latest_checkpoint,
)
