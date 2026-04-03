"""
Evaluation engine for multimodal deepfake detection models.

Provides a single `evaluate` function that works with any model / dataset
layout by accepting the same `unpack_fn` callback used by the trainer.
"""

import numpy as np
import torch
from contextlib import nullcontext
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

from configs.config import FAKE_LABEL


def evaluate(
    model,
    dataloader,
    criterion,
    device,
    unpack_fn,
    gpu_transform=None,
    use_amp=False,
    amp_dtype=torch.float16,
):
    """
    Evaluate a model on validation / test data.

    Args:
        model: The model to evaluate
        dataloader: Validation/test data loader
        criterion: Loss function
        device: Device to evaluate on
        unpack_fn: callable(batch, device) → (model_args_tuple, labels)
        gpu_transform: Optional GPU-side transform applied to the first element
                       of model_args (used by CLIP stream augmentation).
        use_amp: Enable automatic mixed precision.
        amp_dtype: AMP data-type (default float16; use bfloat16 for better
                       numerical stability without GradScaler).

    Returns:
        dict: Dictionary containing:
            - val_loss: Average validation loss
            - accuracy: Classification accuracy
            - roc_auc: ROC-AUC score
            - pr_auc: PR-AUC score
            - all_preds: Predicted labels
            - all_labels: Ground truth labels
            - all_probs: Predicted probabilities
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    is_multiclass = None

    all_preds = []
    all_labels = []
    all_probs = []

    amp_ctx = torch.amp.autocast('cuda', dtype=amp_dtype) if use_amp else nullcontext()

    pbar = tqdm(dataloader, desc='Evaluating', leave=False)

    with torch.no_grad():
        for batch in pbar:
            model_args, labels = unpack_fn(batch, device)

            if gpu_transform is not None:
                model_args = (gpu_transform(model_args[0]),) + model_args[1:]

            with amp_ctx:
                outputs = model(*model_args)

            if outputs.dim() > 1 and outputs.size(1) > 1:
                is_multiclass = True
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_probs.extend(probs[:, FAKE_LABEL].float().cpu().numpy())
            else:
                is_multiclass = False
                outputs_sq = outputs.squeeze()
                labels_float = labels.float()
                loss = criterion(outputs_sq, labels_float)
                probs = torch.sigmoid(outputs_sq)
                preds = (probs > 0.5).long()
                all_probs.extend(probs.float().cpu().numpy())

            running_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{100.0 * correct / total:.2f}%')

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    labels_fake_positive = (all_labels == FAKE_LABEL).astype(int)
    try:
        if is_multiclass:
            roc_auc = roc_auc_score(labels_fake_positive, all_probs)
            pr_auc = average_precision_score(labels_fake_positive, all_probs)
        else:
            probs_fake = 1.0 - all_probs
            roc_auc = roc_auc_score(labels_fake_positive, probs_fake)
            pr_auc = average_precision_score(labels_fake_positive, probs_fake)
    except ValueError:
        roc_auc = 0.0
        pr_auc = 0.0

    return {
        'val_loss': epoch_loss,
        'accuracy': epoch_acc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs,
    }


def compute_metrics(labels, predictions, probabilities=None):
    """
    Compute comprehensive classification metrics.
    """

    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, pos_label=FAKE_LABEL, zero_division=0),
        'recall': recall_score(labels, predictions, pos_label=FAKE_LABEL, zero_division=0),
        'f1': f1_score(labels, predictions, pos_label=FAKE_LABEL, zero_division=0),
    }

    cm = confusion_matrix(labels, predictions)
    metrics['confusion_matrix'] = cm

    if cm.shape == (2, 2):
        tp, fn, fp, tn = cm.ravel()
        metrics.update({
            'true_positive': tp,
            'false_negative': fn,
            'false_positive': fp,
            'true_negative': tn,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        })

    if probabilities is not None:
        labels_fake = (np.array(labels) == FAKE_LABEL).astype(int)
        try:
            metrics['roc_auc'] = roc_auc_score(labels_fake, probabilities)
        except ValueError:
            metrics['roc_auc'] = 0.0
        try:
            metrics['pr_auc'] = average_precision_score(labels_fake, probabilities)
        except ValueError:
            metrics['pr_auc'] = 0.0

    return metrics


def evaluate_by_manipulation_type(file_paths, predictions, labels, probabilities=None):
    """
    Evaluate predictions by manipulation type.
    """

    import os

    manip_types = []
    for fp in file_paths:
        basename = os.path.basename(fp)
        if 'real' in fp.lower():
            manip_types.append('Real')
        else:
            parts = basename.split('_')
            manip_types.append(parts[0] if parts else 'Unknown')

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    if probabilities is not None:
        probabilities = np.asarray(probabilities)

    results = {}
    for mt in sorted(set(manip_types)):
        idxs = [i for i, m in enumerate(manip_types) if m == mt]
        if not idxs:
            continue
        m = compute_metrics(
            labels[idxs],
            predictions[idxs],
            probabilities[idxs] if probabilities is not None else None,
        )
        m['num_samples'] = len(idxs)
        results[mt] = m

    return results
