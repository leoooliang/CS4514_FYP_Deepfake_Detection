"""
Training engine for multimodal deepfake detection models.

Provides a single `train_one_epoch` function that works with any model / dataset
layout by accepting the same `unpack_fn` callback used by the evaluator.
"""

import torch
import torch.nn as nn
import numpy as np
from contextlib import nullcontext
from tqdm import tqdm


def unpack_single(batch, device):
    """
    Unpack a single batch of data.
    """

    inputs, labels = batch
    return (inputs.to(device, non_blocking=True),), labels.to(device, non_blocking=True)


def unpack_dual_stream(batch, device):
    """
    Unpack a batch of data for audio dual-stream input.
    """

    mel, lfcc, labels = batch
    return (
        mel.to(device, non_blocking=True),
        lfcc.to(device, non_blocking=True),
    ), labels.to(device, non_blocking=True)


def unpack_video_dict(batch, device):
    """
    Unpack a batch of data for video dictionary input.
    """

    return (
        batch['visual'].to(device, non_blocking=True),
        batch['mel'].to(device, non_blocking=True),
        batch['lfcc'].to(device, non_blocking=True),
    ), batch['label'].to(device, non_blocking=True)


def make_noise_unpacker(srm_layer, gpu_augment=None, label_aware_augment=None, label_smoothing=0.0):
    """
    Make a noise residual stream unpacker. Pipeline: raw RGB → augment (on [0,1]) → scale [0, 255] → SRM → clamp [-3, 3] → /3.
    """

    def _unpack(batch, device):
        inputs, labels = batch
        inputs = inputs.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True)

        if label_aware_augment is not None:
            inputs = label_aware_augment(inputs, labels_dev)
        elif gpu_augment is not None:
            inputs = gpu_augment(inputs)

        inputs = srm_layer(inputs * 255.0)
        inputs = torch.clamp(inputs, -3.0, 3.0) / 3.0

        labels_t = labels_dev.float()
        if label_smoothing > 0:
            labels_t = torch.where(
                labels_t == 0.0,
                torch.tensor(label_smoothing, device=device),
                torch.tensor(1.0 - label_smoothing, device=device),
            )
        return (inputs,), labels_t

    return _unpack


def _classify_outputs(outputs, labels):
    """
    Classify outputs into predictions and labels for classification.
    """

    if outputs.dim() > 1 and outputs.size(1) > 1:
        preds = torch.argmax(outputs, dim=1)
        return labels, preds
    else:
        outputs_sq = outputs.squeeze()
        preds = (torch.sigmoid(outputs_sq) > 0.5).long()
        labels_binary = (labels > 0.5).long()
        return labels.float(), preds, labels_binary


class EarlyStopping:
    """
    Stop training when the monitored metric stops improving.
    """

    def __init__(self, patience=10, min_delta=0.0, mode='max', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric):
        if self.best_score is None:
            self.best_score = metric
            return False

        improved = (
            metric > self.best_score + self.min_delta
            if self.mode == 'max'
            else metric < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('Early stopping triggered!')
                return True
        return False

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    unpack_fn=unpack_single,
    gpu_transform=None,
    label_aware_transform=None,
    grad_clip=None,
    scaler=None,
    use_amp=False,
    amp_dtype=torch.float16,
):
    """
    Train a model for one epoch with optional mixed-precision (AMP).

    Args:
        model:         nn.Module
        dataloader:    DataLoader
        criterion:     Loss function
        optimizer:     Optimiser
        device:        torch.device
        unpack_fn:     callable(batch, device) → (model_args_tuple, labels)
        gpu_transform: Optional GPU-side transform applied to the first element
                       of model_args (used by CLIP stream augmentation).
        label_aware_transform: Optional label-aware GPU transform with signature
                       ``(tensor, labels) → tensor``.  Used for asymmetric
                       augmentation.  Takes precedence over *gpu_transform*.
        grad_clip:     If not None, clip gradients to this max norm.
        scaler:        torch.amp.GradScaler instance (required when use_amp=True).
        use_amp:       Enable automatic mixed precision.
        amp_dtype:     AMP data-type (default float16; use bfloat16 for better
                       numerical stability without GradScaler).

    Returns:
        epoch_loss: Average training loss
        epoch_acc: Average training accuracy
    """

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    amp_ctx = torch.amp.autocast('cuda', dtype=amp_dtype) if use_amp else nullcontext()

    pbar = tqdm(dataloader, desc='Training', leave=False)

    for batch in pbar:
        model_args, labels = unpack_fn(batch, device)

        if label_aware_transform is not None:
            model_args = (label_aware_transform(model_args[0], labels),) + model_args[1:]
        elif gpu_transform is not None:
            model_args = (gpu_transform(model_args[0]),) + model_args[1:]

        optimizer.zero_grad(set_to_none=True)

        with amp_ctx:
            outputs = model(*model_args)
            result = _classify_outputs(outputs, labels)
            if len(result) == 3:
                loss_labels, preds, labels_for_acc = result
            else:
                loss_labels, preds = result
                labels_for_acc = labels
            
            if outputs.dim() > 1 and outputs.size(1) > 1:
                loss = criterion(outputs, loss_labels)
            else:
                loss = criterion(outputs.squeeze(), loss_labels)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        running_loss += loss.item() * labels.size(0)
        correct += (preds == labels_for_acc).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{100.0 * correct / total:.2f}%')

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc
