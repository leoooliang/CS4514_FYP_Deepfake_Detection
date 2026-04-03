"""
Checkpoint uses for saving and loading model state.
"""

import os
import torch
from pathlib import Path


def save_checkpoint(model, optimizer, epoch, metrics, save_path, scheduler=None, additional_state=None):
    """
    Save a full training checkpoint.
    """

    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    if scheduler is not None:
        ckpt['scheduler_state_dict'] = scheduler.state_dict()
    if additional_state is not None:
        ckpt['additional_state'] = additional_state

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(ckpt, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None, device='cpu', load_optimizer=True):
    """
    Load a checkpoint and optionally restore optimizer / scheduler state.
    """

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Model weights loaded from {checkpoint_path}")

    if optimizer is not None and load_optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print("Optimizer state loaded")

    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        print("Scheduler state loaded")

    if 'metrics' in ckpt:
        print(f"Checkpoint metrics: {ckpt['metrics']}")

    return ckpt


def save_best_model(model, save_path, epoch, metrics):
    """
    Save model weights only, annotated with epoch and metrics.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
    }, save_path)
    print(f"Best model saved to {save_path}")


def load_model_weights_only(model, weights_path, device='cpu'):
    """
    Load only model weights.
    """

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}")

    ckpt = torch.load(weights_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    print(f"Model weights loaded from {weights_path}")
    return ckpt


class ModelCheckpoint:
    """
    Callback that automatically saves best and periodic checkpoints.
    """

    def __init__(self, save_dir, monitor='val_auc', mode='max', save_best_only=True, save_every_n_epochs=None, filename_prefix='checkpoint', verbose=True):
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_every_n_epochs = save_every_n_epochs
        self.filename_prefix = filename_prefix
        self.verbose = verbose

        self.save_dir.mkdir(parents=True, exist_ok=True)
        if mode == 'max':
            self.best_metric = float('-inf')
            self.is_better = lambda n, b: n > b
        else:
            self.best_metric = float('inf')
            self.is_better = lambda n, b: n < b

    def __call__(self, model, optimizer, epoch, metrics, scheduler=None):
        current = metrics.get(self.monitor)
        if current is None:
            print(f"Warning: Metric '{self.monitor}' not found in metrics dict")
            return

        if self.is_better(current, self.best_metric):
            self.best_metric = current
            save_best_model(model, self.save_dir / f'{self.filename_prefix}_best.pth', epoch, metrics)
            if self.verbose:
                print(f"New best {self.monitor}: {current:.4f}")

        if not self.save_best_only or (
            self.save_every_n_epochs and epoch % self.save_every_n_epochs == 0
        ):
            save_checkpoint(
                model, optimizer, epoch, metrics,
                self.save_dir / f'{self.filename_prefix}_epoch_{epoch}.pth', scheduler,
            )

        save_checkpoint(
            model, optimizer, epoch, metrics,
            self.save_dir / f'{self.filename_prefix}_last.pth', scheduler,
        )


def get_latest_checkpoint(checkpoint_dir, prefix='checkpoint'):
    """
    Return the path to the most recent checkpoint in a directory.
    """

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    last = checkpoint_dir / f'{prefix}_last.pth'
    if last.exists():
        return str(last)

    candidates = list(checkpoint_dir.glob(f'{prefix}_epoch_*.pth'))
    if not candidates:
        return None

    epochs = []
    for cp in candidates:
        try:
            epochs.append((int(cp.stem.split('_')[-1]), cp))
        except (ValueError, IndexError):
            continue

    return str(max(epochs, key=lambda x: x[0])[1]) if epochs else None
