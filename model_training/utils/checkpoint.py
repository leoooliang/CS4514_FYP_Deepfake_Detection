"""
Model checkpoint utilities for saving and loading models.

This module contains:
- save_checkpoint: Save model, optimizer, and training state
- load_checkpoint: Load model, optimizer, and training state
- save_best_model: Save best model based on metric
- ModelCheckpoint: Callback for automatic model checkpointing
"""

import os
import torch
from pathlib import Path


def save_checkpoint(model, optimizer, epoch, metrics, save_path, 
                   scheduler=None, additional_state=None):
    """
    Save model checkpoint with training state.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        metrics: Dictionary of metrics (e.g., {'val_loss': 0.5, 'val_auc': 0.95})
        save_path: Path to save checkpoint
        scheduler: Learning rate scheduler (optional)
        additional_state: Additional state dictionary to save (optional)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if additional_state is not None:
        checkpoint['additional_state'] = additional_state
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None, 
                   device='cpu', load_optimizer=True):
    """
    Load model checkpoint and optionally restore training state.
    
    Args:
        model: PyTorch model to load weights into
        checkpoint_path: Path to checkpoint file
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load model to (default: 'cpu')
        load_optimizer: Whether to load optimizer state (default: True)
        
    Returns:
        dict: Checkpoint dictionary containing epoch, metrics, etc.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model weights loaded from {checkpoint_path}")
    
    # Load optimizer state
    if optimizer is not None and load_optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded")
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Scheduler state loaded")
    
    # Print metrics
    if 'metrics' in checkpoint:
        print(f"Checkpoint metrics: {checkpoint['metrics']}")
    
    return checkpoint


def save_best_model(model, save_path, epoch, metrics):
    """
    Save best model (weights only, no optimizer).
    
    Args:
        model: PyTorch model to save
        save_path: Path to save model
        epoch: Current epoch number
        metrics: Dictionary of metrics
    """
    model_state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save model
    torch.save(model_state, save_path)
    print(f"Best model saved to {save_path}")


class ModelCheckpoint:
    """
    Callback for automatic model checkpointing during training.
    
    This class automatically saves the best model based on a specified metric
    and optionally saves periodic checkpoints.
    
    Args:
        save_dir: Directory to save checkpoints
        monitor: Metric to monitor (e.g., 'val_auc', 'val_loss')
        mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        save_best_only: If True, only save when metric improves (default: True)
        save_every_n_epochs: Save checkpoint every N epochs (default: None)
        filename_prefix: Prefix for checkpoint filenames (default: 'checkpoint')
        verbose: Print save messages (default: True)
    """
    
    def __init__(self, save_dir, monitor='val_auc', mode='max', 
                 save_best_only=True, save_every_n_epochs=None,
                 filename_prefix='checkpoint', verbose=True):
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_every_n_epochs = save_every_n_epochs
        self.filename_prefix = filename_prefix
        self.verbose = verbose
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize best metric
        if mode == 'max':
            self.best_metric = float('-inf')
            self.is_better = lambda new, best: new > best
        else:
            self.best_metric = float('inf')
            self.is_better = lambda new, best: new < best
    
    def __call__(self, model, optimizer, epoch, metrics, scheduler=None):
        """
        Check if checkpoint should be saved and save if necessary.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch number
            metrics: Dictionary of metrics
            scheduler: Learning rate scheduler (optional)
        """
        current_metric = metrics.get(self.monitor)
        
        if current_metric is None:
            print(f"Warning: Metric '{self.monitor}' not found in metrics dictionary")
            return
        
        # Check if this is the best model
        is_best = self.is_better(current_metric, self.best_metric)
        
        if is_best:
            self.best_metric = current_metric
            
            # Save best model
            best_path = self.save_dir / f'{self.filename_prefix}_best.pth'
            save_best_model(model, best_path, epoch, metrics)
            
            if self.verbose:
                print(f"New best {self.monitor}: {current_metric:.4f}")
        
        # Save periodic checkpoint
        if not self.save_best_only or (self.save_every_n_epochs and epoch % self.save_every_n_epochs == 0):
            checkpoint_path = self.save_dir / f'{self.filename_prefix}_epoch_{epoch}.pth'
            save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path, scheduler)
        
        # Save last checkpoint (overwrite)
        last_path = self.save_dir / f'{self.filename_prefix}_last.pth'
        save_checkpoint(model, optimizer, epoch, metrics, last_path, scheduler)


def load_model_weights_only(model, weights_path, device='cpu'):
    """
    Load only model weights (no optimizer, scheduler, or other state).
    
    Args:
        model: PyTorch model to load weights into
        weights_path: Path to weights file (.pth)
        device: Device to load model to (default: 'cpu')
        
    Returns:
        dict: Checkpoint dictionary (if available)
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}")
    
    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location=device)
    
    # Check if this is a full checkpoint or just weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Assume the file contains only model weights
        model.load_state_dict(checkpoint)
    
    print(f"Model weights loaded from {weights_path}")
    
    return checkpoint


def get_latest_checkpoint(checkpoint_dir, prefix='checkpoint'):
    """
    Get the path to the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        prefix: Checkpoint filename prefix (default: 'checkpoint')
        
    Returns:
        str: Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # Look for 'last' checkpoint first
    last_checkpoint = checkpoint_dir / f'{prefix}_last.pth'
    if last_checkpoint.exists():
        return str(last_checkpoint)
    
    # Otherwise, find the checkpoint with highest epoch number
    checkpoints = list(checkpoint_dir.glob(f'{prefix}_epoch_*.pth'))
    
    if not checkpoints:
        return None
    
    # Extract epoch numbers and find max
    epochs = []
    for cp in checkpoints:
        try:
            epoch_str = cp.stem.split('_')[-1]
            epochs.append((int(epoch_str), cp))
        except (ValueError, IndexError):
            continue
    
    if not epochs:
        return None
    
    # Return checkpoint with highest epoch
    latest = max(epochs, key=lambda x: x[0])
    return str(latest[1])
