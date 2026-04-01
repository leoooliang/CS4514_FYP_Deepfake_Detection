"""
Training utilities for deepfake detection models.

This module contains:
- train_one_epoch: Generic training loop for single-stream models
- train_one_epoch_dual_stream: Training loop for dual-stream audio models
- train_one_epoch_noise: Training loop for noise residual models with GPU augmentation
- EarlyStopping: Early stopping callback
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


def train_one_epoch(model, dataloader, criterion, optimizer, device, 
                   return_predictions=False, gpu_transform=None):
    """
    Train model for one epoch (single-stream architecture).
    
    Args:
        model: The model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        return_predictions: If True, return predictions and probabilities (slower)
        gpu_transform: Optional GPU-based transform to apply to inputs
    
    Returns:
        If return_predictions=False: (epoch_loss, epoch_acc)
        If return_predictions=True: (epoch_loss, epoch_acc, all_preds, all_labels, all_probs)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    
    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Apply GPU transform if provided
        if gpu_transform is not None:
            inputs = gpu_transform(inputs)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Handle different output shapes
        if outputs.dim() > 1 and outputs.size(1) > 1:
            # Multi-class classification (CrossEntropyLoss)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            if return_predictions:
                all_probs.extend(probs[:, 0].cpu().numpy())  # Probability of Fake (class 0)
        else:
            # Binary classification (BCEWithLogitsLoss)
            outputs = outputs.squeeze()
            labels = labels.float()
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()
            if return_predictions:
                all_probs.extend(probs.cpu().numpy())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        if return_predictions:
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'acc': f'{100.0 * correct / total:.2f}%'})
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    if return_predictions:
        return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)
    else:
        return epoch_loss, epoch_acc


def train_one_epoch_dual_stream(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch with dual-stream input (audio models).
    
    Args:
        model: The model to train (expects dual-stream input)
        dataloader: Training data loader (yields mels, lfccs, labels)
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        tuple: (avg_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    
    for mels, lfccs, labels in progress_bar:
        mels = mels.to(device, non_blocking=True)
        lfccs = lfccs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with dual streams
        outputs = model(mels, lfccs)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        acc = 100.0 * correct / total
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def train_one_epoch_noise(model, dataloader, criterion, optimizer, device, 
                          srm_layer, gpu_augment=None, label_smoothing=0.0):
    """
    Train the model for one epoch with noise residual processing.
    
    This function handles the complete GPU-accelerated pipeline:
    1. Load raw RGB tensors
    2. Apply GPU augmentation (if provided)
    3. Apply SRM filtering on GPU
    4. Truncate noise residuals to [-3, 3]
    5. Forward pass through model
    
    Args:
        model: The model to train
        dataloader: Training data loader (raw RGB tensors)
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        srm_layer: SRM filter layer (on GPU)
        gpu_augment: Optional GPU augmentation module
        label_smoothing: Label smoothing factor (default: 0.0)
    
    Returns:
        tuple: (epoch_loss, epoch_acc)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    
    for inputs, labels in pbar:
        # Move to GPU
        inputs = inputs.to(device, non_blocking=True)
        labels_tensor = labels.to(device, non_blocking=True).float()
        
        # Apply label smoothing if specified
        if label_smoothing > 0:
            labels_smooth = torch.where(
                labels_tensor == 0.0,
                torch.tensor(label_smoothing, device=device),
                torch.tensor(1.0 - label_smoothing, device=device)
            )
        else:
            labels_smooth = labels_tensor
        
        # Zero gradients
        optimizer.zero_grad()
        
        # GPU Pipeline: Augmentation → SRM → Truncation
        # Scale to [0, 255] range
        inputs = inputs * 255.0
        
        # Apply augmentation (if training)
        if gpu_augment is not None:
            inputs = gpu_augment(inputs)
        
        # Extract noise residuals with SRM
        inputs = srm_layer(inputs)
        
        # Truncate noise to [-3, 3] (respects zero-mean Laplacian distribution)
        inputs = torch.clamp(inputs, min=-3.0, max=3.0)
        
        # Forward pass
        outputs = model(inputs).squeeze()
        
        # Calculate loss
        loss = criterion(outputs, labels_smooth)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        correct += (preds == labels_tensor).sum().item()
        total += labels_tensor.size(0)
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.0 * correct / total:.2f}%'})
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


class EarlyStopping:
    """
    Early stopping callback to stop training when validation metric stops improving.
    
    Args:
        patience: Number of epochs to wait before stopping (default: 10)
        min_delta: Minimum change to qualify as improvement (default: 0.0)
        mode: 'min' for loss, 'max' for accuracy/AUC (default: 'max')
        verbose: Print messages (default: True)
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
        """
        Check if training should stop.
        
        Args:
            metric: Current validation metric value
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = metric
            return False
        
        # Check if improvement
        if self.mode == 'max':
            improved = metric > (self.best_score + self.min_delta)
        else:
            improved = metric < (self.best_score - self.min_delta)
        
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
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
