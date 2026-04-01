"""
Evaluation utilities for deepfake detection models.

This module contains:
- validate: Generic validation function for single-stream models
- validate_dual_stream: Validation function for dual-stream audio models
- validate_noise: Validation function for noise residual models
- compute_metrics: Compute comprehensive performance metrics
- evaluate_by_manipulation_type: Per-manipulation-type evaluation
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix
)


def validate(model, dataloader, criterion, device, gpu_transform=None):
    """
    Validate model on validation/test set (single-stream architecture).
    
    Args:
        model: The model to evaluate
        dataloader: Validation/test data loader
        criterion: Loss function
        device: Device to evaluate on
        gpu_transform: Optional GPU-based transform to apply to inputs
    
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
    
    all_preds = []
    all_labels = []
    all_probs = []
    is_multiclass_output = None
    
    pbar = tqdm(dataloader, desc='Validation', leave=False)
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Apply GPU transform if provided
            if gpu_transform is not None:
                inputs = gpu_transform(inputs)
            
            # Forward pass
            outputs = model(inputs)
            
            # Handle different output shapes
            if outputs.dim() > 1 and outputs.size(1) > 1:
                # Multi-class classification (CrossEntropyLoss)
                is_multiclass_output = True
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_probs.extend(probs[:, 0].cpu().numpy())  # Probability of Fake (class 0)
            else:
                # Binary classification (BCEWithLogitsLoss)
                is_multiclass_output = False
                outputs = outputs.squeeze()
                labels_float = labels.float()
                loss = criterion(outputs, labels_float)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()
                all_probs.extend(probs.cpu().numpy())
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'acc': f'{100.0 * correct / total:.2f}%'})
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    # Calculate AUC scores with Fake (class 0) as positive class.
    labels_fake_positive = (all_labels == 0).astype(int)
    try:
        if is_multiclass_output:
            # all_probs already stores P(fake) for CE models
            roc_auc = roc_auc_score(labels_fake_positive, all_probs)
            pr_auc = average_precision_score(labels_fake_positive, all_probs)
        else:
            # For BCE models, sigmoid(outputs) represents P(label=1), so convert to P(fake=0)
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
        'all_probs': all_probs
    }


def validate_dual_stream(model, dataloader, criterion, device):
    """
    Validate the model with dual-stream input (audio models).
    
    Args:
        model: The model to evaluate (expects dual-stream input)
        dataloader: Validation data loader (yields mels, lfccs, labels)
        criterion: Loss function
        device: Device to evaluate on
    
    Returns:
        dict: Dictionary containing:
            - val_loss: Average validation loss
            - accuracy: Classification accuracy
            - pr_auc: PR-AUC (treating Fake as positive class)
            - roc_auc: ROC-AUC (treating Fake as positive class)
            - all_labels: Ground truth labels
            - all_probs: Predicted probabilities for Fake class
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_labels = []
    all_probs = []
    
    progress_bar = tqdm(dataloader, desc='Validation', leave=False)
    
    with torch.no_grad():
        for mels, lfccs, labels in progress_bar:
            mels = mels.to(device, non_blocking=True)
            lfccs = lfccs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(mels, lfccs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Statistics
            running_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Store labels and probabilities for Fake class (class 0)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 0].cpu().numpy())
            
            # Update progress bar
            acc = 100.0 * correct / total
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    val_loss = running_loss / total
    accuracy = correct / total
    
    # Calculate AUC scores (Fake = 0 is positive class)
    try:
        # For binary classification where Fake=0 and Real=1
        # We want probability of Fake (class 0) for positive class
        roc_auc = roc_auc_score(all_labels == 0, all_probs)
        pr_auc = average_precision_score(all_labels == 0, all_probs)
    except ValueError:
        roc_auc = 0.0
        pr_auc = 0.0
    
    return {
        'val_loss': val_loss,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'all_labels': all_labels,
        'all_probs': all_probs
    }


def validate_noise(model, dataloader, criterion, device, srm_layer):
    """
    Validate the model for one epoch with noise residual processing.
    
    This function handles the complete GPU-accelerated pipeline for validation:
    1. Load raw RGB tensors
    2. Apply SRM filtering on GPU
    3. Truncate noise residuals to [-3, 3]
    4. Forward pass through model
    
    Args:
        model: The model to evaluate
        dataloader: Validation data loader (raw RGB tensors)
        criterion: Loss function
        device: Device to evaluate on
        srm_layer: SRM filter layer (on GPU)
    
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
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for inputs, labels in pbar:
            # Move to GPU
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            
            # Scale to [0, 255] range for SRM processing
            inputs = inputs * 255.0
            
            # Extract noise from 3-channel RGB
            inputs = srm_layer(inputs)
            
            # Truncate noise to [-3, 3]
            inputs = torch.clamp(inputs, min=-3.0, max=3.0)
            
            # Forward pass
            outputs = model(inputs).squeeze()
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Get probabilities and predictions
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
    
    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > 0.5).astype(int)
    
    # Calculate metrics
    val_loss = running_loss / total
    accuracy = correct / total
    
    # Calculate AUC scores with Fake (class 0) as positive class.
    labels_fake_positive = (all_labels == 0).astype(int)
    probs_fake = 1.0 - all_probs
    try:
        roc_auc = roc_auc_score(labels_fake_positive, probs_fake)
        pr_auc = average_precision_score(labels_fake_positive, probs_fake)
    except ValueError:
        roc_auc = 0.0
        pr_auc = 0.0
    
    return {
        'val_loss': val_loss,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs
    }


def compute_metrics(labels, predictions, probabilities=None):
    """
    Compute comprehensive performance metrics.
    
    Args:
        labels: Ground truth labels
        predictions: Predicted labels
        probabilities: Predicted probabilities (optional, for AUC scores)
    
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(labels, predictions)
    metrics['precision'] = precision_score(labels, predictions, pos_label=0, zero_division=0)
    metrics['recall'] = recall_score(labels, predictions, pos_label=0, zero_division=0)
    metrics['f1'] = f1_score(labels, predictions, pos_label=0, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    metrics['confusion_matrix'] = cm
    
    # Confusion matrix components
    if cm.shape == (2, 2):
        # confusion_matrix with labels [0,1]:
        # [[fake->fake, fake->real], [real->fake, real->real]]
        fake_tp, fake_fn, fake_fp, fake_tn = cm.ravel()
        metrics['true_negative'] = fake_tn
        metrics['false_positive'] = fake_fp
        metrics['false_negative'] = fake_fn
        metrics['true_positive'] = fake_tp
        
        # Additional metrics
        metrics['specificity'] = fake_tn / (fake_tn + fake_fp) if (fake_tn + fake_fp) > 0 else 0
        metrics['sensitivity'] = fake_tp / (fake_tp + fake_fn) if (fake_tp + fake_fn) > 0 else 0
    
    # AUC scores (if probabilities provided)
    if probabilities is not None:
        try:
            labels_fake_positive = (np.array(labels) == 0).astype(int)
            metrics['roc_auc'] = roc_auc_score(labels_fake_positive, probabilities)
        except ValueError:
            metrics['roc_auc'] = 0.0
        
        try:
            labels_fake_positive = (np.array(labels) == 0).astype(int)
            metrics['pr_auc'] = average_precision_score(labels_fake_positive, probabilities)
        except ValueError:
            metrics['pr_auc'] = 0.0
    
    return metrics


def evaluate_by_manipulation_type(file_paths, predictions, labels, probabilities=None):
    """
    Evaluate performance by manipulation type using pre-computed predictions.
    
    This function analyzes performance for each deepfake manipulation method
    separately (e.g., Deepfakes, FaceSwap, Face2Face, etc.).
    
    Args:
        file_paths: List of file paths for test samples
        predictions: Pre-computed predictions from test evaluation
        labels: Ground truth labels from test evaluation
        probabilities: Predicted probabilities (optional)
        
    Returns:
        dict: Dictionary with manipulation types as keys and metrics as values
    """
    import time
    import os
    start_time = time.time()
    
    num_samples = len(file_paths)
    print(f"Analyzing {num_samples} test samples by manipulation type...")
    
    # Convert to numpy if needed
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    if probabilities is not None and not isinstance(probabilities, np.ndarray):
        probabilities = np.array(probabilities)
    
    # Extract manipulation types from file paths
    print("Mapping samples to manipulation types...")
    manip_types = []
    for fp in file_paths:
        # Extract manipulation type from path
        # Typical path: .../fake/Deepfakes_c23_...
        # or: .../real/...
        basename = os.path.basename(fp)
        if 'real' in fp.lower():
            manip_types.append('Real')
        else:
            # Extract manipulation method from filename
            parts = basename.split('_')
            if len(parts) > 0:
                manip_types.append(parts[0])
            else:
                manip_types.append('Unknown')
    
    # Get unique manipulation types
    unique_types = sorted(set(manip_types))
    print(f"Found {len(unique_types)} manipulation types: {unique_types}")
    
    # Evaluate each manipulation type
    results = {}
    for manip_type in unique_types:
        # Get indices for this manipulation type
        indices = [i for i, mt in enumerate(manip_types) if mt == manip_type]
        
        if len(indices) == 0:
            continue
        
        # Extract data for this manipulation type
        type_labels = labels[indices]
        type_preds = predictions[indices]
        type_probs = probabilities[indices] if probabilities is not None else None
        
        # Compute metrics
        metrics = compute_metrics(type_labels, type_preds, type_probs)
        metrics['num_samples'] = len(indices)
        
        results[manip_type] = metrics
    
    elapsed_time = time.time() - start_time
    print(f"Manipulation type analysis completed in {elapsed_time:.2f} seconds")
    
    return results
