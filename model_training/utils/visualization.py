"""
Visualization utilities for deepfake detection model evaluation.

This module contains:
- plot_confusion_matrix: Plot confusion matrix heatmap
- plot_roc_curve: Plot ROC curve with AUC score
- plot_precision_recall_curve: Plot Precision-Recall curve
- plot_training_history: Plot training/validation metrics over epochs
- plot_manipulation_type_comparison: Compare performance across manipulation types
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix


def plot_confusion_matrix(labels, predictions, class_names=['Fake', 'Real'], 
                          normalize=False, title='Confusion Matrix', 
                          figsize=(8, 6), save_path=None):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        labels: Ground truth labels
        predictions: Predicted labels
        class_names: List of class names (default: ['Fake', 'Real'])
        normalize: Whether to normalize the confusion matrix (default: False)
        title: Plot title (default: 'Confusion Matrix')
        figsize: Figure size (default: (8, 6))
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    cm = confusion_matrix(labels, predictions)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curve(labels, probabilities, title='ROC Curve', 
                   figsize=(8, 6), save_path=None):
    """
    Plot ROC curve with AUC score.
    
    Args:
        labels: Ground truth labels
        probabilities: Predicted probabilities for positive class
        title: Plot title (default: 'ROC Curve')
        figsize: Figure size (default: (8, 6))
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(labels, probabilities, title='Precision-Recall Curve',
                                figsize=(8, 6), save_path=None):
    """
    Plot Precision-Recall curve with AP score.
    
    Args:
        labels: Ground truth labels
        probabilities: Predicted probabilities for positive class
        title: Plot title (default: 'Precision-Recall Curve')
        figsize: Figure size (default: (8, 6))
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(recall, precision, color='blue', lw=2, 
            label=f'PR curve (AP = {pr_auc:.4f})')
    
    # Baseline (proportion of positive class)
    baseline = np.sum(labels) / len(labels)
    ax.plot([0, 1], [baseline, baseline], color='gray', lw=2, 
            linestyle='--', label=f'Baseline ({baseline:.4f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_history(train_losses, val_losses=None, train_accs=None, 
                          val_accs=None, val_aucs=None, figsize=(12, 4), 
                          save_path=None):
    """
    Plot training history (loss, accuracy, AUC).
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch (optional)
        train_accs: List of training accuracies per epoch (optional)
        val_accs: List of validation accuracies per epoch (optional)
        val_aucs: List of validation AUC scores per epoch (optional)
        figsize: Figure size (default: (12, 4))
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    num_plots = 1 + (train_accs is not None) + (val_aucs is not None)
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    
    if num_plots == 1:
        axes = [axes]
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot 1: Loss
    ax = axes[0]
    ax.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    if val_losses is not None:
        ax.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 2: Accuracy (if provided)
    if train_accs is not None:
        ax = axes[1]
        ax.plot(epochs, train_accs, 'b-o', label='Train Acc', linewidth=2, markersize=4)
        if val_accs is not None:
            ax.plot(epochs, val_accs, 'r-s', label='Val Acc', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    
    # Plot 3: AUC (if provided)
    if val_aucs is not None:
        ax = axes[-1]
        ax.plot(epochs, val_aucs, 'g-^', label='Val AUC', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('AUC-ROC', fontsize=11)
        ax.set_title('Validation AUC-ROC', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_manipulation_type_comparison(results_dict, metric='roc_auc', 
                                      title=None, figsize=(10, 6), 
                                      save_path=None):
    """
    Compare performance across different manipulation types.
    
    Args:
        results_dict: Dictionary with manipulation types as keys and metrics as values
        metric: Metric to plot (default: 'roc_auc')
        title: Plot title (default: auto-generated from metric)
        figsize: Figure size (default: (10, 6))
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Extract manipulation types and metric values
    manip_types = []
    metric_values = []
    sample_counts = []
    
    for manip_type, metrics in results_dict.items():
        if metric in metrics:
            manip_types.append(manip_type)
            metric_values.append(metrics[metric])
            sample_counts.append(metrics.get('num_samples', 0))
    
    # Sort by metric value (descending)
    sorted_indices = np.argsort(metric_values)[::-1]
    manip_types = [manip_types[i] for i in sorted_indices]
    metric_values = [metric_values[i] for i in sorted_indices]
    sample_counts = [sample_counts[i] for i in sorted_indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(manip_types)))
    bars = ax.barh(manip_types, metric_values, color=colors)
    
    # Add value labels and sample counts
    for i, (bar, val, count) in enumerate(zip(bars, metric_values, sample_counts)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f} (n={count})',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Formatting
    metric_name = metric.replace('_', ' ').title()
    if title is None:
        title = f'{metric_name} by Manipulation Type'
    
    ax.set_xlabel(metric_name, fontsize=12)
    ax.set_ylabel('Manipulation Type', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([0, min(1.1, max(metric_values) * 1.15)])
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_multiple_roc_curves(labels_list, probs_list, labels_legend, 
                             title='ROC Curve Comparison', figsize=(8, 6), 
                             save_path=None):
    """
    Plot multiple ROC curves on the same plot for comparison.
    
    Args:
        labels_list: List of ground truth label arrays
        probs_list: List of predicted probability arrays
        labels_legend: List of labels for each curve
        title: Plot title (default: 'ROC Curve Comparison')
        figsize: Figure size (default: (8, 6))
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels_list)))
    
    for i, (labels, probs, label) in enumerate(zip(labels_list, probs_list, labels_legend)):
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2, 
                label=f'{label} (AUC = {roc_auc:.4f})')
    
    # Random classifier baseline
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
            label='Random Classifier', alpha=0.7)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_score_distribution(labels, probabilities, bins=50, 
                            title='Score Distribution', figsize=(8, 6), 
                            save_path=None):
    """
    Plot distribution of prediction scores for real vs fake samples.
    
    Args:
        labels: Ground truth labels (0=fake, 1=real)
        probabilities: Predicted probabilities
        bins: Number of histogram bins (default: 50)
        title: Plot title (default: 'Score Distribution')
        figsize: Figure size (default: (8, 6))
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Separate scores by class
    fake_scores = probabilities[labels == 0]
    real_scores = probabilities[labels == 1]
    
    # Plot histograms
    ax.hist(fake_scores, bins=bins, alpha=0.6, label='Fake', color='red', density=True)
    ax.hist(real_scores, bins=bins, alpha=0.6, label='Real', color='blue', density=True)
    
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
