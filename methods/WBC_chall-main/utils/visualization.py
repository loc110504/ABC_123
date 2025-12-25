"""
Visualization utilities for WBC Challenge.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from src.dataset import WBCDataset


def plot_class_distribution(
    labels_df: pd.DataFrame,
    title: str = "Class Distribution",
    save_path: Optional[str] = None
):
    """
    Plot class distribution.
    
    Args:
        labels_df: DataFrame with 'labels' column
        title: Plot title
        save_path: Path to save figure
    """
    class_counts = labels_df['labels'].value_counts().reindex(WBCDataset.CLASSES, fill_value=0)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_counts)), class_counts.values)
    plt.xticks(range(len(class_counts)), class_counts.index, rotation=45, ha='right')
    plt.ylabel('Count')
    plt.xlabel('Class')
    plt.title(title)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None
):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Macro-F1
    axes[0, 1].plot(epochs, history['val_macro_f1'], 'g-', label='Val Macro-F1')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Macro-F1 Score')
    axes[0, 1].set_title('Validation Macro-F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Balanced Accuracy
    axes[1, 0].plot(epochs, history['val_balanced_acc'], 'm-', label='Val Balanced Acc')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Balanced Accuracy')
    axes[1, 0].set_title('Validation Balanced Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate
    axes[1, 1].plot(epochs, history['learning_rate'], 'c-', label='Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10)
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
    """
    if class_names is None:
        class_names = WBCDataset.CLASSES
    
    plt.figure(figsize=figsize)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'}
    )
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Normalized)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_per_class_f1(
    metrics: Dict,
    save_path: Optional[str] = None
):
    """
    Plot per-class F1 scores.
    
    Args:
        metrics: Metrics dictionary from calculate_metrics
        save_path: Path to save figure
    """
    class_names = WBCDataset.CLASSES
    f1_scores = metrics['per_class_f1']
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), f1_scores)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylabel('F1 Score')
    plt.xlabel('Class')
    plt.title('Per-Class F1 Scores')
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        plt.text(bar.get_x() + bar.get_width()/2., score + 0.01,
                f'{score:.3f}',
                ha='center', va='bottom', fontsize=8)
    
    # Add macro-F1 line
    macro_f1 = metrics['macro_f1']
    plt.axhline(y=macro_f1, color='r', linestyle='--', 
                label=f'Macro-F1: {macro_f1:.4f}')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def visualize_sample_images(
    dataset,
    num_samples: int = 16,
    save_path: Optional[str] = None
):
    """
    Visualize sample images from dataset.
    
    Args:
        dataset: WBCDataset instance
        num_samples: Number of samples to visualize
        save_path: Path to save figure
    """
    import torch
    from torchvision.utils import make_grid
    
    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    samples = [dataset[i] for i in indices]
    
    # Create grid
    images = torch.stack([s['image'] for s in samples])
    
    # Denormalize for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    grid = make_grid(images, nrow=4, padding=2, normalize=False)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title(f'Sample Images from {dataset.mode} Set')
    
    # Add labels
    if dataset.mode != 'test':
        labels = [WBCDataset.IDX_TO_CLASS[s['label']] for s in samples]
        for i, label in enumerate(labels):
            row = i // 4
            col = i % 4
            plt.text(col * 98 + 10, row * 98 + 90, label,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

