"""
Evaluation metrics and utilities for WBC Challenge.
"""

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    accuracy_score, balanced_accuracy_score,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple
import pandas as pd

from src.dataset import WBCDataset


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'macro'
) -> Dict:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for F1, precision, recall
    
    Returns:
        Dictionary of metrics
    """
    # Overall metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create per-class dictionary
    per_class_metrics = {}
    for idx, class_name in enumerate(WBCDataset.CLASSES):
        per_class_metrics[class_name] = {
            'f1': per_class_f1[idx],
            'precision': per_class_precision[idx],
            'recall': per_class_recall[idx],
            'support': np.sum(y_true == idx)
        }
    
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'per_class_f1': per_class_f1,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm
    }


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray):
    """Print detailed classification report."""
    target_names = WBCDataset.CLASSES
    print(classification_report(
        y_true, y_pred,
        target_names=target_names,
        zero_division=0
    ))


def print_per_class_metrics(metrics: Dict):
    """Print per-class metrics in a readable format."""
    print("\nPer-Class Metrics:")
    print("-" * 80)
    print(f"{'Class':<10} {'F1':<10} {'Precision':<12} {'Recall':<12} {'Support':<10}")
    print("-" * 80)
    
    for class_name in WBCDataset.CLASSES:
        class_metrics = metrics['per_class_metrics'][class_name]
        print(f"{class_name:<10} "
              f"{class_metrics['f1']:<10.4f} "
              f"{class_metrics['precision']:<12.4f} "
              f"{class_metrics['recall']:<12.4f} "
              f"{class_metrics['support']:<10}")
    
    print("-" * 80)
    print(f"{'Macro Avg':<10} "
          f"{metrics['macro_f1']:<10.4f} "
          f"{metrics['macro_precision']:<12.4f} "
          f"{metrics['macro_recall']:<12.4f}")
    print("-" * 80)


def save_confusion_matrix(cm: np.ndarray, save_path: str, class_names: List[str] = None):
    """Save confusion matrix to file."""
    if class_names is None:
        class_names = WBCDataset.CLASSES
    
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(save_path)
    print(f"Confusion matrix saved to {save_path}")


def save_predictions(
    image_ids: List[str],
    predictions: np.ndarray,
    true_labels: np.ndarray = None,
    save_path: str = 'predictions.csv'
):
    """
    Save predictions to CSV file.
    
    Args:
        image_ids: List of image IDs
        predictions: Predicted class indices
        true_labels: True class indices (optional)
        save_path: Path to save predictions
    """
    results = {
        'image_id': image_ids,
        'predicted_class': [WBCDataset.IDX_TO_CLASS[p] for p in predictions]
    }
    
    if true_labels is not None:
        results['true_class'] = [WBCDataset.IDX_TO_CLASS[t] for t in true_labels]
        results['correct'] = (predictions == true_labels).astype(int)
    
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")

