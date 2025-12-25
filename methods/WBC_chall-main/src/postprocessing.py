"""
Post-processing utilities for WBC Challenge.
Includes temperature scaling, calibration, and class-specific threshold tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for calibrating model predictions.
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize temperature scaling.
        
        Args:
            temperature: Temperature parameter (learned or fixed)
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Model logits [B, num_classes]
        
        Returns:
            Scaled logits
        """
        return logits / self.temperature
    
    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor, 
                  max_iter: int = 50, lr: float = 0.01):
        """
        Learn optimal temperature on validation set.
        
        Args:
            logits: Validation logits [N, num_classes]
            labels: Validation labels [N]
            max_iter: Maximum optimization iterations
            lr: Learning rate
        """
        self.train()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        self.eval()


def apply_temperature_scaling(
    logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    
    Args:
        logits: Model logits
        temperature: Temperature parameter
    
    Returns:
        Scaled logits
    """
    return logits / temperature


def calibrate_temperature(
    model: nn.Module,
    val_loader,
    device: torch.device,
    max_iter: int = 50
) -> float:
    """
    Calibrate temperature on validation set.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to run inference on
        max_iter: Maximum optimization iterations
    
    Returns:
        Optimal temperature value
    """
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Learn temperature
    temperature_scaler = TemperatureScaling()
    temperature_scaler.calibrate(all_logits, all_labels, max_iter=max_iter)
    
    optimal_temp = temperature_scaler.temperature.item()
    print(f"Optimal temperature: {optimal_temp:.4f}")
    
    return optimal_temp


class ClassSpecificThresholdTuning:
    """
    Class-specific threshold tuning for imbalanced classification.
    """
    
    def __init__(self, num_classes: int = 13):
        """
        Initialize threshold tuner.
        
        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
        self.thresholds = np.ones(num_classes) * 0.5  # Default: equal thresholds
    
    def tune_thresholds(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        metric: str = 'f1',
        rare_class_boost: bool = True
    ) -> np.ndarray:
        """
        Tune class-specific thresholds to optimize metric.
        
        Args:
            probs: Predicted probabilities [N, num_classes]
            labels: True labels [N]
            metric: Metric to optimize ('f1', 'precision', 'recall')
            rare_class_boost: If True, prioritize rare classes
        
        Returns:
            Optimal thresholds for each class
        """
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        thresholds = np.zeros(self.num_classes)
        
        for class_idx in range(self.num_classes):
            class_probs = probs[:, class_idx]
            class_labels = (labels == class_idx).astype(int)
            
            # Try different thresholds
            best_threshold = 0.5
            best_score = 0.0
            
            for threshold in np.arange(0.1, 0.9, 0.05):
                preds = (class_probs >= threshold).astype(int)
                
                if metric == 'f1':
                    score = f1_score(class_labels, preds, zero_division=0)
                elif metric == 'precision':
                    score = precision_score(class_labels, preds, zero_division=0)
                elif metric == 'recall':
                    score = recall_score(class_labels, preds, zero_division=0)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            thresholds[class_idx] = best_threshold
            
            # Boost rare classes (lower threshold = higher recall)
            if rare_class_boost and class_idx in [8, 9, 10]:  # PC, PLY, PMY
                thresholds[class_idx] = max(0.1, thresholds[class_idx] - 0.1)
        
        self.thresholds = thresholds
        print(f"Tuned thresholds: {thresholds}")
        
        return thresholds
    
    def predict(self, probs: np.ndarray) -> np.ndarray:
        """
        Predict using class-specific thresholds.
        
        Args:
            probs: Predicted probabilities [N, num_classes]
        
        Returns:
            Predictions [N]
        """
        predictions = np.zeros(len(probs), dtype=int)
        
        for i in range(len(probs)):
            # For each sample, find class with highest probability above threshold
            adjusted_probs = probs[i].copy()
            
            # Apply thresholds
            for class_idx in range(self.num_classes):
                if adjusted_probs[class_idx] < self.thresholds[class_idx]:
                    adjusted_probs[class_idx] = 0.0
            
            # Predict class with highest adjusted probability
            if adjusted_probs.sum() > 0:
                predictions[i] = np.argmax(adjusted_probs)
            else:
                # Fallback to standard argmax if no class passes threshold
                predictions[i] = np.argmax(probs[i])
        
        return predictions


def apply_class_specific_thresholds(
    probs: np.ndarray,
    thresholds: np.ndarray
) -> np.ndarray:
    """
    Apply class-specific thresholds to probabilities.
    
    Args:
        probs: Predicted probabilities [N, num_classes]
        thresholds: Thresholds for each class [num_classes]
    
    Returns:
        Predictions [N]
    """
    tuner = ClassSpecificThresholdTuning(num_classes=len(thresholds))
    tuner.thresholds = thresholds
    return tuner.predict(probs)


def smooth_predictions(
    predictions: np.ndarray,
    window_size: int = 3
) -> np.ndarray:
    """
    Apply moving average smoothing to predictions.
    Useful for time-series or sequential data.
    
    Args:
        predictions: Predictions array
        window_size: Size of smoothing window
    
    Returns:
        Smoothed predictions
    """
    if window_size <= 1:
        return predictions
    
    smoothed = np.zeros_like(predictions)
    half_window = window_size // 2
    
    for i in range(len(predictions)):
        start = max(0, i - half_window)
        end = min(len(predictions), i + half_window + 1)
        
        # Get most common class in window
        window_preds = predictions[start:end]
        smoothed[i] = np.bincount(window_preds).argmax()
    
    return smoothed


def calibrate_predictions(
    probs: np.ndarray,
    labels: np.ndarray,
    method: str = 'isotonic'
) -> callable:
    """
    Calibrate predictions using isotonic regression or Platt scaling.
    
    Args:
        probs: Predicted probabilities [N, num_classes]
        labels: True labels [N]
        method: Calibration method ('isotonic' or 'platt')
    
    Returns:
        Calibration function
    """
    from sklearn.calibration import CalibratedClassifierCV
    
    # For multi-class, calibrate each class separately
    calibrated_probs = np.zeros_like(probs)
    
    for class_idx in range(probs.shape[1]):
        class_probs = probs[:, class_idx]
        class_labels = (labels == class_idx).astype(int)
        
        if method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
        else:
            from sklearn.linear_model import LogisticRegression
            calibrator = LogisticRegression()
        
        calibrator.fit(class_probs.reshape(-1, 1), class_labels)
        calibrated_probs[:, class_idx] = calibrator.predict_proba(
            class_probs.reshape(-1, 1)
        )[:, 1]
    
    # Normalize to ensure probabilities sum to 1
    calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)
    
    return calibrated_probs

