"""
Ensemble prediction utilities for WBC Challenge.
Supports weighted averaging, hard/soft voting, and multi-model ensembles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from tqdm import tqdm

from src.dataset import WBCDataset
from src.models import create_model
from src.config import Config
from src.submission import validate_submission


class ModelEnsemble:
    """Ensemble of multiple models for prediction."""
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        device: torch.device = None
    ):
        """
        Initialize ensemble.
        
        Args:
            models: List of trained models
            weights: Optional weights for each model (default: equal weights)
            device: Device to run inference on
        """
        self.models = models
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set equal weights if not provided
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        # Normalize weights
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]
        
        # Move models to device and set to eval mode
        for model in self.models:
            model.to(self.device)
            model.eval()
    
    def predict(
        self,
        images: torch.Tensor,
        use_tta: bool = False,
        return_probs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate ensemble predictions.
        
        Args:
            images: Input images tensor [B, C, H, W]
            use_tta: Whether to use test-time augmentation
            return_probs: If True, return probabilities in addition to predictions
        
        Returns:
            Predictions (class indices) or (predictions, probabilities) if return_probs=True
        """
        all_probs = []
        
        with torch.no_grad():
            for model, weight in zip(self.models, self.weights):
                if use_tta:
                    # Apply TTA for this model
                    probs_list = []
                    
                    # Original
                    logits = model(images)
                    probs_list.append(F.softmax(logits, dim=1))
                    
                    # Horizontal flip
                    probs_list.append(F.softmax(model(torch.flip(images, [3])), dim=1))
                    
                    # Vertical flip
                    probs_list.append(F.softmax(model(torch.flip(images, [2])), dim=1))
                    
                    # Both flips
                    probs_list.append(F.softmax(model(torch.flip(images, [2, 3])), dim=1))
                    
                    # Average TTA predictions
                    model_probs = torch.stack(probs_list).mean(0)
                else:
                    logits = model(images)
                    model_probs = F.softmax(logits, dim=1)
                
                # Weight the probabilities
                all_probs.append(model_probs * weight)
        
        # Ensemble: weighted average of probabilities
        ensemble_probs = torch.stack(all_probs).sum(0)
        predictions = torch.argmax(ensemble_probs, dim=1)
        
        if return_probs:
            return predictions, ensemble_probs
        return predictions
    
    def predict_batch(
        self,
        test_loader,
        use_tta: bool = False
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Generate predictions for entire test set.
        
        Args:
            test_loader: DataLoader for test set
            use_tta: Whether to use test-time augmentation
        
        Returns:
            Tuple of (image_ids, predictions, probabilities)
        """
        all_image_ids = []
        all_predictions = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Ensemble prediction')
            for batch in pbar:
                images = batch['image'].to(self.device)
                image_ids = batch['image_id']
                
                preds, probs = self.predict(images, use_tta=use_tta, return_probs=True)
                
                all_image_ids.extend(image_ids)
                all_predictions.extend(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        all_probs = np.vstack(all_probs)
        
        return all_image_ids, np.array(all_predictions), all_probs


def load_ensemble_models(
    model_configs: List[Dict],
    checkpoint_paths: List[str],
    device: torch.device = None
) -> List[nn.Module]:
    """
    Load multiple models from checkpoints.
    
    Args:
        model_configs: List of model configuration dicts
        checkpoint_paths: List of checkpoint file paths
        device: Device to load models on
    
    Returns:
        List of loaded models
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = []
    
    for config_dict, checkpoint_path in zip(model_configs, checkpoint_paths):
        # Create model
        model = create_model(config_dict, num_classes=13)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume checkpoint is the state dict itself
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        models.append(model)
    
    return models


def generate_ensemble_submission(
    models: List[nn.Module],
    test_loader,
    device: torch.device,
    output_path: str = 'submission_ensemble.csv',
    weights: Optional[List[float]] = None,
    use_tta: bool = True,
    model_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate submission file from ensemble of models.
    
    Args:
        models: List of trained models
        test_loader: Test data loader
        device: Device to run inference on
        output_path: Path to save submission file
        weights: Optional weights for each model
        use_tta: Whether to use test-time augmentation
        model_names: Optional names for each model (for logging)
    
    Returns:
        Submission DataFrame
    """
    ensemble = ModelEnsemble(models, weights=weights, device=device)
    
    if model_names:
        print(f"\nEnsemble models: {', '.join(model_names)}")
    print(f"Using TTA: {use_tta}")
    if weights:
        print(f"Model weights: {weights}")
    
    all_image_ids, all_predictions, all_probs = ensemble.predict_batch(
        test_loader, use_tta=use_tta
    )
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'ID': all_image_ids,
        'labels': [WBCDataset.IDX_TO_CLASS[p] for p in all_predictions]
    })
    
    # Validate submission
    validate_submission(submission)
    
    # Save submission
    submission.to_csv(output_path, index=False)
    print(f"\nEnsemble submission file saved to {output_path}")
    print(f"Total predictions: {len(submission)}")
    print(f"Class distribution:")
    print(submission['labels'].value_counts().sort_index())
    
    return submission


def compute_ensemble_weights_from_validation_scores(
    validation_scores: List[float],
    method: str = 'linear'
) -> List[float]:
    """
    Compute ensemble weights from validation scores.
    
    Args:
        validation_scores: List of validation F1 scores for each model
        method: Weighting method ('linear', 'exponential', 'rank')
    
    Returns:
        List of normalized weights
    """
    if method == 'linear':
        # Linear weighting: weight proportional to score
        weights = [max(0, score) for score in validation_scores]
    elif method == 'exponential':
        # Exponential weighting: weight = exp(score * temperature)
        temperature = 10.0
        weights = [np.exp(score * temperature) for score in validation_scores]
    elif method == 'rank':
        # Rank-based weighting: higher rank = higher weight
        sorted_indices = sorted(range(len(validation_scores)), 
                               key=lambda i: validation_scores[i], 
                               reverse=True)
        weights = [0.0] * len(validation_scores)
        for rank, idx in enumerate(sorted_indices):
            weights[idx] = len(validation_scores) - rank
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Normalize
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]
    else:
        weights = [1.0 / len(weights)] * len(weights)
    
    return weights


def hard_vote_ensemble(
    predictions_list: List[np.ndarray]
) -> np.ndarray:
    """
    Hard voting ensemble: majority class wins.
    
    Args:
        predictions_list: List of prediction arrays from different models
    
    Returns:
        Ensemble predictions
    """
    # Stack predictions
    predictions_stack = np.stack(predictions_list, axis=1)  # [N, M] where M is num models
    
    # Get majority vote for each sample
    ensemble_preds = []
    for i in range(predictions_stack.shape[0]):
        votes = predictions_stack[i]
        # Get most common class
        ensemble_preds.append(np.bincount(votes).argmax())
    
    return np.array(ensemble_preds)


def soft_vote_ensemble(
    probabilities_list: List[np.ndarray],
    weights: Optional[List[float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Soft voting ensemble: average probabilities, then predict.
    
    Args:
        probabilities_list: List of probability arrays from different models
        weights: Optional weights for each model
    
    Returns:
        Tuple of (predictions, ensemble_probabilities)
    """
    if weights is None:
        weights = [1.0 / len(probabilities_list)] * len(probabilities_list)
    
    # Weighted average of probabilities
    ensemble_probs = np.zeros_like(probabilities_list[0])
    for probs, weight in zip(probabilities_list, weights):
        ensemble_probs += probs * weight
    
    # Predict from ensemble probabilities
    predictions = np.argmax(ensemble_probs, axis=1)
    
    return predictions, ensemble_probs

