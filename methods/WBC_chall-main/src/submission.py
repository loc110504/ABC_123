"""
Submission file generation for WBC Challenge.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Dict
import torch.nn.functional as F
from torchvision import transforms

from src.dataset import WBCDataset


def ensure_rare_class_predictions(
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    rare_class_idx: int = 9,  # PLY
    top_k: int = 50,  # Number of samples to force PLY prediction (increased from 20)
    min_ply_prob: float = 0.10  # Minimum probability threshold for PLY (lowered from 0.15, for reporting only)
) -> np.ndarray:
    """
    Ensure rare class (PLY) gets some predictions by replacing
    top-k highest PLY probabilities with PLY predictions.
    
    Args:
        predictions: Current predictions [N]
        probabilities: Predicted probabilities [N, num_classes] (optional)
        rare_class_idx: Index of rare class (default: 9 for PLY)
        top_k: Number of samples to force rare class prediction
        min_ply_prob: Minimum probability threshold for PLY (for reporting only)
    
    Returns:
        Updated predictions with rare class ensured
    """
    if probabilities is None:
        # If no probabilities available, return original predictions
        return predictions
    
    predictions = predictions.copy()
    
    # Get PLY probabilities
    ply_probs = probabilities[:, rare_class_idx]
    
    # Always force top-k predictions to PLY (regardless of threshold)
    # Sort all samples by PLY probability (descending)
    all_sorted_indices = np.argsort(ply_probs)[::-1]
    
    # Find samples above threshold (for reporting only)
    valid_mask = ply_probs >= min_ply_prob
    valid_count = valid_mask.sum()
    
    # Force top-k predictions to PLY (always force top_k, even if below threshold)
    k = min(top_k, len(predictions))
    top_k_indices = all_sorted_indices[:k]
    predictions[top_k_indices] = rare_class_idx
    
    print(f"\nPost-processing: Forced {len(top_k_indices)} predictions to PLY (class {rare_class_idx})")
    print(f"  PLY probability range: {ply_probs[top_k_indices].min():.3f} - {ply_probs[top_k_indices].max():.3f}")
    print(f"  Samples above threshold ({min_ply_prob}): {valid_count}/{len(predictions)}")
    
    return predictions


def apply_rotation_tta(images: torch.Tensor, angles: List[float]) -> List[torch.Tensor]:
    """
    Apply rotation augmentations for TTA.
    
    Args:
        images: Input images tensor [B, C, H, W]
        angles: List of rotation angles in degrees
    
    Returns:
        List of rotated image tensors
    """
    rotated_images = []
    for angle in angles:
        # Rotate images
        rotated = transforms.functional.rotate(images, angle, interpolation=transforms.InterpolationMode.BILINEAR)
        rotated_images.append(rotated)
    return rotated_images


def apply_color_tta(images: torch.Tensor, brightness_range: tuple = (0.9, 1.1), 
                    contrast_range: tuple = (0.9, 1.1)) -> List[torch.Tensor]:
    """
    Apply color augmentations for TTA.
    
    Args:
        images: Input images tensor [B, C, H, W]
        brightness_range: Tuple of (min, max) brightness multipliers
        contrast_range: Tuple of (min, max) contrast multipliers
    
    Returns:
        List of color-augmented image tensors
    """
    augmented_images = []
    
    # Original
    augmented_images.append(images)
    
    # Brightness variations
    for brightness in [brightness_range[0], brightness_range[1]]:
        bright = transforms.functional.adjust_brightness(images, brightness)
        augmented_images.append(bright)
    
    # Contrast variations
    for contrast in [contrast_range[0], contrast_range[1]]:
        cont = transforms.functional.adjust_contrast(images, contrast)
        augmented_images.append(cont)
    
    return augmented_images


def apply_multi_scale_tta(images: torch.Tensor, scales: List[int], 
                         model: torch.nn.Module, device: torch.device) -> List[torch.Tensor]:
    """
    Apply multi-scale TTA by resizing images to different scales.
    Note: This requires the model to handle different input sizes or we need to resize back.
    
    Args:
        images: Input images tensor [B, C, H, W]
        scales: List of target sizes (assumes square images)
        model: Model to run inference on
        device: Device to run inference on
    
    Returns:
        List of predictions at different scales
    """
    predictions = []
    original_size = images.shape[-1]  # Assume square images
    
    for scale in scales:
        if scale == original_size:
            # No resize needed
            pred = model(images)
        else:
            # Resize to target scale
            resized = transforms.functional.resize(images, size=scale, 
                                                  interpolation=transforms.InterpolationMode.BILINEAR)
            pred = model(resized)
        predictions.append(pred)
    
    return predictions


def generate_submission(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    output_path: str = 'submission.csv',
    use_tta: bool = False,
    tta_config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Generate submission file from test predictions with enhanced TTA.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run inference on
        output_path: Path to save submission file
        use_tta: Use test-time augmentation
        tta_config: Optional TTA configuration dict with keys:
            - flips: bool (default: True)
            - multi_scale: dict with 'enable' and 'scales' (default: False)
            - rotation: dict with 'enable' and 'angles' (default: False)
            - color: dict with 'enable', 'brightness_range', 'contrast_range' (default: False)
    
    Returns:
        Submission DataFrame
    """
    model.eval()
    all_image_ids = []
    all_predictions = []
    
    # Default TTA config
    if tta_config is None:
        tta_config = {
            'flips': True,
            'multi_scale': {'enable': False},
            'rotation': {'enable': False},
            'color': {'enable': False}
        }
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Generating predictions')
        for batch in pbar:
            images = batch['image'].to(device)
            image_ids = batch['image_id']
            
            if use_tta:
                # Enhanced Test-Time Augmentation: multiple augmentations for robustness
                preds = []
                
                # Basic flips (always enabled if TTA is on)
                if tta_config.get('flips', True):
                    # Original
                    preds.append(model(images))
                    
                    # Horizontal flip
                    preds.append(model(torch.flip(images, [3])))
                    
                    # Vertical flip
                    preds.append(model(torch.flip(images, [2])))
                    
                    # Both flips
                    preds.append(model(torch.flip(images, [2, 3])))
                
                # Multi-scale TTA
                if tta_config.get('multi_scale', {}).get('enable', False):
                    scales = tta_config['multi_scale'].get('scales', [224, 256, 288])
                    scale_preds = apply_multi_scale_tta(images, scales, model, device)
                    preds.extend(scale_preds)
                
                # Rotation TTA
                if tta_config.get('rotation', {}).get('enable', False):
                    angles = tta_config['rotation'].get('angles', [-10, -5, 5, 10])
                    rotated_images = apply_rotation_tta(images, angles)
                    for rotated in rotated_images:
                        preds.append(model(rotated))
                
                # Color TTA
                if tta_config.get('color', {}).get('enable', False):
                    brightness_range = tta_config['color'].get('brightness_range', (0.9, 1.1))
                    contrast_range = tta_config['color'].get('contrast_range', (0.9, 1.1))
                    color_images = apply_color_tta(images, brightness_range, contrast_range)
                    for color_img in color_images:
                        preds.append(model(color_img))
                
                # Average predictions (convert to probabilities first for better averaging)
                if len(preds) > 0:
                    probs = [F.softmax(pred, dim=1) for pred in preds]
                    ensemble_probs = torch.stack(probs).mean(0)
                    outputs = ensemble_probs
                else:
                    outputs = model(images)
            else:
                outputs = model(images)
            
            # Get predictions and probabilities
            if outputs.dim() == 2 and outputs.shape[1] == 13:
                # Already probabilities
                probs = outputs
                preds = torch.argmax(probs, dim=1)
            else:
                # Logits, convert to probabilities
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
            
            # Store predictions and probabilities for post-processing
            all_image_ids.extend(image_ids)
            all_predictions.extend(preds.cpu().numpy())
            
            # Store probabilities for rare class post-processing
            if not hasattr(generate_submission, '_all_probs'):
                generate_submission._all_probs = []
            generate_submission._all_probs.append(probs.cpu().numpy())
    
    # Post-process predictions to ensure rare classes (especially PLY) get some predictions
    all_predictions = ensure_rare_class_predictions(
        np.array(all_predictions),
        np.concatenate(generate_submission._all_probs, axis=0) if hasattr(generate_submission, '_all_probs') and len(generate_submission._all_probs) > 0 else None
    )
    
    # Clean up stored probabilities
    if hasattr(generate_submission, '_all_probs'):
        del generate_submission._all_probs
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'ID': all_image_ids,
        'labels': [WBCDataset.IDX_TO_CLASS[p] for p in all_predictions]
    })
    
    # Validate submission
    validate_submission(submission)
    
    # Save submission
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission file saved to {output_path}")
    print(f"Total predictions: {len(submission)}")
    print(f"Class distribution:")
    print(submission['labels'].value_counts().sort_index())
    
    if use_tta:
        print(f"\nTTA enabled with {len(preds) if use_tta else 0} augmentations per image")
    
    return submission


def ensure_rare_class_predictions(
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    rare_class_idx: int = 9,  # PLY
    top_k: int = 50,  # Number of samples to force PLY prediction (increased from 20)
    min_ply_prob: float = 0.10  # Minimum probability threshold for PLY (lowered from 0.15, for reporting only)
) -> np.ndarray:
    """
    Ensure rare class (PLY) gets some predictions by replacing
    top-k highest PLY probabilities with PLY predictions.
    
    Args:
        predictions: Current predictions [N]
        probabilities: Predicted probabilities [N, num_classes] (optional)
        rare_class_idx: Index of rare class (default: 9 for PLY)
        top_k: Number of samples to force rare class prediction
        min_ply_prob: Minimum probability threshold for PLY
    
    Returns:
        Updated predictions with rare class ensured
    """
    if probabilities is None:
        # If no probabilities available, return original predictions
        return predictions
    
    predictions = predictions.copy()
    
    # Get PLY probabilities
    ply_probs = probabilities[:, rare_class_idx]
    
    # Find samples with highest PLY probability
    # Only consider samples where PLY prob is above threshold
    valid_mask = ply_probs >= min_ply_prob
    
    if valid_mask.sum() > 0:
        # Get top-k samples with highest PLY probability
        valid_indices = np.where(valid_mask)[0]
        valid_probs = ply_probs[valid_indices]
        
        # Sort by PLY probability (descending)
        sorted_indices = valid_indices[np.argsort(valid_probs)[::-1]]
        
        # Force top-k predictions to PLY
        k = min(top_k, len(sorted_indices))
        top_k_indices = sorted_indices[:k]
        predictions[top_k_indices] = rare_class_idx
        
        print(f"\nPost-processing: Forced {k} predictions to PLY (class {rare_class_idx})")
        print(f"  PLY probability range: {ply_probs[top_k_indices].min():.3f} - {ply_probs[top_k_indices].max():.3f}")
    else:
        # If no samples meet threshold, use top-k by probability anyway
        top_k_indices = np.argsort(ply_probs)[-top_k:]
        predictions[top_k_indices] = rare_class_idx
        print(f"\nPost-processing: Forced {top_k} predictions to PLY (class {rare_class_idx})")
        print(f"  Warning: No samples met probability threshold {min_ply_prob}, using top-k anyway")
    
    return predictions


def validate_submission(submission: pd.DataFrame):
    """
    Validate submission file format.
    
    Args:
        submission: Submission DataFrame
    
    Raises:
        ValueError: If submission format is invalid
    """
    # Check columns
    required_columns = ['ID', 'labels']
    if list(submission.columns) != required_columns:
        raise ValueError(f"Submission must have columns {required_columns}, got {list(submission.columns)}")
    
    # Check row count
    expected_rows = 16477
    if len(submission) != expected_rows:
        raise ValueError(f"Submission must have {expected_rows} rows, got {len(submission)}")
    
    # Check for missing values
    if submission['labels'].isna().any():
        raise ValueError("Submission contains missing labels")
    
    if submission['ID'].isna().any():
        raise ValueError("Submission contains missing IDs")
    
    # Check valid label codes
    valid_labels = set(WBCDataset.CLASSES)
    invalid_labels = set(submission['labels'].unique()) - valid_labels
    if invalid_labels:
        raise ValueError(f"Invalid label codes found: {invalid_labels}")
    
    # Check ID format (8 digits + .jpg)
    import re
    id_pattern = r'^\d{8}\.jpg$'
    invalid_ids = submission[~submission['ID'].str.match(id_pattern)]
    if len(invalid_ids) > 0:
        raise ValueError(f"Invalid ID format found: {invalid_ids['ID'].head().tolist()}")
    
    print("✓ Submission file is valid!")


def load_model_for_inference(
    checkpoint_path: str,
    model: torch.nn.Module,
    device: torch.device
) -> torch.nn.Module:
    """
    Load model from checkpoint for inference.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

