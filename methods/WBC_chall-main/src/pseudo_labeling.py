"""
Pseudo-labeling utilities for WBC Challenge.
Uses high-confidence predictions from test set to augment training data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

from src.dataset import WBCDataset


def generate_pseudo_labels(
    model: nn.Module,
    test_loader,
    device: torch.device,
    confidence_threshold: float = 0.9,
    min_samples_per_class: Optional[Dict[str, int]] = None,
    max_samples_per_class: Optional[Dict[str, int]] = None,
    use_tta: bool = True
) -> pd.DataFrame:
    """
    Generate pseudo-labels from test set using high-confidence predictions.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run inference on
        confidence_threshold: Minimum confidence (probability) to accept a prediction
        min_samples_per_class: Minimum samples to keep per class (for rare classes)
        max_samples_per_class: Maximum samples to keep per class (to balance dataset)
        use_tta: Whether to use test-time augmentation for more robust predictions
    
    Returns:
        DataFrame with columns ['ID', 'labels'] containing pseudo-labeled samples
    """
    model.eval()
    all_image_ids = []
    all_predictions = []
    all_confidences = []
    
    print(f"Generating pseudo-labels with confidence threshold: {confidence_threshold}")
    if use_tta:
        print("Using test-time augmentation for more robust predictions")
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Generating pseudo-labels')
        for batch in pbar:
            images = batch['image'].to(device)
            image_ids = batch['image_id']
            
            if use_tta:
                # Apply TTA for more robust predictions
                preds = []
                
                # Original
                logits = model(images)
                preds.append(F.softmax(logits, dim=1))
                
                # Horizontal flip
                preds.append(F.softmax(model(torch.flip(images, [3])), dim=1))
                
                # Vertical flip
                preds.append(F.softmax(model(torch.flip(images, [2])), dim=1))
                
                # Both flips
                preds.append(F.softmax(model(torch.flip(images, [2, 3])), dim=1))
                
                # Average probabilities
                probs = torch.stack(preds).mean(0)
            else:
                logits = model(images)
                probs = F.softmax(logits, dim=1)
            
            # Get predictions and confidences
            confidences, predictions = torch.max(probs, dim=1)
            
            all_image_ids.extend(image_ids)
            all_predictions.extend(predictions.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    # Create DataFrame
    pseudo_labels_df = pd.DataFrame({
        'ID': all_image_ids,
        'prediction_idx': all_predictions,
        'confidence': all_confidences
    })
    
    # Convert prediction indices to class labels
    pseudo_labels_df['labels'] = pseudo_labels_df['prediction_idx'].map(
        lambda x: WBCDataset.IDX_TO_CLASS[x]
    )
    
    # Filter by confidence threshold
    high_confidence = pseudo_labels_df[pseudo_labels_df['confidence'] >= confidence_threshold].copy()
    
    print(f"\nTotal test samples: {len(pseudo_labels_df)}")
    print(f"High-confidence samples (>= {confidence_threshold}): {len(high_confidence)}")
    print(f"Confidence distribution:")
    print(f"  Mean: {high_confidence['confidence'].mean():.4f}")
    print(f"  Min: {high_confidence['confidence'].min():.4f}")
    print(f"  Max: {high_confidence['confidence'].max():.4f}")
    
    # Apply class-based filtering
    if min_samples_per_class or max_samples_per_class:
        filtered_samples = []
        
        for class_name in WBCDataset.CLASSES:
            class_samples = high_confidence[high_confidence['labels'] == class_name].copy()
            
            # Sort by confidence (descending)
            class_samples = class_samples.sort_values('confidence', ascending=False)
            
            # Apply min/max constraints
            if min_samples_per_class and class_name in min_samples_per_class:
                min_samples = min_samples_per_class[class_name]
                if len(class_samples) < min_samples:
                    print(f"Warning: Class {class_name} has only {len(class_samples)} high-confidence samples, "
                          f"but minimum is {min_samples}")
            
            if max_samples_per_class and class_name in max_samples_per_class:
                max_samples = max_samples_per_class[class_name]
                class_samples = class_samples.head(max_samples)
            
            filtered_samples.append(class_samples)
        
        high_confidence = pd.concat(filtered_samples, ignore_index=True)
        print(f"\nAfter class-based filtering: {len(high_confidence)} samples")
    
    # Show class distribution
    print("\nPseudo-label class distribution:")
    class_dist = high_confidence['labels'].value_counts().sort_index()
    print(class_dist)
    
    # Return only ID and labels columns
    result = high_confidence[['ID', 'labels']].copy()
    
    return result


def merge_pseudo_labels_with_training(
    training_df: pd.DataFrame,
    pseudo_labels_df: pd.DataFrame,
    balance_classes: bool = False,
    balance_percentile: float = 0.95
) -> pd.DataFrame:
    """
    Merge pseudo-labels with training data.
    
    Args:
        training_df: Original training DataFrame with columns ['ID', 'labels']
        pseudo_labels_df: Pseudo-labeled DataFrame with columns ['ID', 'labels']
        balance_classes: If True, balance the dataset by limiting majority classes.
                         Default is False to preserve all training data.
        balance_percentile: Percentile to use as max samples per class when balancing.
                           Default 0.95 (less aggressive than 0.75).
                           Only used if balance_classes=True.
    
    Returns:
        Combined training DataFrame
    """
    # Combine training and pseudo-labels
    combined = pd.concat([training_df, pseudo_labels_df], ignore_index=True)
    
    # Remove duplicates (if same ID appears in both, keep training label)
    combined = combined.drop_duplicates(subset=['ID'], keep='first')
    
    if balance_classes:
        # Balance classes by limiting majority classes
        class_counts = combined['labels'].value_counts()
        max_samples = class_counts.quantile(balance_percentile)  # Use specified percentile as max
        
        print(f"\nBalancing classes with {balance_percentile*100:.0f}th percentile cap: {max_samples:.0f} samples per class")
        
        balanced_samples = []
        for class_name in WBCDataset.CLASSES:
            class_samples = combined[combined['labels'] == class_name].copy()
            
            if len(class_samples) > max_samples:
                # Sample randomly to max_samples
                original_count = len(class_samples)
                class_samples = class_samples.sample(n=int(max_samples), random_state=42)
                print(f"  {class_name}: {original_count} -> {len(class_samples)} samples")
            
            balanced_samples.append(class_samples)
        
        combined = pd.concat(balanced_samples, ignore_index=True)
    else:
        print("\nClass balancing disabled - preserving all training data")
    
    print(f"\nCombined dataset:")
    print(f"  Original training: {len(training_df)} samples")
    print(f"  Pseudo-labels: {len(pseudo_labels_df)} samples")
    print(f"  Combined: {len(combined)} samples")
    print(f"\nFinal class distribution:")
    print(combined['labels'].value_counts().sort_index())
    
    return combined


def iterative_pseudo_labeling(
    model: nn.Module,
    test_loader,
    device: torch.device,
    training_df: pd.DataFrame,
    iterations: int = 3,
    confidence_threshold: float = 0.9,
    confidence_increment: float = 0.05,
    use_tta: bool = True
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    Perform iterative pseudo-labeling: train -> pseudo-label -> retrain.
    
    Args:
        model: Initial trained model
        test_loader: Test data loader
        device: Device to run inference on
        training_df: Original training DataFrame
        iterations: Number of pseudo-labeling iterations
        confidence_threshold: Starting confidence threshold
        confidence_increment: Increase threshold by this amount each iteration
        use_tta: Whether to use TTA for pseudo-labeling
    
    Returns:
        Tuple of (final training DataFrame, list of pseudo-label DataFrames from each iteration)
    """
    current_training_df = training_df.copy()
    all_pseudo_labels = []
    
    print(f"Starting iterative pseudo-labeling with {iterations} iterations")
    print(f"Initial confidence threshold: {confidence_threshold}")
    
    for iteration in range(iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{iterations}")
        print(f"{'='*60}")
        
        # Generate pseudo-labels
        current_threshold = confidence_threshold + (iteration * confidence_increment)
        print(f"Confidence threshold: {current_threshold:.3f}")
        
        pseudo_labels = generate_pseudo_labels(
            model=model,
            test_loader=test_loader,
            device=device,
            confidence_threshold=current_threshold,
            use_tta=use_tta
        )
        
        if len(pseudo_labels) == 0:
            print(f"No high-confidence samples found at threshold {current_threshold}")
            break
        
        all_pseudo_labels.append(pseudo_labels)
        
        # Merge with training data
        current_training_df = merge_pseudo_labels_with_training(
            training_df=current_training_df,
            pseudo_labels_df=pseudo_labels,
            balance_classes=False  # Preserve all data by default
        )
        
        print(f"\nIteration {iteration + 1} complete. Total training samples: {len(current_training_df)}")
    
    return current_training_df, all_pseudo_labels


def save_pseudo_labels(
    pseudo_labels_df: pd.DataFrame,
    output_path: str
):
    """
    Save pseudo-labels to CSV file.
    
    Args:
        pseudo_labels_df: Pseudo-label DataFrame
        output_path: Path to save CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    pseudo_labels_df[['ID', 'labels']].to_csv(output_path, index=False)
    print(f"Pseudo-labels saved to {output_path}")
    print(f"Total samples: {len(pseudo_labels_df)}")

