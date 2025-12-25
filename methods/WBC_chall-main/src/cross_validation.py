"""
Cross-validation utilities for WBC Challenge.
Supports stratified K-fold CV to ensure rare classes are represented in each fold.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from src.dataset import WBCDataset


def create_stratified_folds(
    labels_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create stratified K-fold splits ensuring rare classes are in each fold.
    
    Args:
        labels_df: DataFrame with 'ID' and 'labels' columns
        n_splits: Number of folds
        random_state: Random seed for reproducibility
        shuffle: Whether to shuffle data before splitting
    
    Returns:
        List of (train_df, val_df) tuples for each fold
    """
    # Convert labels to indices for stratification
    labels_df = labels_df.copy()
    labels_df['label_idx'] = labels_df['labels'].map(WBCDataset.CLASS_TO_IDX)
    
    # Remove any rows with invalid labels
    labels_df = labels_df.dropna(subset=['label_idx'])
    labels_df['label_idx'] = labels_df['label_idx'].astype(int)
    
    # Get features (IDs) and targets (label indices)
    X = labels_df['ID'].values
    y = labels_df['label_idx'].values
    
    # Create stratified K-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        train_df = labels_df.iloc[train_idx][['ID', 'labels']].copy().reset_index(drop=True)
        val_df = labels_df.iloc[val_idx][['ID', 'labels']].copy().reset_index(drop=True)
        
        folds.append((train_df, val_df))
        
        print(f"Fold {fold_idx + 1}/{n_splits}:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val: {len(val_df)} samples")
        print(f"  Train class distribution:")
        print(train_df['labels'].value_counts().sort_index())
        print()
    
    return folds


def create_group_stratified_folds(
    labels_df: pd.DataFrame,
    group_column: str,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create group-stratified folds (e.g., by patient ID).
    Ensures samples from same group are in same fold.
    
    Args:
        labels_df: DataFrame with 'ID', 'labels', and group column
        group_column: Column name for grouping (e.g., 'patient_id')
        n_splits: Number of folds
        random_state: Random seed
    
    Returns:
        List of (train_df, val_df) tuples for each fold
    """
    from sklearn.model_selection import GroupKFold
    
    labels_df = labels_df.copy()
    labels_df['label_idx'] = labels_df['labels'].map(WBCDataset.CLASS_TO_IDX)
    labels_df = labels_df.dropna(subset=['label_idx'])
    labels_df['label_idx'] = labels_df['label_idx'].astype(int)
    
    X = labels_df['ID'].values
    y = labels_df['label_idx'].values
    groups = labels_df[group_column].values
    
    gkf = GroupKFold(n_splits=n_splits)
    
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        train_df = labels_df.iloc[train_idx][['ID', 'labels']].copy().reset_index(drop=True)
        val_df = labels_df.iloc[val_idx][['ID', 'labels']].copy().reset_index(drop=True)
        
        folds.append((train_df, val_df))
        
        print(f"Fold {fold_idx + 1}/{n_splits}:")
        print(f"  Train: {len(train_df)} samples, {len(np.unique(groups[train_idx]))} groups")
        print(f"  Val: {len(val_df)} samples, {len(np.unique(groups[val_idx]))} groups")
        print()
    
    return folds


def cross_validate_model(
    model_factory,
    train_dfs: List[pd.DataFrame],
    val_dfs: List[pd.DataFrame],
    config,
    n_folds: int = 5
) -> Dict:
    """
    Perform cross-validation training and evaluation.
    
    Args:
        model_factory: Function that creates a model
        train_dfs: List of training DataFrames (one per fold)
        val_dfs: List of validation DataFrames (one per fold)
        config: Configuration object
        n_folds: Number of folds
    
    Returns:
        Dictionary with CV results including:
        - fold_scores: List of scores for each fold
        - mean_score: Mean score across folds
        - std_score: Standard deviation of scores
        - fold_models: List of trained models (one per fold)
    """
    from src.training import Trainer
    from src.dataset import create_data_loaders
    from src.evaluation import calculate_metrics
    
    fold_scores = []
    fold_models = []
    
    for fold_idx in range(n_folds):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print(f"{'='*60}\n")
        
        # Create data loaders for this fold
        train_loader, val_loader, _ = create_data_loaders(
            config,
            phase1_df=train_dfs[fold_idx],
            phase2_train_df=None,  # Using CV folds, not phase2
            phase2_eval_df=val_dfs[fold_idx],
            phase2_test_df=None
        )
        
        # Create and train model
        model = model_factory()
        trainer = Trainer(model, config.config, device=config.device)
        trainer.train(train_loader, val_loader)
        
        # Evaluate on validation set
        val_metrics = trainer.evaluate(val_loader)
        fold_score = val_metrics.get('macro_f1', 0.0)
        
        fold_scores.append(fold_score)
        fold_models.append(trainer.model)
        
        print(f"Fold {fold_idx + 1} validation F1: {fold_score:.4f}")
    
    # Calculate statistics
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    results = {
        'fold_scores': fold_scores,
        'mean_score': mean_score,
        'std_score': std_score,
        'fold_models': fold_models
    }
    
    print(f"\n{'='*60}")
    print(f"Cross-Validation Results")
    print(f"{'='*60}")
    print(f"Mean F1: {mean_score:.4f} ± {std_score:.4f}")
    print(f"Fold scores: {fold_scores}")
    
    return results


def save_cv_predictions(
    fold_predictions: List[np.ndarray],
    image_ids: List[str],
    output_dir: str
):
    """
    Save cross-validation predictions for each fold.
    
    Args:
        fold_predictions: List of prediction arrays (one per fold)
        image_ids: List of image IDs
        output_dir: Directory to save predictions
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for fold_idx, predictions in enumerate(fold_predictions):
        fold_df = pd.DataFrame({
            'ID': image_ids,
            'prediction': predictions,
            'label': [WBCDataset.IDX_TO_CLASS[p] for p in predictions]
        })
        
        output_path = output_dir / f'fold_{fold_idx + 1}_predictions.csv'
        fold_df.to_csv(output_path, index=False)
        print(f"Saved fold {fold_idx + 1} predictions to {output_path}")

