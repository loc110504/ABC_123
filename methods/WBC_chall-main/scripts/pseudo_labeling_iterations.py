#!/usr/bin/env python3
"""
Pseudo-Labeling Iterations Script for WBC Challenge

This script runs multiple iterations of pseudo-labeling for any model.
It automatically:
1. Loads the model from previous iteration
2. Generates pseudo-labels from test set
3. Merges with previous expanded dataset
4. Retrains the model
5. Saves checkpoints and submissions

SPLIT METHODS:
==============

NEW SPLIT METHOD (Default, use_newsplit=True):
- Merges: phase1 + phase2_train + phase2_eval → one dataset
- Creates stratified 90/10 train/val split (configurable via --val_split)
- More training data available (includes phase2_eval in training)
- Consistent with new training script (scripts/train_model.py)
- Files saved with '_newsplit' suffix:
  * pseudo_labels_{model}_iter{N}_newsplit.csv
  * expanded_training_{model}_iter{N}_newsplit.csv
  * submission_pseudo_labeling_{model}_iter{N}_newsplit.csv
  * checkpoints/pseudo_labeling_{model}_iter{N}_newsplit/

OLD SPLIT METHOD (Default):
- Training: phase1 + phase2_train (phase2_eval excluded from training)
- Validation: phase2_eval (fixed, 5,350 samples)
- Total training samples: ~33,185
- Files saved without suffix:
  * pseudo_labels_{model}_iter{N}.csv
  * expanded_training_{model}_iter{N}.csv
  * submission_pseudo_labeling_{model}_iter{N}.csv
  * checkpoints/pseudo_labeling_{model}_iter{N}/

NEW SPLIT METHOD (use --use_new_split flag):
- Merges: phase1 + phase2_train + phase2_eval → one dataset
- Creates stratified 90/10 train/val split (configurable via --val_split)
- More training data available (includes phase2_eval in training)
- Files saved with '_newsplit' suffix

Usage Examples:
    # Old split method (default)
    python scripts/pseudo_labeling_iterations.py --model resnet50 --iterations 2 --start_iter 2
    
    # New split method
    python scripts/pseudo_labeling_iterations.py --model resnet50 --use_new_split --iterations 2 --start_iter 2
    
    # Custom validation split ratio (with new split)
    python scripts/pseudo_labeling_iterations.py --model swin_tiny --use_new_split --val_split 0.15 --seed 123
"""

import sys
import os
import argparse
from pathlib import Path
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.config import Config
from src.dataset import (
    create_data_loaders,
    load_label_files,
    merge_all_training_data,
    create_stratified_split,
    CombinedWBCDataset,
    WBCDataset,
    get_train_transform,
    get_val_transform
)
from src.models import create_model
from src.training import Trainer
from src.pseudo_labeling import (
    generate_pseudo_labels,
    merge_pseudo_labels_with_training,
    save_pseudo_labels
)
from src.submission import generate_submission
from torch.utils.data import DataLoader, WeightedRandomSampler


def get_model_config_name(model_name: str) -> str:
    """Map model name to config file name."""
    model_config_map = {
        'resnet50': 'resnet50',
        'swin_tiny': 'swin_tiny',
        'convnext_tiny': 'convnext_tiny',
        'efficientnet_b3': 'efficientnet_b3',
        'efficientnet_b4': 'efficientnet_b4',
        'resnext50': 'resnext50',
        'resnext101': 'resnext101',
        'efficientnetv2': 'efficientnetv2',
        'regnet': 'regnet',
        'deit': 'deit',
        'vit': 'vit',
        'maxvit': 'maxvit',
        'coatnet': 'coatnet',
    }
    return model_config_map.get(model_name.lower(), model_name.lower())


def load_model_from_checkpoint(
    config: Config,
    checkpoint_path: Path,
    device: torch.device
) -> nn.Module:
    """Load model from checkpoint."""
    model = create_model(config.config, num_classes=13)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    val_score = checkpoint.get('best_val_score', 'N/A')
    epoch = checkpoint.get('epoch', 'N/A')
    print(f"  ✓ Loaded model from checkpoint")
    print(f"    Best validation F1: {val_score:.4f}" if isinstance(val_score, (int, float)) else f"    Best validation F1: {val_score}")
    print(f"    Best epoch: {epoch + 1}" if isinstance(epoch, int) else f"    Best epoch: {epoch}")
    
    return model


def create_expanded_data_loaders(
    config: Config,
    expanded_training_df: pd.DataFrame,
    original_training_df: pd.DataFrame,
    phase1_df: pd.DataFrame,
    phase2_eval_df: pd.DataFrame,
    phase2_test_df: pd.DataFrame
) -> tuple:
    """
    Create data loaders from expanded training dataset (OLD METHOD - uses phase2_eval as validation).
    
    OLD SPLIT METHOD:
    - Training: phase1 + phase2_train (original_training_df)
    - Validation: phase2_eval (fixed, not split)
    - This method is preserved for backward compatibility.
    """
    # Split expanded training data into original training IDs and pseudo-label IDs
    original_training_ids = set(original_training_df['ID'].values)
    pseudo_label_ids = set(expanded_training_df['ID'].values) - original_training_ids
    
    # Split the expanded DataFrame
    expanded_original_df = expanded_training_df[expanded_training_df['ID'].isin(original_training_ids)].copy()
    expanded_pseudo_df = expanded_training_df[expanded_training_df['ID'].isin(pseudo_label_ids)].copy()
    
    # Get directory paths
    phase1_dir = config.get('data.phase1_dir')
    phase2_train_dir = config.get('data.phase2_train_dir')
    phase2_eval_dir = config.get('data.phase2_eval_dir')
    phase2_test_dir = config.get('data.phase2_test_dir')
    
    # Further split original training data into phase1 and phase2_train
    if phase1_df is not None and len(phase1_df) > 0:
        phase1_ids = set(phase1_df['ID'].values)
        expanded_phase1_df = expanded_original_df[expanded_original_df['ID'].isin(phase1_ids)].copy()
        expanded_phase2_train_df = expanded_original_df[~expanded_original_df['ID'].isin(phase1_ids)].copy()
    else:
        expanded_phase1_df = pd.DataFrame()
        expanded_phase2_train_df = expanded_original_df.copy()
    
    # Create image directories and DataFrames lists for CombinedWBCDataset
    train_image_dirs = []
    train_dfs = []
    
    if len(expanded_phase1_df) > 0:
        train_image_dirs.append(phase1_dir)
        train_dfs.append(expanded_phase1_df)
    
    if len(expanded_phase2_train_df) > 0:
        train_image_dirs.append(phase2_train_dir)
        train_dfs.append(expanded_phase2_train_df)
    
    # Add pseudo-labels from test directory
    if len(expanded_pseudo_df) > 0:
        train_image_dirs.append(phase2_test_dir)
        train_dfs.append(expanded_pseudo_df)
    
    # Create dataset
    rare_class_boost = config.config.get('augmentation', {}).get('train', {}).get('rare_class_boost', False)
    
    train_dataset_expanded = CombinedWBCDataset(
        image_dirs=train_image_dirs,
        labels_dfs=train_dfs,
        transform=get_train_transform(config.config, aggressive=False),
        mode='train',
        rare_class_boost=rare_class_boost
    )
    
    val_dataset = WBCDataset(
        image_dir=phase2_eval_dir,
        labels_df=phase2_eval_df,
        transform=get_val_transform(config.config),
        mode='val'
    ) if phase2_eval_df is not None and len(phase2_eval_df) > 0 else None
    
    test_dataset = WBCDataset(
        image_dir=phase2_test_dir,
        labels_df=phase2_test_df,
        transform=get_val_transform(config.config),
        mode='test'
    ) if phase2_test_df is not None and len(phase2_test_df) > 0 else None
    
    # Create data loaders with weighted sampling
    train_config = config.config.get('training', {})
    batch_size = train_config.get('batch_size', 32)
    num_workers = train_config.get('num_workers', 4)
    pin_memory = train_config.get('pin_memory', True)
    
    use_class_weights = train_config.get('use_class_weights', True)
    if use_class_weights:
        class_weights = train_dataset_expanded.get_class_weights()
        sample_weights = class_weights[train_dataset_expanded.labels_df['label_idx'].values]
        
        # Get minimum samples per class per epoch
        min_class_samples = train_config.get('min_class_samples_per_epoch', None)
        if min_class_samples is not None and min_class_samples > 0:
            class_counts = train_dataset_expanded.labels_df['label_idx'].value_counts().sort_index()
            num_classes = len(WBCDataset.CLASSES)
            rare_class_indices = []
            for class_idx in range(num_classes):
                current_class_count = class_counts.get(class_idx, 0)
                if current_class_count < min_class_samples:
                    rare_class_indices.append(class_idx)
            
            boost_factor = 5.0
            for idx, class_idx in enumerate(train_dataset_expanded.labels_df['label_idx'].values):
                if class_idx in rare_class_indices:
                    sample_weights[idx] *= boost_factor
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset_expanded),
            replacement=True
        )
        train_loader_expanded = DataLoader(
            train_dataset_expanded,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        train_loader_expanded = DataLoader(
            train_dataset_expanded,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    ) if val_dataset is not None else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    ) if test_dataset is not None else None
    
    return train_loader_expanded, val_loader, test_loader


def create_expanded_data_loaders_newsplit(
    config: Config,
    expanded_training_df: pd.DataFrame,
    merged_train_df: pd.DataFrame,
    merged_val_df: pd.DataFrame,
    phase2_test_df: pd.DataFrame,
    random_state: int = 42
) -> tuple:
    """
    Create data loaders from expanded training dataset using NEW MERGED SPLIT METHOD.
    
    NEW SPLIT METHOD:
    - Merges: phase1 + phase2_train + phase2_eval into one dataset
    - Creates stratified 90/10 train/val split
    - More training data available (includes phase2_eval in training)
    - Consistent validation approach with new training script
    
    Args:
        config: Configuration object
        expanded_training_df: Expanded training DataFrame (includes pseudo-labels)
        merged_train_df: Original merged training split (90% of phase1+phase2_train+phase2_eval)
        merged_val_df: Original merged validation split (10% of phase1+phase2_train+phase2_eval)
        phase2_test_df: Test labels DataFrame
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Split expanded training data into original training IDs and pseudo-label IDs
    merged_train_ids = set(merged_train_df['ID'].values)
    merged_val_ids = set(merged_val_df['ID'].values)
    pseudo_label_ids = set(expanded_training_df['ID'].values) - merged_train_ids - merged_val_ids
    
    # Split the expanded DataFrame
    expanded_original_train_df = expanded_training_df[expanded_training_df['ID'].isin(merged_train_ids)].copy()
    expanded_original_val_df = expanded_training_df[expanded_training_df['ID'].isin(merged_val_ids)].copy()
    expanded_pseudo_df = expanded_training_df[expanded_training_df['ID'].isin(pseudo_label_ids)].copy()
    
    # Get directory paths
    phase1_dir = config.get('data.phase1_dir')
    phase2_train_dir = config.get('data.phase2_train_dir')
    phase2_eval_dir = config.get('data.phase2_eval_dir')
    phase2_test_dir = config.get('data.phase2_test_dir')
    
    # Group by source_dir to determine which directory each sample comes from
    # merged_train_df and merged_val_df have 'source_dir' column from merge_all_training_data
    train_image_dirs = []
    train_dfs = []
    
    # Process training data by source directory
    phase1_train = expanded_original_train_df[expanded_original_train_df['ID'].isin(
        merged_train_df[merged_train_df['source_dir'] == 'phase1']['ID'].values
    )].copy() if 'source_dir' in merged_train_df.columns else pd.DataFrame()
    
    phase2_train_split = expanded_original_train_df[expanded_original_train_df['ID'].isin(
        merged_train_df[merged_train_df['source_dir'] == 'phase2_train']['ID'].values
    )].copy() if 'source_dir' in merged_train_df.columns else pd.DataFrame()
    
    phase2_eval_split = expanded_original_train_df[expanded_original_train_df['ID'].isin(
        merged_train_df[merged_train_df['source_dir'] == 'phase2_eval']['ID'].values
    )].copy() if 'source_dir' in merged_train_df.columns else pd.DataFrame()
    
    # If source_dir not available, try to infer from merged_train_df
    if len(phase1_train) == 0 and len(phase2_train_split) == 0 and len(phase2_eval_split) == 0:
        # Fallback: try to match with original data structure
        phase1_df = pd.read_csv(config.get('data.phase1_labels'))
        phase2_train_df = pd.read_csv(config.get('data.phase2_train_labels'))
        phase2_eval_df = pd.read_csv(config.get('data.phase2_eval_labels'))
        
        phase1_ids = set(phase1_df['ID'].values)
        phase2_train_ids = set(phase2_train_df['ID'].values)
        phase2_eval_ids = set(phase2_eval_df['ID'].values)
        
        phase1_train = expanded_original_train_df[expanded_original_train_df['ID'].isin(phase1_ids)].copy()
        phase2_train_split = expanded_original_train_df[expanded_original_train_df['ID'].isin(phase2_train_ids)].copy()
        phase2_eval_split = expanded_original_train_df[expanded_original_train_df['ID'].isin(phase2_eval_ids)].copy()
    
    if len(phase1_train) > 0:
        train_image_dirs.append(phase1_dir)
        train_dfs.append(phase1_train[['ID', 'labels']])
    
    if len(phase2_train_split) > 0:
        train_image_dirs.append(phase2_train_dir)
        train_dfs.append(phase2_train_split[['ID', 'labels']])
    
    if len(phase2_eval_split) > 0:
        train_image_dirs.append(phase2_eval_dir)
        train_dfs.append(phase2_eval_split[['ID', 'labels']])
    
    # Add pseudo-labels from test directory
    if len(expanded_pseudo_df) > 0:
        train_image_dirs.append(phase2_test_dir)
        train_dfs.append(expanded_pseudo_df[['ID', 'labels']])
    
    # Create training dataset
    rare_class_boost = config.config.get('augmentation', {}).get('train', {}).get('rare_class_boost', False)
    
    train_dataset_expanded = CombinedWBCDataset(
        image_dirs=train_image_dirs,
        labels_dfs=train_dfs,
        transform=get_train_transform(config.config, aggressive=False),
        mode='train',
        rare_class_boost=rare_class_boost
    )
    
    # Create validation dataset from merged_val_df (which may include samples from phase1, phase2_train, phase2_eval)
    val_image_dirs = []
    val_dfs = []
    
    # Group validation data by source directory
    if 'source_dir' in merged_val_df.columns:
        phase1_val = expanded_original_val_df[expanded_original_val_df['ID'].isin(
            merged_val_df[merged_val_df['source_dir'] == 'phase1']['ID'].values
        )].copy()
        phase2_train_val = expanded_original_val_df[expanded_original_val_df['ID'].isin(
            merged_val_df[merged_val_df['source_dir'] == 'phase2_train']['ID'].values
        )].copy()
        phase2_eval_val = expanded_original_val_df[expanded_original_val_df['ID'].isin(
            merged_val_df[merged_val_df['source_dir'] == 'phase2_eval']['ID'].values
        )].copy()
    else:
        # Fallback: infer from original data
        phase1_df = pd.read_csv(config.get('data.phase1_labels'))
        phase2_train_df = pd.read_csv(config.get('data.phase2_train_labels'))
        phase2_eval_df = pd.read_csv(config.get('data.phase2_eval_labels'))
        
        phase1_ids = set(phase1_df['ID'].values)
        phase2_train_ids = set(phase2_train_df['ID'].values)
        phase2_eval_ids = set(phase2_eval_df['ID'].values)
        
        phase1_val = expanded_original_val_df[expanded_original_val_df['ID'].isin(phase1_ids)].copy()
        phase2_train_val = expanded_original_val_df[expanded_original_val_df['ID'].isin(phase2_train_ids)].copy()
        phase2_eval_val = expanded_original_val_df[expanded_original_val_df['ID'].isin(phase2_eval_ids)].copy()
    
    if len(phase1_val) > 0:
        val_image_dirs.append(phase1_dir)
        val_dfs.append(phase1_val[['ID', 'labels']])
    
    if len(phase2_train_val) > 0:
        val_image_dirs.append(phase2_train_dir)
        val_dfs.append(phase2_train_val[['ID', 'labels']])
    
    if len(phase2_eval_val) > 0:
        val_image_dirs.append(phase2_eval_dir)
        val_dfs.append(phase2_eval_val[['ID', 'labels']])
    
    # Create validation dataset (can use CombinedWBCDataset if multiple directories)
    if len(val_dfs) > 1:
        val_dataset = CombinedWBCDataset(
            image_dirs=val_image_dirs,
            labels_dfs=val_dfs,
            transform=get_val_transform(config.config),
            mode='val'
        )
    elif len(val_dfs) == 1:
        val_dataset = WBCDataset(
            image_dir=val_image_dirs[0],
            labels_df=val_dfs[0],
            transform=get_val_transform(config.config),
            mode='val'
        )
    else:
        val_dataset = None
    
    # Create test dataset
    test_dataset = WBCDataset(
        image_dir=phase2_test_dir,
        labels_df=phase2_test_df,
        transform=get_val_transform(config.config),
        mode='test'
    ) if phase2_test_df is not None and len(phase2_test_df) > 0 else None
    
    # Create data loaders with weighted sampling
    train_config = config.config.get('training', {})
    batch_size = train_config.get('batch_size', 32)
    num_workers = train_config.get('num_workers', 4)
    pin_memory = train_config.get('pin_memory', True)
    
    use_class_weights = train_config.get('use_class_weights', True)
    if use_class_weights:
        class_weights = train_dataset_expanded.get_class_weights()
        sample_weights = class_weights[train_dataset_expanded.labels_df['label_idx'].values]
        
        # Get minimum samples per class per epoch
        min_class_samples = train_config.get('min_class_samples_per_epoch', None)
        if min_class_samples is not None and min_class_samples > 0:
            class_counts = train_dataset_expanded.labels_df['label_idx'].value_counts().sort_index()
            num_classes = len(WBCDataset.CLASSES)
            rare_class_indices = []
            for class_idx in range(num_classes):
                current_class_count = class_counts.get(class_idx, 0)
                if current_class_count < min_class_samples:
                    rare_class_indices.append(class_idx)
            
            boost_factor = 5.0
            for idx, class_idx in enumerate(train_dataset_expanded.labels_df['label_idx'].values):
                if class_idx in rare_class_indices:
                    sample_weights[idx] *= boost_factor
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset_expanded),
            replacement=True
        )
        train_loader_expanded = DataLoader(
            train_dataset_expanded,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        train_loader_expanded = DataLoader(
            train_dataset_expanded,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    ) if val_dataset is not None else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    ) if test_dataset is not None else None
    
    return train_loader_expanded, val_loader, test_loader


def run_pseudo_labeling_iteration(
    model_name: str,
    iteration: int,
    project_root: Path,
    confidence_threshold: float = 0.9,
    use_tta: bool = True,
    use_newsplit: bool = False,  # Default to old split method
    val_split_ratio: float = 0.1,
    random_state: int = 42
) -> dict:
    """
    Run a single pseudo-labeling iteration.
    
    Args:
        model_name: Model name (e.g., 'resnet50', 'swin_tiny')
        iteration: Current iteration number (e.g., 2, 3, 4)
        project_root: Project root directory
        confidence_threshold: Confidence threshold for pseudo-labels
        use_tta: Whether to use TTA for pseudo-label generation
        use_newsplit: If True, use new merged split method
                     If False, use old method (phase1+phase2_train for training, phase2_eval for validation) (default: False)
        val_split_ratio: Validation split ratio for new split method (default: 0.1 for 90/10)
        random_state: Random seed for stratified split (default: 42)
    
    Returns:
        Dictionary with results and paths
    """
    previous_iteration = iteration - 1
    
    print(f"\n{'='*70}")
    print(f"Running Iteration {iteration} for {model_name.upper()}")
    print(f"{'='*70}")
    print(f"Split Method: {'NEW (Merged Split)' if use_newsplit else 'OLD (Fixed phase2_eval)'}")
    print(f"{'='*70}")
    
    # Load configuration
    config_name = get_model_config_name(model_name)
    os.environ['WBC_MODEL_CONFIG'] = config_name
    config = Config(project_root=str(project_root))
    
    print(f"Model: {model_name}")
    print(f"Config: {config_name}")
    print(f"Device: {config.device}")
    
    # Load data
    print("\n[1/5] Loading data...")
    phase1_df, phase2_train_df, phase2_eval_df, phase2_test_df = load_label_files(config)
    
    if use_newsplit:
        # NEW METHOD: Merge all training data and create stratified split
        print("  Using NEW merged split method...")
        merged_df = merge_all_training_data(config)
        merged_train_df, merged_val_df = create_stratified_split(
            merged_df,
            val_ratio=val_split_ratio,
            random_state=random_state
        )
        # Store original merged splits for later use (keep source_dir column)
        original_merged_train_df = merged_train_df.copy()
        original_merged_val_df = merged_val_df.copy()
        # For compatibility, create original_training_df from merged_train_df
        original_training_df = merged_train_df[['ID', 'labels']].copy()
    else:
        # OLD METHOD: Use phase1 + phase2_train for training, phase2_eval for validation
        print("  Using OLD split method (phase1+phase2_train / phase2_eval)...")
        original_training_df = pd.concat([phase1_df, phase2_train_df], ignore_index=True)
        merged_train_df = None
        merged_val_df = None
        original_merged_train_df = None
        original_merged_val_df = None
    
    # Use '_newsplit' suffix for checkpoints if using new split method
    checkpoint_suffix = '_newsplit' if use_newsplit else ''
    file_suffix = '_newsplit' if use_newsplit else ''
    
    # Handle iteration 0 (initial training from scratch)
    if iteration == 0:
        print("\n[Iteration 0] Initial training from scratch (no pseudo-labels yet)...")
        
        # Create data loaders for initial training
        if use_newsplit:
            train_loader, val_loader, test_loader = create_data_loaders(
                config,
                use_merged_split=True,
                merged_train_df=merged_train_df,
                merged_val_df=merged_val_df,
                phase2_test_df=phase2_test_df
            )
        else:
            train_loader, val_loader, test_loader = create_data_loaders(
                config,
                phase1_df=phase1_df,
                phase2_train_df=phase2_train_df,
                phase2_eval_df=phase2_eval_df,
                phase2_test_df=phase2_test_df
            )
        
        # Create model from scratch
        model = create_model(config.config, num_classes=13)
        
        # Get class weights
        train_dataset = train_loader.dataset
        class_weights = train_dataset.get_class_weights() if hasattr(train_dataset, 'get_class_weights') else None
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config.config,
            device=config.device,
            class_weights=class_weights
        )
        
        # Train model
        num_epochs = config.config.get('training', {}).get('num_epochs', 50)
        print(f"\n{'='*70}")
        print(f"Training initial model (Iteration 0)...")
        print(f"{'='*70}")
        trainer.train(num_epochs=num_epochs)
        
        # Save checkpoint
        current_checkpoint_dir = project_root / 'outputs' / 'checkpoints' / f'pseudo_labeling_{model_name}_iter0{checkpoint_suffix}'
        current_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_checkpoint_source = trainer.checkpoint_dir / 'best.pth'
        
        if best_checkpoint_source.exists():
            shutil.copy(best_checkpoint_source, current_checkpoint_dir / 'best.pth')
            print(f"\n✓ Model saved to {current_checkpoint_dir / 'best.pth'}")
            print(f"  Best validation F1: {trainer.best_val_score:.4f} at epoch {trainer.best_epoch + 1}")
        else:
            print(f"⚠ Warning: Best checkpoint not found at {best_checkpoint_source}")
        
        # Save initial training data (for reference)
        if use_newsplit:
            initial_training_df = merged_train_df[['ID', 'labels']].copy()
        else:
            initial_training_df = original_training_df
        initial_training_path = project_root / 'outputs' / f'expanded_training_{model_name}_iter0{file_suffix}.csv'
        initial_training_df.to_csv(initial_training_path, index=False)
        print(f"  ✓ Saved initial training data: {len(initial_training_df)} samples")
        
        # Generate submission
        print(f"\n[6/6] Generating submission file...")
        submission_path = project_root / 'outputs' / f'submission_pseudo_labeling_{model_name}_iter0{file_suffix}.csv'
        
        submission = generate_submission(
            model=trainer.model,
            test_loader=test_loader,
            device=config.device,
            output_path=str(submission_path),
            use_tta=True
        )
        
        print(f"✓ Submission saved to {submission_path}")
        
        return {
            'iteration': 0,
            'validation_f1': trainer.best_val_score,
            'improvement': 0.0,  # No previous iteration to compare
            'checkpoint_path': current_checkpoint_dir / 'best.pth',
            'submission_path': submission_path,
            'pseudo_labels_path': None,  # No pseudo-labels for iteration 0
            'expanded_training_path': initial_training_path
        }
    
    # For iteration >= 1: Load model from previous iteration and generate pseudo-labels
    # Create test loader for pseudo-label generation
    if use_newsplit:
        # Use merged split for test loader creation
        _, _, test_loader = create_data_loaders(
            config,
            use_merged_split=True,
            merged_train_df=merged_train_df,
            merged_val_df=merged_val_df,
            phase2_test_df=phase2_test_df
        )
    else:
        # Use old method
        _, _, test_loader = create_data_loaders(
            config,
            phase1_df=phase1_df,
            phase2_train_df=phase2_train_df,
            phase2_eval_df=phase2_eval_df,
            phase2_test_df=phase2_test_df
        )
    
    # Load model from previous iteration
    print(f"\n[2/5] Loading model from Iteration {previous_iteration}...")
    prev_checkpoint_path = project_root / 'outputs' / 'checkpoints' / f'pseudo_labeling_{model_name}_iter{previous_iteration}{checkpoint_suffix}' / 'best.pth'
    
    if not prev_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint from Iteration {previous_iteration} not found at {prev_checkpoint_path}\n"
            f"Please ensure Iteration {previous_iteration} has been completed first."
        )
    
    model = load_model_from_checkpoint(config, prev_checkpoint_path, config.device)
    
    # Generate pseudo-labels
    print(f"\n[3/5] Generating pseudo-labels (confidence >= {confidence_threshold})...")
    pseudo_labels = generate_pseudo_labels(
        model=model,
        test_loader=test_loader,
        device=config.device,
        confidence_threshold=confidence_threshold,
        use_tta=use_tta
    )
    
    # Save pseudo-labels
    pseudo_labels_path = project_root / 'outputs' / f'pseudo_labels_{model_name}_iter{iteration}{file_suffix}.csv'
    save_pseudo_labels(pseudo_labels, str(pseudo_labels_path))
    
    # Merge with previous expanded dataset
    print(f"\n[4/5] Merging with previous expanded dataset...")
    prev_expanded_training_path = project_root / 'outputs' / f'expanded_training_{model_name}_iter{previous_iteration}{file_suffix}.csv'
    
    if prev_expanded_training_path.exists():
        prev_expanded_training_df = pd.read_csv(prev_expanded_training_path)
        print(f"  Loaded previous expanded dataset: {len(prev_expanded_training_df)} samples")
    else:
        # Fallback to original training data
        if use_newsplit:
            # For new split, use merged_train_df as base
            prev_expanded_training_df = original_merged_train_df[['ID', 'labels']].copy()
        else:
            prev_expanded_training_df = original_training_df
        print(f"  Using original training data: {len(prev_expanded_training_df)} samples")
    
    expanded_training_df = merge_pseudo_labels_with_training(
        training_df=prev_expanded_training_df,
        pseudo_labels_df=pseudo_labels,
        balance_classes=False  # Preserve all training data
    )
    
    # Save expanded training data (with '_newsplit' suffix if using new method)
    expanded_training_path = project_root / 'outputs' / f'expanded_training_{model_name}_iter{iteration}{file_suffix}.csv'
    expanded_training_df.to_csv(expanded_training_path, index=False)
    print(f"  ✓ Saved expanded training data: {len(expanded_training_df)} samples")
    
    # Create data loaders for retraining
    print(f"\n[5/5] Creating data loaders and retraining model...")
    if use_newsplit:
        # NEW METHOD: Use merged split approach
        train_loader_expanded, val_loader, test_loader = create_expanded_data_loaders_newsplit(
            config=config,
            expanded_training_df=expanded_training_df,
            merged_train_df=original_merged_train_df,
            merged_val_df=original_merged_val_df,
            phase2_test_df=phase2_test_df,
            random_state=random_state
        )
    else:
        # OLD METHOD: Use original approach
        train_loader_expanded, val_loader, test_loader = create_expanded_data_loaders(
            config=config,
            expanded_training_df=expanded_training_df,
            original_training_df=original_training_df,
            phase1_df=phase1_df,
            phase2_eval_df=phase2_eval_df,
            phase2_test_df=phase2_test_df
        )
    
    # Create model and load from previous iteration
    model_expanded = create_model(config.config, num_classes=13)
    checkpoint = torch.load(prev_checkpoint_path, map_location=config.device)
    model_expanded.load_state_dict(checkpoint['model_state_dict'])
    
    # Create trainer
    trainer = Trainer(
        model=model_expanded,
        train_loader=train_loader_expanded,
        val_loader=val_loader,
        config=config.config,
        device=config.device
    )
    
    # Train
    num_epochs = config.config.get('training', {}).get('num_epochs', 50)
    print(f"\n{'='*70}")
    print(f"Training model with expanded dataset (Iteration {iteration})...")
    print(f"{'='*70}")
    trainer.train(num_epochs=num_epochs)
    
    # Save checkpoint
    current_checkpoint_dir = project_root / 'outputs' / 'checkpoints' / f'pseudo_labeling_{model_name}_iter{iteration}{checkpoint_suffix}'
    current_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_source = trainer.checkpoint_dir / 'best.pth'
    
    if best_checkpoint_source.exists():
        shutil.copy(best_checkpoint_source, current_checkpoint_dir / 'best.pth')
        print(f"\n✓ Model saved to {current_checkpoint_dir / 'best.pth'}")
        print(f"  Best validation F1: {trainer.best_val_score:.4f} at epoch {trainer.best_epoch + 1}")
    else:
        print(f"⚠ Warning: Best checkpoint not found at {best_checkpoint_source}")
    
    # Generate submission
    print(f"\n[6/6] Generating submission file...")
    submission_path = project_root / 'outputs' / f'submission_pseudo_labeling_{model_name}_iter{iteration}{file_suffix}.csv'
    
    submission = generate_submission(
        model=trainer.model,
        test_loader=test_loader,
        device=config.device,
        output_path=str(submission_path),
        use_tta=True
    )
    
    print(f"✓ Submission saved to {submission_path}")
    
    # Compare with previous iteration (skip for iteration 0)
    if iteration > 0:
        print(f"\n{'='*70}")
        print(f"Comparison: Iteration {previous_iteration} vs Iteration {iteration}")
        print(f"{'='*70}")
        
        prev_checkpoint = torch.load(prev_checkpoint_path, map_location='cpu')
        prev_score = prev_checkpoint.get('best_val_score', 0.0)
        current_score = trainer.best_val_score
        
        print(f"Iteration {previous_iteration}: Validation F1 = {prev_score:.4f}")
        print(f"Iteration {iteration}:      Validation F1 = {current_score:.4f}")
        
        improvement = current_score - prev_score
        improvement_pct = (improvement / prev_score) * 100 if prev_score > 0 else 0
        print(f"Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        if improvement > 0:
            print("✓ Iteration improved model performance!")
        elif improvement < -0.001:
            print("⚠ Iteration decreased performance (may need tuning)")
        else:
            print("→ Iteration maintained similar performance")
    else:
        improvement = 0.0
    
    return {
        'iteration': iteration,
        'validation_f1': current_score,
        'improvement': improvement,
        'checkpoint_path': current_checkpoint_dir / 'best.pth',
        'submission_path': submission_path,
        'pseudo_labels_path': pseudo_labels_path,
        'expanded_training_path': expanded_training_path
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run pseudo-labeling iterations for WBC Challenge models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run iteration 2 for ResNet-50
  python scripts/pseudo_labeling_iterations.py --model resnet50 --iterations 1 --start_iter 2
  
  # Run iterations 2 and 3 for Swin-Tiny
  python scripts/pseudo_labeling_iterations.py --model swin_tiny --iterations 2 --start_iter 2
  
  # Run iterations 2-4 for all models
  for model in resnet50 swin_tiny convnext_tiny efficientnet_b3; do
    python scripts/pseudo_labeling_iterations.py --model $model --iterations 3 --start_iter 2
  done
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['resnet50', 'swin_tiny', 'convnext_tiny', 'efficientnet_b3', 'efficientnet_b4',
                 'resnext50', 'resnext101', 'efficientnetv2', 'regnet', 'deit', 'vit', 'maxvit', 'coatnet'],
        help='Model name to run pseudo-labeling for'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=1,
        help='Number of iterations to run (default: 1)'
    )
    
    parser.add_argument(
        '--start_iter',
        type=int,
        default=2,
        help='Starting iteration number (default: 2, assumes iter0 and iter1 are done)'
    )
    
    parser.add_argument(
        '--confidence_threshold',
        type=float,
        default=0.9,
        help='Confidence threshold for pseudo-labels (default: 0.9)'
    )
    
    parser.add_argument(
        '--confidence_increment',
        type=float,
        default=0.0,
        help='Increase confidence threshold by this amount each iteration (default: 0.0)'
    )
    
    parser.add_argument(
        '--no_tta',
        action='store_true',
        help='Disable test-time augmentation for pseudo-label generation'
    )
    
    parser.add_argument(
        '--use_new_split',
        action='store_true',
        help='Use new merged split method (phase1+phase2_train+phase2_eval with stratified split). Default: old split method (phase1+phase2_train / phase2_eval)'
    )
    
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.1,
        help='Validation split ratio for new split method (default: 0.1 for 90/10 split)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for stratified split (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    print("="*70)
    print("WBC Challenge - Pseudo-Labeling Iterations")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Starting iteration: {args.start_iter}")
    print(f"Number of iterations: {args.iterations}")
    print(f"Initial confidence threshold: {args.confidence_threshold}")
    if args.confidence_increment > 0:
        print(f"Confidence increment per iteration: {args.confidence_increment}")
    print(f"TTA enabled: {not args.no_tta}")
    print(f"Split method: {'NEW (Merged 90/10 split)' if args.use_new_split else 'OLD (phase1+phase2_train / phase2_eval)'}")
    if args.use_new_split:
        print(f"Validation split ratio: {args.val_split}")
        print(f"Random seed: {args.seed}")
    print("="*70)
    
    # Run iterations
    results = []
    current_threshold = args.confidence_threshold
    
    for i in range(args.iterations):
        iteration = args.start_iter + i
        current_threshold = args.confidence_threshold + (i * args.confidence_increment)
        
        try:
            result = run_pseudo_labeling_iteration(
                model_name=args.model,
                iteration=iteration,
                project_root=project_root,
                confidence_threshold=current_threshold,
                use_tta=not args.no_tta,
                use_newsplit=args.use_new_split,  # Default: False (old split method)
                val_split_ratio=args.val_split,
                random_state=args.seed
            )
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error in Iteration {iteration}: {e}")
            print(f"Stopping execution. Previous iterations completed successfully.")
            break
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Completed {len(results)} iteration(s):")
    for result in results:
        print(f"  Iteration {result['iteration']}: Validation F1 = {result['validation_f1']:.4f} "
              f"(Improvement: {result['improvement']:+.4f})")
    
    if len(results) > 1:
        total_improvement = results[-1]['validation_f1'] - results[0]['validation_f1']
        print(f"\nTotal improvement from Iteration {results[0]['iteration']} to {results[-1]['iteration']}: "
              f"{total_improvement:+.4f}")
    
    print(f"\n✓ All iterations completed successfully!")
    file_suffix = '_newsplit' if args.use_new_split else ''
    print(f"Checkpoints saved in: outputs/checkpoints/pseudo_labeling_{args.model}_iter*{file_suffix}/")
    print(f"Submissions saved in: outputs/submission_pseudo_labeling_{args.model}_iter*{file_suffix}.csv")
    print(f"Expanded training saved in: outputs/expanded_training_{args.model}_iter*{file_suffix}.csv")
    print(f"Pseudo-labels saved in: outputs/pseudo_labels_{args.model}_iter*{file_suffix}.csv")


if __name__ == '__main__':
    main()

