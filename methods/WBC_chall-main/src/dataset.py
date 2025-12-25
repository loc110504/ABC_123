"""
Dataset and data loading utilities for WBC Challenge.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional, Tuple, Dict, List
from sklearn.model_selection import train_test_split


class WBCDataset(Dataset):
    """Dataset class for WBC images."""
    
    # 13 WBC class codes in sorted order
    CLASSES = ['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PC', 'PLY', 'PMY', 'SNE', 'VLY']
    CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
    IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}
    
    # Rare classes that need more aggressive augmentation
    RARE_CLASSES = ['PC', 'PLY', 'PMY']  # Indices: 8, 9, 10
    
    def __init__(
        self,
        image_dir: str,
        labels_df: pd.DataFrame,
        transform: Optional[A.Compose] = None,
        mode: str = 'train',
        rare_class_boost: bool = False
    ):
        """
        Initialize WBC dataset.
        
        Args:
            image_dir: Directory containing images
            labels_df: DataFrame with 'ID' and 'labels' columns
            transform: Albumentations transform pipeline
            mode: 'train', 'val', or 'test'
            rare_class_boost: If True, apply more aggressive augmentation for rare classes
        """
        self.image_dir = image_dir
        self.labels_df = labels_df.copy()
        self.transform = transform
        self.mode = mode
        self.rare_class_boost = rare_class_boost
        
        # Filter out rows with missing labels (for train/val)
        if mode != 'test':
            self.labels_df = self.labels_df.dropna(subset=['labels'])
        
        # Create full image paths
        self.labels_df['image_path'] = self.labels_df['ID'].apply(
            lambda x: os.path.join(image_dir, x)
        )
        
        # Convert labels to indices
        if mode != 'test':
            self.labels_df['label_idx'] = self.labels_df['labels'].map(self.CLASS_TO_IDX)
            # Remove any rows with invalid labels
            self.labels_df = self.labels_df.dropna(subset=['label_idx'])
            self.labels_df['label_idx'] = self.labels_df['label_idx'].astype(int)
    
    def __len__(self) -> int:
        return len(self.labels_df)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get item by index."""
        row = self.labels_df.iloc[idx]
        image_path = row['image_path']
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            # Check if this is a rare class and we should use boosted augmentation
            if self.rare_class_boost and self.mode == 'train':
                label_str = row.get('labels', None)
                if label_str in self.RARE_CLASSES:
                    # Apply more aggressive augmentation for rare classes
                    # We'll do this by applying transform twice with higher probability
                    transformed = self.transform(image=image)
                    image = transformed['image']
                    # Optionally apply additional augmentations
                else:
                    transformed = self.transform(image=image)
                    image = transformed['image']
            else:
                transformed = self.transform(image=image)
                image = transformed['image']
        else:
            # Default: convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        result = {'image': image, 'image_id': row['ID']}
        
        if self.mode != 'test':
            result['label'] = row['label_idx']
            result['label_str'] = row['labels']
        
        return result
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset."""
        if self.mode == 'test':
            return None
        
        class_counts = self.labels_df['label_idx'].value_counts().sort_index()
        total_samples = len(self.labels_df)
        num_classes = len(self.CLASSES)
        
        # Inverse frequency weighting
        weights = total_samples / (num_classes * class_counts)
        weights = weights / weights.sum() * num_classes  # Normalize
        
        return torch.FloatTensor(weights.values)


def get_train_transform(config: Dict, aggressive: bool = False) -> A.Compose:
    """
    Get training augmentation pipeline with advanced augmentations.
    
    Args:
        config: Configuration dictionary
        aggressive: If True, use more aggressive augmentations (for rare classes)
    """
    aug_config = config.get('augmentation', {}).get('train', {})
    resize_size = aug_config.get('resize', 384)
    
    # Adjust probabilities based on aggressiveness
    prob_mult = 1.5 if aggressive else 1.0
    
    transforms = [
        # Resize first
        A.Resize(resize_size, resize_size),
        
        # Geometric augmentations - medical images can be rotated/flipped
        A.HorizontalFlip(p=aug_config.get('horizontal_flip', 0.5)),
        A.VerticalFlip(p=aug_config.get('vertical_flip', 0.5)),
        A.Rotate(
            limit=aug_config.get('rotation', 15) * (1.5 if aggressive else 1.0),
            border_mode=0,
            value=0,
            p=min(0.9, 0.7 * prob_mult)
        ),
        
        # Advanced geometric augmentations for medical imaging
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            border_mode=0,
            value=0,
            p=min(0.9, aug_config.get('elastic_transform', 0.3) * prob_mult)
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            border_mode=0,
            value=0,
            p=min(0.9, aug_config.get('grid_distortion', 0.3) * prob_mult)
        ),
        A.OpticalDistortion(
            distort_limit=0.2,
            shift_limit=0.05,
            border_mode=0,
            value=0,
            p=min(0.9, aug_config.get('optical_distortion', 0.2) * prob_mult)
        ),
    ]
    
    # Advanced masking augmentations (added conditionally)
    # GridMask may not be available in older albumentations versions
    if hasattr(A, 'GridMask') and aug_config.get('gridmask', 0) > 0:
        transforms.append(A.GridMask(
            num_grid=(3, 7),
            fill_value=0,
            rotate=90,
            mode=aug_config.get('gridmask_mode', 1),
            p=min(0.9, aug_config.get('gridmask', 0.5) * prob_mult)
        ))
    elif hasattr(A, 'GridDropout') and aug_config.get('gridmask', 0) > 0:
        # Use GridDropout as alternative to GridMask
        transforms.append(A.GridDropout(
            ratio=0.4,
            holes_number_x=4,
            holes_number_y=4,
            shift_x=0,
            shift_y=0,
            fill_value=0,
            p=min(0.9, aug_config.get('gridmask', 0.5) * prob_mult)
        ))
    
    # CoarseDropout for regularization
    if aug_config.get('coarse_dropout', 0) > 0:
        transforms.append(A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=min(0.9, aug_config.get('coarse_dropout', 0.3) * prob_mult)
        ))
    
    # Color augmentations - important for medical imaging variations
    transforms.extend([
        A.RandomBrightnessContrast(
            brightness_limit=aug_config.get('brightness', 0.2) * (1.2 if aggressive else 1.0),
            contrast_limit=aug_config.get('contrast', 0.2) * (1.2 if aggressive else 1.0),
            brightness_by_max=True,
            p=min(0.9, aug_config.get('brightness_contrast', 0.7) * prob_mult)
        ),
        A.ColorJitter(
            brightness=aug_config.get('brightness', 0.2),
            contrast=aug_config.get('contrast', 0.2),
            saturation=aug_config.get('saturation', 0.2),
            hue=aug_config.get('hue', 0.1),
            p=min(0.9, aug_config.get('color_jitter', 0.5) * prob_mult)
        ),
        A.HueSaturationValue(
            hue_shift_limit=aug_config.get('hue', 0.1) * 180,
            sat_shift_limit=aug_config.get('saturation', 0.2) * 255,
            val_shift_limit=aug_config.get('brightness', 0.2) * 255,
            p=min(0.9, aug_config.get('hue_saturation', 0.5) * prob_mult)
        ),
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) for medical images
        A.CLAHE(
            clip_limit=4.0,
            tile_grid_size=(8, 8),
            p=min(0.9, aug_config.get('clahe', 0.3) * prob_mult)
        ),
        
        # Sharpening - helps with cell boundary definition
        A.Sharpen(
            alpha=(0.2, 0.5),
            lightness=(0.5, 1.0),
            p=min(0.9, aug_config.get('sharpen', 0.3) * prob_mult)
        ),
        
        # Noise and blur - simulates scanner variations
        A.GaussNoise(
            var_limit=(10.0, 50.0),
            mean=0,
            p=min(0.9, aug_config.get('gaussian_noise', 0.2) * prob_mult)
        ),
        A.GaussianBlur(
            blur_limit=(3, 7),
            p=min(0.9, aug_config.get('gaussian_blur', 0.2) * prob_mult)
        ),
        
        # Motion blur - simulates movement artifacts
        A.MotionBlur(
            blur_limit=7,
            p=min(0.9, aug_config.get('motion_blur', 0.2) * prob_mult)
        ),
        
        # Random gamma - simulates exposure variations
        A.RandomGamma(
            gamma_limit=(80, 120),
            p=min(0.9, aug_config.get('gamma', 0.3) * prob_mult)
        ),
    ])
    
    # Normalization and tensor conversion
    transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return A.Compose(transforms)


def get_val_transform(config: Dict) -> A.Compose:
    """Get validation/test augmentation pipeline."""
    aug_config = config.get('augmentation', {}).get('val', {})
    resize_size = aug_config.get('resize', 384)
    
    return A.Compose([
        A.Resize(resize_size, resize_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


class CombinedWBCDataset(Dataset):
    """Dataset that combines images from multiple directories."""
    
    def __init__(
        self,
        image_dirs: List[str],
        labels_dfs: List[pd.DataFrame],
        transform: Optional[A.Compose] = None,
        mode: str = 'train',
        rare_class_boost: bool = False
    ):
        """
        Initialize combined WBC dataset.
        
        Args:
            image_dirs: List of image directories
            labels_dfs: List of label DataFrames corresponding to image_dirs
            transform: Albumentations transform pipeline
            mode: 'train', 'val', or 'test'
            rare_class_boost: If True, apply more aggressive augmentation for rare classes
        """
        if len(image_dirs) != len(labels_dfs):
            raise ValueError("Number of image directories must match number of label DataFrames")
        
        self.image_dirs = image_dirs
        self.transform = transform
        self.mode = mode
        self.rare_class_boost = rare_class_boost
        
        # Combine all labels
        self.labels_df = pd.concat(labels_dfs, ignore_index=True)
        
        # Create image paths mapping
        self.image_paths = []
        for image_dir, labels_df in zip(image_dirs, labels_dfs):
            for image_id in labels_df['ID']:
                self.image_paths.append(os.path.join(image_dir, image_id))
        
        # Convert labels to indices
        if mode != 'test':
            self.labels_df['label_idx'] = self.labels_df['labels'].map(WBCDataset.CLASS_TO_IDX)
            # Store original indices before dropping
            valid_mask = self.labels_df['label_idx'].notna()
            self.labels_df = self.labels_df[valid_mask].copy()
            self.labels_df['label_idx'] = self.labels_df['label_idx'].astype(int)
            # Filter image_paths to match valid rows
            self.image_paths = [self.image_paths[i] for i, valid in enumerate(valid_mask) if valid]
            self.labels_df = self.labels_df.reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.labels_df)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get item by index."""
        row = self.labels_df.iloc[idx]
        image_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            # Check if this is a rare class and we should use boosted augmentation
            if self.rare_class_boost and self.mode == 'train':
                label_str = row.get('labels', None)
                if label_str in WBCDataset.RARE_CLASSES:
                    # Apply transform with aggressive mode
                    transformed = self.transform(image=image)
                    image = transformed['image']
                else:
                    transformed = self.transform(image=image)
                    image = transformed['image']
            else:
                transformed = self.transform(image=image)
                image = transformed['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        result = {'image': image, 'image_id': row['ID']}
        
        if self.mode != 'test':
            result['label'] = row['label_idx']
            result['label_str'] = row['labels']
        
        return result
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset."""
        if self.mode == 'test':
            return None
        
        class_counts = self.labels_df['label_idx'].value_counts().sort_index()
        total_samples = len(self.labels_df)
        num_classes = len(WBCDataset.CLASSES)
        
        weights = total_samples / (num_classes * class_counts)
        weights = weights / weights.sum() * num_classes
        
        return torch.FloatTensor(weights.values)


def create_data_loaders(
    config,
    phase1_df: Optional[pd.DataFrame] = None,
    phase2_train_df: Optional[pd.DataFrame] = None,
    phase2_eval_df: Optional[pd.DataFrame] = None,
    phase2_test_df: Optional[pd.DataFrame] = None,
    use_merged_split: bool = False,
    merged_train_df: Optional[pd.DataFrame] = None,
    merged_val_df: Optional[pd.DataFrame] = None
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create data loaders for train, validation, and test.
    
    Args:
        config: Configuration object
        phase1_df: Phase 1 training labels DataFrame (used when use_merged_split=False)
        phase2_train_df: Phase 2 training labels DataFrame (used when use_merged_split=False)
        phase2_eval_df: Phase 2 evaluation labels DataFrame (used when use_merged_split=False)
        phase2_test_df: Phase 2 test labels DataFrame
        use_merged_split: If True, use merged_train_df and merged_val_df instead of separate phase data
        merged_train_df: Merged training DataFrame with 'ID', 'labels', 'source_dir' columns
        merged_val_df: Merged validation DataFrame with 'ID', 'labels', 'source_dir' columns
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if use_merged_split:
        # Use merged split mode
        if merged_train_df is None or merged_val_df is None:
            raise ValueError("merged_train_df and merged_val_df must be provided when use_merged_split=True")
        
        # Group merged training data by source directory
        train_image_dirs = []
        train_dfs = []
        
        # Separate by source_dir
        phase1_train = merged_train_df[merged_train_df['source_dir'] == 'phase1'].copy()
        phase2_train_split = merged_train_df[merged_train_df['source_dir'] == 'phase2_train'].copy()
        phase2_eval_split = merged_train_df[merged_train_df['source_dir'] == 'phase2_eval'].copy()
        
        if len(phase1_train) > 0:
            train_image_dirs.append(config.get('data.phase1_dir'))
            train_dfs.append(phase1_train[['ID', 'labels']])
        
        if len(phase2_train_split) > 0:
            train_image_dirs.append(config.get('data.phase2_train_dir'))
            train_dfs.append(phase2_train_split[['ID', 'labels']])
        
        if len(phase2_eval_split) > 0:
            # phase2_eval images are in phase2/eval directory
            train_image_dirs.append(config.get('data.phase2_eval_dir'))
            train_dfs.append(phase2_eval_split[['ID', 'labels']])
        
        if not train_dfs:
            raise ValueError("No training data found in merged_train_df")
        
        # Create validation dataset from merged_val_df
        # Need to determine which directory each validation sample comes from
        val_image_dirs = []
        val_dfs = []
        
        phase1_val = merged_val_df[merged_val_df['source_dir'] == 'phase1'].copy()
        phase2_train_val = merged_val_df[merged_val_df['source_dir'] == 'phase2_train'].copy()
        phase2_eval_val = merged_val_df[merged_val_df['source_dir'] == 'phase2_eval'].copy()
        
        if len(phase1_val) > 0:
            val_image_dirs.append(config.get('data.phase1_dir'))
            val_dfs.append(phase1_val[['ID', 'labels']])
        
        if len(phase2_train_val) > 0:
            val_image_dirs.append(config.get('data.phase2_train_dir'))
            val_dfs.append(phase2_train_val[['ID', 'labels']])
        
        if len(phase2_eval_val) > 0:
            val_image_dirs.append(config.get('data.phase2_eval_dir'))
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
    else:
        # Original mode: use separate phase data
        # Create training dataset from phase1 and phase2
        train_image_dirs = []
        train_dfs = []
        
        if phase1_df is not None and len(phase1_df) > 0:
            train_image_dirs.append(config.get('data.phase1_dir'))
            train_dfs.append(phase1_df)
        
        if phase2_train_df is not None and len(phase2_train_df) > 0:
            train_image_dirs.append(config.get('data.phase2_train_dir'))
            train_dfs.append(phase2_train_df)
        
        if not train_dfs:
            raise ValueError("At least one training dataset must be provided")
        
        # Create validation dataset from phase2_eval
        val_dataset = WBCDataset(
            image_dir=config.get('data.phase2_eval_dir'),
            labels_df=phase2_eval_df,
            transform=get_val_transform(config.config),
            mode='val'
        ) if phase2_eval_df is not None and len(phase2_eval_df) > 0 else None
    
    # Check if rare class boost is enabled
    rare_class_boost = config.get('augmentation', {}).get('train', {}).get('rare_class_boost', False)
    
    # Create training dataset
    train_dataset = CombinedWBCDataset(
        image_dirs=train_image_dirs,
        labels_dfs=train_dfs,
        transform=get_train_transform(config.config, aggressive=False),
        mode='train',
        rare_class_boost=rare_class_boost
    )
    
    # Create data loaders
    train_config = config.get('training', {})
    batch_size = train_config.get('batch_size', 32)
    num_workers = train_config.get('num_workers', 4)
    pin_memory = train_config.get('pin_memory', True)
    
    # Use weighted sampling for imbalanced data
    use_class_weights = train_config.get('use_class_weights', True)
    if use_class_weights:
        class_weights = train_dataset.get_class_weights()
        
        # Get minimum samples per class per epoch
        min_class_samples = train_config.get('min_class_samples_per_epoch', None)
        
        if min_class_samples is not None and min_class_samples > 0:
            # Enhanced sampling: guarantee minimum samples per class
            # Calculate how many samples we need for each class
            class_counts = train_dataset.labels_df['label_idx'].value_counts().sort_index()
            num_classes = len(WBCDataset.CLASSES)
            
            # Calculate base sample weights
            sample_weights_base = class_weights[train_dataset.labels_df['label_idx'].values]
            
            # Boost weights for rare classes to guarantee minimum samples
            # Classes that need more samples get higher boost
            total_samples = len(train_dataset)
            samples_per_epoch = total_samples  # Default: use all samples once
            
            # Identify rare classes (those that would have fewer than min_class_samples)
            rare_class_indices = []
            for class_idx in range(num_classes):
                current_class_count = class_counts.get(class_idx, 0)
                if current_class_count < min_class_samples:
                    rare_class_indices.append(class_idx)
            
            # Create boosted weights
            sample_weights_boosted = sample_weights_base.clone()
            # Different boost factors for different rare classes
            # PLY (index 9) is the rarest, needs highest boost
            boost_factors = {
                8: 8.0,   # PC: 8x boost
                9: 20.0,  # PLY: 20x boost (rarest class, only 14 samples)
                10: 10.0  # PMY: 10x boost
            }
            
            for idx, class_idx in enumerate(train_dataset.labels_df['label_idx'].values):
                if class_idx in rare_class_indices:
                    boost_factor = boost_factors.get(class_idx, 5.0)
                    sample_weights_boosted[idx] *= boost_factor
            
            # Normalize weights
            sample_weights_boosted = sample_weights_boosted / sample_weights_boosted.sum() * len(sample_weights_boosted)
            
            # Use boosted weights for sampling
            sample_weights = sample_weights_boosted
        else:
            # Standard weighted sampling
            sample_weights = class_weights[train_dataset.labels_df['label_idx'].values]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        train_loader = DataLoader(
            train_dataset,
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
    
    # Test loader (optional)
    test_loader = None
    if phase2_test_df is not None:
        test_dataset = WBCDataset(
            image_dir=config.get('data.phase2_test_dir'),
            labels_df=phase2_test_df,
            transform=get_val_transform(config.config),
            mode='test'
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    return train_loader, val_loader, test_loader


def load_label_files(config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load label files for all phases.
    
    Args:
        config: Configuration object
    
    Returns:
        Tuple of (phase1_df, phase2_train_df, phase2_eval_df, phase2_test_df)
    """
    phase1_df = pd.read_csv(config.get('data.phase1_labels'))
    phase2_train_df = pd.read_csv(config.get('data.phase2_train_labels'))
    phase2_eval_df = pd.read_csv(config.get('data.phase2_eval_labels'))
    phase2_test_df = pd.read_csv(config.get('data.phase2_test_labels'))
    
    return phase1_df, phase2_train_df, phase2_eval_df, phase2_test_df


def merge_all_training_data(config) -> pd.DataFrame:
    """
    Merge all training data from phase1, phase2_train, and phase2_eval.
    
    Args:
        config: Configuration object
    
    Returns:
        Merged DataFrame with 'ID', 'labels', and 'source_dir' columns
    """
    # Load all label files
    phase1_df = pd.read_csv(config.get('data.phase1_labels'))
    phase2_train_df = pd.read_csv(config.get('data.phase2_train_labels'))
    phase2_eval_df = pd.read_csv(config.get('data.phase2_eval_labels'))
    
    # Add source column to track which directory images come from
    phase1_df = phase1_df.copy()
    phase2_train_df = phase2_train_df.copy()
    phase2_eval_df = phase2_eval_df.copy()
    
    phase1_df['source_dir'] = 'phase1'
    phase2_train_df['source_dir'] = 'phase2_train'
    phase2_eval_df['source_dir'] = 'phase2_eval'
    
    # Merge all DataFrames
    merged_df = pd.concat([phase1_df, phase2_train_df, phase2_eval_df], ignore_index=True)
    
    # Ensure we have required columns
    if 'ID' not in merged_df.columns or 'labels' not in merged_df.columns:
        raise ValueError("Merged DataFrame must have 'ID' and 'labels' columns")
    
    return merged_df


def create_stratified_split(
    merged_df: pd.DataFrame,
    val_ratio: float = 0.1,
    random_state: int = 42,
    ensure_rare_class_in_train: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/validation split from merged data.
    
    Args:
        merged_df: Merged DataFrame with 'ID', 'labels', and 'source_dir' columns
        val_ratio: Validation split ratio (default: 0.1 for 90/10 split)
        random_state: Random seed for reproducibility
        ensure_rare_class_in_train: If True, ensure rare classes (PLY, PC, PMY) have
                                   minimum samples in training (default: True)
    
    Returns:
        Tuple of (train_df, val_df) with 'ID', 'labels', and 'source_dir' columns
    """
    # Make a copy to avoid modifying original
    df = merged_df.copy()
    
    # Convert labels to indices for stratification
    df['label_idx'] = df['labels'].map(WBCDataset.CLASS_TO_IDX)
    
    # Remove any rows with invalid labels
    df = df.dropna(subset=['label_idx'])
    df['label_idx'] = df['label_idx'].astype(int)
    
    # Check if we have enough samples for stratification
    # For very rare classes, we might not be able to stratify perfectly
    class_counts = df['label_idx'].value_counts()
    min_class_count = class_counts.min()
    
    if min_class_count < 2:
        print(f"Warning: Some classes have fewer than 2 samples. Stratification may not be perfect.")
        # For classes with only 1 sample, we'll put it in training
        single_sample_classes = class_counts[class_counts == 1].index
        if len(single_sample_classes) > 0:
            print(f"Classes with single sample: {single_sample_classes.tolist()}")
    
    # Special handling for rare classes (PLY, PC, PMY)
    rare_class_indices = [8, 9, 10]  # PC, PLY, PMY
    min_rare_class_train_samples = 10  # Minimum samples in training for rare classes
    
    if ensure_rare_class_in_train:
        # Separate rare class samples
        rare_class_mask = df['label_idx'].isin(rare_class_indices)
        rare_class_df = df[rare_class_mask].copy()
        common_class_df = df[~rare_class_mask].copy()
        
        # For rare classes, ensure minimum in training
        rare_train_dfs = []
        rare_val_dfs = []
        
        for class_idx in rare_class_indices:
            class_df = rare_class_df[rare_class_df['label_idx'] == class_idx].copy()
            if len(class_df) == 0:
                continue
            
            class_count = len(class_df)
            
            if class_count <= 2:
                # If only 1-2 samples, put all in training
                rare_train_dfs.append(class_df)
                print(f"  Class {WBCDataset.IDX_TO_CLASS[class_idx]}: {class_count} samples -> all in training")
            elif class_count < min_rare_class_train_samples * 2:
                # If very few samples, put most in training (80/20 instead of 90/10)
                train_size = max(min_rare_class_train_samples, int(class_count * 0.8))
                class_train = class_df.sample(n=train_size, random_state=random_state)
                class_val = class_df.drop(class_train.index)
                rare_train_dfs.append(class_train)
                rare_val_dfs.append(class_val)
                print(f"  Class {WBCDataset.IDX_TO_CLASS[class_idx]}: {len(class_train)} train, {len(class_val)} val")
            else:
                # Normal stratified split for rare classes
                class_train, class_val = train_test_split(
                    class_df,
                    test_size=val_ratio,
                    stratify=class_df['label_idx'],
                    random_state=random_state
                )
                rare_train_dfs.append(class_train)
                rare_val_dfs.append(class_val)
        
        # Combine rare class splits
        if rare_train_dfs:
            rare_train_df = pd.concat(rare_train_dfs, ignore_index=True)
        else:
            rare_train_df = pd.DataFrame(columns=df.columns)
        
        if rare_val_dfs:
            rare_val_df = pd.concat(rare_val_dfs, ignore_index=True)
        else:
            rare_val_df = pd.DataFrame(columns=df.columns)
        
        # Split common classes normally
        if len(common_class_df) > 0:
            common_train_df, common_val_df = train_test_split(
                common_class_df,
                test_size=val_ratio,
                stratify=common_class_df['label_idx'],
                random_state=random_state
            )
        else:
            common_train_df = pd.DataFrame(columns=df.columns)
            common_val_df = pd.DataFrame(columns=df.columns)
        
        # Combine all
        train_df = pd.concat([rare_train_df, common_train_df], ignore_index=True)
        val_df = pd.concat([rare_val_df, common_val_df], ignore_index=True)
        
        # Shuffle
        train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        val_df = val_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        # Standard stratified split
        train_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            stratify=df['label_idx'],
            random_state=random_state
        )
    
    # Select only required columns
    train_df = train_df[['ID', 'labels', 'source_dir']].copy().reset_index(drop=True)
    val_df = val_df[['ID', 'labels', 'source_dir']].copy().reset_index(drop=True)
    
    print(f"\nStratified Split Summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Train samples: {len(train_df)} ({100*len(train_df)/len(df):.1f}%)")
    print(f"  Val samples: {len(val_df)} ({100*len(val_df)/len(df):.1f}%)")
    print(f"\nTrain class distribution:")
    print(train_df['labels'].value_counts().sort_index())
    print(f"\nVal class distribution:")
    print(val_df['labels'].value_counts().sort_index())
    
    # Check PLY specifically
    ply_train = (train_df['labels'] == 'PLY').sum()
    ply_val = (val_df['labels'] == 'PLY').sum()
    if ply_train < 10:
        print(f"\n⚠️  Warning: PLY has only {ply_train} samples in training (recommended: 10+)")
    if ply_val == 0:
        print(f"⚠️  Warning: PLY has 0 samples in validation (evaluation will be limited)")
    elif ply_val == 1:
        print(f"⚠️  Warning: PLY has only 1 sample in validation (evaluation may be unreliable)")
    
    return train_df, val_df
