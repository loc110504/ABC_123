#!/usr/bin/env python
"""
Unified training script for WBC Challenge.
Supports training any of the 5 models: resnet50, swin_tiny, efficientnet_b3, efficientnet_b4, convnext_tiny.
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.dataset import (
    merge_all_training_data,
    create_stratified_split,
    create_data_loaders,
    load_label_files,
    WBCDataset
)
from src.models import create_model
from src.training import Trainer
from src.submission import generate_submission
from src.evaluation import calculate_metrics


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description='Train WBC classification model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['resnet50', 'swin_tiny', 'efficientnet_b3', 'efficientnet_b4', 'convnext_tiny',
                 'resnext50', 'resnext101', 'efficientnetv2', 'regnet', 'deit', 'vit', 'maxvit', 'coatnet'],
        help='Model name to train'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Custom config file path (defaults to configs/config_{model}.yaml)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.1,
        help='Validation split ratio (default: 0.1 for 90/10 split)'
    )
    parser.add_argument(
        '--no_submission',
        action='store_true',
        help='Skip generating submission file'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine config path
    if args.config:
        config_path = args.config
    else:
        config_path = f'configs/config_{args.model}.yaml'
    
    print("=" * 60)
    print(f"WBC Challenge - Training {args.model.upper()}")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Seed: {args.seed}")
    print(f"Validation split: {args.val_split:.1%}")
    print()
    
    # Load configuration
    config = Config(config_path)
    
    # Update checkpoint directory to be model-specific if not already
    checkpoint_dir = config.config.get('paths', {}).get('checkpoint_dir', 'outputs/checkpoints')
    if not checkpoint_dir.endswith(args.model):
        # Make it model-specific
        checkpoint_dir = str(Path(checkpoint_dir) / args.model)
        if 'paths' not in config.config:
            config.config['paths'] = {}
        config.config['paths']['checkpoint_dir'] = checkpoint_dir
    
    # Check device
    device = config.device
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    # Merge all training data
    print("Merging training data...")
    merged_df = merge_all_training_data(config)
    print(f"Total training samples: {len(merged_df)}")
    print()
    
    # Create stratified split
    print("Creating stratified train/val split...")
    train_df, val_df = create_stratified_split(
        merged_df,
        val_ratio=args.val_split,
        random_state=args.seed
    )
    
    # Load test data
    phase2_test_df = pd.read_csv(config.get('data.phase2_test_labels'))
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        config,
        use_merged_split=True,
        merged_train_df=train_df,
        merged_val_df=val_df,
        phase2_test_df=phase2_test_df
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print()
    
    # Create model
    print("Creating model...")
    model = create_model(
        config=config.config,
        num_classes=config.config.get('model', {}).get('num_classes', 13)
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Get class weights from training dataset
    train_dataset = train_loader.dataset
    class_weights = train_dataset.get_class_weights() if hasattr(train_dataset, 'get_class_weights') else None
    
    # Create trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.config,
        device=device,
        class_weights=class_weights
    )
    print()
    
    # Train model
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)
    num_epochs = config.config.get('training', {}).get('num_epochs', 25)
    trainer.train(num_epochs)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    val_loss, val_metrics = trainer.validate()
    
    print(f"\nValidation Results:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Macro-F1: {val_metrics['macro_f1']:.4f}")
    print(f"  Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}")
    print(f"\nPer-Class F1 Scores:")
    for i, (class_name, f1) in enumerate(zip(WBCDataset.CLASSES, val_metrics['per_class_f1'])):
        print(f"  {class_name:3s}: {f1:.4f}")
    
    # Save checkpoint path info
    checkpoint_dir = Path(config.get('paths.checkpoint_dir', 'outputs/checkpoints'))
    checkpoint_path = checkpoint_dir / 'best.pth'
    
    print(f"\nBest model saved to: {checkpoint_path}")
    print(f"Best validation F1: {trainer.best_val_score:.4f} at epoch {trainer.best_epoch + 1}")
    
    # Generate submission
    if not args.no_submission and test_loader is not None:
        print("\n" + "=" * 60)
        print("Generating submission file...")
        print("=" * 60)
        
        submission_path = Path(config.get('paths.submission_dir', 'outputs')) / f'submission_{args.model}.csv'
        
        submission = generate_submission(
            model=trainer.model,
            test_loader=test_loader,
            device=device,
            output_path=str(submission_path),
            use_tta=False  # Can enable TTA if needed
        )
        
        print(f"\nSubmission file saved to: {submission_path}")
        print(f"Total predictions: {len(submission)}")
        print(f"\nClass distribution:")
        print(submission['labels'].value_counts().sort_index())
    else:
        print("\nSkipping submission generation (--no_submission flag or no test data)")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

