#!/usr/bin/env python3
"""
Generate submission with Advanced Test-Time Augmentation (TTA).

This script generates submissions with:
- Multi-scale inference (320, 384, 448)
- Rotation augmentations (-10°, -5°, 5°, 10°)
- Color augmentations (brightness, contrast)
- Standard flips (horizontal, vertical, both)
- PLY post-processing to ensure rare class predictions

Expected improvement: +0.005 to +0.015 over basic TTA
"""

import sys
import os
import argparse
from pathlib import Path
import torch
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.dataset import (
    create_data_loaders, 
    merge_all_training_data, 
    create_stratified_split
)
from src.models import create_model
from src.submission import generate_submission

def main():
    parser = argparse.ArgumentParser(description='Generate submission with Advanced TTA')
    parser.add_argument('--model', type=str, required=True,
                        choices=['resnet50', 'swin_tiny', 'efficientnet_b3', 'efficientnet_b4', 'convnext_tiny',
                                 'resnext50', 'resnext101', 'efficientnetv2', 'regnet', 'deit', 'vit', 'maxvit', 'coatnet'],
                        help='Model name')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (e.g., outputs/checkpoints/pseudo_labeling_resnet50_iter6/best.pth)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output submission path (default: outputs/submission_{model}_advanced_tta.csv)')
    parser.add_argument('--use_old_split', action='store_true',
                        help='Use old split method (phase1+phase2_train / phase2_eval)')
    parser.add_argument('--multi_scale', action='store_true', default=True,
                        help='Enable multi-scale TTA (default: True)')
    parser.add_argument('--rotation', action='store_true', default=True,
                        help='Enable rotation TTA (default: True)')
    parser.add_argument('--color', action='store_true', default=True,
                        help='Enable color TTA (default: True)')
    parser.add_argument('--scales', type=str, default='320,384,448',
                        help='Multi-scale sizes (comma-separated, default: 320,384,448)')
    parser.add_argument('--angles', type=str, default='-10,-5,5,10',
                        help='Rotation angles in degrees (comma-separated, default: -10,-5,5,10)')
    
    args = parser.parse_args()
    
    # Parse scales and angles
    scales = [int(s.strip()) for s in args.scales.split(',')]
    angles = [float(a.strip()) for a in args.angles.split(',')]
    
    # Load config
    config_name = args.model
    os.environ['WBC_MODEL_CONFIG'] = config_name
    config = Config(project_root=str(project_root))
    
    device = config.device
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print()
    
    # Load test data
    phase2_test_df = pd.read_csv(config.get('data.phase2_test_labels'))
    
    # Create test loader
    if args.use_old_split:
        # Old method
        phase1_df = pd.read_csv(config.get('data.phase1_labels'))
        phase2_train_df = pd.read_csv(config.get('data.phase2_train_labels'))
        phase2_eval_df = pd.read_csv(config.get('data.phase2_eval_labels'))
        
        _, _, test_loader = create_data_loaders(
            config,
            phase1_df=phase1_df,
            phase2_train_df=phase2_train_df,
            phase2_eval_df=phase2_eval_df,
            phase2_test_df=phase2_test_df
        )
    else:
        # New merged split method
        merged_df = merge_all_training_data(config)
        train_df, val_df = create_stratified_split(
            merged_df,
            val_ratio=0.1,
            random_state=42,
            ensure_rare_class_in_train=True
        )
        
        _, _, test_loader = create_data_loaders(
            config,
            use_merged_split=True,
            merged_train_df=train_df,
            merged_val_df=val_df,
            phase2_test_df=phase2_test_df
        )
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create model
    model = create_model(config.config, num_classes=13)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded model state dict")
        if 'best_val_score' in checkpoint:
            print(f"  Best validation F1: {checkpoint['best_val_score']:.4f}")
    else:
        # If checkpoint is just the state dict
        model.load_state_dict(checkpoint)
        print(f"  Loaded state dict directly")
    
    model = model.to(device)
    model.eval()
    print(f"  Model loaded and set to evaluation mode")
    print()
    
    # Configure Advanced TTA
    print("="*70)
    print("Advanced TTA Configuration")
    print("="*70)
    tta_config = {
        'flips': True,  # Always enable flips
        'multi_scale': {
            'enable': args.multi_scale,
            'scales': scales
        },
        'rotation': {
            'enable': args.rotation,
            'angles': angles
        },
        'color': {
            'enable': args.color,
            'brightness_range': (0.9, 1.1),
            'contrast_range': (0.9, 1.1)
        }
    }
    
    print(f"Flips: Enabled (horizontal, vertical, both)")
    print(f"Multi-scale: {'Enabled' if args.multi_scale else 'Disabled'}")
    if args.multi_scale:
        print(f"  Scales: {scales}")
    print(f"Rotation: {'Enabled' if args.rotation else 'Disabled'}")
    if args.rotation:
        print(f"  Angles: {angles}°")
    print(f"Color: {'Enabled' if args.color else 'Disabled'}")
    if args.color:
        print(f"  Brightness range: {tta_config['color']['brightness_range']}")
        print(f"  Contrast range: {tta_config['color']['contrast_range']}")
    
    # Calculate total TTA augmentations
    num_augmentations = 4  # Base: original + 3 flips
    if args.multi_scale:
        num_augmentations += len(scales) - 1  # -1 because original scale is already counted
    if args.rotation:
        num_augmentations += len(angles)
    if args.color:
        num_augmentations += 2  # brightness + contrast variations
    
    print(f"\nTotal TTA augmentations per image: {num_augmentations}")
    print("="*70)
    print()
    
    # Generate submission with Advanced TTA
    output_path = args.output or f'outputs/submission_{args.model}_advanced_tta.csv'
    print(f"Generating submission with Advanced TTA...")
    print(f"Output: {output_path}")
    print()
    
    submission = generate_submission(
        model=model,
        test_loader=test_loader,
        device=device,
        output_path=output_path,
        use_tta=True,
        tta_config=tta_config
    )
    
    # Check PLY predictions
    ply_count = (submission['labels'] == 'PLY').sum()
    print(f"\n✓ Submission generated!")
    print(f"  Total predictions: {len(submission)}")
    print(f"  PLY predictions: {ply_count}")
    print(f"  PLY percentage: {100*ply_count/len(submission):.2f}%")
    
    if ply_count == 0:
        print("\n⚠️  WARNING: Still 0 PLY predictions! Post-processing may need adjustment.")
    elif ply_count < 10:
        print(f"\n⚠️  WARNING: Only {ply_count} PLY predictions. Consider increasing top_k in post-processing.")
    else:
        print(f"\n✓ PLY post-processing successful: {ply_count} PLY predictions")
    
    print(f"\nClass distribution:")
    print(submission['labels'].value_counts().sort_index())
    
    print(f"\n{'='*70}")
    print("Advanced TTA submission complete!")
    print(f"{'='*70}")
    print(f"Expected improvement: +0.005 to +0.015 over basic TTA")
    print(f"File saved to: {output_path}")

if __name__ == '__main__':
    main()

