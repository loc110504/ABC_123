#!/usr/bin/env python3
"""
Regenerate submission with PLY post-processing fixes.

This script loads an existing model checkpoint and regenerates the submission
with post-processing to ensure PLY gets predictions.
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
from src.dataset import create_data_loaders, merge_all_training_data, create_stratified_split
from src.models import create_model
from src.submission import generate_submission

def main():
    parser = argparse.ArgumentParser(description='Regenerate submission with PLY fixes')
    parser.add_argument('--model', type=str, required=True,
                        choices=['resnet50', 'swin_tiny', 'efficientnet_b3', 'efficientnet_b4', 'convnext_tiny',
                                 'resnext50', 'resnext101', 'efficientnetv2', 'regnet', 'deit', 'vit', 'maxvit', 'coatnet'],
                        help='Model name')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (e.g., outputs/checkpoints/pseudo_labeling_resnet50_iter5/best.pth)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output submission path (default: outputs/submission_{model}_ply_fixed.csv)')
    parser.add_argument('--use_old_split', action='store_true',
                        help='Use old split method (phase1+phase2_train / phase2_eval)')
    parser.add_argument('--use_tta', action='store_true', default=True,
                        help='Use test-time augmentation (default: True)')
    
    args = parser.parse_args()
    
    # Load config
    config_name = args.model
    os.environ['WBC_MODEL_CONFIG'] = config_name
    config = Config(project_root=str(project_root))
    
    device = config.device
    print(f"Device: {device}")
    
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
    print(f"\nLoading model from {args.checkpoint}...")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create model
    model = create_model(config.config, num_classes=13)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded model state dict")
    else:
        # If checkpoint is just the state dict
        model.load_state_dict(checkpoint)
        print(f"  Loaded state dict directly")
    
    model = model.to(device)
    model.eval()
    print(f"  Model loaded and set to evaluation mode")
    
    # Generate submission with PLY post-processing
    output_path = args.output or f'outputs/submission_{args.model}_ply_fixed.csv'
    print(f"\nGenerating submission with PLY post-processing...")
    print(f"Output: {output_path}")
    
    submission = generate_submission(
        model=model,
        test_loader=test_loader,
        device=device,
        output_path=output_path,
        use_tta=args.use_tta
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

if __name__ == '__main__':
    import os
    main()

