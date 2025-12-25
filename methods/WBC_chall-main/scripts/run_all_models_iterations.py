#!/usr/bin/env python3
"""
Run pseudo-labeling iterations for all models.

This script runs multiple iterations for all available models in sequence.

Usage:
    python scripts/run_all_models_iterations.py --iterations 2 --start_iter 2
    python scripts/run_all_models_iterations.py --iterations 3 --start_iter 2 --models resnet50 swin_tiny
"""

import subprocess
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Run pseudo-labeling iterations for all models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run iteration 2 for all models
  python scripts/run_all_models_iterations.py --iterations 1 --start_iter 2
  
  # Run iterations 2-3 for all models
  python scripts/run_all_models_iterations.py --iterations 2 --start_iter 2
  
  # Run only specific models
  python scripts/run_all_models_iterations.py --iterations 2 --start_iter 2 --models resnet50 swin_tiny
        """
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=1,
        help='Number of iterations to run for each model (default: 1)'
    )
    
    parser.add_argument(
        '--start_iter',
        type=int,
        default=2,
        help='Starting iteration number (default: 2)'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['resnet50', 'swin_tiny', 'convnext_tiny', 'efficientnet_b3', 'efficientnet_b4',
                 'resnext50', 'resnext101', 'efficientnetv2', 'regnet', 'deit', 'vit', 'maxvit', 'coatnet'],
        default=['resnet50', 'swin_tiny', 'convnext_tiny', 'efficientnet_b3', 'efficientnet_b4'],
        help='Models to run (default: all models)'
    )
    
    parser.add_argument(
        '--confidence_threshold',
        type=float,
        default=0.9,
        help='Initial confidence threshold (default: 0.9)'
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
        help='Disable test-time augmentation'
    )
    
    args = parser.parse_args()
    
    # Get script path
    script_path = Path(__file__).parent / 'pseudo_labeling_iterations.py'
    project_root = Path(__file__).parent.parent
    
    print("="*70)
    print("Running Pseudo-Labeling Iterations for All Models")
    print("="*70)
    print(f"Models: {', '.join(args.models)}")
    print(f"Starting iteration: {args.start_iter}")
    print(f"Number of iterations per model: {args.iterations}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    if args.confidence_increment > 0:
        print(f"Confidence increment: {args.confidence_increment}")
    print("="*70)
    
    results = {}
    
    for model in args.models:
        print(f"\n{'='*70}")
        print(f"Processing: {model.upper()}")
        print(f"{'='*70}")
        
        cmd = [
            sys.executable,
            str(script_path),
            '--model', model,
            '--iterations', str(args.iterations),
            '--start_iter', str(args.start_iter),
            '--confidence_threshold', str(args.confidence_threshold),
            '--confidence_increment', str(args.confidence_increment)
        ]
        
        if args.no_tta:
            cmd.append('--no_tta')
        
        try:
            result = subprocess.run(cmd, cwd=str(project_root), check=True)
            results[model] = 'success'
            print(f"✓ {model.upper()} completed successfully")
        except subprocess.CalledProcessError as e:
            results[model] = 'failed'
            print(f"❌ {model.upper()} failed with error code {e.returncode}")
        except KeyboardInterrupt:
            print(f"\n⚠ Interrupted by user. Stopping execution.")
            results[model] = 'interrupted'
            break
    
    # Summary
    print(f"\n{'='*70}")
    print("Final Summary")
    print(f"{'='*70}")
    for model, status in results.items():
        status_symbol = "✓" if status == 'success' else "❌" if status == 'failed' else "⚠"
        print(f"{status_symbol} {model.upper()}: {status}")
    
    successful = sum(1 for s in results.values() if s == 'success')
    print(f"\nCompleted: {successful}/{len(results)} models successfully")
    
    if successful == len(results):
        print("✓ All models completed successfully!")
    else:
        print("⚠ Some models failed. Check the output above for details.")


if __name__ == '__main__':
    main()

