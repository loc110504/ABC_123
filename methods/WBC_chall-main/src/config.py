"""
Configuration management for WBC Challenge.
Handles environment detection (local vs Kaggle) and config loading.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
import torch


class Config:
    """Configuration manager for WBC Challenge."""
    
    def __init__(self, config_path: str = None, project_root: str = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config YAML file. If None, auto-detects environment.
            project_root: Project root directory. If None, auto-detects.
        """
        self.is_kaggle = self._detect_kaggle()
        self.project_root = self._get_project_root(project_root)
        # Get default config path (needs project_root to check file existence)
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self._resolve_paths()
        self._set_paths()
        self._set_device()
    
    def _get_project_root(self, project_root: str = None) -> Path:
        """Get project root directory."""
        if project_root:
            return Path(project_root).resolve()
        
        # Try to find project root by looking for wbc-bench-2026 directory
        current = Path.cwd().resolve()
        
        # If we're in notebooks directory, go up one level
        if current.name == 'notebooks':
            current = current.parent
        
        # Check if wbc-bench-2026 exists here
        if (current / 'wbc-bench-2026').exists():
            return current
        
        # Otherwise, use current directory
        return current
        
    def _detect_kaggle(self) -> bool:
        """Detect if running on Kaggle."""
        return os.path.exists('/kaggle/input') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    
    def _get_default_config_path(self) -> str:
        """Get default config path based on environment."""
        # Check if a model-specific config is requested via environment variable
        model_config = os.environ.get('WBC_MODEL_CONFIG', None)
        if model_config:
            config_path = f'configs/config_{model_config}.yaml'
            # Check if file exists relative to project root (which is set before this is called)
            full_path = self.project_root / config_path
            if full_path.exists():
                print(f"Loading model-specific config: {config_path}")
                return config_path
            else:
                print(f"Warning: Model-specific config not found: {full_path}. Using default config.")
        
        if self.is_kaggle:
            return 'configs/config_kaggle.yaml'
        else:
            return 'configs/config_local.yaml'
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        # Resolve config path relative to project root
        config_path = Path(self.config_path)
        if not config_path.is_absolute():
            config_path = self.project_root / config_path
        
        if not config_path.exists():
            # Return default config if file doesn't exist
            return self._get_default_config()
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _resolve_paths(self):
        """Resolve all data paths relative to project root."""
        if 'data' in self.config:
            data_config = self.config['data']
            for key in data_config:
                if isinstance(data_config[key], str) and not os.path.isabs(data_config[key]):
                    # Resolve relative paths
                    resolved_path = self.project_root / data_config[key]
                    data_config[key] = str(resolved_path.resolve())
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        if self.is_kaggle:
            return {
                'data': {
                    'base_path': '/kaggle/input/wbc-bench-2026',
                    'phase1_dir': '/kaggle/input/wbc-bench-2026/phase1',
                    'phase2_train_dir': '/kaggle/input/wbc-bench-2026/phase2/train',
                    'phase2_eval_dir': '/kaggle/input/wbc-bench-2026/phase2/eval',
                    'phase2_test_dir': '/kaggle/input/wbc-bench-2026/phase2/test',
                    'phase1_labels': '/kaggle/input/wbc-bench-2026/phase1_label.csv',
                    'phase2_train_labels': '/kaggle/input/wbc-bench-2026/phase2_train.csv',
                    'phase2_eval_labels': '/kaggle/input/wbc-bench-2026/phase2_eval.csv',
                    'phase2_test_labels': '/kaggle/input/wbc-bench-2026/phase2_test.csv',
                },
                'model': {
                    'name': 'efficientnet_b3',
                    'num_classes': 13,
                    'pretrained': True,
                },
                'training': {
                    'batch_size': 32,
                    'num_epochs': 25,
                    'learning_rate': 1e-4,
                    'weight_decay': 1e-4,
                    'num_workers': 2,
                    'pin_memory': True,
                },
                'augmentation': {
                    'train': {
                        'resize': 384,
                        'horizontal_flip': 0.5,
                        'vertical_flip': 0.5,
                        'rotation': 15,
                        'color_jitter': 0.2,
                    },
                    'val': {
                        'resize': 384,
                    }
                },
                'paths': {
                    'output_dir': '/kaggle/working',
                    'checkpoint_dir': '/kaggle/working/checkpoints',
                    'submission_dir': '/kaggle/working',
                }
            }
        else:
            # These will be resolved relative to project root
            return {
                'data': {
                    'base_path': 'wbc-bench-2026',
                    'phase1_dir': 'wbc-bench-2026/phase1',
                    'phase2_train_dir': 'wbc-bench-2026/phase2/train',
                    'phase2_eval_dir': 'wbc-bench-2026/phase2/eval',
                    'phase2_test_dir': 'wbc-bench-2026/phase2/test',
                    'phase1_labels': 'wbc-bench-2026/phase1_label.csv',
                    'phase2_train_labels': 'wbc-bench-2026/phase2_train.csv',
                    'phase2_eval_labels': 'wbc-bench-2026/phase2_eval.csv',
                    'phase2_test_labels': 'wbc-bench-2026/phase2_test.csv',
                },
                'model': {
                    'name': 'efficientnet_b3',
                    'num_classes': 13,
                    'pretrained': True,
                },
                'training': {
                    'batch_size': 32,
                    'num_epochs': 25,
                    'learning_rate': 1e-4,
                    'weight_decay': 1e-4,
                    'num_workers': 4,
                    'pin_memory': True,
                },
                'augmentation': {
                    'train': {
                        'resize': 384,
                        'horizontal_flip': 0.5,
                        'vertical_flip': 0.5,
                        'rotation': 15,
                        'color_jitter': 0.2,
                    },
                    'val': {
                        'resize': 384,
                    }
                },
                'paths': {
                    'output_dir': 'outputs',
                    'checkpoint_dir': 'outputs/checkpoints',
                    'submission_dir': 'outputs',
                }
            }
    
    def _set_paths(self):
        """Set up output directories."""
        output_dir = Path(self.config['paths']['output_dir'])
        checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        submission_dir = Path(self.config['paths']['submission_dir'])
        
        # Resolve relative paths
        if not output_dir.is_absolute():
            output_dir = self.project_root / output_dir
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = self.project_root / checkpoint_dir
        if not submission_dir.is_absolute():
            submission_dir = self.project_root / submission_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        submission_dir.mkdir(parents=True, exist_ok=True)
        
        # Update config with resolved paths
        self.config['paths']['output_dir'] = str(output_dir)
        self.config['paths']['checkpoint_dir'] = str(checkpoint_dir)
        self.config['paths']['submission_dir'] = str(submission_dir)
    
    def _set_device(self):
        """Set device (CPU/GPU)."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config['device'] = str(self.device)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot notation (e.g., 'model.name')."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access."""
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self.config
    
    def print_config(self):
        """Print current configuration."""
        print(f"Environment: {'Kaggle' if self.is_kaggle else 'Local'}")
        print(f"Device: {self.device}")
        print(f"Config Path: {self.config_path}")
        print("\nConfiguration:")
        print(yaml.dump(self.config, default_flow_style=False))

