"""
Training utilities for WBC Challenge.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import time
import gc

from src.evaluation import calculate_metrics


class Trainer:
    """Training manager for WBC Challenge."""
    
    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get config value supporting both Config objects and plain dicts.
        Supports dot notation (e.g., 'paths.checkpoint_dir').
        """
        if isinstance(self.config, dict):
            # Plain dict - handle dot notation manually
            keys = key.split('.')
            value = self.config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        else:
            # Config object (or other object with dot notation support)
            # Try to use its get method if available
            if hasattr(self.config, 'get'):
                return self.config.get(key, default)
            else:
                return default
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config,
        device: torch.device,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object (Config instance or dict)
            device: Device to train on
            class_weights: Class weights for loss function
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        from src.models import create_loss_function
        train_config = self._get_config_value('training', {})
        
        # Get focal loss settings
        focal_config = train_config.get('focal_loss', {})
        use_focal_loss = focal_config.get('enable', False) if isinstance(focal_config, dict) else False
        focal_gamma = focal_config.get('gamma', 2.0) if isinstance(focal_config, dict) else 2.0
        focal_alpha = focal_config.get('alpha', None) if isinstance(focal_config, dict) else None
        
        # Get label smoothing
        label_smoothing = train_config.get('label_smoothing', 0.0)
        
        self.criterion = create_loss_function(
            class_weights=class_weights,
            device=device,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            label_smoothing=label_smoothing
        )
        
        # Optimizer
        train_config = self._get_config_value('training', {})
        # Ensure learning_rate and weight_decay are floats (YAML might parse 1e-4 as string)
        lr = train_config.get('learning_rate', 1e-4)
        wd = train_config.get('weight_decay', 1e-4)
        if isinstance(lr, str):
            lr = float(lr)
        if isinstance(wd, str):
            wd = float(wd)
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(lr),
            weight_decay=float(wd)
        )
        
        # Learning rate warmup
        self.lr_warmup_epochs = train_config.get('lr_warmup_epochs', 0)
        self.base_lr = float(lr)
        
        # Scheduler with warmup support
        num_epochs = train_config.get('num_epochs', 25)
        self.total_epochs = num_epochs
        self.use_warmup = self.lr_warmup_epochs > 0
        
        if self.use_warmup:
            # Warmup + CosineAnnealingLR
            # We'll manually handle warmup, then use cosine annealing
            self.warmup_scheduler = None  # Will be set during training
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs - self.lr_warmup_epochs,
                eta_min=1e-6
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=1e-6
            )
        
        # Mixed precision training
        self.use_mixed_precision = train_config.get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Gradient accumulation
        self.gradient_accumulation_steps = train_config.get('gradient_accumulation_steps', 1)
        
        # Training state
        self.best_val_score = 0.0
        self.best_epoch = 0
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_macro_f1': [],
            'val_balanced_acc': [],
            'learning_rate': [],
            'rare_class_f1': []  # Track rare class F1 scores
        }
        
        # Checkpoint directory
        checkpoint_path = self._get_config_value('paths.checkpoint_dir')
        if checkpoint_path is None:
            # Fallback to default if not specified
            checkpoint_path = 'outputs/checkpoints'
        self.checkpoint_dir = Path(checkpoint_path)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Early stopping
        self.early_stopping_patience = train_config.get('early_stopping_patience', 8)  # Increased default
        self.min_improvement_threshold = train_config.get('min_improvement_threshold', 0.001)
        self.early_stopping_counter = 0
        
        # Learning rate warmup
        self.lr_warmup_epochs = train_config.get('lr_warmup_epochs', 0)
        self.base_lr = float(lr)
        
        # Track rare class F1 scores (classes with indices that are typically rare)
        # Based on class distribution: PLY (index 9), PC (index 8), PMY (index 10)
        self.rare_class_indices = [8, 9, 10]  # PC, PLY, PMY
        self.train_history['rare_class_f1'] = []
    
    def train_epoch(self) -> float:
        """Train for one epoch with gradient accumulation and mixed precision."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Clear GPU cache at start of epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Update progress bar
            current_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            pbar.set_postfix({
                'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                'mem': f'{current_mem:.2f}GB'
            })
        
        # Handle remaining gradients if any
        if num_batches % self.gradient_accumulation_steps != 0:
            if self.use_mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        return total_loss / num_batches
    
    def validate(self) -> Tuple[float, Dict]:
        """Validate model with mixed precision support."""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        # Clear GPU cache before validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validating')
            for batch in pbar:
                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Get predictions
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Calculate metrics
        metrics = calculate_metrics(all_labels, all_preds)
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_score': self.best_val_score,
            'train_history': self.train_history
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model at epoch {epoch} with score {self.best_val_score:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_score = checkpoint.get('best_val_score', 0.0)
        self.train_history = checkpoint.get('train_history', self.train_history)
        return checkpoint['epoch']
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """
        Train model.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
        """
        start_epoch = 0
        
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            print(f"Resumed training from epoch {start_epoch}")
        
        # Clear GPU cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            reserved_mem = torch.cuda.memory_reserved(0) / 1e9
            print(f"GPU Memory: {total_mem:.2f} GB total, {reserved_mem:.2f} GB reserved")
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model: {self._get_config_value('model.name', 'unknown')}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.train_loader.batch_size * self.gradient_accumulation_steps}")
        print(f"Mixed precision: {self.use_mixed_precision}")
        print("-" * 50)
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Track rare class F1 scores
            rare_class_f1_mean = 0.0
            if len(self.rare_class_indices) > 0 and 'per_class_f1' in val_metrics:
                rare_class_f1s = [val_metrics['per_class_f1'][idx] for idx in self.rare_class_indices if idx < len(val_metrics['per_class_f1'])]
                rare_class_f1_mean = np.mean(rare_class_f1s) if rare_class_f1s else 0.0
            
            # Update learning rate (with warmup if enabled)
            if self.use_warmup and epoch < self.lr_warmup_epochs:
                # Warmup phase: linear increase from 0 to base_lr
                warmup_lr = self.base_lr * (epoch + 1) / self.lr_warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            else:
                # Cosine annealing phase
                if epoch == self.lr_warmup_epochs:
                    # Reset scheduler at end of warmup
                    self.scheduler = CosineAnnealingLR(
                        self.optimizer,
                        T_max=self.total_epochs - self.lr_warmup_epochs,
                        eta_min=1e-6
                    )
                if epoch >= self.lr_warmup_epochs:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_macro_f1'].append(val_metrics['macro_f1'])
            self.train_history['val_balanced_acc'].append(val_metrics['balanced_accuracy'])
            self.train_history['learning_rate'].append(current_lr)
            if 'rare_class_f1' in self.train_history:
                self.train_history['rare_class_f1'].append(rare_class_f1_mean)
            
            # Check if best model (with minimum improvement threshold)
            improvement = val_metrics['macro_f1'] - self.best_val_score
            is_best = improvement > self.min_improvement_threshold
            
            if is_best:
                self.best_val_score = val_metrics['macro_f1']
                self.best_epoch = epoch
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Save checkpoint
            save_best_only = self._get_config_value('training.save_best_only', True)
            if not save_best_only or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Macro-F1: {val_metrics['macro_f1']:.4f}")
            print(f"Val Balanced Acc: {val_metrics['balanced_accuracy']:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            if len(self.rare_class_indices) > 0:
                print(f"Rare Class F1 (PC, PLY, PMY): {rare_class_f1_mean:.4f}")
            print(f"Per-class F1: {val_metrics['per_class_f1']}")
            print("-" * 50)
            
            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best score: {self.best_val_score:.4f} at epoch {self.best_epoch+1}")
                break
        
        print(f"\nTraining completed!")
        print(f"Best Val Macro-F1: {self.best_val_score:.4f} at epoch {self.best_epoch+1}")
        
        return self.train_history

