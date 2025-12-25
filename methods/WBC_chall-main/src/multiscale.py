"""
Multi-scale training and inference utilities for WBC Challenge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from typing import List, Optional
import numpy as np


def apply_multi_scale_inference(
    model: nn.Module,
    images: torch.Tensor,
    scales: List[int],
    device: torch.device
) -> torch.Tensor:
    """
    Apply multi-scale inference by averaging predictions at different scales.
    
    Args:
        model: Trained model
        images: Input images [B, C, H, W]
        scales: List of scales to test
        device: Device to run inference on
    
    Returns:
        Averaged predictions [B, num_classes]
    """
    model.eval()
    all_predictions = []
    
    original_size = images.shape[-1]  # Assume square images
    
    with torch.no_grad():
        for scale in scales:
            if scale == original_size:
                # No resize needed
                logits = model(images)
            else:
                # Resize to target scale
                resized = transforms.functional.resize(
                    images, 
                    size=scale,
                    interpolation=transforms.InterpolationMode.BILINEAR
                )
                logits = model(resized)
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=1)
            all_predictions.append(probs)
    
    # Average probabilities across scales
    ensemble_probs = torch.stack(all_predictions).mean(0)
    
    return ensemble_probs


def create_multi_scale_transform(base_size: int, scales: List[int], prob: float = 0.5):
    """
    Create a transform that randomly applies multi-scale resizing.
    This should be used in the training augmentation pipeline.
    
    Args:
        base_size: Base image size
        scales: List of scales to randomly choose from
        prob: Probability of applying multi-scale (vs using base_size)
    
    Returns:
        Transform function
    """
    def multi_scale_resize(image):
        if np.random.random() < prob:
            scale = np.random.choice(scales)
            return transforms.functional.resize(
                image,
                size=scale,
                interpolation=transforms.InterpolationMode.BILINEAR
            )
        else:
            return transforms.functional.resize(
                image,
                size=base_size,
                interpolation=transforms.InterpolationMode.BILINEAR
            )
    
    return multi_scale_resize

