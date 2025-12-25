"""
Model definitions for WBC Challenge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Dict


def create_model(config: Dict, num_classes: int = 13) -> nn.Module:
    """
    Create model based on configuration.
    
    Args:
        config: Configuration dictionary
        num_classes: Number of output classes
    
    Returns:
        PyTorch model
    """
    model_name = config.get('model', {}).get('name', 'efficientnet_b3')
    pretrained = config.get('model', {}).get('pretrained', True)
    dropout = config.get('model', {}).get('dropout', 0.2)
    
    # Get image size from config (used for models that require fixed input size)
    img_size = config.get('augmentation', {}).get('train', {}).get('resize', 384)
    if img_size is None:
        img_size = config.get('augmentation', {}).get('val', {}).get('resize', 384)
    
    if 'efficientnetv2' in model_name.lower() or 'efficientnet_v2' in model_name.lower():
        return create_efficientnetv2(model_name, num_classes, pretrained, dropout)
    elif 'efficientnet' in model_name.lower():
        return create_efficientnet(model_name, num_classes, pretrained, dropout)
    elif 'resnext' in model_name.lower():
        return create_resnext(model_name, num_classes, pretrained, dropout)
    elif 'resnet' in model_name.lower():
        return create_resnet(model_name, num_classes, pretrained, dropout)
    elif 'swin' in model_name.lower():
        return create_swin(model_name, num_classes, pretrained, dropout, img_size=img_size)
    elif 'convnext' in model_name.lower():
        return create_convnext(model_name, num_classes, pretrained, dropout)
    elif 'deit' in model_name.lower():
        return create_deit(model_name, num_classes, pretrained, dropout, img_size=img_size)
    elif 'maxvit' in model_name.lower():
        return create_maxvit(model_name, num_classes, pretrained, dropout, img_size=img_size)
    elif 'coatnet' in model_name.lower():
        return create_coatnet(model_name, num_classes, pretrained, dropout, img_size=img_size)
    elif 'regnet' in model_name.lower():
        return create_regnet(model_name, num_classes, pretrained, dropout)
    elif 'vit' in model_name.lower() or 'vision_transformer' in model_name.lower():
        return create_vit(model_name, num_classes, pretrained, dropout, img_size=img_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def create_efficientnet(
    model_name: str = 'efficientnet_b3',
    num_classes: int = 13,
    pretrained: bool = True,
    dropout: float = 0.2
) -> nn.Module:
    """
    Create EfficientNet model.
    
    Args:
        model_name: Model name (efficientnet_b3, efficientnet_b4, etc.)
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout: Dropout rate
    
    Returns:
        EfficientNet model
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout
    )
    return model


def create_resnet(
    model_name: str = 'resnet50',
    num_classes: int = 13,
    pretrained: bool = True,
    dropout: float = 0.2
) -> nn.Module:
    """
    Create ResNet model.
    
    Args:
        model_name: Model name (resnet50, resnet101, etc.)
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout: Dropout rate
    
    Returns:
        ResNet model
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout
    )
    return model


def create_swin(
    model_name: str = 'swin_tiny_patch4_window7_224',
    num_classes: int = 13,
    pretrained: bool = True,
    dropout: float = 0.2,
    img_size: int = 224
) -> nn.Module:
    """
    Create Swin Transformer model.
    
    Args:
        model_name: Model name (swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, etc.)
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout: Dropout rate
        img_size: Input image size (default: 224, but can be overridden for different sizes like 384)
    
    Returns:
        Swin Transformer model
    """
    # Create model with specified image size
    # Note: Swin models have strict image size requirements by default
    # We need to pass img_size and potentially disable strict checking
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
        img_size=img_size
    )
    
    # Disable strict image size checking to allow different input sizes
    # This is needed when using pretrained models with different input sizes
    if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'strict_img_size'):
        model.patch_embed.strict_img_size = False
    elif hasattr(model, 'strict_img_size'):
        model.strict_img_size = False
    
    return model


def create_convnext(
    model_name: str = 'convnext_tiny',
    num_classes: int = 13,
    pretrained: bool = True,
    dropout: float = 0.2
) -> nn.Module:
    """
    Create ConvNeXt model.
    
    Args:
        model_name: Model name (convnext_tiny, convnext_small, convnext_base, etc.)
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout: Dropout rate
    
    Returns:
        ConvNeXt model
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout
    )
    return model


def create_vit(
    model_name: str = 'vit_base_patch16_224',
    num_classes: int = 13,
    pretrained: bool = True,
    dropout: float = 0.2,
    img_size: int = 224
) -> nn.Module:
    """
    Create Vision Transformer model.
    
    Args:
        model_name: Model name (vit_base_patch16_224, etc.)
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout: Dropout rate
        img_size: Input image size (default: 224)
    
    Returns:
        Vision Transformer model
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
        img_size=img_size
    )
    return model


def create_resnext(
    model_name: str = 'resnext50_32x4d',
    num_classes: int = 13,
    pretrained: bool = True,
    dropout: float = 0.2
) -> nn.Module:
    """
    Create ResNeXt model.
    
    Args:
        model_name: Model name (resnext50_32x4d, resnext101_32x8d, etc.)
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout: Dropout rate
    
    Returns:
        ResNeXt model
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout
    )
    return model


def create_efficientnetv2(
    model_name: str = 'efficientnetv2_rw_s',
    num_classes: int = 13,
    pretrained: bool = True,
    dropout: float = 0.2
) -> nn.Module:
    """
    Create EfficientNet-V2 model.
    
    Args:
        model_name: Model name (efficientnetv2_rw_s, efficientnetv2_rw_m, etc.)
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout: Dropout rate
    
    Returns:
        EfficientNet-V2 model
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout
    )
    return model


def create_regnet(
    model_name: str = 'regnetx_006',
    num_classes: int = 13,
    pretrained: bool = True,
    dropout: float = 0.2
) -> nn.Module:
    """
    Create RegNet model.
    
    Args:
        model_name: Model name (regnetx_006, regnetx_016, etc.)
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout: Dropout rate
    
    Returns:
        RegNet model
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout
    )
    return model


def create_deit(
    model_name: str = 'deit_base_distilled_patch16_224',
    num_classes: int = 13,
    pretrained: bool = True,
    dropout: float = 0.2,
    img_size: int = 224
) -> nn.Module:
    """
    Create DeiT (Data-efficient Image Transformer) model.
    
    Args:
        model_name: Model name (deit_base_distilled_patch16_224, etc.)
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout: Dropout rate
        img_size: Input image size (default: 224)
    
    Returns:
        DeiT model
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
        img_size=img_size
    )
    return model


def create_maxvit(
    model_name: str = 'maxvit_tiny_tf_224',
    num_classes: int = 13,
    pretrained: bool = True,
    dropout: float = 0.2,
    img_size: int = 224
) -> nn.Module:
    """
    Create MaxViT model.
    
    Args:
        model_name: Model name (maxvit_tiny_tf_224, etc.)
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout: Dropout rate
        img_size: Input image size (default: 224)
    
    Returns:
        MaxViT model
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
        img_size=img_size
    )
    return model


def create_coatnet(
    model_name: str = 'coatnet_rmlp_1_rw_224',
    num_classes: int = 13,
    pretrained: bool = True,
    dropout: float = 0.2,
    img_size: int = 224
) -> nn.Module:
    """
    Create CoAtNet model.
    
    Args:
        model_name: Model name (coatnet_rmlp_1_rw_224, etc.)
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout: Dropout rate
        img_size: Input image size (default: 224)
    
    Returns:
        CoAtNet model
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
        img_size=img_size
    )
    return model


def get_class_weights_from_dataset(dataset) -> Optional[torch.Tensor]:
    """
    Get class weights from dataset.
    
    Args:
        dataset: WBCDataset instance
    
    Returns:
        Class weights tensor or None
    """
    if hasattr(dataset, 'get_class_weights'):
        return dataset.get_class_weights()
    return None


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss focuses on hard examples (misclassified samples) by down-weighting
    easy examples. This is particularly useful for rare classes.
    
    Paper: https://arxiv.org/abs/1708.02002
    Formula: FL = -α(1-p)^γ * log(p)
    
    where:
    - α: class balancing factor (can be tensor of per-class weights or single float)
    - γ: focusing parameter (gamma), higher values focus more on hard examples
    - p: predicted probability of the true class
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for each class (tensor) or single float.
                   If None, no class weighting. If 'balanced', use inverse frequency.
            gamma: Focusing parameter. Higher values put more focus on hard examples.
                   Default: 2.0 (from paper)
            reduction: Specifies the reduction to apply: 'none' | 'mean' | 'sum'
            label_smoothing: Label smoothing factor (0.0 to 1.0)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Focal Loss.
        
        Args:
            inputs: Logits from model (batch_size, num_classes)
            targets: True class indices (batch_size,)
        
        Returns:
            Focal loss value
        """
        # Calculate log probabilities
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Get probability of true class (pt)
        # Gather the log probability of the true class for each sample
        pt = torch.exp(log_probs.gather(1, targets.unsqueeze(1)).squeeze(1))
        
        # Calculate cross-entropy loss (for alpha weighting if needed)
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        
        # Compute focal loss: FL = -α(1-p_t)^γ * log(p_t)
        # Note: ce_loss already includes -log(p_t) (or smoothed version)
        # So we use: (1-pt)^γ * ce_loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_loss_function(
    class_weights: Optional[torch.Tensor] = None,
    device: torch.device = None,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0
) -> nn.Module:
    """
    Create loss function with optional class weights, focal loss, and label smoothing.
    
    Args:
        class_weights: Class weights tensor for weighted loss
        device: Device to place weights on
        use_focal_loss: Whether to use Focal Loss instead of CrossEntropyLoss
        focal_gamma: Gamma parameter for Focal Loss (focusing parameter)
        focal_alpha: Alpha parameter for Focal Loss (class balancing).
                     If None, uses class_weights. If 'balanced', uses inverse frequency.
        label_smoothing: Label smoothing factor (0.0 to 1.0)
    
    Returns:
        Loss function
    """
    if use_focal_loss:
        # Use Focal Loss
        if focal_alpha is None and class_weights is not None:
            # Use class_weights as alpha for focal loss
            alpha = class_weights.to(device) if device is not None else class_weights
        elif focal_alpha == 'balanced' and class_weights is not None:
            # Use balanced alpha (inverse frequency)
            alpha = class_weights.to(device) if device is not None else class_weights
        else:
            alpha = focal_alpha.to(device) if focal_alpha is not None and device is not None else focal_alpha
        
        return FocalLoss(
            alpha=alpha,
            gamma=focal_gamma,
            reduction='mean',
            label_smoothing=label_smoothing
        )
    else:
        # Use standard CrossEntropyLoss with optional class weights and label smoothing
        if class_weights is not None and device is not None:
            class_weights = class_weights.to(device)
        
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )

