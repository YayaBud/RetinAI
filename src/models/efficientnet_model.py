"""
EfficientNet-B3 based models for retinal disease classification.
"""

import torch
import torch.nn as nn
import timm


def create_efficientnet_model(num_classes, pretrained=True, dropout=0.4):
    """
    Creates EfficientNet-B3 model with custom classification head.
    
    Args:
        num_classes: Number of output classes (5 for DR, 2 for others)
        pretrained: Whether to load ImageNet weights (default: True)
        dropout: Dropout probability for regularization (default: 0.4)
        
    Returns:
        PyTorch model with EfficientNet-B3 backbone
    """
    # Load EfficientNet-B3 from timm
    model = timm.create_model('efficientnet_b3', pretrained=pretrained, num_classes=0)
    
    # Get the number of features from the backbone
    num_features = model.num_features
    
    # Create custom classification head
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.BatchNorm1d(num_features),
        nn.Dropout(p=dropout),
        nn.Linear(num_features, num_classes)
    )
    
    # Ensure all parameters are trainable (fine-tuning enabled)
    for param in model.parameters():
        param.requires_grad = True
    
    return model


def create_cnn1_model(pretrained=True, dropout=0.4):
    """
    Create CNN-1 model for Diabetic Retinopathy detection (5 classes).
    
    Args:
        pretrained: Whether to load ImageNet weights (default: True)
        dropout: Dropout probability (default: 0.4)
        
    Returns:
        EfficientNet-B3 model with 5-class output
    """
    return create_efficientnet_model(num_classes=5, pretrained=pretrained, dropout=dropout)


def create_cnn2_model(pretrained=True, dropout=0.4):
    """
    Create CNN-2 model for Glaucoma detection (binary classification).
    
    Args:
        pretrained: Whether to load ImageNet weights (default: True)
        dropout: Dropout probability (default: 0.4)
        
    Returns:
        EfficientNet-B3 model with 2-class output
    """
    return create_efficientnet_model(num_classes=2, pretrained=pretrained, dropout=dropout)


def create_cnn3_model(pretrained=True, dropout=0.4):
    """
    Create CNN-3 model for Pathologic Myopia detection (binary classification).
    
    Args:
        pretrained: Whether to load ImageNet weights (default: True)
        dropout: Dropout probability (default: 0.4)
        
    Returns:
        EfficientNet-B3 model with 2-class output
    """
    return create_efficientnet_model(num_classes=2, pretrained=pretrained, dropout=dropout)
