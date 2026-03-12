"""
EfficientNet-B3 based models for retinal disease classification.
Supports 4-channel input: [R, G, B, AnomalyMap] from diffusion model.
"""

import torch
import torch.nn as nn
import timm


def create_efficientnet_model(num_classes, pretrained=True, dropout=0.4, in_channels=4):
    """
    Creates EfficientNet-B3 model with custom classification head.
    Supports 4-channel input [R, G, B, Anomaly] from the diffusion bridge.

    Args:
        num_classes: Number of output classes (5 for DR, 2 for others)
        pretrained:  Whether to load ImageNet weights (default: True)
        dropout:     Dropout probability for regularization (default: 0.4)
        in_channels: Input channels — use 4 for anomaly-guided training,
                     3 for standard RGB (default: 4)

    Returns:
        PyTorch model with EfficientNet-B3 backbone
    """
    # FIX 1: this line was completely missing — the model was never created
    # FIX 2: num_classes=0 + global_pool='avg' removes timm's default head
    #         and keeps the built-in global avg pool so the tensor reaching
    #         our classifier is already [N, num_features] (flat), not [N, C, H, W]
    model = timm.create_model(
        'efficientnet_b3',
        pretrained=pretrained,
        num_classes=0,
        global_pool='avg'
    )

    # ── 4-channel input patch ─────────────────────────────────────────────
    if in_channels != 3:
        orig_conv = model.conv_stem

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False
        )

        with torch.no_grad():
            # Copy pretrained RGB weights into first 3 channels
            new_conv.weight[:, :3, :, :] = orig_conv.weight
            # Init anomaly channel as tiny copy of red channel so the network
            # starts RGB-dominant and gradually learns the anomaly signal
            for ch in range(3, in_channels):
                new_conv.weight[:, ch:ch+1, :, :] = orig_conv.weight[:, :1, :, :] * 0.01

        model.conv_stem = new_conv
        print(f"[EfficientNet] First conv patched: 3 → {in_channels} input channels")
    # ─────────────────────────────────────────────────────────────────────

    # FIX 3: num_features must be read AFTER the if-block, not inside it,
    #         so it is always assigned regardless of in_channels value
    num_features = model.num_features

    # FIX 4: removed AdaptiveAvgPool2d and Flatten — timm already outputs
    #         [N, num_features] flat tensors before the classifier.
    #         Having an extra pool produced [N, num_features, 1, 1] which
    #         caused BatchNorm1d to crash during training.
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(num_features),
        nn.Dropout(p=dropout),
        nn.Linear(num_features, num_classes)
    )

    for param in model.parameters():
        param.requires_grad = True

    return model


def create_cnn1_model(pretrained=True, dropout=0.4, in_channels=4):
    """CNN-1: Diabetic Retinopathy detection (5 classes)."""
    return create_efficientnet_model(
        num_classes=5, pretrained=pretrained,
        dropout=dropout, in_channels=in_channels
    )


def create_cnn2_model(pretrained=True, dropout=0.4, in_channels=4):
    """CNN-2: Glaucoma detection (binary classification)."""
    return create_efficientnet_model(
        num_classes=2, pretrained=pretrained,
        dropout=dropout, in_channels=in_channels
    )


def create_cnn3_model(pretrained=True, dropout=0.4, in_channels=4):
    """CNN-3: Pathologic Myopia detection (binary classification)."""
    return create_efficientnet_model(
        num_classes=2, pretrained=pretrained,
        dropout=dropout, in_channels=in_channels
    )