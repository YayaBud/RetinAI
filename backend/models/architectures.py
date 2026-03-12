"""
Model architecture definitions — copied from the training repo.
EfficientNet-B3 (4ch) + MetaMLP.
"""

import torch
import torch.nn as nn
import timm


# ── EfficientNet-B3 with 4-channel input ──────────────────────────────────────

def create_efficientnet_model(num_classes, pretrained=False, dropout=0.4, in_channels=4):
    model = timm.create_model(
        'efficientnet_b3',
        pretrained=pretrained,
        num_classes=0,
        global_pool='avg'
    )

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
            new_conv.weight[:, :3, :, :] = orig_conv.weight
            for ch in range(3, in_channels):
                new_conv.weight[:, ch:ch+1, :, :] = orig_conv.weight[:, :1, :, :] * 0.01
        model.conv_stem = new_conv

    num_features = model.num_features

    model.classifier = nn.Sequential(
        nn.BatchNorm1d(num_features),
        nn.Dropout(p=dropout),
        nn.Linear(num_features, num_classes)
    )
    return model


def create_cnn1_model(pretrained=False, dropout=0.4, in_channels=4):
    """CNN-1: Diabetic Retinopathy (5 classes)."""
    return create_efficientnet_model(5, pretrained=pretrained, dropout=dropout, in_channels=in_channels)


def create_cnn2_model(pretrained=False, dropout=0.4, in_channels=4):
    """CNN-2: Glaucoma (2 classes)."""
    return create_efficientnet_model(2, pretrained=pretrained, dropout=dropout, in_channels=in_channels)


def create_cnn3_model(pretrained=False, dropout=0.4, in_channels=4):
    """CNN-3: Pathologic Myopia (2 classes)."""
    return create_efficientnet_model(2, pretrained=pretrained, dropout=dropout, in_channels=in_channels)


# ── Meta MLP ──────────────────────────────────────────────────────────────────

class MetaMLP(nn.Module):
    def __init__(self, input_dim=4609, hidden_dim=512, num_classes=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)
