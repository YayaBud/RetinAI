"""
Image preprocessing and augmentation transforms.
"""

from torchvision import transforms


# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_preprocessing_transform():
    """
    Returns preprocessing transform pipeline.
    
    Returns:
        torchvision.transforms.Compose with:
        - Resize to 300×300
        - Convert to tensor
        - Normalize with ImageNet mean/std
    """
    return transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_augmentation_transform(
    hflip_prob=0.5,
    vflip_prob=0.5,
    rotation_degrees=15,
    brightness=0.2,
    contrast=0.2
):
    """
    Returns augmentation transform pipeline for training.
    
    Args:
        hflip_prob: Probability of horizontal flip (default: 0.5)
        vflip_prob: Probability of vertical flip (default: 0.5)
        rotation_degrees: Range of rotation in degrees (default: 15)
        brightness: Brightness jitter factor (default: 0.2)
        contrast: Contrast jitter factor (default: 0.2)
    
    Returns:
        torchvision.transforms.Compose with:
        - Random horizontal flip
        - Random vertical flip
        - Random rotation
        - Color jitter (brightness, contrast)
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=hflip_prob),
        transforms.RandomVerticalFlip(p=vflip_prob),
        transforms.RandomRotation(degrees=rotation_degrees),
        transforms.ColorJitter(brightness=brightness, contrast=contrast),
    ])


def get_train_transforms(
    hflip_prob=0.5,
    vflip_prob=0.5,
    rotation_degrees=15,
    brightness=0.2,
    contrast=0.2
):
    """
    Returns combined training transform pipeline with augmentation and preprocessing.
    
    Args:
        hflip_prob: Probability of horizontal flip (default: 0.5)
        vflip_prob: Probability of vertical flip (default: 0.5)
        rotation_degrees: Range of rotation in degrees (default: 15)
        brightness: Brightness jitter factor (default: 0.2)
        contrast: Contrast jitter factor (default: 0.2)
    
    Returns:
        torchvision.transforms.Compose with augmentation + preprocessing
    """
    return transforms.Compose([
        # Augmentation
        transforms.RandomHorizontalFlip(p=hflip_prob),
        transforms.RandomVerticalFlip(p=vflip_prob),
        transforms.RandomRotation(degrees=rotation_degrees),
        transforms.ColorJitter(brightness=brightness, contrast=contrast),
        # Preprocessing
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transforms():
    """
    Returns validation/test transform pipeline (preprocessing only, no augmentation).
    
    Returns:
        torchvision.transforms.Compose with preprocessing only
    """
    return get_preprocessing_transform()
