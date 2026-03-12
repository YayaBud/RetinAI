"""
Utility functions for dataset management.
"""

import numpy as np
from torch.utils.data import Subset


def train_val_split(dataset, val_split=0.2, random_seed=42):
    """
    Split dataset into training and validation sets.
    
    Args:
        dataset: PyTorch Dataset object
        val_split: Fraction of data to use for validation (default: 0.2)
        random_seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple of (train_dataset, val_dataset) as Subset objects
    """
    if not 0 < val_split < 1:
        raise ValueError(f"val_split must be between 0 and 1, got {val_split}")
    
    # Get dataset size
    dataset_size = len(dataset)
    
    if dataset_size == 0:
        raise ValueError("Cannot split empty dataset")
    
    # Create indices
    indices = np.arange(dataset_size)
    
    # Shuffle indices
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    # Calculate split point
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    # Ensure at least one sample in each split
    if train_size == 0 or val_size == 0:
        raise ValueError(f"Dataset too small to split with val_split={val_split}. Dataset size: {dataset_size}")
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Verify no overlap
    assert len(set(train_indices) & set(val_indices)) == 0, "Train and validation sets overlap!"
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    return train_dataset, val_dataset


def get_dataset_indices(subset):
    """
    Get the indices used in a Subset.
    
    Args:
        subset: torch.utils.data.Subset object
        
    Returns:
        List of indices
    """
    return subset.indices



def create_dataloaders(dataset, batch_size=16, val_split=0.2, num_workers=0, random_seed=42):
    """
    Create training and validation DataLoaders.
    
    Args:
        dataset: PyTorch Dataset object
        batch_size: Batch size for DataLoader (default: 16)
        val_split: Fraction of data for validation (default: 0.2)
        num_workers: Number of worker processes (default: 0)
        random_seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader
    
    # Split dataset
    train_dataset, val_dataset = train_val_split(dataset, val_split, random_seed)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
