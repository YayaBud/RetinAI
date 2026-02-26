"""
Checkpoint management for saving and loading model states.
"""

import os
import torch


def save_checkpoint(model, optimizer, epoch, metrics, filepath, scheduler=None):
    """
    Save model checkpoint with training state.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer instance
        epoch: Current epoch number
        metrics: Dictionary of training/validation metrics
        filepath: Path to save checkpoint
        scheduler: Optional learning rate scheduler
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Prepare checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    # Add scheduler state if provided
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Add learning rate
    checkpoint['learning_rate'] = optimizer.param_groups[0]['lr']
    
    # Save checkpoint atomically using temporary file
    temp_filepath = filepath + '.tmp'
    try:
        torch.save(checkpoint, temp_filepath)
        # Atomic rename
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(temp_filepath, filepath)
    except Exception as e:
        # Clean up temp file if save failed
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        raise IOError(f"Failed to save checkpoint: {e}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint and optionally restore training state.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        
    Returns:
        Dictionary with epoch and metrics
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    try:
        checkpoint = torch.load(filepath)
    except Exception as e:
        raise IOError(f"Failed to load checkpoint: {e}")
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Return epoch and metrics info
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
        'learning_rate': checkpoint.get('learning_rate', None)
    }


def get_checkpoint_filename(model_name, checkpoint_dir='checkpoints'):
    """
    Generate unique checkpoint filename for a model.
    
    Args:
        model_name: Name of the model (e.g., 'cnn1_dr', 'cnn2_glaucoma', 'cnn3_pm')
        checkpoint_dir: Directory to save checkpoints (default: 'checkpoints')
        
    Returns:
        Full path to checkpoint file
    """
    filename = f"{model_name}_best.pth"
    return os.path.join(checkpoint_dir, filename)
