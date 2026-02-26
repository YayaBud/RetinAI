"""
Utilities for reproducibility and logging.
"""

import random
import numpy as np
import torch
import logging
import os
from datetime import datetime


def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Set deterministic behavior for CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(log_dir='logs', log_filename=None):
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to save log files (default: 'logs')
        log_filename: Name of log file (default: timestamp-based)
        
    Returns:
        Logger instance
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename if not provided
    if log_filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'training_{timestamp}.log'
    
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filepath}")
    
    return logger


def log_hyperparameters(logger, hyperparameters):
    """
    Log hyperparameters at the start of training.
    
    Args:
        logger: Logger instance
        hyperparameters: Dictionary of hyperparameters
    """
    logger.info("=" * 50)
    logger.info("Hyperparameters:")
    for key, value in hyperparameters.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 50)


def log_epoch_metrics(logger, epoch, num_epochs, train_metrics, val_metrics, lr=None):
    """
    Log metrics after each epoch.
    
    Args:
        logger: Logger instance
        epoch: Current epoch number
        num_epochs: Total number of epochs
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics
        lr: Current learning rate (optional)
    """
    logger.info(f"Epoch [{epoch}/{num_epochs}]")
    logger.info(f"  Train Loss: {train_metrics.get('loss', 'N/A'):.4f}, "
                f"Train Acc: {train_metrics.get('accuracy', 'N/A'):.4f}")
    logger.info(f"  Val Loss: {val_metrics.get('loss', 'N/A'):.4f}, "
                f"Val Acc: {val_metrics.get('accuracy', 'N/A'):.4f}")
    if lr is not None:
        logger.info(f"  Learning Rate: {lr:.6f}")
