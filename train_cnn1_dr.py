"""
Training script for CNN-1: Diabetic Retinopathy Detection (5-class classification)
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.datasets import EyePACSDataset, create_dataloaders
from src.models import create_cnn1_model
from src.training.trainer import Trainer
from src.utils.transforms import get_train_transforms, get_val_transforms
from src.utils.reproducibility import set_random_seeds, setup_logging, log_hyperparameters
from src.utils.checkpoint import save_checkpoint, get_checkpoint_filename


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train CNN-1 for Diabetic Retinopathy Detection')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to EyePACS dataset directory')
    parser.add_argument('--labels_file', type=str, required=True,
                        help='Path to labels CSV file')
    parser.add_argument('--output_dir', type=str, default='outputs/cnn1_dr',
                        help='Path to save checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Setup logging
    logger = setup_logging(log_dir=f'{args.output_dir}/logs', log_filename='train_cnn1_dr.log')
    
    # Log hyperparameters
    hyperparameters = {
        'model': 'CNN-1 (Diabetic Retinopathy)',
        'num_classes': 5,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'val_split': args.val_split,
        'seed': args.seed,
        'device': args.device,
    }
    log_hyperparameters(logger, hyperparameters)
    
    # Create datasets with transforms
    logger.info("Loading EyePACS dataset...")
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    # Create full dataset with training transforms
    full_dataset = EyePACSDataset(
        image_dir=args.data_dir,
        labels_file=args.labels_file,
        transform=None  # We'll apply transforms in DataLoader
    )
    
    # Create datasets with appropriate transforms
    from src.datasets.data_utils import train_val_split
    train_dataset, val_dataset = train_val_split(full_dataset, args.val_split, args.seed)
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create DataLoaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    logger.info("Creating CNN-1 model (EfficientNet-B3, 5 classes)...")
    model = create_cnn1_model(pretrained=True, dropout=0.4)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create checkpoint directory
    checkpoint_dir = f'{args.output_dir}/checkpoints'
    
    # Define checkpoint save function
    def save_checkpoint_fn(model, optimizer, epoch, metrics, scheduler):
        checkpoint_path = get_checkpoint_filename('cnn1_dr', checkpoint_dir)
        save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path, scheduler)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        num_epochs=args.epochs,
        checkpoint_dir=checkpoint_dir,
        use_amp=True
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(save_checkpoint_fn=save_checkpoint_fn, model_name='cnn1_dr')
    
    logger.info("Training completed!")
    logger.info(f"Best Validation Accuracy: {history['best_val_acc']:.4f}")
    logger.info(f"Best Validation Loss: {history['best_val_loss']:.4f}")


if __name__ == '__main__':
    main()
