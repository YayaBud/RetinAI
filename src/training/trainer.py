"""
Training loop engine for CNN models.
"""

import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np


class Trainer:
    """
    Handles model training with validation and checkpointing.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        device: Device to train on (cuda/cpu)
        num_epochs: Number of training epochs
        checkpoint_dir: Directory to save checkpoints
        use_amp: Whether to use automatic mixed precision
    """
    
    def __init__(self, model, train_loader, val_loader, criterion, 
                 optimizer, scheduler, device, num_epochs, 
                 checkpoint_dir='checkpoints', use_amp=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.use_amp = use_amp
        
        # Initialize mixed precision training
        self.scaler = GradScaler() if use_amp else None
        
        # Metrics tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        
        # Move model to device
        self.model.to(self.device)

    
    def train_epoch(self):
        """
        Execute one training epoch.
        
        Returns:
            Dictionary with epoch metrics (loss, accuracy)
        """
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Check for NaN/Inf loss
                if not torch.isfinite(loss):
                    raise ValueError(f"Loss is {loss.item()} at batch {batch_idx}. Training stopped.")
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Check for NaN/Inf loss
                if not torch.isfinite(loss):
                    raise ValueError(f"Loss is {loss.item()} at batch {batch_idx}. Training stopped.")
                
                loss.backward()
                self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }

    
    def validate(self):
        """
        Run validation and return metrics.
        
        Returns:
            Dictionary with validation metrics (loss, accuracy)
        """
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = correct / total
        
        return {
            'loss': val_loss,
            'accuracy': val_acc
        }

    
    def train(self, save_checkpoint_fn=None, model_name='model'):
        """
        Execute full training loop.
        
        Args:
            save_checkpoint_fn: Function to save checkpoints
            model_name: Name of the model for checkpoint naming
            
        Returns:
            Dictionary with training history
        """
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 50)
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            self.train_loss_history.append(train_metrics['loss'])
            
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            self.val_loss_history.append(val_metrics['loss'])
            self.val_acc_history.append(val_metrics['accuracy'])
            
            print(f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Learning Rate: {current_lr:.6f}")
            
            # Track best validation performance
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_loss = val_metrics['loss']
                print(f"New best validation accuracy: {self.best_val_acc:.4f}")
                
                # Save checkpoint if function provided
                if save_checkpoint_fn is not None:
                    metrics = {
                        'best_val_acc': self.best_val_acc,
                        'best_val_loss': self.best_val_loss,
                        'train_loss_history': self.train_loss_history,
                        'val_loss_history': self.val_loss_history,
                        'val_acc_history': self.val_acc_history,
                    }
                    save_checkpoint_fn(
                        self.model,
                        self.optimizer,
                        epoch,
                        metrics,
                        self.scheduler
                    )
        
        print("\nTraining completed!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.4f}")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        
        return {
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
            'val_acc_history': self.val_acc_history,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
        }
