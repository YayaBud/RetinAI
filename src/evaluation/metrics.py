"""
Model evaluation and metrics computation.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
import json


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate model on dataset and compute metrics.
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing:
        - accuracy
        - loss
        - predictions
        - true_labels
        - probabilities (for binary classification)
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions
            if outputs.shape[1] == 2:  # Binary classification
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = (probs > 0.5).long()
                all_probs.extend(probs.cpu().numpy())
            else:  # Multi-class
                preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    result = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'predictions': all_preds,
        'true_labels': all_labels,
    }
    
    if all_probs:
        result['probabilities'] = np.array(all_probs)
    
    return result



def compute_classification_metrics(y_true, y_pred, num_classes):
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        Dictionary with precision, recall, f1, confusion matrix
    """
    # Handle edge case: all predictions are the same class
    unique_preds = np.unique(y_pred)
    
    if num_classes == 2:
        # Binary classification
        # Use zero_division=0 to handle undefined metrics
        precision = precision_score(y_true, y_pred, zero_division=0.0)
        recall = recall_score(y_true, y_pred, zero_division=0.0)
        f1 = f1_score(y_true, y_pred, zero_division=0.0)
    else:
        # Multi-class classification
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0.0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0.0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0.0)
        
        # Per-class metrics
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0.0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0.0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0.0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
    }
    
    if num_classes > 2:
        metrics['per_class_metrics'] = {
            'precision': per_class_precision.tolist(),
            'recall': per_class_recall.tolist(),
            'f1_score': per_class_f1.tolist(),
        }
    
    return metrics



def compute_binary_metrics(y_true, y_scores):
    """
    Compute binary classification metrics including ROC-AUC.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
        
    Returns:
        Dictionary with AUC, sensitivity, specificity, ROC curve
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Compute predictions at 0.5 threshold
    y_pred = (y_scores > 0.5).astype(int)
    
    # Compute confusion matrix for sensitivity/specificity
    cm = confusion_matrix(y_true, y_pred)
    
    # Handle edge cases
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        # Only one class present
        sensitivity = 0.0
        specificity = 0.0
    
    return {
        'auc': float(roc_auc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
        }
    }



def generate_confusion_matrix(y_true, y_pred, num_classes):
    """
    Generate confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        Confusion matrix as numpy array
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return cm



def save_evaluation_results(results, filepath):
    """
    Save evaluation results to disk.
    
    Args:
        results: Dictionary of evaluation metrics
        filepath: Path to save results (JSON format)
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            serializable_results[key] = float(value)
        else:
            serializable_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def generate_summary_report(results, model_name, output_path):
    """
    Generate a comprehensive summary report.
    
    Args:
        results: Dictionary of evaluation metrics
        model_name: Name of the model
        output_path: Path to save the report
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"Evaluation Report: {model_name}\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic metrics
        f.write(f"Accuracy: {results.get('accuracy', 'N/A'):.4f}\n")
        f.write(f"Loss: {results.get('loss', 'N/A'):.4f}\n\n")
        
        # Classification metrics
        if 'precision' in results:
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall: {results['recall']:.4f}\n")
            f.write(f"F1-Score: {results['f1_score']:.4f}\n\n")
        
        # Binary metrics
        if 'auc' in results:
            f.write(f"AUC: {results['auc']:.4f}\n")
            f.write(f"Sensitivity: {results['sensitivity']:.4f}\n")
            f.write(f"Specificity: {results['specificity']:.4f}\n\n")
        
        # Confusion matrix
        if 'confusion_matrix' in results:
            f.write("Confusion Matrix:\n")
            cm = np.array(results['confusion_matrix'])
            f.write(str(cm) + "\n\n")
        
        # Per-class metrics
        if 'per_class_metrics' in results:
            f.write("Per-Class Metrics:\n")
            for metric_name, values in results['per_class_metrics'].items():
                f.write(f"  {metric_name}: {values}\n")
