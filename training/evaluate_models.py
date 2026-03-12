"""
Evaluation script for trained CNN models.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader

from src.datasets import EyePACSDataset, REFUGEDataset, PALMDataset
from src.models import create_cnn1_model, create_cnn2_model, create_cnn3_model
from src.evaluation.metrics import (
    evaluate_model, compute_classification_metrics, compute_binary_metrics,
    generate_confusion_matrix, save_evaluation_results, generate_summary_report
)
from src.utils.transforms import get_val_transforms
from src.utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained CNN models')

    parser.add_argument('--model_type', type=str, required=True,
                        choices=['cnn1_dr', 'cnn2_glaucoma', 'cnn3_pm'])
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--labels_file', type=str, required=True,
                        help='Path to anomaly CSV (image_path, npy_4ch_path, label, split, ...)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--no_4ch', action='store_true',
                        help='Use 3ch RGB mode instead of 4ch anomaly-guided (for ablation)')

    return parser.parse_args()


def main():
    args = parse_args()
    use_4ch = not args.no_4ch
    in_channels = 4 if use_4ch else 3

    print(f"Evaluating {args.model_type}...")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Device: {args.device}")
    print(f"Input mode: {'4ch anomaly-guided' if use_4ch else '3ch RGB'}")

    # In 4ch mode, npy tensors are already preprocessed — no transform needed.
    # In 3ch RGB mode, apply standard val preprocessing.
    val_transform = None if use_4ch else get_val_transforms()

    # FIX 1: correct constructor — (labels_file, transform, use_4ch, split)
    #         old code passed (data_dir, labels_file, transform) → crashed immediately
    # FIX 2: pass in_channels so model arch matches the saved checkpoint
    if args.model_type == 'cnn1_dr':
        dataset = EyePACSDataset(
            labels_file=args.labels_file,
            transform=val_transform,
            use_4ch=use_4ch,
            split='test'
        )
        model = create_cnn1_model(pretrained=False, in_channels=in_channels)
        num_classes = 5

    elif args.model_type == 'cnn2_glaucoma':
        dataset = REFUGEDataset(
            labels_file=args.labels_file,
            transform=val_transform,
            use_4ch=use_4ch,
            split='test'
        )
        model = create_cnn2_model(pretrained=False, in_channels=in_channels)
        num_classes = 2

    elif args.model_type == 'cnn3_pm':
        dataset = PALMDataset(
            labels_file=args.labels_file,
            transform=val_transform,
            use_4ch=use_4ch,
            split='test'
        )
        model = create_cnn3_model(pretrained=False, in_channels=in_channels)
        num_classes = 2

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Test samples: {len(dataset)}")

    print("Loading checkpoint...")
    load_checkpoint(args.checkpoint_path, model)
    model.to(args.device)

    criterion = torch.nn.CrossEntropyLoss()

    print("Evaluating model...")
    eval_results = evaluate_model(model, dataloader, criterion, args.device)

    print(f"Accuracy: {eval_results['accuracy']:.4f}")
    print(f"Loss:     {eval_results['loss']:.4f}")

    classification_metrics = compute_classification_metrics(
        eval_results['true_labels'],
        eval_results['predictions'],
        num_classes
    )

    print(f"Precision: {classification_metrics['precision']:.4f}")
    print(f"Recall:    {classification_metrics['recall']:.4f}")
    print(f"F1-Score:  {classification_metrics['f1_score']:.4f}")

    if num_classes == 2 and 'probabilities' in eval_results:
        binary_metrics = compute_binary_metrics(
            eval_results['true_labels'],
            eval_results['probabilities']
        )
        print(f"AUC:         {binary_metrics['auc']:.4f}")
        print(f"Sensitivity: {binary_metrics['sensitivity']:.4f}")
        print(f"Specificity: {binary_metrics['specificity']:.4f}")
    else:
        binary_metrics = {}

    cm = generate_confusion_matrix(
        eval_results['true_labels'],
        eval_results['predictions'],
        num_classes
    )
    print("\nConfusion Matrix:")
    print(cm)

    all_results = {
        'accuracy': eval_results['accuracy'],
        'loss': eval_results['loss'],
        **classification_metrics,
        **binary_metrics
    }

    os.makedirs(args.output_dir, exist_ok=True)

    results_file = os.path.join(args.output_dir, f'{args.model_type}_results.json')
    save_evaluation_results(all_results, results_file)
    print(f"\nResults saved to: {results_file}")

    report_file = os.path.join(args.output_dir, f'{args.model_type}_report.txt')
    generate_summary_report(all_results, args.model_type, report_file)
    print(f"Report saved to: {report_file}")

    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()