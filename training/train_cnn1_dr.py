import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from src.datasets import EyePACSDataset
from src.models import create_cnn1_model
from src.training.trainer import Trainer
from src.utils.transforms import get_train_transforms, get_val_transforms
from src.utils.reproducibility import set_random_seeds, setup_logging, log_hyperparameters
from src.utils.checkpoint import save_checkpoint, get_checkpoint_filename


LR_CANDIDATES = [4e-4]


def parse_args():
    parser = argparse.ArgumentParser(description='Train CNN-1 for Diabetic Retinopathy Detection')
    parser.add_argument('--labels_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='models/checkpoints/dr')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--no_4ch', action='store_true')
    return parser.parse_args()


def augment_4ch(tensor):
    """Tensor-based augmentation for 4ch npy tensors [4, H, W]."""
    if random.random() > 0.5:
        tensor = TF.hflip(tensor)
    if random.random() > 0.5:
        tensor = TF.vflip(tensor)
    angle = random.uniform(-15, 15)
    tensor = TF.rotate(tensor, angle)
    return tensor


class AugmentedSubset(torch.utils.data.Dataset):
    """Wraps a Subset and applies 4ch augmentation on-the-fly."""
    def __init__(self, subset, augment=False):
        self.subset = subset
        self.augment = augment

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.augment:
            image = augment_4ch(image)
        return image, label


def compute_class_weights(dataset, num_classes, device):
    """Compute inverse-frequency class weights from dataset labels."""
    counts = torch.zeros(num_classes)
    for _, label in dataset:
        counts[label] += 1
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights.to(device)


def run_single_lr(args, lr, use_4ch, in_channels, train_indices, val_indices,
                  train_dataset_full, val_dataset_full, logger):

    lr_tag = f"lr_{str(lr).replace('.', '_').replace('-', 'n')}"
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints', lr_tag)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"\n{'='*50}")
    logger.info(f"Training with lr={lr}")
    logger.info(f"{'='*50}")

    train_subset = Subset(train_dataset_full, train_indices.indices)
    val_subset   = Subset(val_dataset_full,   val_indices.indices)

    # Wrap with augmentation for train only
    train_dataset = AugmentedSubset(train_subset, augment=use_4ch)
    val_dataset   = AugmentedSubset(val_subset,   augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    # Compute class weights from training subset
    logger.info("Computing class weights...")
    class_weights = compute_class_weights(train_dataset, num_classes=5, device=args.device)
    logger.info(f"Class weights: {class_weights.cpu().tolist()}")

    model     = create_cnn1_model(pretrained=True, dropout=0.4, in_channels=in_channels)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    def save_checkpoint_fn(model, optimizer, epoch, metrics, scheduler):
        checkpoint_path = get_checkpoint_filename(f'cnn1_dr_{lr_tag}', checkpoint_dir)
        save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path, scheduler)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=args.device, num_epochs=args.epochs,
        checkpoint_dir=checkpoint_dir, use_amp=True
    )

    history = trainer.train(save_checkpoint_fn=save_checkpoint_fn,
                            model_name=f'cnn1_dr_{lr_tag}')

    result = {
        'lr': lr,
        'best_val_acc': history['best_val_acc'],
        'best_val_loss': history['best_val_loss'],
        'epochs_trained': len(history['val_acc_history']),
    }

    logger.info(f"lr={lr} → best_val_acc={result['best_val_acc']:.4f} "
                f"best_val_loss={result['best_val_loss']:.4f} "
                f"epochs={result['epochs_trained']}")

    return result


def main():
    args = parse_args()
    use_4ch = not args.no_4ch
    in_channels = 4 if use_4ch else 3

    set_random_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(log_dir=f'{args.output_dir}/logs',
                           log_filename='train_cnn1_dr_sweep.log')

    log_hyperparameters(logger, {
        'model': 'CNN-1 Diabetic Retinopathy — LR Sweep',
        'lr_candidates': LR_CANDIDATES,
        'input_mode': f'{in_channels}ch anomaly-guided' if use_4ch else 'RGB only',
        'num_classes': 5,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'seed': args.seed,
        'device': args.device,
    })

    train_transform = None if use_4ch else get_train_transforms()
    val_transform   = None if use_4ch else get_val_transforms()

    logger.info("Loading EyePACS dataset...")

    train_dataset_full = EyePACSDataset(
        labels_file=args.labels_file, transform=train_transform,
        use_4ch=use_4ch, split='train'
    )
    val_dataset_full = EyePACSDataset(
        labels_file=args.labels_file, transform=val_transform,
        use_4ch=use_4ch, split='train'
    )

    n_total = len(train_dataset_full)
    n_val   = int(0.2 * n_total)
    n_train = n_total - n_val

    train_indices, val_indices = torch.utils.data.random_split(
        range(n_total), [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    logger.info(f"Train: {n_train} samples | Val: {n_val} samples")
    logger.info(f"LR sweep: {LR_CANDIDATES}")

    all_results = []

    for lr in LR_CANDIDATES:
        set_random_seeds(args.seed)   # reset seed for fair comparison
        result = run_single_lr(
            args, lr, use_4ch, in_channels,
            train_indices, val_indices,
            train_dataset_full, val_dataset_full,
            logger
        )
        all_results.append(result)

    # ── Save comparison JSON ──────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, 'lr_sweep_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # ── Print comparison table ────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("LR SWEEP RESULTS — CNN-1 Diabetic Retinopathy")
    logger.info("="*60)
    logger.info(f"{'LR':<12} {'Val Acc':>10} {'Val Loss':>10} {'Epochs':>8}")
    logger.info("-"*44)
    for r in sorted(all_results, key=lambda x: x['best_val_acc'], reverse=True):
        logger.info(f"{r['lr']:<12} {r['best_val_acc']:>10.4f} "
                    f"{r['best_val_loss']:>10.4f} {r['epochs_trained']:>8}")

    best = max(all_results, key=lambda x: x['best_val_acc'])
    logger.info(f"\nBEST LR: {best['lr']} → Val Acc: {best['best_val_acc']:.4f}")
    logger.info(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()
