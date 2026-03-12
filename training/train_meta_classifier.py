"""
Meta-classifier training script.
Stacks outputs from CNN-1 (DR), CNN-2 (Glaucoma), CNN-3 (PM)
plus the anomaly score into a small MLP for final ensemble prediction.

Input features per sample (3841 total):
    - 1280 backbone embeddings from CNN-1 (DR)
    - 1280 backbone embeddings from CNN-2 (Glaucoma)
    - 1280 backbone embeddings from CNN-3 (PM)
    - 1 anomaly score from CSV

Output: 3-class prediction [DR, Glaucoma, PM]
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from src.models import create_cnn1_model, create_cnn2_model, create_cnn3_model
from src.datasets import EyePACSDataset, REFUGEDataset, PALMDataset
from src.utils.checkpoint import load_checkpoint
from src.utils.reproducibility import set_random_seeds, setup_logging


# ── Meta MLP ─────────────────────────────────────────────────────────────────

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


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, dataset, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0, pin_memory=True)

    model.eval()
    all_feats = []

    for images, _ in tqdm(loader, desc="Extracting features", leave=False):
        images = images.to(device)
        feats = model.forward_features(images)          # backbone embeddings (batch, 1280)
        # safety: pool spatial dims if not already flat
        if feats.dim() == 4:
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)
        all_feats.append(feats.cpu().numpy())

    return np.concatenate(all_feats, axis=0)


def build_feature_matrix(dr_feats, glaucoma_feats, pm_feats, anomaly_scores):
    return np.concatenate([
        dr_feats,
        glaucoma_feats,
        pm_feats,
        anomaly_scores.reshape(-1, 1)
    ], axis=1).astype(np.float32)


# ── Meta dataset ──────────────────────────────────────────────────────────────

class MetaDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels   = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ── Training ──────────────────────────────────────────────────────────────────

def train_meta(model, train_loader, val_loader, criterion, optimizer,
               num_epochs, device, patience, logger):

    best_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):

        model.train()
        running_loss, correct, total = 0, 0, 0

        for features, labels in train_loader:

            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        model.eval()

        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for features, labels in val_loader:

                features, labels = features.to(device), labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, pred = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (pred == labels).sum().item()

        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_loader)

        logger.info(
            f"Epoch {epoch+1:3d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:

            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0

            logger.info(f"  → New best val acc: {best_acc:.4f}")

        else:

            epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)

    return best_acc


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dr_checkpoint', required=True)
    parser.add_argument('--glaucoma_checkpoint', required=True)
    parser.add_argument('--pm_checkpoint', required=True)

    parser.add_argument('--dr_labels', required=True)
    parser.add_argument('--glaucoma_labels', required=True)
    parser.add_argument('--pm_labels', required=True)

    parser.add_argument('--output_dir', default='models/checkpoints/meta')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=20)

    parser.add_argument('--device',
        default='cuda' if torch.cuda.is_available() else 'cpu')

    return parser.parse_args()


def main():

    args = parse_args()

    set_random_seeds(42)

    os.makedirs(args.output_dir, exist_ok=True)

    # Delete previous meta checkpoint
    checkpoint_path = os.path.join(args.output_dir, 'meta_classifier_best.pth')

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Old meta checkpoint deleted — OK done")
    else:
        print("No previous meta checkpoint found")

    logger = setup_logging(
        log_dir=f'{args.output_dir}/logs',
        log_filename='train_meta.log'
    )

    device = args.device

    logger.info("Loading CNN backbones...")

    cnn1 = create_cnn1_model(pretrained=False, in_channels=4)
    load_checkpoint(args.dr_checkpoint, cnn1)
    cnn1.to(device).eval()

    cnn2 = create_cnn2_model(pretrained=False, in_channels=4)
    load_checkpoint(args.glaucoma_checkpoint, cnn2)
    cnn2.to(device).eval()

    cnn3 = create_cnn3_model(pretrained=False, in_channels=4)
    load_checkpoint(args.pm_checkpoint, cnn3)
    cnn3.to(device).eval()

    logger.info("Loading datasets...")

    dr_dataset = EyePACSDataset(args.dr_labels, use_4ch=True, split='train')
    glaucoma_dataset = REFUGEDataset(args.glaucoma_labels, use_4ch=True, split='test')
    pm_dataset = PALMDataset(args.pm_labels, use_4ch=True, split='train')

    logger.info("Extracting CNN features...")

    dr_feats_dr = extract_features(cnn1, dr_dataset, device)
    dr_feats_gl = extract_features(cnn2, dr_dataset, device)
    dr_feats_pm = extract_features(cnn3, dr_dataset, device)

    gl_feats_dr = extract_features(cnn1, glaucoma_dataset, device)
    gl_feats_gl = extract_features(cnn2, glaucoma_dataset, device)
    gl_feats_pm = extract_features(cnn3, glaucoma_dataset, device)

    pm_feats_dr = extract_features(cnn1, pm_dataset, device)
    pm_feats_gl = extract_features(cnn2, pm_dataset, device)
    pm_feats_pm = extract_features(cnn3, pm_dataset, device)

    dr_anomaly = dr_dataset.labels_df['anomaly_score'].values.astype(np.float32)
    gl_anomaly = glaucoma_dataset.labels_df['anomaly_score'].values.astype(np.float32)
    pm_anomaly = pm_dataset.labels_df['anomaly_score'].values.astype(np.float32)

    dr_features = build_feature_matrix(dr_feats_dr, dr_feats_gl, dr_feats_pm, dr_anomaly)
    gl_features = build_feature_matrix(gl_feats_dr, gl_feats_gl, gl_feats_pm, gl_anomaly)
    pm_features = build_feature_matrix(pm_feats_dr, pm_feats_gl, pm_feats_pm, pm_anomaly)

    all_features = np.concatenate([dr_features, gl_features, pm_features])

    # ── FEATURE NORMALIZATION
    mean = all_features.mean(axis=0)
    std = all_features.std(axis=0) + 1e-8
    all_features = (all_features - mean) / std

    all_labels = np.array(
        [0]*len(dr_features) +
        [1]*len(gl_features) +
        [2]*len(pm_features)
    )

    # ── SOFTENED CLASS WEIGHTS
    class_counts = np.bincount(all_labels)
    class_weights = np.sqrt(len(all_labels) / (len(class_counts) * class_counts))

    logger.info(f"Class counts: {class_counts}")
    logger.info(f"Class weights: {class_weights}")

    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    dataset = MetaDataset(all_features, all_labels)

    n_total = len(dataset)
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size)

    meta_model = MetaMLP().to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    optimizer = optim.AdamW(meta_model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = train_meta(
        meta_model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        args.epochs,
        device,
        args.patience,
        logger
    )

    checkpoint_path = os.path.join(args.output_dir, 'meta_classifier_best.pth')

    torch.save({
        'model_state_dict': meta_model.state_dict(),
        'best_val_acc': best_acc
    }, checkpoint_path)

    logger.info(f"Meta classifier saved: {checkpoint_path}")
    logger.info(f"Best Val Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
