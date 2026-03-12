"""
Full evaluation script for all 4 models:
  - CNN-1 Diabetic Retinopathy
  - CNN-2 Glaucoma
  - CNN-3 Pathologic Myopia
  - Meta-Classifier
"""

import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm

from src.datasets import EyePACSDataset, REFUGEDataset, PALMDataset
from src.models import create_cnn1_model, create_cnn2_model, create_cnn3_model
from src.utils.checkpoint import load_checkpoint
from src.utils.reproducibility import set_random_seeds, setup_logging
from train_meta_classifier import MetaMLP, build_feature_matrix, extract_features


@torch.no_grad()
def run_inference(model, dataset, device, batch_size=32):

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0, pin_memory=True)

    model.eval()

    all_probs, all_preds, all_labels = [], [], []

    for images, labels in tqdm(loader, desc='  Inference', leave=False):

        images = images.to(device)

        logits = model(images)

        probs = torch.softmax(logits, dim=1).cpu().numpy()

        preds = np.argmax(probs, axis=1)

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return (
        np.concatenate(all_probs),
        np.concatenate(all_preds),
        np.concatenate(all_labels)
    )


def compute_metrics(labels, preds, probs, class_names, logger, model_name):

    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS — {model_name}")
    logger.info(f"{'='*60}")

    acc = accuracy_score(labels, preds)
    logger.info(f"Accuracy:  {acc:.4f}")

    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)

    logger.info(f"F1 Macro:    {f1_macro:.4f}")
    logger.info(f"F1 Weighted: {f1_weighted:.4f}")

    try:
        if len(class_names) == 2:
            auc = roc_auc_score(labels, probs[:, 1])
            logger.info(f"AUC-ROC:   {auc:.4f}")
        else:
            from sklearn.preprocessing import label_binarize
            labels_bin = label_binarize(labels, classes=list(range(len(class_names))))
            auc = roc_auc_score(labels_bin, probs, multi_class='ovr', average='macro')
            logger.info(f"AUC-ROC (macro OvR): {auc:.4f}")
    except:
        auc = None

    logger.info("\nPer-class Sensitivity & Specificity:")

    cm = confusion_matrix(labels, preds)

    for i, cls in enumerate(class_names):

        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        logger.info(f"  {cls:<20} Sensitivity: {sens:.4f}  Specificity: {spec:.4f}")

    logger.info("\nConfusion Matrix:")

    for row in cm:
        logger.info(row.tolist())

    report = classification_report(labels, preds,
                                   target_names=class_names,
                                   zero_division=0)

    for line in report.split('\n'):
        logger.info(line)

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'auc_roc': auc,
        'confusion_matrix': cm.tolist()
    }


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dr_checkpoint', required=True)
    parser.add_argument('--glaucoma_checkpoint', required=True)
    parser.add_argument('--pm_checkpoint', required=True)
    parser.add_argument('--meta_checkpoint', required=True)

    parser.add_argument('--dr_labels', required=True)
    parser.add_argument('--glaucoma_labels', required=True)
    parser.add_argument('--pm_labels', required=True)

    parser.add_argument('--output_dir', default='evaluation_results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--device',
        default='cuda' if torch.cuda.is_available() else 'cpu')

    return parser.parse_args()


def main():

    args = parse_args()

    set_random_seeds(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    logger = setup_logging(log_dir=args.output_dir,
                           log_filename='evaluation_results.log')

    device = args.device

    all_results = {}

    # CNN-1 DR
    logger.info("\nEvaluating CNN-1 (Diabetic Retinopathy)...")

    dr_full = EyePACSDataset(args.dr_labels, use_4ch=True, split='train')

    n_total = len(dr_full)
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val

    _, val_indices = torch.utils.data.random_split(
        range(n_total),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    dr_eval = Subset(dr_full, val_indices.indices)

    cnn1 = create_cnn1_model(pretrained=False, in_channels=4)
    load_checkpoint(args.dr_checkpoint, cnn1)
    cnn1.to(device)

    dr_probs, dr_preds, dr_labels = run_inference(cnn1, dr_eval, device, args.batch_size)

    dr_anomaly = dr_full.labels_df.iloc[val_indices.indices]['anomaly_score'].values

    all_results['cnn1_dr'] = compute_metrics(
        dr_labels, dr_preds, dr_probs,
        ['Grade0','Grade1','Grade2','Grade3','Grade4'],
        logger,
        "CNN-1 DR"
    )

    # CNN-2 Glaucoma
    logger.info("\nEvaluating CNN-2 (Glaucoma)...")

    glaucoma_eval = REFUGEDataset(args.glaucoma_labels, use_4ch=True, split='test')

    cnn2 = create_cnn2_model(pretrained=False, in_channels=4)
    load_checkpoint(args.glaucoma_checkpoint, cnn2)
    cnn2.to(device)

    glaucoma_probs, glaucoma_preds, glaucoma_labels = run_inference(
        cnn2, glaucoma_eval, device, args.batch_size
    )

    glaucoma_anomaly = glaucoma_eval.labels_df['anomaly_score'].values

    all_results['cnn2_glaucoma'] = compute_metrics(
        glaucoma_labels, glaucoma_preds, glaucoma_probs,
        ['non_glaucoma','glaucoma'],
        logger,
        "CNN-2 Glaucoma"
    )

    # CNN-3 PM
    logger.info("\nEvaluating CNN-3 (Pathologic Myopia)...")

    pm_eval = PALMDataset(args.pm_labels, use_4ch=True, split='val')

    cnn3 = create_cnn3_model(pretrained=False, in_channels=4)
    load_checkpoint(args.pm_checkpoint, cnn3)
    cnn3.to(device)

    pm_probs, pm_preds, pm_labels = run_inference(
        cnn3, pm_eval, device, args.batch_size
    )

    pm_anomaly = pm_eval.labels_df['anomaly_score'].values

    all_results['cnn3_pm'] = compute_metrics(
        pm_labels, pm_preds, pm_probs,
        ['non_pm','pm'],
        logger,
        "CNN-3 PM"
    )

    # META CLASSIFIER
    logger.info("\nEvaluating Meta-Classifier...")

    # extract backbone embeddings (1280-dim each)
    dr_feats_dr = extract_features(cnn1, dr_eval, device)
    dr_feats_gl = extract_features(cnn2, dr_eval, device)
    dr_feats_pm = extract_features(cnn3, dr_eval, device)

    gl_feats_dr = extract_features(cnn1, glaucoma_eval, device)
    gl_feats_gl = extract_features(cnn2, glaucoma_eval, device)
    gl_feats_pm = extract_features(cnn3, glaucoma_eval, device)

    pm_feats_dr = extract_features(cnn1, pm_eval, device)
    pm_feats_gl = extract_features(cnn2, pm_eval, device)
    pm_feats_pm = extract_features(cnn3, pm_eval, device)

    dr_features = build_feature_matrix(dr_feats_dr, dr_feats_gl, dr_feats_pm, dr_anomaly)
    gl_features = build_feature_matrix(gl_feats_dr, gl_feats_gl, gl_feats_pm, glaucoma_anomaly)
    pm_features = build_feature_matrix(pm_feats_dr, pm_feats_gl, pm_feats_pm, pm_anomaly)

    meta_features = np.concatenate([dr_features, gl_features, pm_features])

    # ── match training normalization
    mean = meta_features.mean(axis=0)
    std = meta_features.std(axis=0) + 1e-8
    meta_features = (meta_features - mean) / std

    meta_labels = np.array(
        [0]*len(dr_features) +
        [1]*len(gl_features) +
        [2]*len(pm_features),
        dtype=np.int64
    )

    meta_ckpt = torch.load(args.meta_checkpoint)

    meta_model = MetaMLP(input_dim=4609, hidden_dim=512, num_classes=3, dropout=0.3).to(device)

    meta_model.load_state_dict(meta_ckpt['model_state_dict'])
    meta_model.eval()

    with torch.no_grad():

        features_tensor = torch.from_numpy(meta_features).float().to(device)

        logits = meta_model(features_tensor)

        # ---- CLASS PRIOR CORRECTION ----
        class_counts = np.array([
            len(dr_features),
            len(gl_features),
            len(pm_features)
        ], dtype=np.float32)

        priors = class_counts / class_counts.sum()
        log_priors = torch.log(torch.tensor(priors)).to(device)

        logits = logits - log_priors
        # --------------------------------

        meta_probs = torch.softmax(logits, dim=1).cpu().numpy()

        meta_preds = np.argmax(meta_probs, axis=1)

    all_results['meta_classifier'] = compute_metrics(
        meta_labels, meta_preds, meta_probs,
        ['DR','Glaucoma','PM'],
        logger,
        "Meta Classifier"
    )

    results_path = os.path.join(args.output_dir,'all_results.json')

    with open(results_path,'w') as f:
        json.dump(all_results,f,indent=2)

    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
