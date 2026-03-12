"""
prepare_glaucoma_combined.py
=============================
Combines REFUGE + G1020 + ORIGA glaucoma datasets into a single
unified 4-channel dataset ready for CNN-2 training.

Steps:
  1. Parse labels from all 3 datasets
  2. Generate anomaly maps using best.pt diffusion model
  3. Create 4-channel .npy files (RGB + anomaly map)
  4. Merge into single CSV with stratified train/val/test split
  5. Print final class distribution summary

Usage:
    python3 prepare_glaucoma_combined.py

Output:
    /home/amr3/Downloads/RetinAI/data/glaucoma_combined/
        glaucoma/          ← glaucoma images (symlinked)
        non_glaucoma/      ← non_glaucoma images (symlinked)
        4ch/               ← 4-channel .npy files
        anomaly_maps/      ← PNG anomaly maps
        glaucoma_combined_summary.csv
"""

import os
import sys
import json
import csv
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.amp import autocast
from sklearn.model_selection import train_test_split
from diffusers import UNet2DModel, DDPMScheduler

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE          = os.path.dirname(os.path.abspath(__file__))
ARCHIVE       = os.path.join(BASE, "data", "archive")
OUT_DIR       = os.path.join(BASE, "data", "glaucoma_combined")
CHECKPOINT    = os.path.join(BASE, "models", "checkpoints_diffusion", "checkpoints", "best.pt")

REFUGE_SPLITS = ["train", "val", "test"]
REFUGE_BASE   = os.path.join(ARCHIVE, "REFUGE")
G1020_BASE    = os.path.join(ARCHIVE, "G1020")
ORIGA_BASE    = os.path.join(ARCHIVE, "ORIGA")

IMAGE_SIZE    = 256
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE   = "cuda" if DEVICE == "cuda" else "cpu"
N_SAMPLES     = 5       # noise samples to average for anomaly map
TIMESTEP      = 500
SEED          = 42

# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path):
    print(f"Loading diffusion model from: {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    model = UNet2DModel(
        sample_size=IMAGE_SIZE,
        in_channels=3, out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D","DownBlock2D","AttnDownBlock2D","AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D","AttnUpBlock2D","UpBlock2D","UpBlock2D"),
    ).to(DEVICE)

    state_dict = ckpt['model_state_dict']
    cleaned    = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    model.eval()
    epoch = ckpt.get('epoch', '?')
    loss  = ckpt.get('best_val_loss', '?')
    print(f"  Loaded epoch={epoch}, best_val_loss={loss}")
    return model

# ── Transform ─────────────────────────────────────────────────────────────────

def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

# ── Anomaly map generation ────────────────────────────────────────────────────

@torch.no_grad()
def generate_anomaly_map(model, scheduler, image_tensor):
    t  = torch.tensor([TIMESTEP], device=DEVICE).long()
    ac = scheduler.alphas_cumprod[TIMESTEP].to(DEVICE)

    residuals = []
    for _ in range(N_SAMPLES):
        noise = torch.randn_like(image_tensor)
        noisy = scheduler.add_noise(image_tensor, noise, t)
        with autocast(device_type=DEVICE_TYPE):
            pred_noise = model(noisy, t).sample
        recon    = (noisy - (1 - ac).sqrt() * pred_noise.float()) / ac.sqrt()
        recon    = recon.clamp(-1, 1)
        orig_np  = ((image_tensor.squeeze().permute(1,2,0).cpu().float().numpy()+1)/2).clip(0,1)
        recon_np = ((recon.squeeze().permute(1,2,0).cpu().float().numpy()+1)/2).clip(0,1)
        residuals.append(np.abs(orig_np - recon_np).mean(axis=2))

    return np.mean(residuals, axis=0), orig_np  # [H,W], [H,W,3]

# ── Parse dataset labels ──────────────────────────────────────────────────────

def parse_refuge():
    """Parse REFUGE train+val only (test has no labels). Returns list of (img_path, label_str)."""
    records = []
    for split in ["train", "val"]:  # test split has no Label field
        index_path = os.path.join(REFUGE_BASE, split, "index.json")
        img_dir    = os.path.join(REFUGE_BASE, split, "Images")
        if not os.path.exists(index_path):
            print(f"  REFUGE {split}: index.json not found, skipping")
            continue
        with open(index_path) as f:
            index = json.load(f)
        for entry in index.values():
            if "Label" not in entry:
                continue  # skip entries without label
            img_path = os.path.join(img_dir, entry["ImgName"])
            if not os.path.exists(img_path):
                continue
            label = "glaucoma" if entry["Label"] == 1 else "non_glaucoma"
            records.append((img_path, label, "REFUGE"))
    print(f"  REFUGE: {len(records)} images")
    return records


def parse_g1020():
    """Parse G1020 using G1020.csv. Returns list of (img_path, label_str)."""
    records  = []
    csv_path = os.path.join(G1020_BASE, "G1020.csv")
    img_dir  = os.path.join(G1020_BASE, "Images")
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(img_dir, row["imageID"])
            if not os.path.exists(img_path):
                continue
            label = "glaucoma" if int(row["binaryLabels"]) == 1 else "non_glaucoma"
            records.append((img_path, label, "G1020"))
    print(f"  G1020: {len(records)} images")
    return records


def parse_origa():
    """Parse ORIGA using OrigaList.csv. Returns list of (img_path, label_str)."""
    records  = []
    csv_path = os.path.join(ORIGA_BASE, "OrigaList.csv")
    img_dir  = os.path.join(ORIGA_BASE, "Images")
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(img_dir, row["Filename"])
            if not os.path.exists(img_path):
                continue
            label = "glaucoma" if int(row["Glaucoma"]) == 1 else "non_glaucoma"
            records.append((img_path, label, "ORIGA"))
    print(f"  ORIGA: {len(records)} images")
    return records

# ── Process single image ──────────────────────────────────────────────────────

def process_image(img_path, label, source, model, scheduler, transform,
                  out_4ch_dir, out_amap_dir):
    """
    Generate anomaly map + 4ch npy for one image.
    Returns (npy_path, amap_path, anomaly_score) or None on failure.
    """
    try:
        img    = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        amap, orig_np = generate_anomaly_map(model, scheduler, tensor)

        # Build output filename — prefix with source to avoid collisions
        basename    = os.path.splitext(os.path.basename(img_path))[0]
        safe_name   = f"{source}_{basename}"

        # 4-channel npy: stack RGB (HxWx3) + anomaly map (HxWx1)
        amap_resized = np.expand_dims(amap, axis=2)         # [H,W,1]
        four_ch      = np.concatenate([orig_np, amap_resized], axis=2)           # [H,W,4]
        four_ch      = np.transpose(four_ch, (2, 0, 1)).astype(np.float32)      # [4,H,W]

        npy_path  = os.path.join(out_4ch_dir,  label, f"{safe_name}_4ch.npy")
        amap_path = os.path.join(out_amap_dir, label, f"{safe_name}_anomaly.npy")

        os.makedirs(os.path.dirname(npy_path),  exist_ok=True)
        os.makedirs(os.path.dirname(amap_path), exist_ok=True)

        np.save(npy_path,  four_ch)
        np.save(amap_path, amap)

        anomaly_score = float(amap.mean())
        return npy_path, amap_path, anomaly_score

    except Exception as e:
        print(f"  Error processing {img_path}: {e}")
        return None

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_4ch_dir  = os.path.join(OUT_DIR, "4ch")
    out_amap_dir = os.path.join(OUT_DIR, "anomaly_maps")

    # ── Enable Flash SDPA for Blackwell ───────────────────────────────────
    if DEVICE == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)

    # ── Parse all datasets ────────────────────────────────────────────────
    print("\nParsing datasets...")
    records = []
    records.extend(parse_refuge())
    records.extend(parse_g1020())
    records.extend(parse_origa())

    total     = len(records)
    n_glau    = sum(1 for r in records if r[1] == "glaucoma")
    n_nonglau = sum(1 for r in records if r[1] == "non_glaucoma")
    print(f"\nTotal: {total} images | glaucoma: {n_glau} | non_glaucoma: {n_nonglau}")

    # ── Load model ────────────────────────────────────────────────────────
    model     = load_model(CHECKPOINT)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    transform = get_transform()

    # ── Process all images ────────────────────────────────────────────────
    print(f"\nGenerating anomaly maps + 4ch npy files...")
    print(f"Device: {DEVICE} | Timestep: {TIMESTEP} | N-samples: {N_SAMPLES}\n")

    rows = []  # for final CSV
    for img_path, label, source in tqdm(records, desc="Processing"):
        result = process_image(
            img_path, label, source,
            model, scheduler, transform,
            out_4ch_dir, out_amap_dir
        )
        if result is None:
            continue
        npy_path, amap_path, anomaly_score = result
        rows.append({
            "image_path":   img_path,
            "npy_4ch_path": npy_path,
            "amap_path":    amap_path,
            "label":        label,
            "source":       source,
            "anomaly_score": anomaly_score,
            "split":        ""  # filled below
        })

    print(f"\nProcessed {len(rows)}/{total} images successfully")

    # ── Stratified train/val/test split ───────────────────────────────────
    print("\nCreating stratified split (70/15/15)...")
    all_idx    = list(range(len(rows)))
    all_labels = [r["label"] for r in rows]

    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.30,
        stratify=all_labels,
        random_state=SEED
    )
    temp_labels = [all_labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50,
        stratify=temp_labels,
        random_state=SEED
    )

    for i in train_idx: rows[i]["split"] = "train"
    for i in val_idx:   rows[i]["split"] = "val"
    for i in test_idx:  rows[i]["split"] = "test"

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "glaucoma_combined_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_path", "npy_4ch_path", "amap_path",
            "label", "source", "anomaly_score", "split"
        ])
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("DONE — Combined Glaucoma Dataset Summary")
    print("="*60)
    for split in ["train", "val", "test"]:
        split_rows = [r for r in rows if r["split"] == split]
        n_g  = sum(1 for r in split_rows if r["label"] == "glaucoma")
        n_ng = sum(1 for r in split_rows if r["label"] == "non_glaucoma")
        print(f"  {split:<6}: {len(split_rows):>4} total | glaucoma: {n_g:>3} | non_glaucoma: {n_ng:>4}")

    print(f"\nCSV saved to: {csv_path}")
    print(f"4ch npy:      {out_4ch_dir}")
    print(f"Anomaly maps: {out_amap_dir}")
    print("\nNext step: update train_cnn2_glaucoma.py --labels_file to use this CSV")


if __name__ == "__main__":
    main()
