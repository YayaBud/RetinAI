"""
generate_anomaly_maps.py
========================
BRIDGE SCRIPT: Diffusion Model → CNN Training

This script:
  1. Loads the trained diffusion model (best.pt)
  2. Loops through ALL images in EyePACS / REFUGE / PALM datasets
  3. For each image:
       - Computes a 256x256 anomaly map (multi-timestep averaging)
       - Computes a scalar anomaly score (mean of map)
       - Saves anomaly map as .png alongside original
       - Saves a 4-channel .npy tensor [R, G, B, Anomaly] for CNN input
  4. Saves a summary CSV: image_path, anomaly_score, label

HOW TO RUN:
  python generate_anomaly_maps.py

EXPECTED RUNTIME:
  ~2-4 seconds per image on GPU. For 88k EyePACS images, consider
  running with --dataset eyepacs only first, then glaucoma, then palm.
  Use the DATASET argument at the bottom to control which to run.
"""

import os
import csv
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from glob import glob
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler

# ============================================================
# CONFIGURATION — EDIT THESE PATHS TO MATCH YOUR SYSTEM
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_ROOTS = {
    "eyepacs":  os.path.join(BASE_DIR, "data", "diabetic_retinopathy", "organized"),
    "glaucoma": os.path.join(BASE_DIR, "data", "REFUGE", "organized"),
    "palm":     os.path.join(BASE_DIR, "data", "palm", "organized"),
}

ANOMALY_OUTPUT_ROOTS = {
    "eyepacs":  os.path.join(BASE_DIR, "data", "eyepacs_4ch"),
    "glaucoma": os.path.join(BASE_DIR, "data", "refuge_4ch"),
    "palm":     os.path.join(BASE_DIR, "data", "palm_4ch"),
}

BEST_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "checkpoints_diffusion",
    "checkpoints",
    "best.pt"
)

# Supported image extensions in your datasets
IMG_EXTENSIONS = ("*.jpeg", "*.jpg", "*.png", "*.PNG", "*.JPG", "*.JPEG")

IMAGE_SIZE  = 256
BATCH_SIZE  = 1       # Process one at a time for anomaly map accuracy
NUM_STEPS   = 20      # Timestep averaging steps (higher = more stable, slower)
TIMESTEP    = 30      # Fixed low timestep — captures fine-grained anomalies
device      = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# DIFFUSION MODEL SETUP
# ============================================================

print(f"Loading diffusion model from: {BEST_MODEL}")
print(f"Device: {device}")

diffusion_model = UNet2DModel(
    sample_size=IMAGE_SIZE,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 256),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
).to(device)

diffusion_model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
diffusion_model.eval()

scheduler = DDPMScheduler(num_train_timesteps=1000)

print("Diffusion model loaded successfully.\n")

# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)   # → [-1, 1]
])

# ============================================================
# CORE: ANOMALY MAP COMPUTATION
# ============================================================

def compute_anomaly_map(image_tensor: torch.Tensor, num_steps: int = NUM_STEPS) -> tuple:
    """
    Given a single image tensor of shape [3, H, W] (normalized to [-1,1]),
    returns:
        anomaly_map   : np.ndarray [H, W]  float32, range ~[0, ∞)
        anomaly_score : float scalar — mean of the map (used for CSV summary)

    Method:
        - Add noise at fixed low timestep t=30 (captures structural anomalies)
        - Average noise prediction error over NUM_STEPS random trials
        - Healthy images → low error (diffusion knows how to reconstruct)
        - Diseased images → high error in pathological regions
    """
    image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]
    total_map = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE, device=device)

    t = torch.tensor([TIMESTEP], device=device).long()

    for _ in range(num_steps):
        noise = torch.randn_like(image_tensor)
        noisy = scheduler.add_noise(image_tensor, noise, t)

        with torch.no_grad():
            pred_noise = diffusion_model(noisy, t).sample

        # Per-pixel squared error averaged across RGB channels → [1, H, W]
        error_map = torch.mean((pred_noise - noise) ** 2, dim=1)
        total_map += error_map

    anomaly_map = (total_map / num_steps).squeeze().cpu().numpy()   # [H, W]

    # Normalize map to [0, 1] for saving as PNG
    map_min, map_max = anomaly_map.min(), anomaly_map.max()
    if map_max > map_min:
        anomaly_map_norm = (anomaly_map - map_min) / (map_max - map_min)
    else:
        anomaly_map_norm = anomaly_map

    anomaly_score = float(anomaly_map.mean())

    return anomaly_map_norm, anomaly_score


def build_4ch_tensor(image_tensor: torch.Tensor, anomaly_map_norm: np.ndarray) -> np.ndarray:
    """
    Combines original 3-channel image with anomaly map into a 4-channel tensor.

    image_tensor    : [3, H, W] normalized [-1, 1]
    anomaly_map_norm: [H, W]    normalized [0, 1]

    Returns:
        tensor_4ch : np.ndarray [4, H, W] float32
                     Channels: [R_norm, G_norm, B_norm, Anomaly_norm]

    The EfficientNet-B3 first conv layer is modified to accept 4 channels.
    The anomaly channel is already [0,1], consistent with normalized RGB.
    """
    # Convert image tensor from [-1,1] to [0,1] for consistency
    img_01 = (image_tensor.cpu().numpy() + 1.0) / 2.0   # [3, H, W]

    # Anomaly map [H, W] → [1, H, W]
    anom_ch = anomaly_map_norm[np.newaxis, :, :]         # [1, H, W]

    # Stack to [4, H, W]
    tensor_4ch = np.concatenate([img_01, anom_ch], axis=0).astype(np.float32)

    return tensor_4ch

# ============================================================
# DATASET SCANNER — finds all images + labels from folder tree
# ============================================================

def scan_dataset(root_dir: str) -> list:
    """
    Scans a dataset folder structured as:
        root_dir/
            split/          ← optional (train / val / test)
                class_name/
                    image.jpg

    Returns list of (image_path, label_str, split_str)
    If no split subfolder exists, split_str = "all"
    """
    entries = []

    # Check if root has split folders (train/val/test) or direct class folders
    subdirs = [d for d in os.listdir(root_dir)
               if os.path.isdir(os.path.join(root_dir, d))]

    splits = ["train", "val", "test"]
    has_splits = any(s in subdirs for s in splits)

    if has_splits:
        scan_dirs = [(os.path.join(root_dir, s), s)
                     for s in subdirs if s in splits]
    else:
        scan_dirs = [(root_dir, "all")]

    for split_dir, split_name in scan_dirs:
        class_dirs = [d for d in os.listdir(split_dir)
                      if os.path.isdir(os.path.join(split_dir, d))]

        for cls in class_dirs:
            cls_dir = os.path.join(split_dir, cls)
            images = []
            for ext in IMG_EXTENSIONS:
                images.extend(glob(os.path.join(cls_dir, ext)))

            for img_path in images:
                entries.append((img_path, cls, split_name))

    return entries

# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

def process_dataset(dataset_name: str, src_root: str, out_root: str):
    """
    Processes all images in one dataset:
      - Generates anomaly maps
      - Saves anomaly map PNG
      - Saves 4-channel .npy tensor
      - Writes summary CSV
    """

    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name.upper()}")
    print(f"  Source : {src_root}")
    print(f"  Output : {out_root}")
    print(f"{'='*60}")

    if not os.path.exists(src_root):
        print(f"  [SKIP] Source directory not found: {src_root}")
        return

    entries = scan_dataset(src_root)
    print(f"  Found {len(entries)} images\n")

    # CSV summary file
    csv_path = os.path.join(out_root, f"{dataset_name}_anomaly_summary.csv")
    os.makedirs(out_root, exist_ok=True)

    csv_rows = []

    for img_path, label, split in tqdm(entries, desc=dataset_name):

        try:
            # ── Load image ──────────────────────────────────────
            img_pil = Image.open(img_path).convert("RGB")
            img_tensor = transform(img_pil)           # [3, 256, 256]

            # ── Compute anomaly map ──────────────────────────────
            anomaly_map_norm, anomaly_score = compute_anomaly_map(img_tensor)

            # ── Build 4-channel tensor ───────────────────────────
            tensor_4ch = build_4ch_tensor(img_tensor, anomaly_map_norm)

            # ── Determine output paths ───────────────────────────
            stem = os.path.splitext(os.path.basename(img_path))[0]
            out_dir = os.path.join(out_root, split, label)
            os.makedirs(out_dir, exist_ok=True)

            # Save anomaly map as PNG (for visual inspection)
            amap_path = os.path.join(out_dir, f"{stem}_anomaly.png")
            plt.imsave(amap_path, anomaly_map_norm, cmap="jet")

            # Save 4-channel numpy array (used by CNN DataLoader)
            npy_path = os.path.join(out_dir, f"{stem}_4ch.npy")
            np.save(npy_path, tensor_4ch)

            # ── Log to CSV ───────────────────────────────────────
            csv_rows.append({
                "image_path"   : img_path,
                "npy_4ch_path" : npy_path,
                "amap_path"    : amap_path,
                "label"        : label,
                "split"        : split,
                "anomaly_score": round(anomaly_score, 6),
            })

        except Exception as e:
            print(f"\n  [ERROR] {img_path}: {e}")
            continue

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["image_path", "npy_4ch_path", "amap_path",
                      "label", "split", "anomaly_score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n  Done. {len(csv_rows)} images processed.")
    print(f"  Summary CSV: {csv_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate anomaly maps + 4ch tensors for CNN training"
    )
    parser.add_argument(
        "--dataset",
        choices=["eyepacs", "glaucoma", "palm", "all"],
        default="all",
        help="Which dataset to process (default: all)"
    )
    args = parser.parse_args()

    datasets_to_run = (
        list(DATASET_ROOTS.keys()) if args.dataset == "all"
        else [args.dataset]
    )

    for name in datasets_to_run:
        process_dataset(
            dataset_name=name,
            src_root=DATASET_ROOTS[name],
            out_root=ANOMALY_OUTPUT_ROOTS[name],
        )

    print("\n\nAll datasets processed. Ready for CNN training!")
    print("Next step: train_cnn1_dr.py, train_cnn2_glaucoma.py, train_cnn3_pm.py")