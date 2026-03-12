"""
test_run.py
===========
1-epoch test run for all 3 CNNs + anomaly map visualization.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import subprocess
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = BASE_DIR
os.chdir(BASE_DIR)

LABELS = {
    "eyepacs":  os.path.join(BASE_DIR, "data", "eyepacs_4ch", "eyepacs_anomaly_summary.csv"),
    "glaucoma": os.path.join(BASE_DIR, "data", "glaucoma_combined", "glaucoma_combined_summary.csv"),
    "palm":     os.path.join(BASE_DIR, "data", "palm_4ch", "palm_anomaly_summary.csv"),
}

# ── Step 1: Visualize anomaly maps ────────────────────────────────────────────
print("="*60)
print("ANOMALY MAP VISUALIZATION")
print("="*60)

import pandas as pd

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Anomaly Map Check — Sample Images", fontsize=16)

datasets = [
    ("EyePACS (DR)",  LABELS["eyepacs"],  "eyepacs"),
    ("Glaucoma",      LABELS["glaucoma"], "glaucoma"),
    ("PALM (PM)",     LABELS["palm"],     "palm"),
]

for row_idx, (name, csv_path, key) in enumerate(datasets):
    df     = pd.read_csv(csv_path)
    sample = df.sample(1, random_state=42).iloc[0]

    npy_4ch   = np.load(sample["npy_4ch_path"])   # [4, H, W]
    amap_path = sample["amap_path"]
    amap      = np.load(amap_path)                 # [H, W]

    rgb = npy_4ch[:3].transpose(1, 2, 0)           # [H, W, 3]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    # Col 0: RGB
    axes[row_idx, 0].imshow(rgb)
    axes[row_idx, 0].set_title(f"{name}\nRGB")
    axes[row_idx, 0].axis("off")

    # Col 1: Anomaly map
    im = axes[row_idx, 1].imshow(amap, cmap="jet")
    axes[row_idx, 1].set_title(f"Anomaly Map\nscore={sample['anomaly_score']:.4f}")
    axes[row_idx, 1].axis("off")
    plt.colorbar(im, ax=axes[row_idx, 1])

    # Col 2: Overlay
    axes[row_idx, 2].imshow(rgb)
    axes[row_idx, 2].imshow(amap, cmap="jet", alpha=0.4)
    axes[row_idx, 2].set_title(f"Overlay\nlabel={sample['label']}")
    axes[row_idx, 2].axis("off")

    print(f"  {name}: amap min={amap.min():.4f} max={amap.max():.4f} mean={amap.mean():.4f}")

out_viz = os.path.join(BASE_DIR, "test_anomaly_maps.png")
plt.tight_layout()
plt.savefig(out_viz, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_viz}")

# ── Step 2: 1-epoch training test ─────────────────────────────────────────────
print("\n" + "="*60)
print("1-EPOCH TRAINING TEST")
print("="*60)

runs = [
    ("CNN-1 DR",      "train_cnn1_dr.py",      LABELS["eyepacs"]),
    ("CNN-2 Glaucoma","train_cnn2_glaucoma.py", LABELS["glaucoma"]),
    ("CNN-3 PM",      "train_cnn3_pm.py",       LABELS["palm"]),
]

for name, script, labels_file in runs:
    print(f"\n--- {name} ---")
    result = subprocess.run([
        sys.executable, script,
        "--labels_file", labels_file,
        "--epochs",      "1",
        "--batch_size",  "16",
    ], capture_output=False)
    if result.returncode == 0:
        print(f"  {name}: OK")
    else:
        print(f"  {name}: FAILED (returncode={result.returncode})")

print("\n" + "="*60)
print("TEST RUN COMPLETE")
print(f"Check anomaly maps: {out_viz}")
print("="*60)
