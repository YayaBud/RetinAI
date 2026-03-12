"""
run_on_5090.py
==============
Full pipeline — one script to rule them all.

Steps:
  0. GPU check
  1. Install requirements
  2. Generate 4ch npy files for EyePACS (DR) and PALM datasets
     (Glaucoma combined already done via prepare_glaucoma_combined.py)
  3. Train CNN-1 (Diabetic Retinopathy) — weighted loss for class imbalance
  4. Train CNN-2 (Glaucoma) — new combined dataset, weighted loss
  5. Train CNN-3 (Pathologic Myopia)
  6. Train Meta-Classifier

HOW TO USE:
  python3 run_on_5090.py

  To skip steps already done, set the SKIP_* flags below to True.
"""

import os
import subprocess
import sys
import torch

# ============================================================
# PATHS — everything relative to this script's directory
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = BASE_DIR
DIFFUSION_CHECKPOINT = os.path.join(BASE_DIR, "models", "checkpoints_diffusion", "checkpoints", "best.pt")

LABELS = {
    "eyepacs":  os.path.join(BASE_DIR, "data", "eyepacs_4ch", "eyepacs_anomaly_summary.csv"),
    "glaucoma": os.path.join(BASE_DIR, "data", "glaucoma_combined", "glaucoma_combined_summary.csv"),
    "palm":     os.path.join(BASE_DIR, "data", "palm_4ch", "palm_anomaly_summary.csv"),
}

# Best LRs from sweep
BEST_LR = {
    "dr":       "0.0004",
    "glaucoma": "0.0003",
    "pm":       "0.0003",
}

# CNN checkpoint paths (auto-named by LR sweep)
CHECKPOINTS = {
    "dr":       os.path.join(BASE_DIR, "models", "checkpoints", "dr", "checkpoints", "lr_0_0004", "cnn1_dr_lr_0_0004_best.pth"),
    "glaucoma": os.path.join(BASE_DIR, "models", "checkpoints", "glaucoma", "checkpoints", "lr_0_0003", "cnn2_glaucoma_lr_0_0003_best.pth"),
    "pm":       os.path.join(BASE_DIR, "models", "checkpoints", "pm", "checkpoints", "lr_0_0003", "cnn3_pm_lr_0_0003_best.pth"),
}

# ============================================================
# AUTO-RESUME — replaces manual SKIP_* flags
# Checks each checkpoint: if it exists AND was trained for
# >= MIN_EPOCHS, that step is skipped automatically.
# The 1-epoch test run checkpoints will NOT be skipped.
# To force a re-run, just delete the checkpoint file.
# ============================================================

MIN_EPOCHS = 50

def checkpoint_is_complete(path):
    """Returns True if checkpoint exists and was trained >= MIN_EPOCHS."""
    if not os.path.exists(path):
        return False
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        epochs_trained = len(ckpt.get("val_acc_history", []))
        if epochs_trained >= MIN_EPOCHS:
            print(f"  [AUTO-RESUME] Complete ({epochs_trained} epochs): {os.path.basename(path)}")
            return True
        else:
            print(f"  [AUTO-RESUME] Only {epochs_trained} epochs found (need {MIN_EPOCHS}) — re-training")
            return False
    except Exception as e:
        print(f"  [AUTO-RESUME] Unreadable checkpoint: {e} — re-training")
        return False

SKIP_INSTALL         = True
SKIP_4CH_EYEPACS     = True   # set True if eyepacs_4ch already exists
SKIP_4CH_PALM        = True   # set True if palm_4ch already exists
SKIP_CNN1_DR         = checkpoint_is_complete(CHECKPOINTS["dr"])
SKIP_CNN2_GLAUCOMA   = checkpoint_is_complete(CHECKPOINTS["glaucoma"])
SKIP_CNN3_PM         = checkpoint_is_complete(CHECKPOINTS["pm"])
SKIP_META            = os.path.exists(os.path.join(BASE_DIR, "models", "checkpoints", "meta", "meta_classifier_best.pth"))

# ============================================================
# STEP 0 — GPU CHECK
# ============================================================

print("="*60)
print("STEP 0 — GPU CHECK")
print("="*60)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram     = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU   : {gpu_name}")
    print(f"  VRAM  : {vram:.1f} GB")
    print(f"  CUDA  : {torch.version.cuda}")
else:
    print("  [ERROR] No CUDA GPU detected! Check drivers.")
    sys.exit(1)

os.chdir(BASE_DIR)

# ============================================================
# STEP 1 — INSTALL REQUIREMENTS
# ============================================================

if not SKIP_INSTALL:
    print("\n" + "="*60)
    print("STEP 1 — INSTALLING REQUIREMENTS")
    print("="*60)
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "diffusers",
        "accelerate", "timm", "tqdm",
        "Pillow", "numpy", "matplotlib",
        "pandas", "openpyxl", "scikit-learn",
        "bitsandbytes",
        "--quiet"
    ], check=True)
    print("  Requirements installed.")
else:
    print("\nSTEP 1 — Skipped (SKIP_INSTALL=True)")

# ============================================================
# STEP 2A — GENERATE 4CH NPY FOR EYEPACS (DR)
# ============================================================

if not SKIP_4CH_EYEPACS:
    print("\n" + "="*60)
    print("STEP 2A — GENERATING 4CH NPY: EyePACS (DR)")
    print("="*60)
    print("  This will take a while — 35k images through diffusion model...")
    subprocess.run([
        sys.executable, "generate_4ch_eyepacs.py",
        "--checkpoint",  DIFFUSION_CHECKPOINT,
        "--input_dir",   os.path.join(BASE_DIR, "data", "diabetic_retinopathy", "organized"),
        "--output_dir",  os.path.join(BASE_DIR, "data", "eyepacs_4ch"),
        "--n-samples",   "5",
        "--timestep",    "500",
    ], check=True)
    print("  EyePACS 4ch done.")
else:
    print("\nSTEP 2A — Skipped (SKIP_4CH_EYEPACS=True)")

# ============================================================
# STEP 2B — GENERATE 4CH NPY FOR PALM (Pathologic Myopia)
# ============================================================

if not SKIP_4CH_PALM:
    print("\n" + "="*60)
    print("STEP 2B — GENERATING 4CH NPY: PALM")
    print("="*60)
    subprocess.run([
        sys.executable, "generate_4ch_palm.py",
        "--checkpoint",  DIFFUSION_CHECKPOINT,
        "--input_dir",   os.path.join(BASE_DIR, "data", "palm"),
        "--output_dir",  os.path.join(BASE_DIR, "data", "palm_4ch"),
        "--n-samples",   "5",
        "--timestep",    "500",
    ], check=True)
    print("  PALM 4ch done.")
else:
    print("\nSTEP 2B — Skipped (SKIP_4CH_PALM=True)")

# ============================================================
# STEP 3 — TRAIN CNN-1 (Diabetic Retinopathy)
# ============================================================

if not SKIP_CNN1_DR:
    print("\n" + "="*60)
    print("STEP 3 — TRAINING CNN-1 (Diabetic Retinopathy)")
    print("="*60)
    subprocess.run([
        sys.executable, "train_cnn1_dr.py",
        "--labels_file", LABELS["eyepacs"],
        "--epochs",      "100",
        "--batch_size",  "32",
    ], check=True)
    print("  CNN-1 DR training done.")
else:
    print("\nSTEP 3 — Skipped (SKIP_CNN1_DR=True)")

# ============================================================
# STEP 4 — TRAIN CNN-2 (Glaucoma)
# ============================================================

if not SKIP_CNN2_GLAUCOMA:
    print("\n" + "="*60)
    print("STEP 4 — TRAINING CNN-2 (Glaucoma)")
    print("  Using combined REFUGE+G1020+ORIGA dataset")
    print("="*60)
    subprocess.run([
        sys.executable, "train_cnn2_glaucoma.py",
        "--labels_file", LABELS["glaucoma"],
        "--epochs",      "100",
        "--batch_size",  "32",
    ], check=True)
    print("  CNN-2 Glaucoma training done.")
else:
    print("\nSTEP 4 — Skipped (SKIP_CNN2_GLAUCOMA=True)")

# ============================================================
# STEP 5 — TRAIN CNN-3 (Pathologic Myopia)
# ============================================================

if not SKIP_CNN3_PM:
    print("\n" + "="*60)
    print("STEP 5 — TRAINING CNN-3 (Pathologic Myopia)")
    print("="*60)
    subprocess.run([
        sys.executable, "train_cnn3_pm.py",
        "--labels_file", LABELS["palm"],
        "--epochs",      "100",
        "--batch_size",  "32",
    ], check=True)
    print("  CNN-3 PM training done.")
else:
    print("\nSTEP 5 — Skipped (SKIP_CNN3_PM=True)")

# ============================================================
# STEP 6 — TRAIN META-CLASSIFIER
# ============================================================

if not SKIP_META:
    print("\n" + "="*60)
    print("STEP 6 — TRAINING META-CLASSIFIER")
    print("="*60)
    subprocess.run([
        sys.executable, "train_meta_classifier.py",
        "--dr_checkpoint",       CHECKPOINTS["dr"],
        "--glaucoma_checkpoint", CHECKPOINTS["glaucoma"],
        "--pm_checkpoint",       CHECKPOINTS["pm"],
        "--dr_labels",           LABELS["eyepacs"],
        "--glaucoma_labels",     LABELS["glaucoma"],
        "--pm_labels",           LABELS["palm"],
    ], check=True)
    print("  Meta-classifier training done.")
else:
    print("\nSTEP 6 — Skipped (SKIP_META=True)")

# ============================================================
# DONE
# ============================================================

print("\n" + "="*60)
print("ALL DONE!")
print("="*60)
print("Checkpoints saved in:")
print(f"  {os.path.join(BASE_DIR, 'models', 'checkpoints', 'dr')}")
print(f"  {os.path.join(BASE_DIR, 'models', 'checkpoints', 'glaucoma')}")
print(f"  {os.path.join(BASE_DIR, 'models', 'checkpoints', 'pm')}")
print(f"  {os.path.join(BASE_DIR, 'models', 'checkpoints', 'meta')}")
print("\nNext step: run evaluate_all.py")
