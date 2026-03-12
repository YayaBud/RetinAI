"""
generate_4ch_palm.py
=====================
Generates 4-channel .npy files for the PALM pathologic myopia dataset.

Structure expected:
    data/palm/organized/train/pm/*.jpg
    data/palm/organized/train/non_pm/*.jpg
    data/palm/organized/val/pm/*.jpg
    data/palm/organized/val/non_pm/*.jpg
    data/palm/organized/palm_labels.csv  (image_id, label, split)

Output:
    data/palm_4ch/
        train/pm/*_4ch.npy
        train/non_pm/*_4ch.npy
        val/pm/*_4ch.npy
        val/non_pm/*_4ch.npy
        palm_anomaly_summary.csv

Usage:
    python3 generate_4ch_palm.py \
        --checkpoint models/checkpoints_diffusion/checkpoints/best.pt \
        --input_dir  data/palm/organized \
        --output_dir data/palm_4ch
"""

import os
import csv
import argparse
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.amp import autocast
from diffusers import UNet2DModel, DDPMScheduler

IMAGE_SIZE  = 256
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if DEVICE == "cuda" else "cpu"


def load_model(checkpoint_path):
    print(f"Loading model: {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = UNet2DModel(
        sample_size=IMAGE_SIZE, in_channels=3, out_channels=3,
        layers_per_block=2, block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D","DownBlock2D","AttnDownBlock2D","AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D","AttnUpBlock2D","UpBlock2D","UpBlock2D"),
    ).to(DEVICE)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Loaded. Epoch={ckpt.get('epoch','?')} | Loss={ckpt.get('best_val_loss','?')}")
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])


@torch.no_grad()
def generate_anomaly_map(model, scheduler, tensor, timestep, n_samples):
    t  = torch.tensor([timestep], device=DEVICE).long()
    ac = scheduler.alphas_cumprod[timestep].to(DEVICE)
    residuals = []
    for _ in range(n_samples):
        noise = torch.randn_like(tensor)
        noisy = scheduler.add_noise(tensor, noise, t)
        with autocast(device_type=DEVICE_TYPE):
            pred_noise = model(noisy, t).sample
        recon    = (noisy - (1 - ac).sqrt() * pred_noise.float()) / ac.sqrt()
        recon    = recon.clamp(-1, 1)
        orig_np  = ((tensor.squeeze().permute(1,2,0).cpu().float().numpy()+1)/2).clip(0,1)
        recon_np = ((recon.squeeze().permute(1,2,0).cpu().float().numpy()+1)/2).clip(0,1)
        residuals.append(np.abs(orig_np - recon_np).mean(axis=2))
    amap = np.mean(residuals, axis=0)
    orig_np = ((tensor.squeeze().permute(1,2,0).cpu().float().numpy()+1)/2).clip(0,1)
    return amap, orig_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--input_dir",   required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--n-samples",   type=int, default=5)
    parser.add_argument("--timestep",    type=int, default=500)
    args = parser.parse_args()

    if DEVICE == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)

    os.makedirs(args.output_dir, exist_ok=True)
    model     = load_model(args.checkpoint)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    transform = get_transform()

    # Collect all images from train/ and val/ splits
    all_images = []  # list of (img_path, label_str, split_str)
    for split in ("train", "val"):
        for class_name in ("pm", "non_pm"):
            class_dir = os.path.join(args.input_dir, split, class_name)
            if not os.path.exists(class_dir):
                print(f"  WARNING: {class_dir} not found, skipping")
                continue
            for ext in ("*.jpeg", "*.jpg", "*.png", "*.PNG", "*.JPG"):
                for p in glob(os.path.join(class_dir, ext)):
                    all_images.append((p, class_name, split))

    print(f"Found {len(all_images)} PALM images")

    rows = []
    for img_path, label_str, split in tqdm(all_images, desc="PALM 4ch"):
        try:
            image_id = os.path.splitext(os.path.basename(img_path))[0]
            label    = 1 if label_str == "pm" else 0

            img    = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)

            amap, orig_np = generate_anomaly_map(
                model, scheduler, tensor, args.timestep, args.n_samples
            )

            # 4ch npy: [4, H, W] — channels first for PyTorch
            four_ch  = np.concatenate([orig_np, np.expand_dims(amap, 2)], axis=2)  # [H, W, 4]
            four_ch  = np.transpose(four_ch, (2, 0, 1)).astype(np.float32)          # [4, H, W]
            out_dir  = os.path.join(args.output_dir, split, label_str)
            os.makedirs(out_dir, exist_ok=True)
            npy_path  = os.path.join(out_dir, f"{image_id}_4ch.npy")
            amap_path = os.path.join(out_dir, f"{image_id}_anomaly.npy")
            np.save(npy_path,  four_ch)
            np.save(amap_path, amap)

            rows.append({
                "image_path":    img_path,
                "npy_4ch_path":  npy_path,
                "amap_path":     amap_path,
                "label":         label_str,
                "split":         split,
                "anomaly_score": float(amap.mean()),
            })
        except Exception as e:
            print(f"  Skipping {img_path}: {e}")

    # Save CSV
    csv_path = os.path.join(args.output_dir, "palm_anomaly_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_path","npy_4ch_path","amap_path","label","split","anomaly_score"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} images processed.")
    print(f"CSV: {csv_path}")

    # Print split/class distribution
    from collections import Counter
    dist = Counter((r["split"], r["label"]) for r in rows)
    for (split, label), count in sorted(dist.items()):
        print(f"  {split}/{label}: {count}")


if __name__ == "__main__":
    main()
