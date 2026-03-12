"""
generate_anomaly_maps.py
========================
Generates anomaly maps from a trained diffusion model checkpoint.
Supports both best.pt and last.pt.

Usage:
    # Use best.pt (recommended)
    python3 generate_anomaly_maps.py --checkpoint checkpoints/best.pt --input /path/to/images --output anomaly_maps_best

    # Use last.pt
    python3 generate_anomaly_maps.py --checkpoint checkpoints/last.pt --input /path/to/images --output anomaly_maps_last

    # Run both at once
    python3 generate_anomaly_maps.py --both --input /path/to/images

    # Tune timestep (default 500, try 200-800)
    python3 generate_anomaly_maps.py --checkpoint checkpoints/best.pt --input /path/to/images --timestep 300

    # Average over multiple noise samples for smoother maps (default 5)
    python3 generate_anomaly_maps.py --checkpoint checkpoints/best.pt --input /path/to/images --n-samples 10
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from glob import glob
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.amp import autocast
from diffusers import UNet2DModel, DDPMScheduler


# ── Config ────────────────────────────────────────────────────────────────────

IMAGE_SIZE = 256        # must match training resolution
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if DEVICE == "cuda" else "cpu"


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path):
    """Load UNet model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    try:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    epoch     = ckpt.get('epoch', '?')
    best_loss = ckpt.get('best_val_loss', '?')
    print(f"  Epoch: {epoch} | Best val loss: {best_loss}")

    model = UNet2DModel(
        sample_size=IMAGE_SIZE,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
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
    ).to(DEVICE)

    # Strip _orig_mod. prefix if model was saved with torch.compile
    state_dict = ckpt['model_state_dict']
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    model.eval()
    print(f"  Model loaded successfully on {DEVICE}")
    return model


# ── Image transform ───────────────────────────────────────────────────────────

def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])


# ── Core anomaly map generation ───────────────────────────────────────────────

@torch.no_grad()
def generate_anomaly_map(model, scheduler, image_tensor, timestep=500, n_samples=5):
    """
    Generate anomaly map by averaging residuals over multiple noise samples.

    Args:
        model:        trained UNet
        scheduler:    DDPMScheduler
        image_tensor: [1, 3, H, W] normalized tensor on DEVICE
        timestep:     noise level to use (200=light noise, 800=heavy noise)
        n_samples:    number of noise samples to average (more = smoother map)

    Returns:
        anomaly_map: [H, W] numpy array, higher = more anomalous
        reconstruction: [H, W, 3] numpy array for visualization
    """
    t = torch.tensor([timestep], device=DEVICE).long()
    ac = scheduler.alphas_cumprod[timestep].to(DEVICE)

    residuals = []
    last_recon = None

    for _ in range(n_samples):
        noise = torch.randn_like(image_tensor)
        noisy = scheduler.add_noise(image_tensor, noise, t)

        with autocast(device_type=DEVICE_TYPE):
            pred_noise = model(noisy, t).sample

        # x0 prediction from noise prediction
        recon = (noisy - (1 - ac).sqrt() * pred_noise.float()) / ac.sqrt()
        recon = recon.clamp(-1, 1)

        orig_np  = ((image_tensor.squeeze().permute(1,2,0).cpu().float().numpy() + 1) / 2).clip(0, 1)
        recon_np = ((recon.squeeze().permute(1,2,0).cpu().float().numpy() + 1) / 2).clip(0, 1)

        residual = np.abs(orig_np - recon_np).mean(axis=2)
        residuals.append(residual)
        last_recon = recon_np

    anomaly_map = np.mean(residuals, axis=0)
    return anomaly_map, last_recon, orig_np


# ── Save individual result ────────────────────────────────────────────────────

def save_result(orig_np, recon_np, anomaly_map, save_path, image_name, timestep, n_samples):
    """Save a 3-panel visualization: original | reconstruction | anomaly map."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{image_name} | t={timestep} | n_samples={n_samples}', fontsize=12)

    axes[0].imshow(orig_np)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(recon_np)
    axes[1].set_title('Reconstruction')
    axes[1].axis('off')

    # Normalize anomaly map per-image for better visualization
    vmax = max(anomaly_map.max(), 0.05)  # floor at 0.05 to avoid flat maps
    im = axes[2].imshow(anomaly_map, cmap='jet', vmin=0, vmax=vmax)
    axes[2].set_title(f'Anomaly Map (max={anomaly_map.max():.3f})')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

    # Also save raw anomaly map as numpy for downstream use
    np.save(save_path.replace('.png', '_raw.npy'), anomaly_map)


# ── Run inference on a folder ─────────────────────────────────────────────────

def run_inference(checkpoint_path, input_dir, output_dir, timestep=500, n_samples=5):
    """Run anomaly map generation on all images in input_dir."""
    os.makedirs(output_dir, exist_ok=True)

    model     = load_model(checkpoint_path)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    transform = get_transform()

    # Find all images
    images = []
    for ext in ("*.jpeg", "*.jpg", "*.png", "*.PNG", "*.JPG", "*.JPEG"):
        images.extend(glob(os.path.join(input_dir, "**", ext), recursive=True))

    if len(images) == 0:
        raise RuntimeError(f"No images found in {input_dir}")

    print(f"\nFound {len(images)} images")
    print(f"Timestep: {timestep} | Samples per image: {n_samples}")
    print(f"Output: {output_dir}\n")

    all_scores = []

    for img_path in tqdm(images, desc="Generating anomaly maps"):
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)

            anomaly_map, recon_np, orig_np = generate_anomaly_map(
                model, scheduler, tensor,
                timestep=timestep,
                n_samples=n_samples
            )

            image_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path  = os.path.join(output_dir, f"{image_name}_anomaly.png")

            save_result(orig_np, recon_np, anomaly_map, save_path, image_name, timestep, n_samples)

            score = float(anomaly_map.mean())
            all_scores.append((image_name, score, anomaly_map.max()))

        except Exception as e:
            print(f"  Skipping {img_path}: {e}")

    # Save summary
    summary_path = os.path.join(output_dir, "anomaly_scores.txt")
    all_scores.sort(key=lambda x: x[1], reverse=True)
    with open(summary_path, "w") as f:
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Timestep: {timestep} | N-samples: {n_samples}\n")
        f.write(f"{'Image':<50} {'Mean Score':>12} {'Max Score':>12}\n")
        f.write("-" * 76 + "\n")
        for name, mean_s, max_s in all_scores:
            f.write(f"{name:<50} {mean_s:>12.6f} {max_s:>12.6f}\n")

    print(f"\nDone. Results saved to: {output_dir}")
    print(f"Anomaly scores saved to: {summary_path}")
    print(f"Highest anomaly score: {all_scores[0][0]} ({all_scores[0][1]:.6f})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate anomaly maps from diffusion model")

    parser.add_argument("--checkpoint",  type=str, default=None,
                        help="Path to checkpoint (.pt file)")
    parser.add_argument("--both",        action="store_true",
                        help="Run both best.pt and last.pt")
    parser.add_argument("--input",       type=str, required=True,
                        help="Input image folder")
    parser.add_argument("--output",      type=str, default=None,
                        help="Output folder (auto-named if not set)")
    parser.add_argument("--timestep",    type=int, default=500,
                        help="Noise timestep (default=500, try 200-800)")
    parser.add_argument("--n-samples",   type=int, default=5,
                        help="Noise samples to average per image (default=5)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory (default=checkpoints)")

    args = parser.parse_args()

    # Enable Flash SDPA for Blackwell
    if DEVICE == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)

    if args.both:
        # Run best.pt
        best_ckpt   = os.path.join(args.checkpoint_dir, "best.pt")
        best_output = args.output or f"anomaly_maps_best_t{args.timestep}"
        print("=" * 60)
        print("Running BEST.PT")
        print("=" * 60)
        run_inference(best_ckpt, args.input, best_output, args.timestep, args.n_samples)

        # Run last.pt
        last_ckpt   = os.path.join(args.checkpoint_dir, "last.pt")
        last_output = f"anomaly_maps_last_t{args.timestep}"
        print("\n" + "=" * 60)
        print("Running LAST.PT")
        print("=" * 60)
        run_inference(last_ckpt, args.input, last_output, args.timestep, args.n_samples)

    elif args.checkpoint:
        output = args.output or f"anomaly_maps_t{args.timestep}"
        run_inference(args.checkpoint, args.input, output, args.timestep, args.n_samples)

    else:
        parser.error("Provide --checkpoint or --both")


if __name__ == "__main__":
    main()
