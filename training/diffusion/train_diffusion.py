"""
train_diffusion_512_final.py
=============================
512×512 version of train_diffusion.py — optimized for RTX 5090 (Blackwell).

Changes from 256 og script:
  A. IMAGE_SIZE / CROP_SIZE = 512
  B. EPOCHS = 8 — ~75-90 min/epoch, fits in 10hr window
  C. BATCH_SIZE = 4, ACCUM_STEPS = 8 — effective batch 32, avoids OOM
  D. BF16 instead of FP16 — better dynamic range, native on Blackwell
  E. AdamW8bit — saves ~3GB VRAM on optimizer states
  F. Gradient checkpointing ENABLED — required at 512×512
  G. torch.compile DISABLED — Inductor OOM at 512×512, not worth it
  H. GradScaler disabled for BF16 — not needed
  I. CHECKPOINT_DIR = checkpoints_512_final — clean separate run
  J. VIS_EVERY = 1 — visualize every epoch (only 8 total)
"""

import os
import csv
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from diffusers.utils.import_utils import is_xformers_available

from glob import glob
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers import UNet2DModel, DDPMScheduler


# ── SNR-weighted loss ─────────────────────────────────────────────────────────

def snr_weighted_loss(pred_noise, noise, alphas_cumprod, timesteps, gamma=5.0):
    ac     = alphas_cumprod[timesteps.cpu()].to(pred_noise.device).float()
    snr    = ac / (1.0 - ac + 1e-8)
    weight = torch.clamp(snr, max=gamma) / (snr + 1e-8)
    loss_per_sample = F.mse_loss(
        pred_noise.float(), noise.float(), reduction='none'
    )
    loss_per_sample = loss_per_sample.mean(dim=[1, 2, 3])
    return (weight * loss_per_sample).mean()


# ── Dataset ───────────────────────────────────────────────────────────────────

def make_transform(image_size, crop_size, is_train):
    base = [
        transforms.Resize((image_size + 28, image_size + 28)),
        transforms.RandomCrop(crop_size) if is_train else transforms.CenterCrop(crop_size),
    ]
    augment = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.1, hue=0.05
        ),
    ] if is_train else []
    return transforms.Compose(
        base + augment + [
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ]
    )


class RetinaDataset(Dataset):
    def __init__(self, folder, image_size=512, crop_size=512, is_train=True):
        self.images = []
        for ext in ("*.jpeg", "*.jpg", "*.png",
                    "*.PNG",  "*.JPG",  "*.JPEG"):
            self.images.extend(
                glob(os.path.join(folder, "**", ext), recursive=True)
            )
        self.transform = make_transform(image_size, crop_size, is_train)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                img = Image.open(self.images[idx]).convert("RGB")
                return self.transform(img), self.images[idx]
            except Exception:
                print(f"Skipping corrupted: {self.images[idx]}")
                idx = (idx + 1) % len(self.images)
        raise RuntimeError("Too many consecutive corrupted images")


def collate_fn(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
    DATA_TRAIN     = os.path.join(BASE_DIR, "data", "train")
    DATA_VAL       = os.path.join(BASE_DIR, "data", "val")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_512_final")

    IMAGE_SIZE    = 512
    CROP_SIZE     = 512
    EPOCHS        = 15           # ~75-90 min/epoch → fits in 10hr window
    BATCH_SIZE    = 6           # bs=4 + grad checkpointing fits in 31GB at 512×512
    ACCUM_STEPS   = 6           # effective batch = 32
    LR            = 2e-4
    WARMUP_EPOCHS = 2           # short warmup for short run
    VIS_EVERY     = 1           # visualize every epoch — only 8 total
    NUM_VIS       = 4
    SNR_GAMMA     = 5.0

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu"

    # BF16 — native on Blackwell, better than FP16
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"✓ AMP dtype: {amp_dtype}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

    if device == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        print("✓ Flash Attention (SDPA) enabled")

    # ── Datasets ──────────────────────────────────────────────────────────

    train_dataset = RetinaDataset(DATA_TRAIN, IMAGE_SIZE, CROP_SIZE, is_train=True)
    val_dataset   = RetinaDataset(DATA_VAL,   IMAGE_SIZE, CROP_SIZE, is_train=False)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Dataset is empty — check DATA_TRAIN / DATA_VAL paths.")

    gpu_gen = torch.cuda.get_device_capability()[0] if device == "cuda" else 0
    pin = (device == "cuda") and (gpu_gen < 12)
    if not pin and device == "cuda":
        print("⚠ pin_memory disabled (Blackwell GPU) — using non_blocking transfers instead")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=pin, collate_fn=collate_fn,
        drop_last=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=pin, collate_fn=collate_fn,
        persistent_workers=True
    )

    # ── Model ─────────────────────────────────────────────────────────────

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
    ).to(device)

    if device == "cuda":
        alloc   = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"VRAM after model load: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved")

    # Required at 512×512 to keep VRAM bounded
    model.enable_gradient_checkpointing()
    print("✓ Gradient checkpointing enabled")

    if is_xformers_available():
        try:
            model.enable_xformers_memory_efficient_attention()
            print("✓ xformers enabled")
        except Exception as e:
            print(f"xformers found but could not enable: {e}")
            print("  → Flash Attention (SDPA) is active instead")
    else:
        print("xformers not detected — Flash Attention (SDPA) active")

    # torch.compile DISABLED — Inductor buffers OOM at 512×512
    print("torch.compile disabled — avoids Inductor OOM at 512×512")

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    alphas_cumprod  = noise_scheduler.alphas_cumprod.cpu()

    # AdamW8bit — saves ~3GB VRAM on optimizer states
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=1e-4)
        print("✓ AdamW8bit enabled")
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        print("⚠ bitsandbytes not found — using AdamW. Run: pip install bitsandbytes")

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=max(EPOCHS - WARMUP_EPOCHS, 1),
                                     eta_min=1e-6)
    # GradScaler disabled for BF16 — not needed
    scaler = GradScaler(device, enabled=(amp_dtype == torch.float16))

    # ── Resume ────────────────────────────────────────────────────────────

    start_epoch   = 0
    best_val_loss = float("inf")
    train_losses  = []
    val_losses    = []

    last_checkpoint = os.path.join(CHECKPOINT_DIR, "last.pt")
    best_checkpoint = os.path.join(CHECKPOINT_DIR, "best.pt")
    loss_csv        = os.path.join(CHECKPOINT_DIR, "loss.csv")

    if os.path.exists(last_checkpoint):
        print("Resuming from last checkpoint...")
        try:
            ckpt = torch.load(last_checkpoint, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(last_checkpoint, map_location=device, weights_only=False)

        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']

        if start_epoch >= WARMUP_EPOCHS and 'scheduler_state_dict' in ckpt:
            lr_scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        if os.path.exists(loss_csv):
            with open(loss_csv) as f:
                for row in csv.DictReader(f):
                    train_losses.append(float(row['train_loss']))
                    val_losses.append(float(row['val_loss']))

        print(f"  Resumed epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    if not os.path.exists(loss_csv):
        with open(loss_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr"])

    # ── Visualization ─────────────────────────────────────────────────────

    raw_vis    = next(iter(val_loader))[0]
    vis_images = raw_vis[:min(NUM_VIS, len(raw_vis))].cpu()
    n_vis      = len(vis_images)

    @torch.no_grad()
    def save_reconstructions(epoch):
        model.eval()
        t_val = 500
        t     = torch.tensor([t_val], device=device).long()
        ac    = noise_scheduler.alphas_cumprod[t_val].to(device)

        fig, axes = plt.subplots(n_vis, 3, figsize=(12, n_vis * 3))
        if n_vis == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle(f'Epoch {epoch} — Original | Reconstruction | Residual')

        for i in range(n_vis):
            img   = vis_images[i].unsqueeze(0).to(device)
            noise = torch.randn_like(img)
            noisy = noise_scheduler.add_noise(img, noise, t)
            with autocast(device_type=device_type, dtype=amp_dtype):
                pred_noise = model(noisy, t).sample

            recon    = (noisy - (1 - ac).sqrt() * pred_noise.float()) / ac.sqrt()
            recon    = recon.clamp(-1, 1)
            orig_np  = ((img.squeeze().permute(1,2,0).cpu().float().numpy()+1)/2).clip(0,1)
            recon_np = ((recon.squeeze().permute(1,2,0).cpu().float().numpy()+1)/2).clip(0,1)
            resid_np = np.abs(orig_np - recon_np).mean(axis=2)

            axes[i,0].imshow(orig_np);  axes[i,0].axis('off')
            axes[i,1].imshow(recon_np); axes[i,1].axis('off')
            im = axes[i,2].imshow(resid_np, cmap='jet', vmin=0, vmax=0.3)
            axes[i,2].axis('off')
            plt.colorbar(im, ax=axes[i,2], fraction=0.046)
            if i == 0:
                axes[i,0].set_title('Original')
                axes[i,1].set_title('Reconstruction')
                axes[i,2].set_title('Residual (Anomaly)')

        plt.tight_layout()
        plt.savefig(os.path.join(CHECKPOINT_DIR, f'recon_epoch_{epoch:04d}.png'),
                    dpi=80, bbox_inches='tight')
        plt.close()

    # ── Training loop ─────────────────────────────────────────────────────

    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"\nStarting — {EPOCHS} epochs | batch={BATCH_SIZE} | accum={ACCUM_STEPS} | "
          f"effective_batch={BATCH_SIZE * ACCUM_STEPS} | "
          f"img={IMAGE_SIZE}x{IMAGE_SIZE} | lr={LR} | snr_gamma={SNR_GAMMA}")
    print(f"Device: {device} | Params: {sum(p.numel() for p in model.parameters()):,}\n")

    for epoch in range(start_epoch, EPOCHS):

        in_warmup = epoch < WARMUP_EPOCHS
        if in_warmup:
            for pg in optimizer.param_groups:
                pg['lr'] = LR * (epoch + 1) / WARMUP_EPOCHS

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:4d}/{EPOCHS}")

        optimizer.zero_grad()
        _diag_done = False

        for step, (batch, _) in enumerate(pbar):
            batch     = batch.to(device, non_blocking=True)
            noise     = torch.randn_like(batch)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (batch.shape[0],), device=device
            ).long()
            noisy = noise_scheduler.add_noise(batch, noise, timesteps)

            with autocast(device_type=device_type, dtype=amp_dtype):
                pred_noise = model(noisy, timesteps).sample

            if not _diag_done and epoch == start_epoch:
                alloc   = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"\nVRAM after first forward: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved")
                _diag_done = True

            loss = snr_weighted_loss(
                pred_noise, noise, alphas_cumprod, timesteps, SNR_GAMMA
            )

            loss = loss / ACCUM_STEPS
            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * ACCUM_STEPS
            pbar.set_postfix(loss=f"{loss.item() * ACCUM_STEPS:.5f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # Flush leftover gradients
        if len(train_loader) % ACCUM_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss /= len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch, _ in val_loader:
                batch     = batch.to(device, non_blocking=True)
                noise     = torch.randn_like(batch)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (batch.shape[0],), device=device
                ).long()
                noisy = noise_scheduler.add_noise(batch, noise, timesteps)
                with autocast(device_type=device_type, dtype=amp_dtype):
                    pred_noise = model(noisy, timesteps).sample
                loss = snr_weighted_loss(
                    pred_noise, noise, alphas_cumprod, timesteps, SNR_GAMMA
                )
                val_loss += loss.item()

        val_loss  /= len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:4d} | Train: {train_loss:.6f} | "
              f"Val: {val_loss:.6f} | LR: {current_lr:.2e}")

        if not in_warmup:
            lr_scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        with open(loss_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, current_lr])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict':    scaler.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_val_loss':        best_val_loss,
            }, best_checkpoint)
            print(f"  ✓ best.pt (val={best_val_loss:.6f})")

        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict':    scaler.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'best_val_loss':        best_val_loss,
        }, last_checkpoint)

        # Visualize every epoch — only 8 total, don't miss any
        if epoch % VIS_EVERY == 0:
            save_reconstructions(epoch)

        plt.figure(figsize=(10, 4))
        plt.plot(train_losses, label='Train', alpha=0.7)
        plt.plot(val_losses,   label='Val',   alpha=0.7)
        plt.xlabel('Epoch'); plt.ylabel('SNR-Weighted Loss')
        plt.title('Diffusion Training Loss — 512×512')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(CHECKPOINT_DIR, 'loss_curve.png'))
        plt.close()

    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"Best model: {best_checkpoint}")


if __name__ == "__main__":
    main()
