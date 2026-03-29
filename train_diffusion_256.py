"""
train_diffusion_256.py
======================
Pixel-Space DDPM for Retinal Anomaly Detection — 256×256
Hardware: RTX 5090 32GB

PIPELINE (everything feeds the next, nothing fights):
  Image
    → RETFound (frozen, in-memory cached) → conditioning vector
    → Simplex noise at random t → noisy image
    → UNet (conditioned) → pred_noise
    → one-step x0 estimate → reconstruction
    → |orig - recon| × retinal_mask → raw residual
    → median-subtract + blur + normalize → clean residual
    → seg head (orig+recon, 6ch) → learned anomaly map [starts epoch 10]
    → BOTH saved as PNG for CNN 4th channel

HOW COMPONENTS COMPLEMENT EACH OTHER:
  RETFound  → tells UNet "this specific retina looks like THIS"
               reconstruction stays faithful to structure, not average anatomy
  Simplex   → noise at lesion spatial scale → model learns to denoise at lesion freq
               same noise at train AND inference → no distribution mismatch
  Retinal mask → excludes black border from loss AND residual
                 stops border artifacts dominating anomaly map
  Seg head  → takes (orig, recon) and learns WHICH residuals = real anomaly
               starts at epoch 10 when reconstruction is already decent
               CutPaste uses same-image patches + color jitter + feathered edges
               → anatomically plausible anomalies, not random foreign patches
  Post-processing → median subtract removes vessel-edge baseline noise
                    Gaussian blur removes salt-and-pepper
                    → clean map ready for CNN

WHAT WAS REMOVED AND WHY:
  - MC Dropout: UNet2DConditionModel has no Dropout layers → was pure noise
  - THOR mid-trajectory blending: broke DDIM trajectory → seam artifacts
  - DDIM at training: inconsistent with simplex training noise
  - Focal + Laplacian + uncertainty loss: competing gradients, net negative
  - Cross-image CutPaste hard edges: taught seg head to detect boundary not anomaly

LOSS:
  Diffusion: 0.6 × SNR-weighted noise MSE + 0.4 × multiscale MSE on pred_x0
  Seg head:  BCE on CutPaste masks, weight ramps 0→0.3 over epochs 10-20
"""

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from glob import glob
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers import UNet2DConditionModel, DDPMScheduler
from diffusers.utils.import_utils import is_xformers_available
from sklearn.metrics import roc_auc_score, average_precision_score


# ─────────────────────────────────────────────────────────────────────────────
# 1. SIMPLEX NOISE
#    Spatially correlated multi-octave noise matching lesion scales.
#    Used identically at train AND inference — no distribution mismatch.
# ─────────────────────────────────────────────────────────────────────────────

def generate_simplex_noise(shape, device, frequency=8, octaves=4):
    """
    Multi-octave smooth random field — approximates simplex noise.
    frequency : base grid freq. 8 ≈ matches microaneurysm scale at 256px.
    octaves   : 4 gives micro + macro coverage.
    Per-sample normalised → stable training signal.
    """
    B, C, H, W = shape
    noise = torch.zeros(B, C, H, W, device=device)
    total_amp = 0.0
    for octave in range(octaves):
        freq = frequency * (2 ** octave)
        amp  = 0.5 ** octave
        gh   = max(2, H // freq)
        gw   = max(2, W // freq)
        grid = torch.randn(B, C, gh, gw, device=device)
        noise += amp * F.interpolate(grid, size=(H, W), mode='bicubic', align_corners=False)
        total_amp += amp
    noise = noise / total_amp
    # Per-sample normalisation (more stable than batch-level)
    mean = noise.mean(dim=[1, 2, 3], keepdim=True)
    std  = noise.std(dim=[1, 2, 3],  keepdim=True) + 1e-8
    return (noise - mean) / std


def add_simplex_noise(x, timesteps, alphas_cumprod, frequency=8, octaves=4):
    """Standard DDPM forward process but with simplex noise instead of Gaussian."""
    device = x.device
    ac     = alphas_cumprod[timesteps.cpu()].to(device).float().view(-1, 1, 1, 1)
    noise  = generate_simplex_noise(x.shape, device, frequency=frequency, octaves=octaves)
    noisy  = ac.sqrt() * x.float() + (1 - ac).sqrt() * noise
    return noisy, noise


# ─────────────────────────────────────────────────────────────────────────────
# 2. RETFOUND CONDITIONER
#    Frozen ViT-Large pretrained on 1.6M retinal images.
#    Only the projection MLP trains.
#    CachedConditioner wraps it — runs ViT once per image ever, caches to RAM.
#    At 256px, ~25k images × 768 floats × 4 bytes ≈ 75MB cache. Fine.
# ─────────────────────────────────────────────────────────────────────────────

class RETFoundConditioner(nn.Module):
    def __init__(self, cross_attention_dim=768):
        super().__init__()
        self.vit = None
        self._load_retfound()
        # Trainable: RETFound hidden dim 1024 → UNet cross-attn dim 768
        self.proj = nn.Sequential(
            nn.Linear(1024, 768),
            nn.GELU(),
            nn.Linear(768, cross_attention_dim),
            nn.LayerNorm(cross_attention_dim),
        )

    def _load_retfound(self):
        candidates = [
            "/home/amr3/RETFound_cfp_weights.pth",
            "/home/amr3/Downloads/RETFound_cfp_weights.pth",
            os.path.expanduser("~/RETFound_cfp_weights.pth"),
            os.path.expanduser("~/Downloads/RETFound_cfp_weights.pth"),
        ]
        try:
            import timm
            self.vit = timm.create_model(
                'vit_large_patch16_224', pretrained=False,
                num_classes=0, global_pool='token',
            )
            ckpt_path = next((p for p in candidates if os.path.exists(p)), None)
            if ckpt_path:
                state       = torch.load(ckpt_path, map_location='cpu')
                model_state = state.get('model', state)
                vit_state   = self.vit.state_dict()
                filtered    = {k: v for k, v in model_state.items()
                               if k in vit_state and v.shape == vit_state[k].shape}
                self.vit.load_state_dict(filtered, strict=False)
                print(f"RETFound loaded from {ckpt_path} ({len(filtered)}/{len(vit_state)} keys)")
            else:
                print("WARNING: RETFound weights not found — using random ViT-Large init.")
                print("Expected at: /home/amr3/RETFound_cfp_weights.pth")
            for p in self.vit.parameters():
                p.requires_grad_(False)
            self.vit.eval()
            print("RETFound frozen.")
        except ImportError:
            print("timm not found — falling back to torchvision ViT-L/16")
            from torchvision.models import vit_l_16, ViT_L_16_Weights
            vit_full = vit_l_16(weights=ViT_L_16_Weights.DEFAULT)
            self.vit = nn.Sequential(*list(vit_full.children())[:-1])
            for p in self.vit.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def extract_features(self, x):
        """x: (B,3,H,W) in [-1,1]. Returns (B,1024) CLS token."""
        x_r  = F.interpolate(x.float(), size=(224, 224), mode='bicubic', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_n  = ((x_r + 1) / 2 - mean) / std
        feats = self.vit(x_n)
        if feats.dim() == 3:
            feats = feats[:, 0]
        return feats  # (B, 1024)

    def forward(self, x):
        with torch.no_grad():
            feats = self.extract_features(x)
        return self.proj(feats.float()).unsqueeze(1)  # (B, 1, 768)


class CachedConditioner:
    """
    Caches the FROZEN ViT features (1024-d) to RAM, NOT the projection output.

    Why this matters:
      The old implementation cached the output of conditioner(ub), i.e. the
      result of self.proj(feats). That tensor carries a live autograd graph
      back through proj. On the second epoch PyTorch tries to backward through
      an already-freed graph → RuntimeError: Trying to backward through the
      graph a second time. Even if it didn't crash, proj would only learn on
      the very first forward pass of each image and be frozen thereafter.

    Fix: cache only extract_features() output (no_grad, 1024-d CLS token).
      On every forward call we re-run the trainable proj on the cached ViT
      features, so proj gets fresh gradients every step as intended.

    Key = image path. Value = (1024,) float32 tensor on CPU.
    Cache persists across epochs — same image never runs ViT twice.
    At 256px, ~25k images × 1024 floats × 4 bytes ≈ 100 MB. Fine.
    """
    def __init__(self, conditioner):
        self.conditioner = conditioner
        self._vit_cache = {}   # path → (1024,) cpu tensor, NO grad

    def __call__(self, batch_tensor, batch_paths=None):
        device = batch_tensor.device

        # No paths → augmented / synthetic images: run ViT fresh, no caching
        if batch_paths is None:
            with torch.no_grad():
                vit_feats = self.conditioner.extract_features(batch_tensor)
            return self.conditioner.proj(vit_feats.float()).unsqueeze(1)  # (B,1,768)

        results          = [None] * len(batch_paths)
        uncached_idx     = []
        uncached_tensors = []

        # 1. Serve cached ViT features
        for i, path in enumerate(batch_paths):
            if path in self._vit_cache:
                results[i] = self._vit_cache[path].to(device, non_blocking=True)
            else:
                uncached_idx.append(i)
                uncached_tensors.append(batch_tensor[i])

        # 2. Compute and cache missing ViT features (no_grad — ViT is frozen)
        if uncached_tensors:
            ub = torch.stack(uncached_tensors).to(device)
            with torch.no_grad():
                fresh_feats = self.conditioner.extract_features(ub)  # (N, 1024)
            for j, idx in enumerate(uncached_idx):
                cpu_feat = fresh_feats[j].cpu()
                self._vit_cache[batch_paths[idx]] = cpu_feat
                results[idx] = fresh_feats[j]

        # 3. Stack raw ViT features and run through trainable proj EVERY step
        #    This is where gradients flow — proj learns on every forward pass.
        vit_tensor = torch.stack(results)                              # (B, 1024)
        return self.conditioner.proj(vit_tensor.float()).unsqueeze(1)  # (B, 1, 768)

    def train(self): self.conditioner.train()
    def eval(self):  self.conditioner.eval()

    def parameters(self):
        return self.conditioner.parameters()

    @property
    def proj(self):
        return self.conditioner.proj


# ─────────────────────────────────────────────────────────────────────────────
# 3. RETINAL CIRCULAR MASK
#    Fundus images have a circular field — outside is black border.
#    This mask excludes the border from loss AND residual computation.
#    Stops border artifacts dominating the anomaly map.
# ─────────────────────────────────────────────────────────────────────────────

def make_retinal_mask(images, margin=0.05):
    """
    images: (B, 3, H, W) in [-1, 1]
    Returns: (B, 1, H, W) float mask — 1=inside retina, 0=black border
    margin: shrinks circle slightly to avoid edge artifacts
    """
    B, C, H, W = images.shape
    device = images.device

    # Detect background: pixels near black (-1 in normalised space)
    img_01   = (images.float() + 1) / 2       # → [0, 1]
    is_bg    = (img_01.mean(dim=1, keepdim=True) < 0.05).float()  # (B,1,H,W)
    fg_mask  = 1.0 - is_bg                    # foreground pixels

    # Build circular mask centred on image
    cy, cx = H / 2, W / 2
    r      = min(H, W) / 2 * (1.0 - margin)
    ys     = torch.arange(H, device=device).float() - cy
    xs     = torch.arange(W, device=device).float() - cx
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    circle = ((yy**2 + xx**2) <= r**2).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # Combined: inside circle AND not background
    mask = circle * fg_mask  # (B, 1, H, W)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# 4. CUTPASTE SYNTHETIC ANOMALY (fixed version)
#    Previous version: pasted patches from DIFFERENT images → taught seg head
#    to detect foreign anatomy, not real anomalies.
#
#    This version:
#      - Patches from the SAME image (anatomically plausible)
#      - Color jitter on the patch (creates color-based anomaly signal)
#      - Feathered edges (no hard boundary → seg head can't cheat)
#      - Small patch sizes only (8–32px at 256px = realistic lesion scale)
# ─────────────────────────────────────────────────────────────────────────────

def cutpaste_synthetic_anomaly(images, min_px=8, max_px=32):
    """
    images: (B, 3, H, W) in [-1, 1]
    Returns: (anomalous, masks) both same shape as images / (B,1,H,W)
    """
    B, C, H, W = images.shape
    device     = images.device
    anomalous  = images.clone()
    masks      = torch.zeros(B, 1, H, W, device=device)

    for i in range(B):
        ph = torch.randint(min_px, max_px + 1, (1,)).item()
        pw = torch.randint(min_px, max_px + 1, (1,)).item()

        # Source from same image — anatomically consistent
        sy = torch.randint(0, H - ph, (1,)).item()
        sx = torch.randint(0, W - pw, (1,)).item()
        dy = torch.randint(0, H - ph, (1,)).item()
        dx = torch.randint(0, W - pw, (1,)).item()

        # Extract patch, apply color jitter to make it "wrong"
        patch = images[i, :, sy:sy+ph, sx:sx+pw].clone()  # (3, ph, pw)
        # PIL jitter needs uint8 — do manual brightness/contrast shift instead
        shift = (torch.rand(1, device=device) - 0.5) * 0.6   # brightness
        scale = (torch.rand(1, device=device) * 0.6) + 0.7    # contrast
        patch = (patch * scale + shift).clamp(-1, 1)

        # Feathered blend mask (cosine falloff at edges)
        fy = torch.linspace(0, torch.pi, ph, device=device)
        fx = torch.linspace(0, torch.pi, pw, device=device)
        gy, gx = torch.meshgrid(fy, fx, indexing='ij')
        feather = (gy.sin() * gx.sin()).clamp(0, 1).unsqueeze(0)  # (1,ph,pw)

        # Blend patch into image
        anomalous[i, :, dy:dy+ph, dx:dx+pw] = (
            feather * patch +
            (1 - feather) * images[i, :, dy:dy+ph, dx:dx+pw]
        )
        # Soft mask (feathered)
        masks[i, 0, dy:dy+ph, dx:dx+pw] = feather

    return anomalous, masks


# ─────────────────────────────────────────────────────────────────────────────
# 5. LOSS FUNCTIONS
#    Two losses for diffusion. One for seg head. Clean weights, no fighting.
# ─────────────────────────────────────────────────────────────────────────────

def snr_weighted_loss(pred_noise, noise, alphas_cumprod, timesteps, gamma=2.0):
    """
    SNR-weighted MSE on predicted noise.
    SNR clamped at gamma to avoid overwhelming low-t steps.
    This is the primary training signal for the UNet.
    """
    ac     = alphas_cumprod[timesteps.cpu()].to(pred_noise.device).float().view(-1, 1, 1, 1)
    snr    = ac / (1.0 - ac + 1e-8)
    weight = torch.clamp(snr, max=gamma) / (snr + 1e-8)
    mse    = F.mse_loss(pred_noise.float(), noise.float(), reduction='none').mean(dim=[1, 2, 3])
    return (weight.squeeze() * mse).mean()


def multiscale_mse_loss(pred_x0, x0, retinal_mask=None):
    """
    MSE at 3 scales: full (256), half (128), quarter (64).
    At 64px a 3px microaneurysm is proportionally 4× bigger → model can't ignore it.
    Applied to reconstructed x0, not noise — directly measures reconstruction quality.
    Masked to retinal disc only.
    """
    def masked_mse(p, t, mask):
        err = (p.float() - t.float()) ** 2
        if mask is not None:
            err = err * mask
            return err.sum() / (mask.sum() * p.shape[1] + 1e-8)
        return err.mean()

    p2 = F.avg_pool2d(pred_x0.float(), 2)
    t2 = F.avg_pool2d(x0.float(), 2)
    p4 = F.avg_pool2d(p2, 2)
    t4 = F.avg_pool2d(t2, 2)

    m2 = F.avg_pool2d(retinal_mask.float(), 2)  if retinal_mask is not None else None
    m4 = F.avg_pool2d(m2, 2)                    if m2 is not None            else None

    l1 = masked_mse(pred_x0, x0, retinal_mask)
    l2 = masked_mse(p2, t2, m2)
    l4 = masked_mse(p4, t4, m4)
    return 0.4 * l1 + 0.35 * l2 + 0.25 * l4


def diffusion_loss(pred_noise, noise, pred_x0, x0, retinal_mask,
                   alphas_cumprod, timesteps, snr_gamma=2.0):
    """
    Combined diffusion loss:
      0.6 × SNR-weighted noise MSE  — main training signal
      0.4 × Multiscale MSE on x0   — small lesion fidelity
    Retinal mask applied to multiscale (border excluded from x0 loss).
    """
    l_snr = snr_weighted_loss(pred_noise, noise, alphas_cumprod, timesteps, snr_gamma)
    l_ms  = multiscale_mse_loss(pred_x0, x0, retinal_mask)
    return 0.6 * l_snr + 0.4 * l_ms, {'snr': l_snr.item(), 'ms': l_ms.item()}


# ─────────────────────────────────────────────────────────────────────────────
# 6. SEGMENTATION HEAD
#    6-channel input: original(3) + reconstruction(3)
#    Learns which reconstruction differences = real anomaly vs noise.
#    Starts contributing to loss at epoch SEG_START_EPOCH (default 10).
#    Weight ramps from 0 → SEG_LOSS_MAX over SEG_RAMP_EPOCHS epochs.
# ─────────────────────────────────────────────────────────────────────────────

class AnomalySegHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6,  32, 3, padding=1), nn.BatchNorm2d(32),  nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32),  nn.GELU(),
            nn.Conv2d(32,  1, 1),
            nn.Sigmoid(),
        )

    def forward(self, original, reconstruction):
        return self.net(torch.cat([original, reconstruction], dim=1))  # (B,1,H,W)


# ─────────────────────────────────────────────────────────────────────────────
# 6.5 DIFF-MAMBA INTEGRATION (SS2D UNet Wrapper)
# ─────────────────────────────────────────────────────────────────────────────

class SS2DBlock(nn.Module):
    """
    VMamba-inspired 2D Selective Scan (SS2D) approximation.
    Uses 4-directional spatial scanning via bidirectional LSTMs 
    for global contextual awareness with linear complexity.
    """
    def __init__(self, channels):
        super().__init__()
        self.h_scan = nn.LSTM(channels, channels // 2, batch_first=True, bidirectional=True)
        self.v_scan = nn.LSTM(channels, channels // 2, batch_first=True, bidirectional=True)
        self.merge = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Horizontal scan
        h_seq = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        h_out, _ = self.h_scan(h_seq)
        h_out = h_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # Vertical scan
        v_seq = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        v_out, _ = self.v_scan(v_seq)
        v_out = v_out.reshape(B, W, H, C).permute(0, 3, 2, 1)
        
        # Merge the 4 directions
        merged = torch.cat([h_out, v_out], dim=1)
        attention_mask = self.merge(merged)
        
        return x + self.proj(x * attention_mask)


class DiffMambaUNet(nn.Module):
    """
    Wraps standard Diffusion UNet and integrates VMamba-inspired SS2D blocks.
    Applies the spatial sweep at the bottleneck tensor (mid_block) directly.
    """
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.ss2d_bottleneck = SS2DBlock(channels=512)
        
        def mid_block_hook(module, args, output):
            # output is a tuple if return_dict=False in diffusers context, but 
            # the hook signature for diffusers mid_block returns the tensor
            val = output[0] if isinstance(output, tuple) else output
            return self.ss2d_bottleneck(val)
            
        self.unet.mid_block.register_forward_hook(mid_block_hook)

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet(sample, timestep, encoder_hidden_states)
    
    def enable_gradient_checkpointing(self):
        self.unet.enable_gradient_checkpointing()


def seg_loss_weight(epoch, seg_start=10, seg_max=0.3, ramp_epochs=10):
    """
    Returns current seg head loss weight.
    0 before seg_start, linearly ramps to seg_max over ramp_epochs.
    Smooth ramp prevents seg head from destabilising diffusion training early.
    """
    if epoch < seg_start:
        return 0.0
    ramp = min(1.0, (epoch - seg_start) / max(ramp_epochs, 1))
    return seg_max * ramp


# ─────────────────────────────────────────────────────────────────────────────
# 7. ANOMALY MAP POST-PROCESSING
#    Applied to raw |orig - recon| residual before saving as PNG.
#    Makes the map clean and usable as CNN 4th channel.
# ─────────────────────────────────────────────────────────────────────────────

def postprocess_residual(orig_np, recon_np, retinal_mask_np=None):
    """
    orig_np, recon_np : (H, W, 3) float32 in [0, 1]
    retinal_mask_np   : (H, W) float32 — 1=retina, 0=border (optional)
    Returns clean anomaly map (H, W) float32 in [0, 1]
    """
    residual = np.abs(orig_np - recon_np).mean(axis=2)   # (H, W)

    # Zero out border
    if retinal_mask_np is not None:
        residual = residual * retinal_mask_np

    # Median subtract — removes vessel-edge baseline noise
    median = np.median(residual[residual > 0]) if (residual > 0).any() else 0.0
    residual = (residual - median).clip(0)

    # Small Gaussian blur — removes salt-and-pepper artifacts
    from scipy.ndimage import gaussian_filter
    residual = gaussian_filter(residual, sigma=1.5)

    # Normalize to [0, 1]
    rmax = residual.max()
    if rmax > 0:
        residual = residual / rmax

    return residual.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 8. METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_ssim(img1, img2):
    def rgb2gray(img):
        return 0.2989*img[...,0] + 0.5870*img[...,1] + 0.1140*img[...,2]
    g1 = rgb2gray(img1).astype(np.float64)
    g2 = rgb2gray(img2).astype(np.float64)
    C1, C2 = 0.01**2, 0.03**2
    mu1, mu2 = g1.mean(), g2.mean()
    s1  = ((g1-mu1)**2).mean()
    s2  = ((g2-mu2)**2).mean()
    s12 = ((g1-mu1)*(g2-mu2)).mean()
    return float(((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2)))


def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64))**2)
    return 100.0 if mse < 1e-10 else float(10*np.log10(1.0/mse))


@torch.no_grad()
def compute_val_metrics(model, cached_cond, seg_head, val_loader,
                         alphas_cumprod, device, amp_dtype, device_type,
                         noise_scheduler, max_batches=15,
                         simplex_freq=8, simplex_octaves=4):
    model.eval(); seg_head.eval()
    ssim_scores, psnr_scores = [], []
    all_preds, all_gts = [], []

    T_INFER = 500

    for batch_idx, (batch, paths) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        batch = batch.to(device, non_blocking=True)
        B     = batch.shape[0]

        # Reconstruction quality
        cond  = cached_cond(batch, paths)
        t_vec = torch.full((B,), T_INFER, device=device, dtype=torch.long)
        noisy, noise = add_simplex_noise(batch, t_vec, alphas_cumprod,
                                          simplex_freq, simplex_octaves)
        ac = alphas_cumprod[T_INFER].to(device).float()

        with autocast(device_type=device_type, dtype=amp_dtype):
            pred_noise = model(noisy.to(amp_dtype), t_vec,
                               encoder_hidden_states=cond).sample

        pred_x0 = ((noisy.float() - (1-ac).sqrt() * pred_noise.float())
                   / (ac.sqrt() + 1e-8)).clamp(-1, 1)

        for i in range(min(B, 4)):
            orig_np  = ((batch[i].permute(1,2,0).cpu().float().numpy()+1)/2).clip(0,1)
            recon_np = ((pred_x0[i].permute(1,2,0).cpu().float().numpy()+1)/2).clip(0,1)
            ssim_scores.append(compute_ssim(orig_np, recon_np))
            psnr_scores.append(compute_psnr(orig_np, recon_np))

        # Anomaly detection quality (CutPaste synthetic)
        anomalous, gt_masks = cutpaste_synthetic_anomaly(batch)
        t_anom = torch.full((B,), 300, device=device, dtype=torch.long)
        noisy_a, _ = add_simplex_noise(anomalous, t_anom, alphas_cumprod,
                                        simplex_freq, simplex_octaves)
        ac300 = alphas_cumprod[300].to(device).float()
        cond_a = cached_cond(anomalous, None)  # no cache for augmented images

        with autocast(device_type=device_type, dtype=amp_dtype):
            pred_n  = model(noisy_a.to(amp_dtype), t_anom,
                            encoder_hidden_states=cond_a).sample
        recon_a = ((noisy_a.float() - (1-ac300).sqrt()*pred_n.float())
                   / (ac300.sqrt()+1e-8)).clamp(-1, 1)

        with autocast(device_type=device_type, dtype=amp_dtype):
            pred_maps = seg_head(anomalous.to(amp_dtype), recon_a.to(amp_dtype))

        all_preds.append(pred_maps.squeeze(1).cpu().float().numpy().reshape(B, -1))
        all_gts.append(gt_masks.squeeze(1).cpu().float().numpy().reshape(B, -1))

    pixel_auroc = pixel_ap = 0.0
    if all_preds:
        preds = np.concatenate(all_preds).flatten()
        gts   = (np.concatenate(all_gts).flatten() > 0.5).astype(int)  # threshold soft masks
        try:
            pixel_auroc = float(roc_auc_score(gts, preds))
            pixel_ap    = float(average_precision_score(gts, preds))
        except Exception as e:
            print(f"  AUROC/AP failed: {e}")

    return {
        'ssim':        float(np.mean(ssim_scores)) if ssim_scores else 0.0,
        'psnr':        float(np.mean(psnr_scores)) if psnr_scores else 0.0,
        'pixel_auroc': pixel_auroc,
        'pixel_ap':    pixel_ap,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9. DATASET
# ─────────────────────────────────────────────────────────────────────────────

def make_transform(crop_size, is_train):
    base = [transforms.Resize((crop_size + 32, crop_size + 32),
                               interpolation=transforms.InterpolationMode.BICUBIC)]
    if is_train:
        aug = [
            transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0), ratio=(0.9, 1.1),
                                          interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        ]
    else:
        aug = [transforms.CenterCrop(crop_size)]
    return transforms.Compose(base + aug + [
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])


class RetinaDataset(Dataset):
    def __init__(self, folder, crop_size=256, is_train=True, bad_files_txt=None):
        all_images = []
        for ext in ("*.jpeg", "*.jpg", "*.png", "*.PNG", "*.JPG", "*.JPEG"):
            all_images.extend(glob(os.path.join(folder, "**", ext), recursive=True))
        if bad_files_txt and os.path.exists(bad_files_txt):
            with open(bad_files_txt) as f:
                bad = set(l.strip() for l in f)
            before = len(all_images)
            all_images = [p for p in all_images if p not in bad]
            print(f"Filtered {before - len(all_images)} bad files — {len(all_images)} remaining")
        self.images    = all_images
        self.transform = make_transform(crop_size, is_train)

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                img = Image.open(self.images[idx]).convert("RGB")
                return self.transform(img), self.images[idx]
            except Exception:
                idx = (idx + 1) % len(self.images)
        raise RuntimeError("Too many consecutive bad images")


def collate_fn(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


# ─────────────────────────────────────────────────────────────────────────────
# 10. VISUALIZATION
#     Saves 5-column grid: Orig | Recon | Raw Residual | Clean Residual | Seg Map
#     Also saves individual anomaly map PNGs for CNN consumption.
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def save_visualizations(epoch, model, cached_cond, seg_head,
                          vis_images, vis_paths,
                          alphas_cumprod, device, amp_dtype, device_type,
                          checkpoint_dir, anomaly_maps_raw_dir, anomaly_maps_seg_dir,
                          noise_scheduler, n_vis,
                          simplex_freq=8, simplex_octaves=4):
    model.eval(); seg_head.eval()

    T_INFER = 500
    rows = []

    for i in range(n_vis):
        orig  = vis_images[i].unsqueeze(0)  # (1,3,H,W)
        cond  = cached_cond(orig, [vis_paths[i]])
        t_vec = torch.tensor([T_INFER], device=device, dtype=torch.long)

        noisy, _ = add_simplex_noise(orig, t_vec, alphas_cumprod,
                                      simplex_freq, simplex_octaves)
        ac = alphas_cumprod[T_INFER].to(device).float()

        with autocast(device_type=device_type, dtype=amp_dtype):
            pred_noise = model(noisy.to(amp_dtype), t_vec,
                               encoder_hidden_states=cond).sample

        pred_x0 = ((noisy.float() - (1-ac).sqrt() * pred_noise.float())
                   / (ac.sqrt() + 1e-8)).clamp(-1, 1)

        # Retinal mask
        rmask = make_retinal_mask(orig)  # (1,1,H,W)

        # Seg head anomaly map
        with autocast(device_type=device_type, dtype=amp_dtype):
            seg_map = seg_head(orig.to(amp_dtype), pred_x0.to(amp_dtype))

        # To numpy
        orig_np   = ((orig.squeeze().permute(1,2,0).cpu().float().numpy()+1)/2).clip(0,1)
        recon_np  = ((pred_x0.squeeze().permute(1,2,0).cpu().float().numpy()+1)/2).clip(0,1)
        mask_np   = rmask.squeeze().cpu().float().numpy()
        seg_np    = seg_map.squeeze().cpu().float().numpy()

        # Raw residual (mask only, no post-processing)
        raw_resid = np.abs(orig_np - recon_np).mean(axis=2) * mask_np

        # Clean residual (post-processed)
        clean_resid = postprocess_residual(orig_np, recon_np, mask_np)

        rows.append((orig_np, recon_np, raw_resid, clean_resid, seg_np))

        # Save individual anomaly map PNGs for CNN
        raw_png  = (raw_resid / (raw_resid.max() + 1e-8) * 255).astype(np.uint8)
        seg_png  = (seg_np * 255).astype(np.uint8)
        Image.fromarray(raw_png).save(
            os.path.join(anomaly_maps_raw_dir, f"epoch{epoch:04d}_img{i}.png"))
        Image.fromarray(seg_png).save(
            os.path.join(anomaly_maps_seg_dir, f"epoch{epoch:04d}_img{i}.png"))

    # Grid figure: 5 columns
    fig, axes = plt.subplots(n_vis, 5, figsize=(20, n_vis * 4))
    if n_vis == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f'Epoch {epoch} — Orig | Recon | Raw Residual | Clean Residual | Seg Map',
                 fontsize=11)

    for i, (orig_np, recon_np, raw_resid, clean_resid, seg_np) in enumerate(rows):
        axes[i,0].imshow(orig_np);   axes[i,0].axis('off')
        axes[i,1].imshow(recon_np);  axes[i,1].axis('off')
        im1 = axes[i,2].imshow(raw_resid,   cmap='jet', vmin=0, vmax=raw_resid.max()+1e-8)
        axes[i,2].axis('off'); plt.colorbar(im1, ax=axes[i,2], fraction=0.046)
        im2 = axes[i,3].imshow(clean_resid, cmap='jet', vmin=0, vmax=1)
        axes[i,3].axis('off'); plt.colorbar(im2, ax=axes[i,3], fraction=0.046)
        im3 = axes[i,4].imshow(seg_np,      cmap='jet', vmin=0, vmax=1)
        axes[i,4].axis('off'); plt.colorbar(im3, ax=axes[i,4], fraction=0.046)
        if i == 0:
            for ax, t in zip(axes[i], ['Original', 'Reconstruction',
                                        'Raw Residual', 'Clean Residual', 'Seg Map']):
                ax.set_title(t)

    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, f'recon_epoch_{epoch:04d}.png'),
                dpi=80, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Paths ─────────────────────────────────────────────────────────────
    BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
    DATA_TRAIN     = "/home/amr3/Downloads/RetinAI/data/diabetic_retinopathy/organized/train_split"
    DATA_VAL       = "/home/amr3/Downloads/RetinAI/data/diabetic_retinopathy/organized/val_split"
    BAD_FILES_TXT  = "/home/amr3/Downloads/RetinAI/data/diabetic_retinopathy/organized/bad_files.txt"
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_256")

    # ── Hyperparameters ───────────────────────────────────────────────────
    CROP_SIZE         = 256
    EPOCHS            = 60
    BATCH_SIZE        = 16         # 256px is light — push batch size up
    ACCUM_STEPS       = 2          # effective batch = 32
    LR_UNET           = 3e-5
    LR_CONDITIONER    = 1e-4
    LR_SEGHEAD        = 1e-4
    WARMUP_EPOCHS     = 5
    VIS_EVERY         = 2
    EVAL_EVERY        = 5
    NUM_VIS           = 4
    SNR_GAMMA         = 2.0
    SIMPLEX_FREQ      = 8
    SIMPLEX_OCTAVES   = 4
    SEG_START_EPOCH   = 10         # seg head silent until reconstruction is decent
    SEG_LOSS_MAX      = 0.3        # max seg head loss weight
    SEG_RAMP_EPOCHS   = 10         # ramp from 0 → 0.3 over 10 epochs
    NUM_WORKERS_TRAIN = 16
    NUM_WORKERS_VAL   = 8
    PREFETCH_FACTOR   = 4

    # ── Setup ─────────────────────────────────────────────────────────────
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu"
    amp_dtype   = (torch.bfloat16
                   if (device == "cuda" and torch.cuda.is_bf16_supported())
                   else torch.float16)

    print(f"Device: {device} | AMP: {amp_dtype}")
    print(f"Resolution: {CROP_SIZE}×{CROP_SIZE} | Effective batch: {BATCH_SIZE*ACCUM_STEPS}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ANOMALY_MAPS_RAW = os.path.join(CHECKPOINT_DIR, "anomaly_maps_raw")
    ANOMALY_MAPS_SEG = os.path.join(CHECKPOINT_DIR, "anomaly_maps_seg")
    os.makedirs(ANOMALY_MAPS_RAW, exist_ok=True)
    os.makedirs(ANOMALY_MAPS_SEG, exist_ok=True)

    # ── Noise scheduler ───────────────────────────────────────────────────
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",  # cosine — better than linear
    )
    alphas_cumprod = noise_scheduler.alphas_cumprod.cpu()

    # ── UNet ─────────────────────────────────────────────────────────────
    print("Building Diff-Mamba UNet (256×256, SS2D pixel-space)...")
    base_unet = UNet2DConditionModel(
        sample_size=CROP_SIZE,
        in_channels=3, out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ),
        up_block_types=(
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
        ),
        cross_attention_dim=768,
    )
    
    model = DiffMambaUNet(base_unet).to(device)

    model.enable_gradient_checkpointing()

    # Flash attention for 5090
    if device == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    if is_xformers_available():
        try:
            model.enable_xformers_memory_efficient_attention()
            print("xformers enabled")
        except Exception as e:
            print(f"xformers skipped: {e}")

    print(f"UNet params: {sum(p.numel() for p in model.parameters()):,}")

    # ── RETFound conditioner + cache ──────────────────────────────────────
    print("Building RETFound conditioner...")
    conditioner  = RETFoundConditioner(cross_attention_dim=768).to(device)
    cached_cond  = CachedConditioner(conditioner)

    # ── Seg head ──────────────────────────────────────────────────────────
    print("Building segmentation head...")
    seg_head = AnomalySegHead().to(device)

    # ── Optimizer — separate LRs ──────────────────────────────────────────
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit([
            {'params': model.parameters(),            'lr': LR_UNET},
            {'params': conditioner.proj.parameters(), 'lr': LR_CONDITIONER},
            {'params': seg_head.parameters(),         'lr': LR_SEGHEAD},
        ], weight_decay=1e-4)
        print("AdamW8bit enabled")
    except ImportError:
        optimizer = torch.optim.AdamW([
            {'params': model.parameters(),            'lr': LR_UNET},
            {'params': conditioner.proj.parameters(), 'lr': LR_CONDITIONER},
            {'params': seg_head.parameters(),         'lr': LR_SEGHEAD},
        ], weight_decay=1e-4)
        print("Standard AdamW")

    lr_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(EPOCHS - WARMUP_EPOCHS, 1), eta_min=1e-6)
    scaler = GradScaler(device, enabled=(amp_dtype == torch.float16))

    # ── Datasets ──────────────────────────────────────────────────────────
    train_dataset = RetinaDataset(DATA_TRAIN, CROP_SIZE, is_train=True,
                                   bad_files_txt=BAD_FILES_TXT)
    val_dataset   = RetinaDataset(DATA_VAL,   CROP_SIZE, is_train=False)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset empty — check DATA_TRAIN path")

    gpu_gen = torch.cuda.get_device_capability()[0] if device == "cuda" else 0
    pin     = (device == "cuda") and (gpu_gen < 12)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS_TRAIN, pin_memory=pin,
        collate_fn=collate_fn, drop_last=True,
        persistent_workers=True, prefetch_factor=PREFETCH_FACTOR,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS_VAL, pin_memory=pin,
        collate_fn=collate_fn,
        persistent_workers=True, prefetch_factor=PREFETCH_FACTOR,
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")
    best_auroc    = 0.0
    train_losses  = []
    val_losses    = []
    metrics_history = []

    last_ckpt       = os.path.join(CHECKPOINT_DIR, "last.pt")
    best_loss_ckpt  = os.path.join(CHECKPOINT_DIR, "best_loss.pt")
    best_auroc_ckpt = os.path.join(CHECKPOINT_DIR, "best_auroc.pt")
    loss_csv        = os.path.join(CHECKPOINT_DIR, "loss.csv")
    metrics_csv     = os.path.join(CHECKPOINT_DIR, "metrics.csv")

    if os.path.exists(last_ckpt):
        print("Resuming from last.pt...")
        try:
            ckpt = torch.load(last_ckpt, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)

        model.load_state_dict(ckpt['model'])
        conditioner.proj.load_state_dict(ckpt['conditioner_proj'])
        seg_head.load_state_dict(ckpt['seg_head'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])

        # Always override LRs after optimizer load
        for pg, lr in zip(optimizer.param_groups,
                           [LR_UNET, LR_CONDITIONER, LR_SEGHEAD]):
            pg['lr'] = lr

        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        best_auroc    = ckpt.get('best_auroc', 0.0)

        if 'scheduler' in ckpt and start_epoch >= WARMUP_EPOCHS:
            lr_scheduler.load_state_dict(ckpt['scheduler'])

        if os.path.exists(loss_csv):
            with open(loss_csv) as f:
                for row in csv.DictReader(f):
                    train_losses.append(float(row['train_loss']))
                    val_losses.append(float(row['val_loss']))

        print(f"Resumed epoch {start_epoch} | best_loss={best_val_loss:.6f} | best_auroc={best_auroc:.4f}")
    else:
        print("Starting fresh")

    if not os.path.exists(loss_csv):
        with open(loss_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","train_loss","val_loss","lr_unet","seg_weight"])
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","ssim","psnr","pixel_auroc","pixel_ap"])

    # Fixed val subset for visualization
    raw_vis       = next(iter(val_loader))
    vis_images    = raw_vis[0][:min(NUM_VIS, len(raw_vis[0]))].to(device)
    vis_paths     = raw_vis[1][:min(NUM_VIS, len(raw_vis[1]))]
    n_vis         = len(vis_images)

    if device == "cuda":
        torch.cuda.empty_cache()
        print(f"VRAM before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    print(f"\nStarting — {EPOCHS} epochs | {CROP_SIZE}×{CROP_SIZE} | "
          f"batch {BATCH_SIZE}×{ACCUM_STEPS}={BATCH_SIZE*ACCUM_STEPS} | "
          f"seg starts epoch {SEG_START_EPOCH}\n")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS):

        # Warmup
        in_warmup = epoch < WARMUP_EPOCHS
        if in_warmup:
            wf = (epoch + 1) / WARMUP_EPOCHS
            for pg, base_lr in zip(optimizer.param_groups,
                                    [LR_UNET, LR_CONDITIONER, LR_SEGHEAD]):
                pg['lr'] = base_lr * wf

        sw = seg_loss_weight(epoch, SEG_START_EPOCH, SEG_LOSS_MAX, SEG_RAMP_EPOCHS)

        # ── Train ─────────────────────────────────────────────────────────
        model.train(); conditioner.train(); seg_head.train()
        train_loss = 0.0
        comp_snr   = 0.0
        comp_ms    = 0.0
        comp_seg   = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{EPOCHS} [256]")

        for step, (batch, paths) in enumerate(pbar):
            batch = batch.to(device, non_blocking=True)

            # RETFound conditioning — cached after first epoch
            cond = cached_cond(batch, paths)  # (B, 1, 768)

            # Retinal mask — excludes black border from loss
            retinal_mask = make_retinal_mask(batch)  # (B,1,H,W)

            # Simplex noise forward process
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (batch.shape[0],), device=device).long()
            noisy, noise = add_simplex_noise(batch, timesteps, alphas_cumprod,
                                              SIMPLEX_FREQ, SIMPLEX_OCTAVES)

            # UNet denoising
            with autocast(device_type=device_type, dtype=amp_dtype):
                pred_noise = model(noisy.to(amp_dtype), timesteps,
                                   encoder_hidden_states=cond).sample

            # One-step x0 estimate for multiscale loss
            # noisy.detach() is CRITICAL — noisy is a constant input each step,
            # no gradient flows through it. Without detach, autograd holds the
            # entire noisy→pred_noise graph in memory across accum steps,
            # and step 2 tries to backward through step 1's freed graph.
            ac_t    = alphas_cumprod[timesteps.cpu()].to(device).float().view(-1,1,1,1)
            pred_x0 = ((noisy.detach().float() - (1-ac_t).sqrt() * pred_noise.float())
                       / (ac_t.sqrt() + 1e-8)).clamp(-1, 1)

            # Diffusion loss — single scalar, single .backward() call below
            d_loss, comp = diffusion_loss(
                pred_noise, noise, pred_x0, batch,
                retinal_mask, alphas_cumprod, timesteps, SNR_GAMMA
            )

            # Seg head loss — only when sw > 0 (after epoch SEG_START_EPOCH)
            s_loss = torch.tensor(0.0, device=device)
            if sw > 0 and (step % 4 == 0):
                anomalous, anom_masks = cutpaste_synthetic_anomaly(batch)
                with torch.no_grad():
                    t_anom = torch.full((batch.shape[0],), 300, device=device, dtype=torch.long)
                    noisy_a, _ = add_simplex_noise(anomalous, t_anom, alphas_cumprod,
                                                    SIMPLEX_FREQ, SIMPLEX_OCTAVES)
                    ac300 = alphas_cumprod[300].to(device).float()
                    ca    = cached_cond(anomalous, None)
                    with autocast(device_type=device_type, dtype=amp_dtype):
                        pn = model(noisy_a.to(amp_dtype), t_anom,
                                   encoder_hidden_states=ca).sample
                    recon_a = ((noisy_a.float() - (1-ac300).sqrt()*pn.float())
                               / (ac300.sqrt()+1e-8)).clamp(-1, 1)

                with autocast(device_type=device_type, dtype=amp_dtype):
                    pred_mask = seg_head(anomalous.to(amp_dtype), recon_a.to(amp_dtype))
                s_loss = F.binary_cross_entropy(pred_mask.float(), anom_masks.float())

            # Single backward through combined loss
            # CRITICAL: must be ONE .backward() call — d_loss and s_loss share
            # no graph (s_loss is computed under torch.no_grad for the UNet part)
            # so combining them is safe and avoids any double-backward
            combined = (d_loss + sw * s_loss) / ACCUM_STEPS
            scaler.scale(combined).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) +
                    list(conditioner.proj.parameters()) +
                    list(seg_head.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += d_loss.item() + sw * s_loss.item()
            comp_snr   += comp['snr']
            comp_ms    += comp['ms']
            comp_seg   += s_loss.item()

            pbar.set_postfix(
                snr=f"{comp['snr']:.4f}", ms=f"{comp['ms']:.4f}",
                seg=f"{s_loss.item():.4f}", sw=f"{sw:.2f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}"
            )

        # Flush leftover gradients if dataset not divisible by ACCUM_STEPS
        if len(train_loader) % ACCUM_STEPS != 0:
            try:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) +
                    list(conditioner.proj.parameters()) +
                    list(seg_head.parameters()), 1.0)
                scaler.step(optimizer); scaler.update()
            except Exception:
                pass
            optimizer.zero_grad()

        n_steps     = len(train_loader)
        train_loss /= n_steps
        comp_snr   /= n_steps
        comp_ms    /= n_steps
        comp_seg   /= n_steps

        # ── Validate ──────────────────────────────────────────────────────
        model.eval(); conditioner.eval(); seg_head.eval()
        val_loss = 0.0

        with torch.inference_mode():
            for batch, paths in val_loader:
                batch = batch.to(device, non_blocking=True)
                cond  = cached_cond(batch, paths)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (batch.shape[0],), device=device).long()
                noisy, noise = add_simplex_noise(batch, timesteps, alphas_cumprod,
                                                  SIMPLEX_FREQ, SIMPLEX_OCTAVES)
                with autocast(device_type=device_type, dtype=amp_dtype):
                    pred_noise = model(noisy.to(amp_dtype), timesteps,
                                       encoder_hidden_states=cond).sample
                val_loss += snr_weighted_loss(
                    pred_noise, noise, alphas_cumprod, timesteps, SNR_GAMMA).item()

        val_loss  /= len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d} | Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
              f"LR: {current_lr:.1e} | "
              f"snr:{comp_snr:.4f} ms:{comp_ms:.4f} seg:{comp_seg:.4f} sw:{sw:.2f}")

        if not in_warmup:
            lr_scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        with open(loss_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, current_lr, sw])

        # ── Metrics every EVAL_EVERY ──────────────────────────────────────
        metrics = None
        if epoch % EVAL_EVERY == 0:
            print(f"  Computing metrics...")
            metrics = compute_val_metrics(
                model, cached_cond, seg_head, val_loader,
                alphas_cumprod, device, amp_dtype, device_type,
                noise_scheduler, max_batches=15,
                simplex_freq=SIMPLEX_FREQ, simplex_octaves=SIMPLEX_OCTAVES,
            )
            metrics_history.append({'epoch': epoch, **metrics})
            print(f"  SSIM:{metrics['ssim']:.4f} PSNR:{metrics['psnr']:.2f}dB "
                  f"AUROC:{metrics['pixel_auroc']:.4f} AP:{metrics['pixel_ap']:.4f}")
            with open(metrics_csv, "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{metrics['ssim']:.6f}",
                    f"{metrics['psnr']:.4f}", f"{metrics['pixel_auroc']:.6f}",
                    f"{metrics['pixel_ap']:.6f}"])

        # ── Checkpoints ───────────────────────────────────────────────────
        ckpt_data = {
            'epoch':            epoch,
            'model':            model.state_dict(),
            'conditioner_proj': conditioner.proj.state_dict(),
            'seg_head':         seg_head.state_dict(),
            'optimizer':        optimizer.state_dict(),
            'scaler':           scaler.state_dict(),
            'scheduler':        lr_scheduler.state_dict(),
            'best_val_loss':    best_val_loss,
            'best_auroc':       best_auroc,
        }
        if val_loss < best_val_loss:
            best_val_loss = val_loss; ckpt_data['best_val_loss'] = best_val_loss
            torch.save(ckpt_data, best_loss_ckpt)
            print(f"  best_loss.pt saved (val={best_val_loss:.6f})")
        if metrics and metrics['pixel_auroc'] > best_auroc:
            best_auroc = metrics['pixel_auroc']; ckpt_data['best_auroc'] = best_auroc
            torch.save(ckpt_data, best_auroc_ckpt)
            print(f"  best_auroc.pt saved (AUROC={best_auroc:.4f})")
        torch.save(ckpt_data, last_ckpt)

        # ── Visualize ─────────────────────────────────────────────────────
        if epoch % VIS_EVERY == 0:
            torch.cuda.empty_cache()
            save_visualizations(
                epoch, model, cached_cond, seg_head,
                vis_images, vis_paths,
                alphas_cumprod, device, amp_dtype, device_type,
                CHECKPOINT_DIR, ANOMALY_MAPS_RAW, ANOMALY_MAPS_SEG,
                noise_scheduler, n_vis, SIMPLEX_FREQ, SIMPLEX_OCTAVES,
            )

        # ── Loss curve ────────────────────────────────────────────────────
        plt.figure(figsize=(10, 4))
        plt.plot(train_losses, label='Train', alpha=0.8)
        plt.plot(val_losses,   label='Val',   alpha=0.8)
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.title('256×256 — RETFound + Simplex + RetinalMask + SegHead')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(CHECKPOINT_DIR, 'loss_curve.png'))
        plt.close()

        if len(metrics_history) >= 2:
            ep_e = [m['epoch'] for m in metrics_history]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].plot(ep_e, [m['pixel_auroc'] for m in metrics_history], 'b-o', label='AUROC')
            axes[0].plot(ep_e, [m['pixel_ap']    for m in metrics_history], 'r-o', label='AP')
            axes[0].set_title('Anomaly Detection'); axes[0].legend(); axes[0].grid(True)
            axes[1].plot(ep_e, [m['ssim'] for m in metrics_history], 'g-o')
            axes[1].set_title('SSIM'); axes[1].grid(True)
            axes[2].plot(ep_e, [m['psnr'] for m in metrics_history], 'm-o')
            axes[2].set_title('PSNR (dB)'); axes[2].grid(True)
            plt.suptitle('Metrics Over Training'); plt.tight_layout()
            plt.savefig(os.path.join(CHECKPOINT_DIR, 'metrics_curve.png'))
            plt.close()

    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"Best AUROC: {best_auroc:.4f}")
    print(f"Anomaly maps (raw): {ANOMALY_MAPS_RAW}")
    print(f"Anomaly maps (seg): {ANOMALY_MAPS_SEG}")


if __name__ == "__main__":
    main()