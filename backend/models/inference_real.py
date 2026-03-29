"""
RetinaAI Inference Engine
=========================
Three EfficientNet-B3 CNNs + Diffusion-based Anomaly Detector + Meta-Classifier.

Pipeline:
  1. Diff-Mamba UNet (conditioned via RETFound projection) generates anomaly
     map + anomaly score from RGB fundus image using simplex noise
  2. 4-channel tensor [R, G, B, anomaly_map] is fed to 3 CNNs
  3. CNN backbone features (1536 each) + anomaly score → MetaMLP (4609 → 3)
  4. Per-disease probabilities + meta-classifier output → final result
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from models.architectures import (
    create_cnn1_model,
    create_cnn2_model,
    create_cnn3_model,
    MetaMLP,
)

# Lazy import — diffusers is heavy
UNet2DConditionModel = None
DDPMScheduler = None

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
INPUT_SIZE = 256  # diffusion model resolution; CNNs also trained at 256


def _load_diffusers():
    """Lazy-import diffusers so the module can still be imported even if
    diffusers is missing (for tests, etc)."""
    global UNet2DConditionModel, DDPMScheduler
    if UNet2DConditionModel is None:
        from diffusers import UNet2DConditionModel as _U, DDPMScheduler as _S
        UNet2DConditionModel = _U
        DDPMScheduler = _S


# ─── Simplex noise (matches training) ────────────────────────────────────────

def generate_simplex_noise(shape, device, frequency=8, octaves=4):
    """Multi-octave smooth random field — same as training."""
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
    mean = noise.mean(dim=[1, 2, 3], keepdim=True)
    std  = noise.std(dim=[1, 2, 3],  keepdim=True) + 1e-8
    return (noise - mean) / std


# ─── Conditioner projection (loaded from checkpoint) ─────────────────────────

class ConditionerProjection(nn.Module):
    """Trainable projection MLP from training checkpoint."""
    def __init__(self, cross_attention_dim=768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1024, 768),
            nn.GELU(),
            nn.Linear(768, cross_attention_dim),
            nn.LayerNorm(cross_attention_dim),
        )

    def forward(self, device, batch_size=1):
        dummy = torch.zeros(batch_size, 1024, device=device)
        return self.proj(dummy).unsqueeze(1)


# ─── Anomaly Segmentation Head (loaded from checkpoint) ──────────────────────

class AnomalySegHead(nn.Module):
    """Trained to distinguish real lesions from benign reconstruction noise.
    Takes 6-channel input: original(3) + reconstruction(3) → 1-channel sigmoid map.
    Weights saved in checkpoint under 'seg_head'."""
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
        return self.net(torch.cat([original, reconstruction], dim=1))


class RetinaInference:
    """Wraps all four model stages."""

    def __init__(self):
        self.is_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._diffusion_model = None
        self._noise_scheduler = None
        self._conditioner = None
        self._seg_head = None
        self._dr_model = None
        self._glaucoma_model = None
        self._myopia_model = None
        self._meta_model = None

    # ─── Loading ──────────────────────────────────────────────────────────

    def load_models(self) -> None:
        device = self.device

        # ── Stage 1: Diffusion UNet (UNet2DConditionModel) ────────────────
        _load_diffusers()
        diff_path = WEIGHTS_DIR / "diffusion_best.pt"
        print(f"Loading diffusion model from {diff_path} …")
        try:
            ckpt = torch.load(str(diff_path), map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(str(diff_path), map_location=device, weights_only=False)

        # Build the same UNet2DConditionModel as training
        self._diffusion_model = UNet2DConditionModel(
            sample_size=INPUT_SIZE,
            in_channels=3,
            out_channels=3,
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
        ).to(device)

        # Load state dict — checkpoint saves raw UNet weights under 'model'
        raw_sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        if not isinstance(raw_sd, dict):
            raise RuntimeError("Cannot find state dict in diffusion checkpoint")
        state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in raw_sd.items()
        }
        self._diffusion_model.load_state_dict(state_dict)
        self._diffusion_model.eval()

        # Noise scheduler — cosine schedule matching training
        self._noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
        )

        # Load conditioner projection (from checkpoint)
        self._conditioner = ConditionerProjection(cross_attention_dim=768).to(device)
        if "conditioner_proj" in ckpt:
            self._conditioner.proj.load_state_dict(ckpt["conditioner_proj"])
        self._conditioner.eval()

        # Load anomaly segmentation head
        self._seg_head = AnomalySegHead().to(device)
        if "seg_head" in ckpt:
            self._seg_head.load_state_dict(ckpt["seg_head"])
            print("[OK] Anomaly seg head loaded.")
        else:
            print("WARNING: seg_head weights not found in checkpoint.")
        self._seg_head.eval()
        print("[OK] Diffusion model loaded.")

        # ── Stage 2a: DR EfficientNet-B3 ─────────────────────────────────
        dr_path = WEIGHTS_DIR / "cnn1_dr_best.pth"
        print(f"Loading DR model from {dr_path} …")
        self._dr_model = create_cnn1_model(pretrained=False, in_channels=4)
        self._load_cnn_weights(self._dr_model, dr_path)
        self._dr_model.to(device).eval()
        print("[OK] DR model loaded.")

        # ── Stage 2b: Glaucoma EfficientNet-B3 ───────────────────────────
        gl_path = WEIGHTS_DIR / "cnn2_glaucoma_best.pth"
        print(f"Loading Glaucoma model from {gl_path} …")
        self._glaucoma_model = create_cnn2_model(pretrained=False, in_channels=4)
        self._load_cnn_weights(self._glaucoma_model, gl_path)
        self._glaucoma_model.to(device).eval()
        print("[OK] Glaucoma model loaded.")

        # ── Stage 2c: Pathologic Myopia EfficientNet-B3 ──────────────────
        pm_path = WEIGHTS_DIR / "cnn3_pm_best.pth"
        print(f"Loading PM model from {pm_path} …")
        self._myopia_model = create_cnn3_model(pretrained=False, in_channels=4)
        self._load_cnn_weights(self._myopia_model, pm_path)
        self._myopia_model.to(device).eval()
        print("[OK] PM model loaded.")

        # ── Stage 3: Meta-classifier ─────────────────────────────────────
        meta_path = WEIGHTS_DIR / "meta_classifier_best.pth"
        print(f"Loading Meta-classifier from {meta_path} …")
        meta_ckpt = torch.load(str(meta_path), map_location=device, weights_only=True)
        self._meta_model = MetaMLP(input_dim=4609, hidden_dim=512, num_classes=3)
        self._meta_model.load_state_dict(meta_ckpt["model_state_dict"])
        self._meta_model.to(device).eval()
        print("[OK] Meta-classifier loaded.")

        self.is_loaded = True
        print("All models loaded successfully.")

    @staticmethod
    def _load_cnn_weights(model: nn.Module, path: Path):
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    # ─── Inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> dict:
        """
        Run full pipeline on a preprocessed image.

        Parameters
        ----------
        image : np.ndarray  shape (256, 256, 3), dtype float32, values in [0, 1]

        Returns
        -------
        dict with keys:
            attention_map   np.ndarray  (256, 256)  float32  0-1
            anomaly_score   float                             0-1
            predictions     dict  {disease_name: {probability, severity, description}}
        """
        device = self.device

        # ── Stage 1: Anomaly detection via diffusion ──────────────────────
        attention_map, anomaly_score = self._run_diffusion(image)

        # ── Build 4-channel tensor [4, H, W] in [0,1] range ──────────────
        # image is (H, W, 3) in [0,1], attention_map is (H, W) in [0,1]
        four_ch = np.concatenate(
            [image, attention_map[:, :, np.newaxis]], axis=2
        )  # (H, W, 4)
        four_ch_tensor = (
            torch.from_numpy(four_ch)
            .permute(2, 0, 1)  # (4, H, W)
            .unsqueeze(0)      # (1, 4, H, W)
            .float()
            .to(device)
        )

        # ── Stage 2: Disease-specific CNNs ───────────────────────────────
        dr_logits = self._dr_model(four_ch_tensor)            # (1, 5)
        glaucoma_logits = self._glaucoma_model(four_ch_tensor) # (1, 2)
        myopia_logits = self._myopia_model(four_ch_tensor)     # (1, 2)

        # Per-disease probabilities
        dr_probs = F.softmax(dr_logits, dim=1).squeeze(0).cpu().numpy()       # (5,)
        glaucoma_probs = F.softmax(glaucoma_logits, dim=1).squeeze(0).cpu().numpy()  # (2,)
        myopia_probs = F.softmax(myopia_logits, dim=1).squeeze(0).cpu().numpy()      # (2,)

        # "Positive" probability — any DR (classes 1-4) vs class 0
        dr_positive_prob = float(1.0 - dr_probs[0])
        glaucoma_positive_prob = float(glaucoma_probs[1])
        myopia_positive_prob = float(myopia_probs[1])

        # ── Stage 3: Meta-classifier ─────────────────────────────────────
        # Extract 1536-dim backbone features from each CNN
        dr_feats = self._extract_backbone_features(self._dr_model, four_ch_tensor)
        gl_feats = self._extract_backbone_features(self._glaucoma_model, four_ch_tensor)
        pm_feats = self._extract_backbone_features(self._myopia_model, four_ch_tensor)

        # Concatenate: [1536 + 1536 + 1536 + 1] = 4609
        anomaly_tensor = torch.tensor([[anomaly_score]], device=device)
        meta_input = torch.cat([dr_feats, gl_feats, pm_feats, anomaly_tensor], dim=1)  # (1, 4609)

        meta_logits = self._meta_model(meta_input)  # (1, 3)
        meta_probs = F.softmax(meta_logits, dim=1).squeeze(0).cpu().numpy()  # (3,) = [DR, Glaucoma, PM]

        # Use meta-classifier to weight per-CNN probabilities.
        # The meta-classifier is a ROUTING model — it tells us which disease
        # is the primary diagnosis. We use it to SUPPRESS false positives
        # for non-primary diseases rather than inflate them via max().
        #
        # Strategy: weighted blend where the meta-classifier's routing
        # confidence modulates the CNN's per-disease probability.
        # For the primary disease (highest meta score), we trust the CNN.
        # For non-primary diseases, the meta acts as a suppressor.
        primary_idx = int(np.argmax(meta_probs))  # 0=DR, 1=Glaucoma, 2=PM

        cnn_probs_arr = np.array([dr_positive_prob, glaucoma_positive_prob, myopia_positive_prob])

        final_probs = np.zeros(3)
        for i in range(3):
            if i == primary_idx:
                # Primary disease: trust the CNN probability directly
                final_probs[i] = cnn_probs_arr[i]
            else:
                # Non-primary: blend CNN with meta routing confidence as suppressor
                # If meta says <5% chance it's this disease, strongly suppress
                final_probs[i] = cnn_probs_arr[i] * meta_probs[i]

        dr_final = float(np.nan_to_num(np.clip(final_probs[0], 0, 1), nan=0.0))
        glaucoma_final = float(np.nan_to_num(np.clip(final_probs[1], 0, 1), nan=0.0))
        myopia_final = float(np.nan_to_num(np.clip(final_probs[2], 0, 1), nan=0.0))

        # DR severity from the 5-class output
        dr_class = int(np.argmax(dr_probs))

        return {
            "attention_map": attention_map,
            "anomaly_score": anomaly_score,
            "predictions": {
                "diabetic_retinopathy": self._dr_result(dr_final, dr_class),
                "glaucoma": self._glaucoma_result(glaucoma_final),
                "pathologic_myopia": self._myopia_result(myopia_final),
            },
        }

    # ─── Stage implementations ───────────────────────────────────────────

    @staticmethod
    def _make_retinal_mask(image_01: np.ndarray, margin: float = 0.02) -> np.ndarray:
        """Build circular retinal mask. image_01: (H,W,3) in [0,1]. Returns (H,W) float32."""
        H, W = image_01.shape[:2]
        fg = (image_01.mean(axis=2) > 0.05).astype(np.float32)
        cy, cx = H / 2, W / 2
        r = min(H, W) / 2 * (1.0 - margin)
        ys = np.arange(H, dtype=np.float32) - cy
        xs = np.arange(W, dtype=np.float32) - cx
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        circle = ((yy**2 + xx**2) <= r**2).astype(np.float32)
        return circle * fg

    def _run_diffusion(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Generate anomaly map via diffusion reconstruction + trained seg head.
        The seg head was trained to identify real lesions vs reconstruction noise.
        image: (H, W, 3) float32 [0, 1]
        Returns (attention_map (H, W) [0,1], anomaly_score float [0,1])
        """
        device = self.device
        TIMESTEP = 500

        # Normalize to [-1, 1] for diffusion model
        img_tensor = (
            torch.from_numpy(image)
            .permute(2, 0, 1)        # (3, H, W)
            .unsqueeze(0)             # (1, 3, H, W)
            .float()
            .to(device)
        )
        img_tensor = img_tensor * 2.0 - 1.0  # [0,1] → [-1,1]

        t = torch.tensor([TIMESTEP], device=device).long()
        ac = self._noise_scheduler.alphas_cumprod[TIMESTEP].to(device)

        # Generate conditioning from projection MLP
        cond = self._conditioner(device)  # (1, 1, 768)

        # Single-sample reconstruction
        noise = generate_simplex_noise(img_tensor.shape, device)
        noisy = ac.sqrt() * img_tensor + (1 - ac).sqrt() * noise

        pred_noise = self._diffusion_model(noisy, t,
                                            encoder_hidden_states=cond).sample.float()
        recon = (noisy - (1 - ac).sqrt() * pred_noise) / (ac.sqrt() + 1e-8)
        recon = recon.clamp(-1, 1)

        # Use trained seg head: (orig, recon) → learned anomaly map
        with torch.no_grad():
            seg_map = self._seg_head(img_tensor.float(), recon.float())  # (1,1,H,W)

        anomaly_map = seg_map.squeeze().cpu().numpy().astype(np.float32)  # (H,W) in [0,1]

        # Apply retinal mask to exclude borders
        retinal_mask = self._make_retinal_mask(image)
        anomaly_map = anomaly_map * retinal_mask

        # Percentile-based normalization: stretch relative differences
        # so the visualization shows where the seg head sees MORE anomaly
        fg_vals = anomaly_map[retinal_mask > 0.5]
        if len(fg_vals) > 0:
            p_low = np.percentile(fg_vals, 70)   # bottom 70% → near zero
            p_high = np.percentile(fg_vals, 99)  # top 1% → fully bright
            if p_high - p_low > 1e-6:
                anomaly_map = ((anomaly_map - p_low) / (p_high - p_low)).clip(0, 1)
            else:
                anomaly_map = np.zeros_like(anomaly_map)
        anomaly_map = anomaly_map * retinal_mask  # re-mask after stretch

        anomaly_score = float(np.nan_to_num(anomaly_map.mean(), nan=0.0))
        return anomaly_map, anomaly_score

    @staticmethod
    def _extract_backbone_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Run image through EfficientNet backbone (forward_features) to get
        the 1536-dim embedding vector before the classifier head.
        """
        feats = model.forward_features(x)  # (1, C, H, W) or (1, C)
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        return feats  # (1, 1536)

    # ─── Result formatters ────────────────────────────────────────────────

    @staticmethod
    def _dr_result(prob: float, dr_class: int = -1) -> dict:
        """Maps DR probability + class → severity label."""
        # Use the 5-class prediction for severity when available
        severity_map = {
            0: ("No DR", "No signs of diabetic retinopathy detected."),
            1: ("Mild DR", "Microaneurysms present. No vision-threatening changes."),
            2: ("Moderate DR", "More than just microaneurysms. Some vessel changes present."),
            3: ("Severe DR", "Any of the 4-2-1 rule criteria met. High risk of progression."),
            4: ("Proliferative DR", "Neovascularisation detected. Immediate referral recommended."),
        }
        if dr_class in severity_map:
            severity, description = severity_map[dr_class]
        else:
            # Fallback based on probability
            if prob < 0.2:
                severity, description = severity_map[0]
            elif prob < 0.4:
                severity, description = severity_map[1]
            elif prob < 0.6:
                severity, description = severity_map[2]
            elif prob < 0.8:
                severity, description = severity_map[3]
            else:
                severity, description = severity_map[4]
        return {"probability": round(prob, 4), "severity": severity, "description": description}

    @staticmethod
    def _glaucoma_result(prob: float) -> dict:
        if prob < 0.4:
            severity = "No Glaucoma"
            description = "Optic nerve appearance within normal limits."
        elif prob < 0.7:
            severity = "Glaucoma Suspect"
            description = "Cup-to-disc ratio or nerve fibre layer suggests further evaluation."
        else:
            severity = "Glaucoma Detected"
            description = "Significant optic nerve damage consistent with glaucoma."
        return {"probability": round(prob, 4), "severity": severity, "description": description}

    @staticmethod
    def _myopia_result(prob: float) -> dict:
        if prob < 0.4:
            severity = "No Pathologic Myopia"
            description = "Retinal appearance within normal limits."
        elif prob < 0.7:
            severity = "Myopia Suspect"
            description = "Some retinal stretching observed. Monitor recommended."
        else:
            severity = "Pathologic Myopia"
            description = "Significant myopic degeneration detected."
        return {"probability": round(prob, 4), "severity": severity, "description": description}
