import os
import csv
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from diffusers import UNet2DModel, DDPMScheduler

# =========================
# PATHS (ADJUSTED TO YOUR DIR)
# =========================

BASE_DIR = os.getcwd()
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
BEST_MODEL = os.path.join(CHECKPOINT_DIR, "best.pt")
LOSS_CSV = os.path.join(CHECKPOINT_DIR, "loss.csv")

VAL_DIR = os.path.join(BASE_DIR, "val")

IMAGE_SIZE =256
BATCH_SIZE = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# DATASET
# =========================

class RetinaDataset(Dataset):
    def __init__(self, folder):
        self.images = glob(os.path.join(folder, "*.jpeg"))
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        return self.transform(img)

# =========================
# LOAD MODEL
# =========================

model = UNet2DModel(
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

model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
model.eval()

scheduler = DDPMScheduler(num_train_timesteps=1000)

print("Model loaded.")

# =========================
# 1️⃣ LOSS ANALYSIS
# =========================

epochs, train_losses, val_losses = [], [], []

with open(LOSS_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        train_losses.append(float(row["train_loss"]))
        val_losses.append(float(row["val_loss"]))

epochs = np.array(epochs)
train_losses = np.array(train_losses)
val_losses = np.array(val_losses)

def smooth(x, k=3):
    return np.convolve(x, np.ones(k)/k, mode='same')

best_epoch = epochs[np.argmin(val_losses)]

plt.figure(figsize=(8,5))
plt.plot(epochs, train_losses, alpha=0.4, label="Train")
plt.plot(epochs, val_losses, alpha=0.4, label="Val")
plt.plot(epochs, smooth(train_losses), label="Train (Smoothed)")
plt.plot(epochs, smooth(val_losses), label="Val (Smoothed)")
plt.axvline(best_epoch, linestyle='--', label=f"Best Epoch {best_epoch}")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training & Validation Loss")
plt.savefig(os.path.join(CHECKPOINT_DIR, "loss_analysis.png"))
plt.close()

print("Loss analysis saved.")

# =========================
# 2️⃣ RECONSTRUCTION GRID
# =========================

dataset = RetinaDataset(VAL_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

batch = next(iter(loader)).to(device)

noise = torch.randn_like(batch)
timesteps = torch.randint(
    0, scheduler.config.num_train_timesteps,
    (batch.shape[0],), device=device
).long()

noisy = scheduler.add_noise(batch, noise, timesteps)

with torch.no_grad():
    pred_noise = model(noisy, timesteps).sample

# Get alpha values
alphas_cumprod = scheduler.alphas_cumprod.to(device)

alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
sqrt_alpha = torch.sqrt(alpha_t)
sqrt_one_minus_alpha = torch.sqrt(1 - alpha_t)

# Correct x0 estimate
x0_pred = (noisy - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha

# Clamp to valid range
x0_pred = torch.clamp(x0_pred, -1, 1)

recon = x0_pred


grid = make_grid(recon[:8], normalize=True)

plt.figure(figsize=(8,8))
plt.imshow(grid.permute(1,2,0).cpu())
plt.axis("off")
plt.title("Reconstruction Samples")
plt.savefig(os.path.join(CHECKPOINT_DIR, "reconstruction_samples.png"))
plt.close()

print("Reconstruction grid saved.")

# =========================
# 3️⃣ ANOMALY MAP
# =========================

sample = dataset[0].unsqueeze(0).to(device)

def compute_multi_timestep_anomaly(model, scheduler, image, device, num_steps=20):
    model.eval()
    image = image.unsqueeze(0).to(device)

    total_map = 0

    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    for _ in range(num_steps):

        noise = torch.randn_like(image)

        t = torch.tensor([30],device=device).long()  # Fixed timestep for consistency

        noisy = scheduler.add_noise(image, noise, t)

        with torch.no_grad():
            pred_noise = model(noisy, t).sample

        # Per-pixel error
        error_map = torch.mean((pred_noise - noise)**2, dim=1)

        total_map += error_map

    anomaly_map = total_map / num_steps

    return anomaly_map.squeeze().cpu()


anomaly = compute_multi_timestep_anomaly(model, scheduler, dataset[0], device)

plt.imshow(anomaly, cmap='jet')
plt.colorbar()
plt.title("Anomaly Map (Multi-step)")
plt.savefig(os.path.join(CHECKPOINT_DIR, "anomaly_map.png"))
plt.close()


print("Anomaly map saved.")

print("Evaluation complete.")
