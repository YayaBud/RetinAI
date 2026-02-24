import os
import csv
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from PIL import Image
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.amp import autocast, GradScaler
from diffusers import UNet2DModel, DDPMScheduler

# =========================
# CONFIGURATION
# =========================
def main():
        
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATA_TRAIN = os.path.join(BASE_DIR, "train")
    DATA_VAL = os.path.join(BASE_DIR, "val")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

    IMAGE_SIZE = 256
    EPOCHS = 60
    BATCH_SIZE = 4
    LR = 1e-4
    EARLY_STOP_PATIENCE = 10
    SAVE_EVERY = 5

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # =========================
    # DATASET
    # =========================

    class RetinaDataset(Dataset):
        def __init__(self, folder):
            self.images = glob(os.path.join(folder, "*.jpeg"))
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],
                                    [0.5, 0.5, 0.5])
            ])

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            while True:
                img_path = self.images[idx]
                try:
                    img = Image.open(img_path).convert("RGB")
                    return self.transform(img)
                except Exception:
                    print(f"Skipping corrupted image: {img_path}")
                    idx = (idx + 1) % len(self.images)

    # Create datasets
    train_dataset = RetinaDataset(DATA_TRAIN)
    val_dataset = RetinaDataset(DATA_VAL)

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Train or Val dataset is empty.")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # =========================
    # MODEL
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

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler(device)

    # =========================
    # RESUME SUPPORT
    # =========================

    start_epoch = 0
    best_val_loss = float("inf")

    last_checkpoint = os.path.join(CHECKPOINT_DIR, "last.pt")
    best_checkpoint = os.path.join(CHECKPOINT_DIR, "best.pt")
    loss_csv = os.path.join(CHECKPOINT_DIR, "loss.csv")

    if os.path.exists(last_checkpoint):
        print("Resuming from last checkpoint...")
        model.load_state_dict(torch.load(last_checkpoint, map_location=device))

        if os.path.exists(loss_csv):
            with open(loss_csv) as f:
                rows = list(csv.reader(f))
                if len(rows) > 1:
                    start_epoch = int(rows[-1][0]) + 1

    # =========================
    # LOSS LOGGING
    # =========================

    if not os.path.exists(loss_csv):
        with open(loss_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])

    train_losses = []
    val_losses = []
    epochs_without_improvement = 0

    # =========================
    # TRAINING LOOP
    # =========================

    for epoch in range(start_epoch, EPOCHS):

        # ---------- TRAIN ----------
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            batch = batch.to(device)

            noise = torch.randn_like(batch)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch.shape[0],),
                device=device
            ).long()

            noisy_images = noise_scheduler.add_noise(batch, noise, timesteps)

            optimizer.zero_grad()

            with autocast(device_type="cuda"):
                noise_pred = model(noisy_images, timesteps).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        # ---------- VALIDATION ----------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                noise = torch.randn_like(batch)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch.shape[0],),
                    device=device
                ).long()

                noisy_images = noise_scheduler.add_noise(batch, noise, timesteps)

                with autocast(device_type="cuda"):
                    noise_pred = model(noisy_images, timesteps).sample
                    loss = torch.nn.functional.mse_loss(noise_pred, noise)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        # ---------- LOG ----------
        with open(loss_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss])

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # ---------- SAVE LAST ----------
        if epoch % SAVE_EVERY == 0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss
        }, last_checkpoint)


        # ---------- SAVE BEST ----------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss
        }, last_checkpoint)
            epochs_without_improvement = 0
            print("Best model updated.")
        else:
            epochs_without_improvement += 1

        # ---------- EARLY STOP ----------
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break

        # ---------- PLOT ----------
        plt.figure()
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Val")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.savefig(os.path.join(CHECKPOINT_DIR, "loss_curve.png"))
        plt.close()

    print("Training complete.")

if __name__ == "__main__":
    main()
