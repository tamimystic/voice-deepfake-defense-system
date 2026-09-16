import sys, os, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
torch.set_num_threads(6)
torch.set_num_interop_threads(2)
torch.manual_seed(42)
np.random.seed(42)

from src.models.rawnet_mini import RawNetMini
from src.training.losses import FocalLoss
from src.training.metrics import compute_metrics
from src.utils.config import path_config, train_config
from src.utils.logger import logger

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
TARGET_SAMPLES = 64000
MODEL_NAME = "rawnet_mini"

out_dir = path_config.data_processed
art_dir = path_config.artifacts
ckpt_dir = path_config.checkpoints

train_labels_path = os.path.join(out_dir, "train_labels.npy")
dev_labels_path = os.path.join(out_dir, "dev_labels.npy")
train_manifest_path = os.path.join(out_dir, "train_meta.csv")

import pandas as pd
from src.data.audio_loader import AudioLoader
from src.features.preprocessor import SignalPreprocessor

train_meta = pd.read_csv(train_manifest_path)
full_manifest = pd.read_parquet(os.path.join(out_dir, "protocol_manifest.parquet"))
dev_df = full_manifest[full_manifest.split == "dev"].reset_index(drop=True)
train_df = full_manifest[full_manifest.split == "train"].reset_index(drop=True)


class RawWaveformDataset(Dataset):
    def __init__(self, manifest_df, augment=False):
        self.df = manifest_df.reset_index(drop=True)
        self.loader = AudioLoader()
        self.prep = SignalPreprocessor()
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            y, sr = self.loader.load_raw(row["file_path"])
            y_clean = self.prep.process_pipeline(y, is_training=self.augment)
        except Exception:
            y_clean = np.zeros(TARGET_SAMPLES, dtype=np.float32)
        if len(y_clean) != TARGET_SAMPLES:
            if len(y_clean) < TARGET_SAMPLES:
                y_clean = np.pad(y_clean, (0, TARGET_SAMPLES - len(y_clean)), mode="wrap")
            else:
                y_clean = y_clean[:TARGET_SAMPLES]
        x = torch.from_numpy(y_clean.astype(np.float32)).unsqueeze(0)
        label = torch.tensor(int(row["is_spoof"]), dtype=torch.long)
        return x, label


labels_arr = np.load(train_labels_path, mmap_mode="r")
class_counts = np.bincount(labels_arr)
sample_weights = torch.from_numpy((1.0 / class_counts)[labels_arr].astype(np.float32))
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_ds = RawWaveformDataset(train_df, augment=True)
dev_ds = RawWaveformDataset(dev_df, augment=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

logger.info(f"Train: {len(train_ds)} | Dev: {len(dev_ds)} | Batches/epoch: {len(train_loader)}")
logger.info(f"Note: RawNet loads audio on-the-fly (no pre-extracted feature store needed)")

device = torch.device("cpu")
model = RawNetMini(num_classes=2).to(device)
criterion = FocalLoss(alpha=train_config.focal_alpha, gamma=train_config.focal_gamma)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"RawNet-Mini parameters: {n_params:,}")

best_eer = float("inf")
best_path = os.path.join(art_dir, f"{MODEL_NAME}_best.pth")
history = {"epoch": [], "train_loss": [], "val_eer": [], "val_auc": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, t0 = 0.0, time.time()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    train_loss = total_loss / len(train_ds)
    scheduler.step()

    model.eval()
    probs_all, targets_all = [], []
    with torch.no_grad():
        for xb, yb in dev_loader:
            probs_all.append(torch.softmax(model(xb), dim=1)[:, 1].numpy())
            targets_all.append(yb.numpy())
    m = compute_metrics(np.concatenate(targets_all), np.concatenate(probs_all))

    logger.info(
        f"[Ep {epoch:02d}/{EPOCHS}] loss={train_loss:.4f} "
        f"EER={m['eer']*100:.2f}% AUC={m['auc']:.4f} "
        f"acc={m['accuracy']*100:.1f}% time={time.time()-t0:.0f}s"
    )
    history["epoch"].append(epoch)
    history["train_loss"].append(round(train_loss, 6))
    history["val_eer"].append(round(m["eer"], 6))
    history["val_auc"].append(round(m["auc"], 6))

    if m["eer"] < best_eer:
        best_eer = m["eer"]
        torch.save(model.state_dict(), best_path)
        ep_path = os.path.join(ckpt_dir, f"{MODEL_NAME}_ep{epoch:02d}_eer{best_eer*100:.2f}.pth")
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "eer": best_eer}, ep_path)
        logger.info(f"  Best saved: EER={best_eer*100:.2f}%")

json.dump(history, open(os.path.join(art_dir, f"{MODEL_NAME}_history.json"), "w"), indent=2)
logger.info(f"Done. Best EER on dev: {best_eer*100:.2f}%")
