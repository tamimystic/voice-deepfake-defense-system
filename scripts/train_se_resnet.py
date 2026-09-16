import sys, os, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
torch.set_num_threads(6)
torch.set_num_interop_threads(2)
torch.manual_seed(42)
np.random.seed(42)

from src.models.resnet_spec import SEResNet18
from src.training.losses import FocalLoss
from src.training.metrics import compute_metrics
from src.utils.config import path_config, train_config
from src.utils.logger import logger

BATCH_SIZE = 16
EPOCHS = 20
LR = 5e-4
WEIGHT_DECAY = 1e-4
MEL_SHAPE = (128, 251)
MODEL_NAME = "se_resnet_mel"

out_dir = path_config.data_processed
art_dir = path_config.artifacts
ckpt_dir = path_config.checkpoints

train_mel_path = os.path.join(out_dir, "train_mel.npy")
train_labels_path = os.path.join(out_dir, "train_labels.npy")

for p in [train_mel_path, train_labels_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing: {p}")

dev_mel_path = os.path.join(out_dir, "dev_mel.npy")
dev_labels_path = os.path.join(out_dir, "dev_labels.npy")

if not os.path.exists(dev_mel_path):
    logger.info("Building dev Mel feature store...")
    import pandas as pd
    from src.data.audio_loader import AudioLoader
    from src.features.preprocessor import SignalPreprocessor
    from src.features.spectral import SpectralFeatureExtractor

    manifest = pd.read_parquet(os.path.join(out_dir, "protocol_manifest.parquet"))
    dev_df = manifest[manifest.split == "dev"].reset_index(drop=True)
    N = len(dev_df)
    loader, prep, spec = AudioLoader(), SignalPreprocessor(), SpectralFeatureExtractor()
    dev_mel_mm = np.lib.format.open_memmap(dev_mel_path, mode="w+", dtype=np.float32, shape=(N, *MEL_SHAPE))
    for i, (_, row) in enumerate(dev_df.iterrows()):
        try:
            y, sr = loader.load_raw(row["file_path"])
            y_c = prep.process_pipeline(y, is_training=False)
            mel = spec.extract_log_mel(y_c, sr=sr)
            if mel.shape[1] != MEL_SHAPE[1]:
                mel = np.pad(mel, ((0,0),(0,MEL_SHAPE[1]-mel.shape[1])), mode="edge") if mel.shape[1] < MEL_SHAPE[1] else mel[:,:MEL_SHAPE[1]]
            dev_mel_mm[i] = mel.astype(np.float32)
        except Exception:
            dev_mel_mm[i] = 0.0
        if (i+1) % 2000 == 0:
            logger.info(f"  Dev Mel: {i+1}/{N}")
    dev_mel_mm.flush()
    logger.info("Dev Mel store complete.")


class MemmapMelDataset(Dataset):
    def __init__(self, mel_path, labels_path, augment=False):
        self.X = np.load(mel_path, mmap_mode="r")
        self.y = np.load(labels_path, mmap_mode="r")
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].copy().astype(np.float32)
        if self.augment:
            if np.random.rand() < 0.5:
                t_w = np.random.randint(1, max(2, int(0.1 * x.shape[1])))
                t_s = np.random.randint(0, x.shape[1] - t_w)
                x[:, t_s:t_s + t_w] = x.mean()
            if np.random.rand() < 0.5:
                f_w = np.random.randint(1, max(2, int(0.15 * x.shape[0])))
                f_s = np.random.randint(0, x.shape[0] - f_w)
                x[f_s:f_s + f_w, :] = x.mean()
        return torch.from_numpy(x).unsqueeze(0), torch.tensor(int(self.y[idx]), dtype=torch.long)


labels_arr = np.load(train_labels_path, mmap_mode="r")
class_counts = np.bincount(labels_arr)
sample_weights = torch.from_numpy((1.0 / class_counts)[labels_arr].astype(np.float32))
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_ds = MemmapMelDataset(train_mel_path, train_labels_path, augment=True)
dev_ds = MemmapMelDataset(dev_mel_path, dev_labels_path, augment=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

logger.info(f"Train: {len(train_ds)} | Dev: {len(dev_ds)} | Batches/epoch: {len(train_loader)}")

device = torch.device("cpu")
model = SEResNet18(in_channels=1, num_classes=2).to(device)
criterion = FocalLoss(alpha=train_config.focal_alpha, gamma=train_config.focal_gamma)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"SE-ResNet-18 parameters: {n_params:,}")

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
