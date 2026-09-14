import os, torch, numpy as np, torch.nn as nn
from torch.utils.data import DataLoader
from src.training.losses import FocalLoss
from src.training.metrics import compute_metrics
from src.utils.config import train_config, path_config
from src.utils.logger import logger

class AudioForensicsTrainer:
    def __init__(self, model: nn.Module, model_name: str = "model", lr: float = None, weight_decay: float = None):
        self.model = model
        self.model_name = model_name
        self.device = torch.device(train_config.device)
        self.model.to(self.device)
        self.criterion = FocalLoss(alpha=train_config.focal_alpha, gamma=train_config.focal_gamma)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr or train_config.learning_rate, weight_decay=weight_decay or train_config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=train_config.epochs, eta_min=1e-6)

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for features, targets in train_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * features.size(0)
        return total_loss / len(train_loader.dataset)

    def evaluate(self, val_loader: DataLoader) -> dict:
        self.model.eval()
        all_targets = []
        all_probs = []
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                logits = self.model(features)
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_targets.append(targets.numpy())
                all_probs.append(probs.cpu().numpy())
        y_true = np.concatenate(all_targets)
        y_prob = np.concatenate(all_probs)
        return compute_metrics(y_true, y_prob)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = None) -> dict:
        num_epochs = epochs or train_config.epochs
        best_eer = float("inf")
        history = {"train_loss": [], "val_eer": [], "val_auc": [], "val_acc": []}
        save_path = os.path.join(path_config.artifacts, f"{self.model_name}_best.pth")
        for ep in range(1, num_epochs + 1):
            loss = self.train_epoch(train_loader)
            metrics = self.evaluate(val_loader)
            self.scheduler.step()
            history["train_loss"].append(loss)
            history["val_eer"].append(metrics["eer"])
            history["val_auc"].append(metrics["auc"])
            history["val_acc"].append(metrics["accuracy"])
            logger.info(f"[{self.model_name}] Epoch {ep:02d}/{num_epochs:02d} - Loss: {loss:.4f} - Val EER: {metrics['eer']:.4f} - Val AUC: {metrics['auc']:.4f}")
            if metrics["eer"] < best_eer:
                best_eer = metrics["eer"]
                torch.save(self.model.state_dict(), save_path)
        return history
