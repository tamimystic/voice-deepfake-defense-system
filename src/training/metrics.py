import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support

def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> tuple[float, float]:
    y_true = np.concatenate([np.zeros(len(bonafide_scores)), np.ones(len(spoof_scores))])
    y_scores = np.concatenate([bonafide_scores, spoof_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]
    return float(eer), float(threshold)

def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_pred_prob >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = np.mean(y_true == y_pred)
    auc = roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0.5
    bon_scores = y_pred_prob[y_true == 0]
    spf_scores = y_pred_prob[y_true == 1]
    eer, eer_thresh = compute_eer(bon_scores, spf_scores) if len(bon_scores) > 0 and len(spf_scores) > 0 else (0.0, 0.5)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "auc": float(auc),
        "eer": float(eer),
        "eer_threshold": float(eer_thresh)
    }
