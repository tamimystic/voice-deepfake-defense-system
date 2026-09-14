import torch, torch.nn as nn, torch.nn.functional as F

class WeightedEnsemble(nn.Module):
    def __init__(self, weights: list[float] = None):
        super().__init__()
        w = weights or [0.45, 0.35, 0.20]
        self.weights = nn.Parameter(torch.tensor(w, dtype=torch.float32), requires_grad=False)

    def forward(self, logits_list: list[torch.Tensor]) -> torch.Tensor:
        probs_list = [F.softmax(lg, dim=1) for lg in logits_list]
        w_norm = self.weights / self.weights.sum()
        ensemble_prob = sum(w * p for w, p in zip(w_norm, probs_list))
        return ensemble_prob
