import torch
import numpy as np
from typing import List
from fashion_reco.compat.model import SetEncoder


class CompatibilityReranker:
    def __init__(self, model: SetEncoder, embeddings: np.ndarray, device: str):
        self.model = model
        self.emb = embeddings
        self.device = device

    def score_outfit(self, item_indices: List[int]) -> float:
        X = torch.from_numpy(self.emb[item_indices]).float().unsqueeze(0)
        mask = torch.ones((1, X.shape[1]), dtype=torch.bool)

        X = X.to(self.device)
        mask = mask.to(self.device)

        with torch.no_grad():
            score = self.model(X, mask).item()
        return score
