import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """
    Permutation-invariant attention pooling over set elements.
    Produces a weighted sum of element features.
    """
    def __init__(self, d_hidden: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        h: (B, N, H)
        mask: (B, N) bool; True for real items, False for padding
        """
        logits = self.score(h).squeeze(-1)  # (B, N)
        logits = logits.masked_fill(~mask, -1e9)
        w = torch.softmax(logits, dim=1)    # (B, N)
        pooled = (h * w.unsqueeze(-1)).sum(dim=1)  # (B, H)
        return pooled


class SetEncoder(nn.Module):
    """
    DeepSets-style encoder:
      - phi: per-item transform
      - attention pooling
      - rho: compatibility head
    """
    def __init__(self, d_in=512, d_hidden=256):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        )
        self.pool = AttentionPool(d_hidden)
        self.rho = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        mask: (B, N) bool
        returns logits: (B,)
        """
        h = self.phi(x)               # (B, N, H)
        pooled = self.pool(h, mask)   # (B, H)
        logits = self.rho(pooled).squeeze(-1)  # (B,)
        return logits





