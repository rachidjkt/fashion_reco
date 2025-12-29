import torch
import torch.nn as nn


class SetEncoder(nn.Module):
    def __init__(self, d_in=512, d_hidden=256):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
        )

        self.rho = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x: (B, N, D)
        mask: (B, N) bool, True for real items
        """
        h = self.phi(x)  # (B, N, H)

        # masked mean pool
        mask_f = mask.unsqueeze(-1).float()  # (B, N, 1)
        h = (h * mask_f).sum(dim=1) / (mask_f.sum(dim=1).clamp_min(1.0))

        score = self.rho(h).squeeze(-1)  # (B,)
        return score





