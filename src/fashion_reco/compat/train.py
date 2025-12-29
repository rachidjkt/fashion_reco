from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from fashion_reco.config import DATA_DIR
from fashion_reco.compat.model import SetEncoder


# ----------------------------
# Config
# ----------------------------

@dataclass
class TrainConfig:
    pairs_path: Path = DATA_DIR / "processed" / "compat_pairs.jsonl"
    embeddings_path: Path = DATA_DIR / "processed" / "embeddings.npy"
    out_dir: Path = DATA_DIR / "processed" / "artifacts"

    epochs: int = 5
    batch_size: int = 256           # number of outfits per batch
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2

    max_items: int = 8              # cap outfit length (Polyvore ~4-6; cap for speed)
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Data
# ----------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


class OutfitDataset(Dataset):
    """
    Each example is:
      - items: list[int] of item indices into embeddings.npy
      - label: 0/1
    """
    def __init__(self, pairs: List[Dict[str, Any]], embeddings: np.ndarray, max_items: int):
        self.pairs = pairs
        self.emb = embeddings
        self.max_items = max_items

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ex = self.pairs[i]
        idxs = ex["items"][: self.max_items]
        label = float(ex["label"])

        x = self.emb[idxs]  # (N, 512) float32 numpy
        x = torch.from_numpy(x)  # torch float32

        y = torch.tensor(label, dtype=torch.float32)
        return x, y


def collate_pad(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pads variable-length sets to (B, Nmax, D) and returns a mask (B, Nmax).

    Returns:
      X: (B, Nmax, D)
      mask: (B, Nmax) bool, True for real items, False for padding
      y: (B,)
    """
    xs, ys = zip(*batch)
    B = len(xs)
    D = xs[0].shape[-1]
    Nmax = max(x.shape[0] for x in xs)

    X = torch.zeros((B, Nmax, D), dtype=torch.float32)
    mask = torch.zeros((B, Nmax), dtype=torch.bool)
    y = torch.stack(ys)

    for i, x in enumerate(xs):
        n = x.shape[0]
        X[i, :n] = x
        mask[i, :n] = True

    return X, mask, y


# ----------------------------
# Metrics
# ----------------------------

@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return float((preds == y).float().mean().item())


# ----------------------------
# Train / Eval
# ----------------------------

def train_one_epoch(model, loader, optimizer, criterion, device: str) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for X, mask, y in loader:
        X = X.to(device)
        mask = mask.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(X, mask)  # (B,)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_acc += accuracy_from_logits(logits, y)
        n_batches += 1

    return {"loss": total_loss / n_batches, "acc": total_acc / n_batches}


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device: str) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for X, mask, y in loader:
        X = X.to(device)
        mask = mask.to(device)
        y = y.to(device)

        logits = model(X, mask)
        loss = criterion(logits, y)

        total_loss += float(loss.item())
        total_acc += accuracy_from_logits(logits, y)
        n_batches += 1

    return {"loss": total_loss / n_batches, "acc": total_acc / n_batches}


def main():
    cfg = TrainConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load embeddings (memmap keeps RAM sane)
    emb = np.load(cfg.embeddings_path, mmap_mode="r")  # shape (N_items, 512)

    # Load pairs and split into train/val
    pairs = load_jsonl(cfg.pairs_path)

    # Simple split: last 10% as val (you can improve later)
    split = int(len(pairs) * 0.9)
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    train_ds = OutfitDataset(train_pairs, emb, max_items=cfg.max_items)
    val_ds = OutfitDataset(val_pairs, emb, max_items=cfg.max_items)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_pad,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_pad,
        pin_memory=True,
    )

    model = SetEncoder(d_in=512, d_hidden=256).to(cfg.device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = -1.0
    best_path = cfg.out_dir / "compat_setencoder.pt"

    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device)
        va = eval_one_epoch(model, val_loader, criterion, cfg.device)

        print(f"Epoch {epoch}/{cfg.epochs} | "
              f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
              f"val loss {va['loss']:.4f} acc {va['acc']:.4f}")

        if va["acc"] > best_val:
            best_val = va["acc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg.__dict__,
                },
                best_path,
            )
            print(f"  saved best -> {best_path} (val acc {best_val:.4f})")

    print("Done.")


if __name__ == "__main__":
    main()
