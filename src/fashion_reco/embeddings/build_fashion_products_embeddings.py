from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from fashion_reco.config import DATA_DIR
from fashion_reco.reco.clip_embedder import CLIPImageEmbedder


def open_rgb(path: str):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def main():
    processed = DATA_DIR / "processed"
    items_path = processed / "fashion_products_items.parquet"
    if not items_path.exists():
        raise FileNotFoundError(f"Missing {items_path}. Run prepare_fashion_products.py first.")

    df = pd.read_parquet(items_path)
    df["item_id"] = df["item_id"].astype(str)
    df["image_path"] = df["image_path"].astype(str)

    # --- knobs (set via env vars) ---
    device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(os.environ.get("BATCH", "32"))     # try 16/32; if unstable use 8
    save_every = int(os.environ.get("SAVE_EVERY", "50")) # batches
    dim = int(os.environ.get("D", "512"))

    out_items = processed / "fashion_products_items_indexed.parquet"
    out_emb = processed / "fashion_products_embeddings.npy"
    out_done = processed / "fashion_products_embeddings.done.npy"

    print("[emb] device:", device)
    print("[emb] rows:", len(df))
    print("[emb] batch_size:", batch_size)

    embedder = CLIPImageEmbedder(device=device)

    # Write embeddings to disk (memmap) so RAM stays flat
    n = len(df)
    if out_emb.exists() and out_done.exists():
        E = np.memmap(out_emb, mode="r+", dtype="float32", shape=(n, dim))
        done = np.load(out_done)
        if done.shape[0] != n:
            raise RuntimeError("done-mask length mismatch; delete embeddings + done file and rerun.")
        print(f"[emb] resume: {int(done.sum())}/{n} done")
    else:
        E = np.memmap(out_emb, mode="w+", dtype="float32", shape=(n, dim))
        done = np.zeros((n,), dtype=np.bool_)
        np.save(out_done, done)
        print("[emb] starting fresh")

    imgs = []
    idxs = []
    bad = 0
    batches = 0

    def flush():
        nonlocal batches
        if not imgs:
            return
        # embed_many expects list[PIL.Image]
        vecs = embedder.embed_many(imgs, batch_size=len(imgs)).astype(np.float32)  # (B,512)
        for j, row_i in enumerate(idxs):
            E[row_i] = vecs[j]
            done[row_i] = True
        imgs.clear()
        idxs.clear()

        batches += 1
        if batches % save_every == 0:
            E.flush()
            np.save(out_done, done)
            print(f"[emb] progress: {int(done.sum())}/{n}  bad={bad}")

    for i, row in enumerate(df.itertuples(index=False)):
        if done[i]:
            continue

        img = open_rgb(row.image_path)
        if img is None:
            bad += 1
            done[i] = True
            continue

        imgs.append(img)
        idxs.append(i)

        if len(imgs) >= batch_size:
            flush()

    flush()
    E.flush()
    np.save(out_done, done)

    # Save indexed items (same row order as embeddings)
    df.to_parquet(out_items, index=False)

    print("[emb] done:", int(done.sum()), "/", n, "bad:", bad)
    print("[emb] saved items:", out_items)
    print("[emb] saved embeddings:", out_emb)


if __name__ == "__main__":
    main()
