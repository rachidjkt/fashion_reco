from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
import open_clip

from fashion_reco.config import DATA_DIR


BATCH_SIZE = 64        # safe for laptop GPU
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model = model.to(device)
    model.eval()

    # Load items
    items_path = DATA_DIR / "processed" / "items.parquet"
    df = pd.read_parquet(items_path)

    images = df["image_path"].tolist()
    item_ids = df["item_id"].tolist()

    embeddings = []
    item_id_to_idx = {}

    with torch.no_grad():
        for i in range(0, len(images), BATCH_SIZE):
            batch_imgs = images[i:i + BATCH_SIZE]

            imgs = []
            for p in batch_imgs:
                if p is None:
                    imgs.append(torch.zeros(3, 224, 224))
                else:
                    img = Image.open(p).convert("RGB")
                    imgs.append(preprocess(img))

            imgs = torch.stack(imgs).to(device)
            feats = model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)

            embeddings.append(feats.cpu().numpy())

    embeddings = np.vstack(embeddings).astype("float32")

    # Save artifacts
    out_dir = DATA_DIR / "processed"
    np.save(out_dir / "embeddings.npy", embeddings)

    for idx, item_id in enumerate(item_ids):
        item_id_to_idx[item_id] = idx

    with (out_dir / "item_id_to_idx.json").open("w") as f:
        json.dump(item_id_to_idx, f)

    print("Embeddings shape:", embeddings.shape)
    print("Saved embeddings + index")


if __name__ == "__main__":
    main()
