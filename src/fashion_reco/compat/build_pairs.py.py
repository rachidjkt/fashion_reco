from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from fashion_reco.config import DATA_DIR

RANDOM_SEED = 42
NEG_PER_POS = 1          # 1 negative per positive
HARD_NEG_K = 50          # sample K same-category candidates and pick most similar
SPLITS = ["train", "valid"]


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def pick_hard_negative(
    orig_idx: int,
    cat_pool: List[int],
    emb: np.ndarray,
    outfit_indices_set: set[int],
    k: int = 50,
) -> Optional[int]:
    """
    Choose a hard negative from cat_pool:
      - not the original item
      - not already in the outfit
      - maximize cosine similarity to original (dot product because CLIP embeddings are normalized)
    Returns None if no valid replacement found.
    """
    if not cat_pool:
        return None

    # sample candidate indices
    m = min(k, len(cat_pool))
    # random sample with replacement to keep it fast even for huge pools
    candidates = [cat_pool[np.random.randint(0, len(cat_pool))] for _ in range(m)]
    candidates = [c for c in candidates if c != orig_idx and c not in outfit_indices_set]
    if not candidates:
        return None

    v = emb[orig_idx]         # (512,)
    C = emb[candidates]       # (m, 512)
    sims = C @ v              # (m,)
    best = candidates[int(np.argmax(sims))]
    return best


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    processed = DATA_DIR / "processed"

    # Load item table + id->idx
    items_df = pd.read_parquet(processed / "items.parquet")
    item_id_to_idx: Dict[str, int] = json.load((processed / "item_id_to_idx.json").open("r"))

    # Embeddings (memmap)
    emb = np.load(processed / "embeddings.npy", mmap_mode="r")

    # Group item indices by normalized category
    cat_to_indices: Dict[str, List[int]] = {}
    for _, row in items_df.iterrows():
        item_id = str(row["item_id"])
        cat = str(row["category"])
        idx = item_id_to_idx.get(item_id)
        if idx is None:
            continue
        cat_to_indices.setdefault(cat, []).append(idx)

    pairs: List[Dict[str, Any]] = []

    for split in SPLITS:
        in_path = processed / f"outfits_{split}.jsonl"

        for outfit in load_jsonl(in_path):
            # IMPORTANT FIX:
            # Filter items first, then derive indices from the same list to keep positions aligned
            filtered_items = [it for it in outfit["items"] if it["item_id"] in item_id_to_idx]
            indices = [item_id_to_idx[it["item_id"]] for it in filtered_items]

            if len(indices) < 2:
                continue

            # positive example
            pairs.append({"items": indices, "label": 1})

            # negatives
            for _ in range(NEG_PER_POS):
                neg = indices.copy()

                # choose which position to replace
                j = random.randrange(len(neg))
                cat = filtered_items[j]["category"]
                if cat not in cat_to_indices:
                    continue

                orig_idx = neg[j]
                outfit_set = set(neg)

                replacement = pick_hard_negative(
                    orig_idx=orig_idx,
                    cat_pool=cat_to_indices[cat],
                    emb=emb,
                    outfit_indices_set=outfit_set,
                    k=HARD_NEG_K,
                )

                if replacement is None:
                    continue

                neg[j] = replacement
                pairs.append({"items": neg, "label": 0})

    out_path = processed / "compat_pairs.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    # quick label stats
    labels = [p["label"] for p in pairs]
    pos = sum(labels)
    neg = len(labels) - pos
    print("Saved pairs:", len(pairs))
    print(f"Pos: {pos} | Neg: {neg}")
    print("Path:", out_path)


if __name__ == "__main__":
    main()
