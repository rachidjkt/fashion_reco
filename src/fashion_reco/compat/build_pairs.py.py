from __future__ import annotations
import json
import random
from pathlib import Path

import pandas as pd
from fashion_reco.config import DATA_DIR

RANDOM_SEED = 42
NEG_RATIO = 1.0  # 1 negative per positive


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main():
    random.seed(RANDOM_SEED)

    processed = DATA_DIR / "processed"

    items_df = pd.read_parquet(processed / "items.parquet")
    item_id_to_idx = json.load((processed / "item_id_to_idx.json").open())

    # group items by category for category-aware negatives
    cat_to_indices = {}
    for _, row in items_df.iterrows():
        cat_to_indices.setdefault(row["category"], []).append(
            item_id_to_idx[row["item_id"]]
        )

    pairs = []

    for split in ["train", "valid"]:
        for outfit in load_jsonl(processed / f"outfits_{split}.jsonl"):
            indices = [
                item_id_to_idx[it["item_id"]]
                for it in outfit["items"]
                if it["item_id"] in item_id_to_idx
            ]

            if len(indices) < 2:
                continue

            # positive
            pairs.append({"items": indices, "label": 1})

            # negative(s)
            for _ in range(int(NEG_RATIO)):
                neg = indices.copy()
                j = random.randrange(len(neg))
                cat = outfit["items"][j]["category"]

                if cat in cat_to_indices:
                    neg[j] = random.choice(cat_to_indices[cat])
                    pairs.append({"items": neg, "label": 0})

    out = processed / "compat_pairs.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print("Saved pairs:", len(pairs))
    print("Path:", out)


if __name__ == "__main__":
    main()
