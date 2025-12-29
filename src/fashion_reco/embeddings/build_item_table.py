from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from fashion_reco.config import DATA_DIR


SPLITS = ["train", "valid", "test"]


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main():
    processed = DATA_DIR / "processed"

    item_rows = {}

    for split in SPLITS:
        path = processed / f"outfits_{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)

        for outfit in load_jsonl(path):
            for it in outfit["items"]:
                item_id = it["item_id"]
                if item_id not in item_rows:
                    item_rows[item_id] = {
                        "item_id": item_id,
                        "category": it["category"],
                        "category_raw": it.get("category_raw"),
                        "image_path": it.get("image_path"),
                    }

    df = pd.DataFrame(item_rows.values())
    out_path = processed / "items.parquet"
    df.to_parquet(out_path, index=False)

    print("Items:", len(df))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
