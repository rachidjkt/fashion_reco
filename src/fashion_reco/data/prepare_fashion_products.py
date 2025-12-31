from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fashion_reco.config import DATA_DIR


def main():
    """
    Expected folder:
      data/raw/fashion_products/
        images/
        styles.csv

    Output:
      data/processed/fashion_products_items.parquet
    """
    raw_root = DATA_DIR / "raw" / "fashion_products" / "fashion-dataset"
    images_dir = raw_root / "images"
    styles_csv = raw_root / "styles.csv"

    if not styles_csv.exists():
        raise FileNotFoundError(f"Missing {styles_csv}")

    df = pd.read_csv(
    styles_csv,
    engine="python",       
    on_bad_lines="skip",   # skip malformed rows
)

    # Typical columns: id, gender, masterCategory, subCategory, articleType, baseColour...
    df["id"] = df["id"].astype(str)

    # Keep men items (you can allow both via UI later)
    men = df[df["gender"].astype(str).str.lower().str.contains("men", na=False)].copy()

    def img_path(pid: str) -> str:
        # In this dataset images are often named like: 12345.jpg
        p = images_dir / f"{pid}.jpg"
        return str(p) if p.exists() else ""

    men["image_path"] = men["id"].apply(img_path)
    men = men[men["image_path"] != ""]

    # Map to your categories
    def map_cat(row) -> str:
        article = str(row.get("articleType", "")).lower()
        master = str(row.get("masterCategory", "")).lower()

        if "shoe" in article or "footwear" in master:
            return "shoes"
        if any(x in article for x in ["jean", "trouser", "pant", "short", "bottom"]):
            return "bottom"
        if any(x in article for x in ["jacket", "coat", "blazer", "hoodie", "sweater", "outerwear"]):
            return "outerwear"
        if any(x in article for x in ["shirt", "t-shirt", "tee", "top", "polo"]):
            return "top"
        if any(x in article for x in ["watch", "belt", "cap", "hat", "bag", "sunglass", "wallet"]):
            return "accessory"
        return "other"

    men["category"] = men.apply(map_cat, axis=1)

    out = DATA_DIR / "processed" / "fashion_products_items.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    keep = men[["id", "gender", "masterCategory", "subCategory", "articleType", "category", "image_path"]].copy()
    keep.rename(columns={"id": "item_id"}, inplace=True)
    keep.to_parquet(out, index=False)

    print("Saved:", out)
    print("Rows:", len(keep))
    print("Category counts:\n", keep["category"].value_counts().head(20))
    print("styles rows:", len(df))
    print("images dir:", images_dir, "exists:", images_dir.exists())


if __name__ == "__main__":
    main()
