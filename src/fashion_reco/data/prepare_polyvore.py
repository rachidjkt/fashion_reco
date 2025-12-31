from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterable, Optional, Tuple

from fashion_reco.config import DATA_DIR


# ----------------------------
# Helpers / config
# ----------------------------

@dataclass(frozen=True)
class PolyvorePaths:
    root: Path                 # .../data/raw/polyvore_outfits
    disjoint: Path             # .../disjoint
    images_dir: Path           # .../images
    item_metadata_json: Path   # .../polyvore_item_metadata.json
    categories_csv: Path       # .../categories.csv

    train_json: Path
    valid_json: Path
    test_json: Path


def guess_polyvore_root() -> Path:
    """
    We assume you placed the dataset under:
      data/raw/polyvore_outfits/...
    But we also try a couple common alternatives.
    """
    candidates = [
        DATA_DIR / "raw" / "polyvore_outfits",
        DATA_DIR / "raw" / "polyvore" / "polyvore_outfits",
        DATA_DIR / "raw" / "polyvore",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not find Polyvore root under data/raw/. "
        "Expected something like data/raw/polyvore_outfits/..."
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_categories_csv(path: Path) -> Dict[str, str]:
    """
    categories.csv typically maps category_id -> category_name.
    We load it defensively (different Kaggle uploads can have slight header differences).
    """
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # Try common column names
        for row in reader:
            # Heuristic: pick the first two columns if names are unknown
            keys = list(row.keys())
            if not keys:
                continue
            # Common:
            # - category_id / category_name
            # - id / name
            cid = row.get("category_id") or row.get("id") or row.get(keys[0])
            cname = row.get("category_name") or row.get("name") or row.get(keys[1]) if len(keys) > 1 else None
            if cid is None or cname is None:
                continue
            mapping[str(cid)] = str(cname)
    return mapping


def normalize_category(raw_name: str) -> str:
    """
    Map Polyvore category names to a small canonical set.
    This makes downstream modeling MUCH easier.

    You can tweak this mapping later without breaking the whole pipeline.
    """
    n = raw_name.strip().lower()

    # shoes
    if any(k in n for k in ["shoe", "boot", "sneaker", "heel", "loafer", "flat", "sandal"]):
        return "shoes"

    # outerwear
    if any(k in n for k in ["coat", "jacket", "parka", "blazer", "cardigan", "outerwear"]):
        return "outerwear"

    # tops
    if any(k in n for k in ["shirt", "t-shirt", "tee", "top", "blouse", "sweater", "hoodie", "tank"]):
        return "top"

    # bottoms
    if any(k in n for k in ["pant", "jean", "trouser", "skirt", "short", "legging"]):
        return "bottom"

    # dresses / jumpsuits -> treat as top+bottom combined; keep as "dress" for now
    if any(k in n for k in ["dress", "jumpsuit", "romper"]):
        return "dress"

    # accessories
    if any(k in n for k in ["bag", "belt", "hat", "cap", "scarf", "watch", "jewelry", "necklace", "earring", "bracelet", "sunglass"]):
        return "accessory"

    return "other"


def build_paths() -> PolyvorePaths:
    root = guess_polyvore_root()
    disjoint = root / "disjoint"
    images_dir = root / "images"
    item_metadata_json = root / "polyvore_item_metadata.json"
    categories_csv = root / "categories.csv"

    train_json = disjoint / "train.json"
    valid_json = disjoint / "valid.json"
    test_json = disjoint / "test.json"

    # Some Kaggle versions drop extensions; handle that too
    if not train_json.exists():
        train_json = disjoint / "train"
    if not valid_json.exists():
        valid_json = disjoint / "valid"
    if not test_json.exists():
        test_json = disjoint / "test"

    for p in [disjoint, item_metadata_json, categories_csv, train_json, valid_json, test_json]:
        if not p.exists():
            raise FileNotFoundError(f"Missing expected path: {p}")

    # images are optional for Day 2, but we check existence for later
    return PolyvorePaths(
        root=root,
        disjoint=disjoint,
        images_dir=images_dir,
        item_metadata_json=item_metadata_json,
        categories_csv=categories_csv,
        train_json=train_json,
        valid_json=valid_json,
        test_json=test_json,
    )


def load_item_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Expect either:
    - dict: { "item_id": {...}, ... }
    - list: [ { "item_id": "...", ... }, ... ]
    We'll convert into dict by item_id.
    """
    data = load_json(path)
    out: Dict[str, Dict[str, Any]] = {}

    if isinstance(data, dict):
        # Sometimes keys are item ids directly
        # Sometimes it's {item_id: {...}} where {...} might not include item_id
        for k, v in data.items():
            if isinstance(v, dict):
                item_id = str(v.get("item_id") or k)
                out[item_id] = v
        return out

    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict) and "item_id" in row:
                out[str(row["item_id"])] = row
        return out

    raise ValueError("polyvore_item_metadata.json has an unexpected structure.")


def find_image_path(images_dir: Path, item_id: str) -> Optional[str]:
    """
    FAST: only checks flat files like images/<item_id>.jpg (no recursive glob).
    We can add a proper index later if your images are nested.
    """
    if not images_dir.exists():
        return None

    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = images_dir / f"{item_id}{ext}"
        if p.exists():
            return str(p)

    return None


def convert_split(
    split_name: str,
    split_json_path: Path,
    item_meta: Dict[str, Dict[str, Any]],
    cat_id_to_name: Dict[str, str],
    images_dir: Path,
) -> Tuple[Iterable[Dict[str, Any]], Dict[str, int]]:
    """
    Returns:
      - iterable of normalized outfit records
      - stats dict
    """
    raw = load_json(split_json_path)
    if not isinstance(raw, list):
        raise ValueError(f"{split_json_path.name} expected to be a list, got {type(raw)}")

    stats = {
        "outfits": 0,
        "items_total": 0,
        "unique_items": 0,
        "missing_item_metadata": 0,
        "missing_category": 0,
        "missing_images": 0,
    }
    seen_items = set()

    def outfits_iter():
        nonlocal stats, seen_items
        for i, outfit in enumerate(raw):
            set_id = str(outfit.get("set_id", f"{split_name}_{i:08d}"))
            items = outfit.get("items", [])
            norm_items = []

            for it in items:
                item_id = str(it.get("item_id"))
                seen_items.add(item_id)

                meta = item_meta.get(item_id)
                if meta is None:
                    stats["missing_item_metadata"] += 1
                    # keep it but mark unknown
                    raw_cat_name = "unknown"
                    norm_cat = "other"
                else:
                    # try to find a category id/name field
                    # different dumps use different keys, so we try several
                    cat_id = (
                        meta.get("category_id")
                        or meta.get("category")
                        or meta.get("categoryid")
                        or meta.get("categoryId")
                    )

                    raw_cat_name = None
                    if cat_id is not None:
                        raw_cat_name = cat_id_to_name.get(str(cat_id))
                    if raw_cat_name is None:
                        # sometimes metadata stores category name directly
                        raw_cat_name = meta.get("category_name") or meta.get("categoryName")

                    if raw_cat_name is None:
                        stats["missing_category"] += 1
                        raw_cat_name = "unknown"

                    norm_cat = normalize_category(str(raw_cat_name))

                img_path = find_image_path(images_dir, item_id)
                if images_dir.exists() and img_path is None:
                    stats["missing_images"] += 1

                norm_items.append(
                    {
                        "item_id": item_id,
                        "category_raw": str(raw_cat_name),
                        "category": norm_cat,
                        "image_path": img_path,  # can be None for now
                        "order_index": it.get("index"),
                    }
                )

            stats["outfits"] += 1
            stats["items_total"] += len(norm_items)

            yield {
                "outfit_id": f"{split_name}_{set_id}",
                "split": split_name,
                "set_id": set_id,
                "items": norm_items,
            }

    records = outfits_iter()
    stats["unique_items"] = len(seen_items)
    return records, stats


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    paths = build_paths()
    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_id_to_name = load_categories_csv(paths.categories_csv)
    item_meta = load_item_metadata(paths.item_metadata_json)

    print("Polyvore root:", paths.root)
    print("Loaded categories:", len(cat_id_to_name))
    print("Loaded item metadata:", len(item_meta))
    print("Images dir exists:", paths.images_dir.exists())

    for split_name, split_path in [
        ("train", paths.train_json),
        ("valid", paths.valid_json),
        ("test", paths.test_json),
    ]:
        records, stats = convert_split(
            split_name=split_name,
            split_json_path=split_path,
            item_meta=item_meta,
            cat_id_to_name=cat_id_to_name,
            images_dir=paths.images_dir,
        )

        out_path = out_dir / f"outfits_{split_name}.jsonl"
        write_jsonl(out_path, records)
        seen = set()
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                for it in obj["items"]:
                    seen.add(it["item_id"])
        stats["unique_items"] = len(seen)

        # Print sanity stats
        avg_items = stats["items_total"] / max(stats["outfits"], 1)
        print("\n==", split_name.upper(), "==")
        print("Outfits:", stats["outfits"])
        print("Avg items/outfit:", round(avg_items, 2))
        print("Unique items:", stats["unique_items"])
        print("Missing item metadata:", stats["missing_item_metadata"])
        print("Missing category:", stats["missing_category"])
        if paths.images_dir.exists():
            print("Missing images:", stats["missing_images"])
        print("Wrote:", out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()