from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from fashion_reco.config import DATA_DIR
from fashion_reco.reco.clip_embedder import CLIPImageEmbedder


@dataclass
class ClosetPaths:
    root: Path
    images_dir: Path
    index_path: Path
    emb_path: Path


def get_closet_paths() -> ClosetPaths:
    root = DATA_DIR / "user_closet"
    return ClosetPaths(
        root=root,
        images_dir=root / "images",
        index_path=root / "index.json",
        emb_path=root / "embeddings.npy",
    )


def ensure_closet_dirs():
    p = get_closet_paths()
    p.images_dir.mkdir(parents=True, exist_ok=True)
    if not p.index_path.exists():
        p.index_path.write_text("[]", encoding="utf-8")


def load_index() -> list[dict]:
    p = get_closet_paths()
    ensure_closet_dirs()
    return json.loads(p.index_path.read_text(encoding="utf-8"))


def save_index(items: list[dict]):
    p = get_closet_paths()
    p.index_path.write_text(json.dumps(items, indent=2), encoding="utf-8")


def load_embeddings() -> Optional[np.ndarray]:
    p = get_closet_paths()
    if p.emb_path.exists():
        return np.load(p.emb_path)
    return None


def save_embeddings(E: np.ndarray):
    p = get_closet_paths()
    p.emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(p.emb_path, E)


def rebuild_closet_embeddings(embedder: CLIPImageEmbedder) -> tuple[list[dict], np.ndarray]:
    """
    Rebuild embeddings from data/user_closet/images.
    Keeps existing category assignments by filename if present.
    """
    p = get_closet_paths()
    ensure_closet_dirs()

    old = load_index()
    old_cat = {it.get("filename"): it.get("category", "other") for it in old}

    items = []
    imgs = []

    for fp in sorted(p.images_dir.glob("*")):
        if fp.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        try:
            im = Image.open(fp).convert("RGB")
        except Exception:
            continue

        items.append(
            {
                "filename": fp.name,
                "path": str(fp),
                "category": old_cat.get(fp.name, "other"),
            }
        )
        imgs.append(im)

    if not imgs:
        save_index([])
        save_embeddings(np.zeros((0, 512), dtype=np.float32))
        return [], np.zeros((0, 512), dtype=np.float32)

    E = embedder.embed_many(imgs, batch_size=16).astype(np.float32)
    save_index(items)
    save_embeddings(E)
    return items, E
