import json
import numpy as np
import torch
import pandas as pd

from fashion_reco.config import DATA_DIR
from fashion_reco.compat.model import SetEncoder
from fashion_reco.reco.retriever import CLIPRetriever
from fashion_reco.reco.reranker import CompatibilityReranker
from fashion_reco.reco.pipeline import OutfitRecommender


def main():
    processed = DATA_DIR / "processed"
    print("[demo] starting...")
    print("[demo] processed:", processed)

    # ---- Load embeddings + id maps ----
    emb = np.load(processed / "embeddings.npy", mmap_mode="r")
    print("[demo] embeddings shape:", emb.shape)

    item_id_to_idx = json.load((processed / "item_id_to_idx.json").open("r", encoding="utf-8"))
    idx_to_item_id = {v: k for k, v in item_id_to_idx.items()}
    print("[demo] items indexed:", len(item_id_to_idx))

    # ---- Load item metadata from PARQUET ----
    items_df = pd.read_parquet(processed / "items.parquet")
    print("[demo] items.parquet rows:", len(items_df))
    print("[demo] items.parquet columns:", list(items_df.columns))

    # FAST category map construction (no iterrows)
    # Keep only rows whose item_id is in item_id_to_idx
    items_df["item_id"] = items_df["item_id"].astype(str)
    items_df = items_df[items_df["item_id"].isin(item_id_to_idx)]

    # map: item_index -> category
    item_categories = {
        item_id_to_idx[iid]: cat
        for iid, cat in zip(items_df["item_id"].tolist(), items_df["category"].tolist())
    }
    print("[demo] categories mapped:", len(item_categories))
    print("[demo] unique categories (sample):", sorted(set(item_categories.values()))[:20])

    # ---- Load compatibility model ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[demo] device:", device)

    model = SetEncoder(d_in=512, d_hidden=256)
    state = torch.load(processed / "artifacts" / "compat_setencoder.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print("[demo] loaded model checkpoint")

    # ---- Build recommender ----
    retriever = CLIPRetriever(emb)
    reranker = CompatibilityReranker(model, emb, device)

    reco = OutfitRecommender(
        retriever=retriever,
        reranker=reranker,
        item_categories=item_categories,
    )
    print("[demo] built pipeline")

    # ---- Seed outfit ----
    seed_items = list(item_id_to_idx.values())[:3]
    print("[demo] seed indices:", seed_items)
    print("[demo] seed item_ids:", [idx_to_item_id[i] for i in seed_items])

    # 1) NO FILTER run (guarantee output)
    print("\n[demo] recommend (no category filter)...")
    results = reco.recommend(
        seed_items=seed_items,
        allowed_categories=None,   # <-- requires pipeline tweak below
        top_k=10,
    )
    print("[demo] got results:", len(results))
    for idx, score in results[:10]:
        print(f"  item_id={idx_to_item_id[idx]}  score={score:.4f}  cat={item_categories.get(idx)}")

    # 2) FILTERED run (optional)
    allowed = ["top", "bottom", "shoes", "outerwear", "accessory", "dress"]
    print("\n[demo] recommend (filtered categories:", allowed, ")...")
    results2 = reco.recommend(
        seed_items=seed_items,
        allowed_categories=allowed,
        top_k=10,
        candidate_k=500,
        max_per_category=1,
    )
    print("[demo] got filtered results:", len(results2))
    for idx, score in results2[:10]:
        print(f"  item_id={idx_to_item_id[idx]}  score={score:.4f}  cat={item_categories.get(idx)}")


if __name__ == "__main__":
    main()
