import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch

from fashion_reco.config import DATA_DIR
from fashion_reco.compat.model import SetEncoder
from fashion_reco.reco.retriever import CLIPRetriever
from fashion_reco.reco.reranker import CompatibilityReranker


def load_outfits_jsonl(path: Path, max_items: int = 6) -> List[List[str]]:
    outfits: List[List[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            ids = [str(it["item_id"]) for it in ex["items"]][:max_items]
            if len(ids) >= 3:
                outfits.append(ids)
    return outfits
def multi_query_retrieve(
    retriever,
    emb: np.ndarray,
    seed: List[int],
    k: int,
) -> List[int]:
    """
    Retrieve candidates using multiple query embeddings and union results.
    """
    candidates = set()

    # 1) mean embedding
    mean_q = np.mean(emb[seed], axis=0)
    candidates.update(retriever.retrieve(mean_q, k=k))

    # 2) max embedding
    max_q = np.max(emb[seed], axis=0)
    candidates.update(retriever.retrieve(max_q, k=k))

    # 3) individual seed items
    for idx in seed:
        candidates.update(retriever.retrieve(emb[idx], k=k))

    return list(candidates)

def mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def rank_of(target: int, ordered: List[int]) -> int:
    """0-based rank; returns large number if not found."""
    try:
        return ordered.index(target)
    except ValueError:
        return 10**9


def mrr_from_rank(r: int) -> float:
    return 0.0 if r >= 10**8 else 1.0 / (r + 1)


def main():
    random.seed(0)
    np.random.seed(0)

    processed = DATA_DIR / "processed"
    valid_path = processed / "outfits_valid.jsonl"

    print("[eval] loading embeddings + item index")
    emb = np.load(processed / "embeddings.npy", mmap_mode="r")
    item_id_to_idx: Dict[str, int] = json.load((processed / "item_id_to_idx.json").open("r", encoding="utf-8"))
    print("[eval] embeddings:", emb.shape, "| indexed items:", len(item_id_to_idx))

    print("[eval] loading item categories")
    items_df = pd.read_parquet(processed / "items.parquet")
    items_df["item_id"] = items_df["item_id"].astype(str)
    items_df = items_df[items_df["item_id"].isin(item_id_to_idx)]
    item_categories = {item_id_to_idx[iid]: cat for iid, cat in zip(items_df["item_id"], items_df["category"])}

    allowed = ["top", "bottom", "shoes", "outerwear", "accessory", "dress", "other"]
    allowed_set = set(allowed)

    print("[eval] loading compatibility model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SetEncoder(d_in=512, d_hidden=256)
    model.load_state_dict(torch.load(processed / "artifacts" / "compat_setencoder.pt", map_location="cpu"))
    model.to(device)
    model.eval()
    print("[eval] device:", device)

    retriever = CLIPRetriever(emb)
    reranker = CompatibilityReranker(model, emb, device)

    print("[eval] loading validation outfits")
    outfits = load_outfits_jsonl(valid_path, max_items=6)
    random.shuffle(outfits)
    outfits = outfits[:200]
    print("[eval] eval outfits:", len(outfits))

    retrieve_k = 2000      # big pool improves chance GT is included
    pool_n = 500           # rerank top N clip candidates
    K_list = [10, 50]

    recall_clip = {K: 0 for K in K_list}
    recall_rerank = {K: 0 for K in K_list}
    mrr_clip = []
    mrr_rerank = []
    rank_improvement = []  # clip_rank - rerank_rank (positive = better)

    used = 0
    gt_in_pool = 0

    print("[eval] running outfit completion eval (Recall@K, MRR)...")

    for ids in outfits:
        # map to indices
        idxs = [item_id_to_idx[i] for i in ids if i in item_id_to_idx]
        if len(idxs) < 3:
            continue

        # choose a blank position; true item is gt
        blank = random.randrange(len(idxs))
        gt = idxs[blank]
        seed = [x for j, x in enumerate(idxs) if j != blank]
        seed_set = set(seed)

        # retrieve candidates from seed
        query_emb = np.mean(emb[seed], axis=0)
        candidates = multi_query_retrieve(
            retriever=retriever,
            emb=emb,
            seed=seed,
            k=retrieve_k // 2,   # per-query budget
        )  

        # filter: remove seed, keep allowed categories
        candidates = [c for c in candidates if c not in seed_set and item_categories.get(c) in allowed_set]

        # ensure GT is included in pool (otherwise can't evaluate ranking)
        if gt not in candidates:
            continue
        gt_in_pool += 1

        # CLIP order
        sims = emb[candidates] @ query_emb
        clip_order = [candidates[i] for i in np.argsort(-sims)]

        # take pool_n for reranking (but keep GT inside)
        pool = clip_order[:pool_n]
        if gt not in pool:
            # force include GT by replacing last element (keeps pool size)
            pool[-1] = gt

        # Rerank by compatibility
        comp_scores = []
        for c in pool:
            comp_scores.append(float(reranker.score_outfit(seed + [c])))

        rerank_order = [pool[i] for i in np.argsort(-np.array(comp_scores))]

        # ranks
        r_clip = rank_of(gt, pool)            # rank within pool by CLIP
        r_rerank = rank_of(gt, rerank_order)  # rank within pool by rerank

        # metrics
        for K in K_list:
            if r_clip < K:
                recall_clip[K] += 1
            if r_rerank < K:
                recall_rerank[K] += 1

        mrr_clip.append(mrr_from_rank(r_clip))
        mrr_rerank.append(mrr_from_rank(r_rerank))
        rank_improvement.append(float(r_clip - r_rerank))

        used += 1

    print("\n========== RESULTS (completion) ==========")
    print(f"Target eval outfits: 200")
    print(f"Evaluated (GT in retrieved pool): {used}")
    print(f"GT coverage rate: {gt_in_pool}/200 ({100*gt_in_pool/200:.1f}%)")
    for K in K_list:
        if used > 0:
            print(f"Recall@{K} CLIP:   {recall_clip[K]/used:.3f}")
            print(f"Recall@{K} Rerank: {recall_rerank[K]/used:.3f}")
    print(f"MRR CLIP:   {mean(mrr_clip):.3f}")
    print(f"MRR Rerank: {mean(mrr_rerank):.3f}")
    print(f"Mean rank improvement (clip - rerank): {mean(rank_improvement):.2f}")


if __name__ == "__main__":
    main()
