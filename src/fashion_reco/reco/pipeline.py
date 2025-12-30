import numpy as np
from typing import List, Optional, Tuple

from fashion_reco.reco.retriever import CLIPRetriever
from fashion_reco.reco.reranker import CompatibilityReranker
from fashion_reco.reco.constraints import filter_by_category, dedupe_by_category


class OutfitRecommender:
    def __init__(
        self,
        retriever: CLIPRetriever,
        reranker: CompatibilityReranker,
        item_categories: dict,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.item_categories = item_categories

    def recommend(
        self,
        seed_items: List[int],
        allowed_categories: Optional[List[str]] = None,
        top_k: int = 10,
        candidate_k: int = 300,
        max_per_category: int = 1,
        exclude_seed: bool = True,
    ) -> List[Tuple[int, float]]:
        """
        Returns list of (candidate_item_index, compatibility_score) sorted desc.
        """
        if len(seed_items) < 1:
            raise ValueError("seed_items must contain at least 1 item index")

        seed_set = set(seed_items)

        # 1) Build query embedding = mean of seed item embeddings
        query_emb = np.mean(self.reranker.emb[seed_items], axis=0)

        # 2) Retrieve broad candidate pool
        candidates = self.retriever.retrieve(query_emb, k=candidate_k)

        # 3) Optionally remove items already in the seed
        if exclude_seed:
            candidates = [c for c in candidates if c not in seed_set]

        # 4) Category filter (optional)
        candidates = filter_by_category(candidates, self.item_categories, allowed_categories)

        # 5) Prevent category spam (optional)
        if max_per_category is not None and max_per_category > 0:
            candidates = dedupe_by_category(candidates, self.item_categories, max_per_cat=max_per_category)

        # 6) Rerank by compatibility (score seed + [candidate])
        scored: List[Tuple[int, float]] = []
        for c in candidates:
            outfit = seed_items + [c]
            score = float(self.reranker.score_outfit(outfit))
            scored.append((c, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
