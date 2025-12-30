import numpy as np
from typing import List


class CLIPRetriever:
    def __init__(self, embeddings: np.ndarray):
        """
        embeddings: (N_items, 512), L2-normalized
        """
        self.emb = embeddings

    def retrieve(self, query_emb: np.ndarray, k: int = 200) -> List[int]:
        """
        query_emb: (512,)
        returns indices of top-k most similar items
        """
        sims = self.emb @ query_emb  # cosine similarity
        topk = np.argsort(-sims)[:k]
        return topk.tolist()
