from __future__ import annotations
from typing import Iterable, Optional,List, Dict, Optional


def filter_by_category(
    candidate_indices: Iterable[int],
    item_categories: dict[int, str],
    allowed: Optional[list[str]],
) -> list[int]:
    """
    If allowed is None/empty -> no filtering.
    """
    if not allowed:
        return list(candidate_indices)

    allowed_set = set(allowed)
    out = []
    for i in candidate_indices:
        if item_categories.get(i) in allowed_set:
            out.append(i)
    return out


def dedupe_by_category(
    candidate_indices: List[int],
    item_categories: Dict[int, str],
    max_per_cat: int = 1,
) -> List[int]:
    counts: Dict[str, int] = {}
    out: List[int] = []
    for idx in candidate_indices:
        cat = item_categories.get(idx, "unknown")
        c = counts.get(cat, 0)
        if c >= max_per_cat:
            continue
        counts[cat] = c + 1
        out.append(idx)
    return out
