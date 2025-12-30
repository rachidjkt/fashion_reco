from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from fashion_reco.config import DATA_DIR
from fashion_reco.compat.model import SetEncoder


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_set_map(split_path: Path) -> Dict[str, List[str]]:
    """
    Reads split json (train/valid/test) and returns:
      set_id -> list of item_id strings ordered by 'index' (1..N)
    """
    data = load_json(split_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {split_path}")

    out: Dict[str, List[str]] = {}
    for ex in data:
        set_id = str(ex["set_id"])
        items = ex["items"]
        # sort by index then keep item_id
        items_sorted = sorted(items, key=lambda x: int(x["index"]))
        out[set_id] = [str(it["item_id"]) for it in items_sorted]
    return out


def parse_set_id_from_question(question: Any) -> Optional[str]:
    """
    Your question entries look like '222049137_1' (set_id_position).
    FITB JSON has: keys ['question','blank_position','answers'].
    """
    if not isinstance(question, list) or len(question) == 0:
        return None

    first = str(question[0])
    if "_" not in first:
        return None

    set_id = first.split("_", 1)[0]
    return set_id

def norm_token(t: str) -> str:
    s = str(t)
    if "_" in s:
        return s.split("_", 1)[0]
    return s


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processed = DATA_DIR / "processed"
    raw_root = DATA_DIR / "raw" / "polyvore" / "polyvore_outfits"

    fitb_path = raw_root / "disjoint" / "fill_in_blank_test.json"
    if not fitb_path.exists():
        fitb_path = raw_root / "disjoint" / "fill_in_blank_test"

    # Build set_id -> ordered item_ids map (plain ids)
    set_map: Dict[str, List[str]] = {}
    for name in ["train", "valid", "test"]:
        p = raw_root / "disjoint" / f"{name}.json"
        if p.exists():
            set_map.update(load_set_map(p))

    item_id_to_idx: Dict[str, int] = json.load((processed / "item_id_to_idx.json").open("r"))
    emb = np.load(processed / "embeddings.npy", mmap_mode="r")

    ckpt_path = processed / "artifacts" / "compat_setencoder.pt"
    state = torch.load(ckpt_path, map_location="cpu")
    model = SetEncoder(d_in=512, d_hidden=256)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    data = load_json(fitb_path)
    if not isinstance(data, list):
        raise ValueError("Expected FITB test json to be a list")

    print("EXAMPLE[0]:", data[0])

    correct = 0
    total = 0
    skipped = 0

    # diagnostics
    total_candidates = 0
    valid_candidates_total = 0
    examples_with_lt2_valid = 0
    examples_with_missing = 0

    GT_IS_LAST = False

    with torch.no_grad():
        for ex in data:
            question = ex.get("question")
            blank_pos = ex.get("blank_position")
            answers = ex.get("answers")

            if question is None or blank_pos is None or answers is None:
                skipped += 1
                continue
            if not isinstance(answers, list) or len(answers) < 2:
                skipped += 1
                continue

            set_id = parse_set_id_from_question(question)
            if set_id is None or set_id not in set_map:
                skipped += 1
                continue

            base_item_ids = set_map[set_id]
            if not base_item_ids:
                skipped += 1
                continue

            # blank_pos might be 1-indexed; try both
            try:
                bp_raw = int(blank_pos)
            except Exception:
                skipped += 1
                continue

            bp = None
            for cand in [bp_raw - 1, bp_raw]:
                if 0 <= cand < len(base_item_ids):
                    bp = cand
                    break
            if bp is None:
                skipped += 1
                continue

            # Ground-truth candidate index in original answers list
            gt_idx = (len(answers) - 1) if GT_IS_LAST else 0

            # Score only valid candidates
            scores = []
            valid_map = []  # maps "scores index" -> original candidate index

            any_missing = False

            for i, a in enumerate(answers):
                total_candidates += 1
                a_id = norm_token(a)

                # candidate item must exist
                if a_id not in item_id_to_idx:
                    any_missing = True
                    continue

                outfit_item_ids = base_item_ids.copy()
                outfit_item_ids[bp] = a_id

                # all outfit items must exist
                if any(iid not in item_id_to_idx for iid in outfit_item_ids):
                    any_missing = True
                    continue

                idxs = [item_id_to_idx[iid] for iid in outfit_item_ids]
                X = torch.from_numpy(np.array(emb[idxs], dtype=np.float32)).unsqueeze(0)
                mask = torch.ones((1, X.shape[1]), dtype=torch.bool)

                X = X.to(device)
                mask = mask.to(device)

                logit = model(X, mask).item()
                scores.append(logit)
                valid_map.append(i)
                valid_candidates_total += 1

            if any_missing:
                examples_with_missing += 1

            # Need at least 2 candidates to make it a meaningful ranking problem
            if len(scores) < 2:
                examples_with_lt2_valid += 1
                skipped += 1
                continue

            pred_local = int(np.argmax(scores))
            pred_idx = valid_map[pred_local]

            if pred_idx == gt_idx:
                correct += 1
            total += 1

    acc = correct / max(total, 1)
    print(f"Filtered FITB accuracy: {acc:.4f} ({correct}/{total}), skipped={skipped}")
    print("Total candidates in file:", total_candidates)
    print("Valid candidates scored:", valid_candidates_total)
    print("Examples with any missing candidates:", examples_with_missing)
    print("Examples skipped (<2 valid candidates):", examples_with_lt2_valid)


if __name__ == "__main__":
    main()
