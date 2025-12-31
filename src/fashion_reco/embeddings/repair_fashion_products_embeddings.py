from pathlib import Path
import numpy as np
import pandas as pd

from fashion_reco.config import DATA_DIR

D = 512

def is_npy(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(6) == b"\x93NUMPY"

def main():
    processed = DATA_DIR / "processed"

    # Use your indexed table if present, else fallback to items parquet
    items_idx = processed / "fashion_products_items_indexed.parquet"
    items = processed / "fashion_products_items.parquet"
    items_path = items_idx if items_idx.exists() else items

    df = pd.read_parquet(items_path)
    n = len(df)

    src = processed / "fashion_products_embeddings.npy"
    out = processed / "fashion_products_embeddings_float32.npy"

    if not src.exists():
        raise FileNotFoundError(src)

    print("[repair] items rows:", n)
    print("[repair] src:", src)

    # If it's already a valid .npy, just load + resave float32
    if is_npy(src):
        print("[repair] detected real .npy header")
        arr = np.load(src)
        arr = np.asarray(arr, dtype=np.float32)
        print("[repair] loaded:", arr.shape, arr.dtype)
        np.save(out, arr)
        print("[repair] saved:", out)
        return

    # Otherwise treat it as raw float32 memmap (common when using np.memmap)
    expected_bytes = n * D * 4
    actual_bytes = src.stat().st_size
    print("[repair] detected raw file (no .npy header)")
    print("[repair] expected bytes:", expected_bytes, "actual:", actual_bytes)

    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"Size mismatch. Expected {expected_bytes} bytes for shape ({n},{D}) float32 "
            f"but file is {actual_bytes} bytes. "
            f"Check N (rows) or embedding dim."
        )

    mm = np.memmap(src, mode="r", dtype="float32", shape=(n, D))
    arr = np.asarray(mm)  # converts to real ndarray (loads into RAM once)
    print("[repair] memmap view:", mm.shape, mm.dtype)

    np.save(out, arr.astype(np.float32))
    print("[repair] saved real .npy:", out)

if __name__ == "__main__":
    main()
