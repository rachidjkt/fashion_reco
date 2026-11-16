# Fashion Recommender

A personal outfit-recommendation prototype built to explore a practical question: given clothes I already like, what other pieces might go with them?

The project combines **CLIP image embeddings**, **candidate retrieval**, a **PyTorch outfit-compatibility model**, and a **Streamlit interface**. The implementation separates visual similarity from outfit compatibility rather than treating them as the same signal.

## What is implemented

- Image uploads as recommendation seeds.
- A NumPy-based retrieval stage over image embeddings.
- A DeepSets-style compatibility model with attention pooling.
- Category filters and limits on results per category.
- A personal closet mode with image indexing.
- An optional fashion-products catalog using retrieval only.
- Optional Open-Meteo weather lookup to suggest clothing categories.
- Dataset preparation, embedding extraction, training, and evaluation scripts.

**Status:** experimental prototype. The repository contains source code, but not the datasets, generated embeddings, or trained checkpoint needed to launch the full application. No benchmark scores are claimed here.

## How it works

1. Prepare item metadata and outfit records from the Polyvore dataset.
2. Encode catalog images with OpenCLIP ViT-B-32.
3. Train a set encoder to score outfit compatibility.
4. Retrieve visually related candidates and apply category constraints.
5. Rerank candidates using the compatibility model.
6. Display images and scores in Streamlit.

The optional fashion-products mode uses visual retrieval without the Polyvore-trained compatibility reranker. Compatibility learned on one catalog should not be assumed to transfer to another.

## Code map

| Area | Entry point |
|---|---|
| Streamlit application | [frontend/app.py](frontend/app.py) |
| Dataset preparation | [src/fashion_reco/data/prepare_polyvore.py](src/fashion_reco/data/prepare_polyvore.py) |
| Embedding extraction | [src/fashion_reco/embeddings/extract_embeddings.py](src/fashion_reco/embeddings/extract_embeddings.py) |
| Compatibility model | [src/fashion_reco/compat/model.py](src/fashion_reco/compat/model.py) |
| Training | [src/fashion_reco/compat/train.py](src/fashion_reco/compat/train.py) |
| Recommendation pipeline | [src/fashion_reco/reco/pipeline.py](src/fashion_reco/reco/pipeline.py) |
| Ranking evaluation | [src/fashion_reco/reco/eval_rerank.py](src/fashion_reco/reco/eval_rerank.py) |

## Local setup

Python 3.12 is a reasonable starting point for this prototype. There is not yet a pinned, verified dependency environment; `pyproject.toml` is currently empty.

Create and activate a virtual environment, then install the libraries imported by the main pipeline:

```bash
git clone https://github.com/rachidjkt/fashion_reco.git
cd fashion_reco
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on macOS/Linux, or `.venv\Scripts\Activate.ps1` in Windows PowerShell.

```bash
python -m pip install numpy pandas pyarrow requests streamlit torch torchvision pillow open_clip_torch
```

Choose a PyTorch build appropriate for your hardware. OpenCLIP may download pretrained weights when initialized.

Because packaging is unfinished, expose the source directory before running modules:

```bash
# macOS / Linux, from the repository root
export PYTHONPATH="$PWD/src"
```

```powershell
# Windows PowerShell, from the repository root
$env:PYTHONPATH = (Join-Path $PWD "src")
```

### Required local artifacts

The application expects these files under `data/processed/`, as determined by the current [config.py](src/fashion_reco/config.py):

- `embeddings.npy`
- `item_id_to_idx.json`
- `items.parquet`
- `artifacts/compat_setencoder.pt`

The paths are relative to the repository root in the current implementation. Raw Polyvore files are expected under `data/raw/`; consult the preparation scripts for their exact layouts. The FITB evaluator uses the more specific `data/raw/polyvore/polyvore_outfits/` layout.

Obtain datasets separately under their applicable terms, then prepare the metadata, item table, embeddings, training pairs, and model checkpoint. The pair-building script is currently named `src/fashion_reco/compat/build_pairs.py.py`.

Once these artifacts exist:

```bash
python -m streamlit run frontend/app.py
```

A fresh clone alone is not a complete runnable demo yet.

## Evaluation and interpretation

The ranking evaluator reports Recall@10, Recall@50, and mean reciprocal rank for retrieval and compatibility reranking. The FITB script evaluates a filtered fill-in-the-blank task.

These are experimental diagnostics:

- The ranking evaluator excludes targets missing from its retrieved candidates and can force a target into the final reranking pool.
- FITB evaluation skips some examples with unavailable candidates and assumes the first answer is correct.
- These choices must be checked against the dataset definition before reporting benchmark results.
- A sigmoid-transformed compatibility score is not a calibrated probability.
- No measured latency, production scale, or improvement over a baseline is claimed.

## Next improvements

- Add a reproducible dependency lock and a small example dataset.
- Normalize data paths and complete the artifact-generation instructions.
- Evaluate retrieval coverage and ranking separately on an untouched test split.
- Add checks for missing images, empty catalogs, and category constraints.
- Publish a short demo and reproducible evaluation artifacts.

## Technologies

Python · PyTorch · OpenCLIP · NumPy · pandas · Parquet · Streamlit · Open-Meteo
