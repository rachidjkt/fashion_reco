from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
from PIL import Image

from fashion_reco.config import DATA_DIR

# Polyvore compat model
from fashion_reco.compat.model import SetEncoder

# Your existing recommender parts
from fashion_reco.reco.retriever import CLIPRetriever
from fashion_reco.reco.reranker import CompatibilityReranker
from fashion_reco.reco.pipeline import OutfitRecommender
from fashion_reco.reco.clip_embedder import CLIPImageEmbedder

# Closet helpers (if your file names differ, tell me)
from fashion_reco.reco.closet import (
    ensure_closet_dirs,
    get_closet_paths,
    load_index as closet_load_index,
    load_embeddings as closet_load_embeddings,
    rebuild_closet_embeddings,
    save_index as closet_save_index,
)

st.set_page_config(page_title="fashion_reco", layout="wide")
st.title("🧥 fashion_reco — Outfit Recommender")


# -----------------------
# Utils
# -----------------------
def sigmoid(x: float) -> float:
    x = float(x)
    return 1.0 / (1.0 + np.exp(-x))


def safe_open_image(path: str) -> Optional[Image.Image]:
    try:
        p = Path(path)
        if not p.exists():
            return None
        return Image.open(p).convert("RGB")
    except Exception:
        return None


@st.cache_data(ttl=3600)
def geocode_city(city: str) -> Optional[Tuple[float, float, str]]:
    if not city.strip():
        return None
    url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(url, params={"name": city, "count": 1, "language": "en", "format": "json"}, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json()
    res = data.get("results")
    if not res:
        return None
    lat = float(res[0]["latitude"])
    lon = float(res[0]["longitude"])
    label = f"{res[0].get('name','')}, {res[0].get('country','')}".strip(", ")
    return lat, lon, label


@st.cache_data(ttl=600)
def current_weather(lat: float, lon: float) -> Optional[dict]:
    url = "https://api.open-meteo.com/v1/forecast"
    r = requests.get(
        url,
        params={"latitude": lat, "longitude": lon, "current": "temperature_2m,precipitation,rain,showers,snowfall"},
        timeout=10,
    )
    if r.status_code != 200:
        return None
    return r.json().get("current")


def weather_to_constraints(w: dict) -> List[str]:
    temp = float(w.get("temperature_2m", 0.0))
    precip = float(w.get("precipitation", 0.0))
    cats = {"top", "bottom", "shoes"}

    if temp <= 12:
        cats.add("outerwear")
    if temp >= 25:
        cats.discard("outerwear")
    if precip > 0:
        cats.add("outerwear")
        cats.add("shoes")
    return sorted(cats)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


# -----------------------
# Load systems
# -----------------------
@st.cache_resource
def load_embedder(device: str) -> CLIPImageEmbedder:
    return CLIPImageEmbedder(device=device)


@st.cache_resource
def load_polyvore_system():
    processed = DATA_DIR / "processed"

    emb = np.load(processed / "embeddings.npy", mmap_mode="r")
    item_id_to_idx: Dict[str, int] = json.load((processed / "item_id_to_idx.json").open("r", encoding="utf-8"))

    items_df = pd.read_parquet(processed / "items.parquet")
    items_df["item_id"] = items_df["item_id"].astype(str)

    item_meta: Dict[int, dict] = {}
    for _, row in items_df.iterrows():
        iid = row["item_id"]
        if iid not in item_id_to_idx:
            continue
        idx = item_id_to_idx[iid]
        item_meta[idx] = {
            "item_id": iid,
            "category": str(row["category"]),
            "image_path": str(row["image_path"]),
        }

    item_categories = {idx: m["category"] for idx, m in item_meta.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SetEncoder(d_in=512, d_hidden=256)
    ckpt_path = processed / "artifacts" / "compat_setencoder.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(device)
    model.eval()

    retriever = CLIPRetriever(emb)
    reranker = CompatibilityReranker(model, emb, device)
    reco = OutfitRecommender(retriever=retriever, reranker=reranker, item_categories=item_categories)

    return reco, emb, item_meta, item_categories, device


@st.cache_resource
def load_mens_catalog_if_available():
    processed = DATA_DIR / "processed"
    items_path = processed / "fashion_products_items_indexed.parquet"
    emb_path = processed / "fashion_products_embeddings_float32.npy"

    if not items_path.exists() or not emb_path.exists():
        return None

    df = pd.read_parquet(items_path)
    emb = np.load(emb_path, mmap_mode="r")

    item_meta = {
        i: {
            "item_id": str(df.loc[i, "item_id"]),
            "category": str(df.loc[i, "category"]),
            "image_path": str(df.loc[i, "image_path"]),
        }
        for i in range(len(df))
    }
    item_categories = {i: item_meta[i]["category"] for i in item_meta.keys()}
    retriever = CLIPRetriever(emb)

    return {"emb": emb, "item_meta": item_meta, "item_categories": item_categories, "retriever": retriever}


reco_poly, emb_poly, meta_poly, cats_poly, device = load_polyvore_system()
embedder = load_embedder(device)
mens = load_mens_catalog_if_available()

ensure_closet_dirs()
closet_paths = get_closet_paths()

ALL_CATS = ["top", "bottom", "shoes", "outerwear", "accessory", "dress", "other"]


# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("Mode")
modes = ["Inspiration (Polyvore: rerank)", "My Closet (rerank)"]
if mens:
    modes.append("Mens Catalog (retrieval-only)")
mode = st.sidebar.radio("Catalog", modes)

st.sidebar.caption(
    "Polyvore/Closet uses **compatibility rerank** (trained on Polyvore outfits). "
    "Mens catalog uses **retrieval-only** to avoid invalid generalization."
)

st.sidebar.header("Weather (optional)")
use_weather = st.sidebar.checkbox("Auto-suggest constraints from weather", value=False)
city = st.sidebar.text_input("City", "Gatineau")

weather_info = None
auto_cats = None
if use_weather:
    geo = geocode_city(city)
    if geo:
        lat, lon, label = geo
        w = current_weather(lat, lon)
        if w:
            weather_info = (label, w)
            auto_cats = weather_to_constraints(w)

st.sidebar.header("Constraints")
default_cats = auto_cats if auto_cats else ["top", "bottom", "shoes"]
allowed_categories = st.sidebar.multiselect("Allowed categories", options=ALL_CATS, default=default_cats)

top_k = st.sidebar.slider("Top-K results", 3, 12, 6)
candidate_k = st.sidebar.slider("Candidate pool size", 200, 3000, 800, step=100)
max_per_cat = st.sidebar.slider("Max per category", 1, 4, 1)

st.sidebar.header("Evaluation")
summary_path = DATA_DIR / "processed" / "artifacts" / "eval_summary.json"
with st.sidebar.expander("Show latest evaluation", expanded=False):
    if summary_path.exists():
        st.json(json.loads(summary_path.read_text(encoding="utf-8")))
    else:
        st.caption("No eval_summary.json found yet. Generate it from your eval script.")

st.sidebar.header("Closet")
st.sidebar.caption(str(closet_paths.images_dir))
if st.sidebar.button("Rebuild closet embeddings"):
    with st.spinner("Embedding closet images..."):
        closet_items, closet_emb = rebuild_closet_embeddings(embedder)
    st.sidebar.success(f"Closet indexed: {len(closet_items)} items")

with st.sidebar.expander("Assign closet categories", expanded=False):
    closet_items = closet_load_index()
    if not closet_items:
        st.caption("Closet empty.")
    else:
        show = closet_items[:25]
        updates = {}
        for it in show:
            cols = st.columns([1, 2])
            img = safe_open_image(it["path"])
            if img:
                cols[0].image(img, width=70)
            cur = it.get("category", "other")
            idx = ALL_CATS.index(cur) if cur in ALL_CATS else ALL_CATS.index("other")
            new_cat = cols[1].selectbox(it["filename"], ALL_CATS, index=idx, key=f"cat_{it['filename']}")
            updates[it["filename"]] = new_cat

        if st.button("Save closet categories"):
            merged = []
            for it in closet_items:
                it2 = dict(it)
                it2["category"] = updates.get(it["filename"], it.get("category", "other"))
                merged.append(it2)
            closet_save_index(merged)
            st.success("Saved categories.")


# -----------------------
# Main Layout
# -----------------------
left, right = st.columns(2)

with left:
    st.subheader("Upload (1–3 images)")
    uploads = st.file_uploader("Drop images", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)

    seed_imgs: List[Image.Image] = []
    if uploads:
        for u in uploads[:3]:
            im = Image.open(u).convert("RGB")
            seed_imgs.append(im)
            st.image(im, width=260)

    if weather_info:
        label, w = weather_info
        st.caption(f"🌦️ {label}: {w.get('temperature_2m')}°C, precip {w.get('precipitation')}mm")
        st.caption(f"Suggested constraints: {auto_cats}")

    st.markdown("---")
    st.subheader("Optional: add uploads to closet")
    if uploads and st.button("Save uploads to closet"):
        for u in uploads:
            out = closet_paths.images_dir / Path(u.name).name
            out.write_bytes(u.getbuffer())
        st.success("Saved. Use 'Rebuild closet embeddings' in sidebar.")

with right:
    st.subheader("Recommendations")

    if st.button("Recommend") and seed_imgs:
        with st.spinner("Encoding uploads..."):
            seed_E = embedder.embed_many(seed_imgs, batch_size=8).astype(np.float32)

        query = seed_E.mean(axis=0)
        query = query / (np.linalg.norm(query) + 1e-12)

        # Select catalog
        if mode == "Inspiration (Polyvore: rerank)":
            emb = emb_poly
            item_meta = meta_poly
            item_categories = cats_poly
            retriever = reco_poly.retriever
            rerank_enabled = True

        elif mode == "My Closet (rerank)":
            closet_items = closet_load_index()
            closet_emb = closet_load_embeddings()
            if closet_emb is None or closet_emb.shape[0] == 0:
                st.error("Closet is empty. Add images and rebuild embeddings.")
                st.stop()

            emb = closet_emb
            item_meta = {
                i: {
                    "item_id": closet_items[i]["filename"],
                    "category": closet_items[i].get("category", "other"),
                    "image_path": closet_items[i]["path"],
                }
                for i in range(len(closet_items))
            }
            item_categories = {i: item_meta[i]["category"] for i in item_meta.keys()}
            retriever = CLIPRetriever(emb)
            rerank_enabled = True

        else:
            emb = mens["emb"]
            item_meta = mens["item_meta"]
            item_categories = mens["item_categories"]
            retriever = mens["retriever"]
            rerank_enabled = False

        # ---------- Closest items (trust) ----------
        st.markdown("### Closest items (transparency)")
        for ui, uE in enumerate(seed_E):
            top = retriever.retrieve(uE, k=3)
            cols = st.columns(3)
            for j, idx in enumerate(top):
                m = item_meta.get(idx)
                if not m:
                    continue
                img = safe_open_image(m["image_path"])
                if img:
                    cols[j].image(img, width=140)
                sim = cosine_sim(np.array(emb[idx], dtype=np.float32), uE)
                cols[j].caption(f"{m['item_id']} | {m['category']} | sim={sim:.3f}")

        # pseudo seed from query
        pseudo_seed = retriever.retrieve(query, k=3)

        # ---------- Recommend ----------
        if rerank_enabled:
            # Build reranker model from polyvore ckpt (works for closet too)
            processed = DATA_DIR / "processed"
            model = SetEncoder(d_in=512, d_hidden=256)
            model.load_state_dict(torch.load(processed / "artifacts" / "compat_setencoder.pt", map_location="cpu"))
            model.to(device)
            model.eval()

            reranker = CompatibilityReranker(model, emb, device)
            reco = OutfitRecommender(retriever=retriever, reranker=reranker, item_categories=item_categories)

            results = reco.recommend(
                seed_items=pseudo_seed,
                allowed_categories=allowed_categories,
                top_k=int(top_k),
                candidate_k=int(candidate_k),
                max_per_category=int(max_per_cat),
            )
            st.caption("Ranking = CLIP retrieval + compatibility rerank (Polyvore-trained)")
        else:
            # retrieval-only
            candidates = retriever.retrieve(query, k=int(candidate_k))
            if allowed_categories:
                allowed = set(allowed_categories)
                candidates = [i for i in candidates if item_categories.get(i) in allowed]
            results = []
            for idx in candidates[: int(top_k)]:
                score = cosine_sim(np.array(emb[idx], dtype=np.float32), query)
                results.append((idx, score))
            st.caption("Ranking = CLIP retrieval only")

        st.markdown("---")
        st.markdown("### Results")

        for idx, score in results:
            m = item_meta.get(idx)
            if not m:
                continue

            img = safe_open_image(m["image_path"])
            cat = m.get("category", "unknown")
            iid = m.get("item_id", str(idx))

            sim = cosine_sim(np.array(emb[idx], dtype=np.float32), query)

            if rerank_enabled:
                compat_p = sigmoid(float(score))
                score_line = f"Compat (sigmoid): **{compat_p:.3f}** | logit: {float(score):.3f}"
            else:
                score_line = f"Similarity: **{float(score):.3f}**"

            c1, c2 = st.columns([1, 2])
            if img:
                c1.image(img, width=220)

            c2.markdown(
                f"**Item:** `{iid}`  \n"
                f"**Category:** `{cat}`  \n"
                f"{score_line}  \n"
                f"**CLIP similarity to upload:** `{sim:.3f}`"
            )

            # reasoning bullets
            bullets = [f"Visually similar to your upload (CLIP sim {sim:.3f})."]
            if rerank_enabled:
                bullets.append(f"Selected for outfit compatibility (p≈{compat_p:.3f}).")
            if allowed_categories:
                bullets.append(f"Respects constraints: {', '.join(allowed_categories)}.")

            c2.markdown("**Why this recommendation:**")
            for b in bullets:
                c2.markdown(f"- {b}")

            st.markdown("---")
