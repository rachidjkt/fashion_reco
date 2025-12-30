import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from fashion_reco.config import DATA_DIR
from fashion_reco.compat.model import SetEncoder
from fashion_reco.reco.retriever import CLIPRetriever
from fashion_reco.reco.reranker import CompatibilityReranker
from fashion_reco.reco.pipeline import OutfitRecommender
from fashion_reco.reco.clip_embedder import CLIPImageEmbedder
from fashion_reco.reco.closet import (
    ensure_closet_dirs,
    get_closet_paths,
    load_index,
    load_embeddings,
    rebuild_closet_embeddings,
)

st.set_page_config(layout="wide")
st.title("🧥 Outfit Recommendation Demo")


@st.cache_resource
def load_dataset_system():
    processed = DATA_DIR / "processed"

    emb = np.load(processed / "embeddings.npy", mmap_mode="r")
    item_id_to_idx = json.load((processed / "item_id_to_idx.json").open("r", encoding="utf-8"))
    idx_to_item_id = {v: k for k, v in item_id_to_idx.items()}

    items_df = pd.read_parquet(processed / "items.parquet")
    items_df["item_id"] = items_df["item_id"].astype(str)
    items_df = items_df[items_df["item_id"].isin(item_id_to_idx)]

    item_categories = {
        item_id_to_idx[iid]: cat
        for iid, cat in zip(items_df["item_id"], items_df["category"])
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SetEncoder(d_in=512, d_hidden=256)
    model.load_state_dict(torch.load(processed / "artifacts" / "compat_setencoder.pt", map_location="cpu"))
    model.to(device)
    model.eval()

    retriever = CLIPRetriever(emb)
    reranker = CompatibilityReranker(model, emb, device)
    reco = OutfitRecommender(retriever=retriever, reranker=reranker, item_categories=item_categories)

    return reco, emb, idx_to_item_id, item_categories, device


@st.cache_resource
def load_embedder(device: str):
    return CLIPImageEmbedder(device=device)


reco_ds, emb_ds, idx_to_item_id, item_categories_ds, device = load_dataset_system()
embedder = load_embedder(device)


# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Mode")
mode = st.sidebar.radio("Recommendation mode", ["Inspiration (dataset)", "My Closet (only)"])

st.sidebar.header("Constraints")
allowed_categories = st.sidebar.multiselect(
    "Allowed categories",
    options=["top", "bottom", "shoes", "outerwear", "accessory", "dress", "other"],
    default=["top", "bottom", "shoes"],
)
top_k = st.sidebar.slider("Number of recommendations", 3, 10, 6)
candidate_k = st.sidebar.slider("Candidate pool size", 200, 2000, 800, step=100)
max_per_cat = st.sidebar.slider("Max per category", 1, 3, 1)

st.sidebar.header("Closet")
ensure_closet_dirs()
closet_paths = get_closet_paths()
st.sidebar.caption(f"Closet folder: {closet_paths.images_dir}")


# ----------------------------
# UI layout
# ----------------------------
colL, colR = st.columns(2)

with colL:
    st.subheader("Upload your outfit items (1–3 images)")
    uploads = st.file_uploader(
        "Drag & drop or browse",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    seed_imgs = []
    if uploads:
        for u in uploads[:3]:
            im = Image.open(u).convert("RGB")
            seed_imgs.append(im)
            st.image(im, width=240)

    st.markdown("---")
    st.subheader("Optional: Add to closet")
    add_to_closet = st.checkbox("Save uploaded images into my closet folder")
    if add_to_closet and uploads:
        if st.button("Save to closet"):
            ensure_closet_dirs()
            for u in uploads:
                # store with original name; if collisions bother you, add a timestamp
                out_path = closet_paths.images_dir / Path(u.name).name
                out_path.write_bytes(u.getbuffer())
            st.success("Saved to closet. Now click 'Rebuild closet index' in the sidebar.")
    st.caption("Tip: Closet mode uses only items saved in data/user_closet/images.")

with st.sidebar:
    if st.button("Rebuild closet index"):
        with st.spinner("Embedding closet images with CLIP..."):
            closet_items, closet_emb = rebuild_closet_embeddings(embedder)
        st.success(f"Closet indexed: {len(closet_items)} items")


with colR:
    st.subheader("Recommendations")
    if st.button("Recommend") and seed_imgs:
        # 1) Embed uploaded images with CLIP
        with st.spinner("Encoding uploaded images with CLIP..."):
            seed_E = embedder.embed_many(seed_imgs, batch_size=8).astype(np.float32)
        query_emb = seed_E.mean(axis=0)

        # 2) Choose candidate universe
        if mode == "Inspiration (dataset)":
            reco = reco_ds
            # dataset retriever already inside reco_ds uses emb_ds
            # but we need to override retrieval query => use reco.retriever directly
            candidate_indices = reco.retriever.retrieve(query_emb, k=int(candidate_k))
            universe = "dataset"
        else:
            closet_items = load_index()
            closet_emb = load_embeddings()
            if closet_emb is None or closet_emb.shape[0] == 0:
                st.error("Your closet is empty. Add items and rebuild the closet index first.")
                st.stop()

            # closet retrieval: dot product since both are normalized
            sims = closet_emb @ query_emb
            order = np.argsort(-sims)[: int(candidate_k)]
            candidate_indices = list(order)
            universe = "closet"

        # 3) Build a lightweight recommender wrapper for closet mode
        # (reuses same compatibility model, but different embeddings)
        if universe == "closet":
            # create a temporary retriever/reranker on closet embeddings
            # categories are unknown for closet items; treat as "other"
            closet_item_categories = {i: "other" for i in range(len(candidate_indices) + 1000000)}  # safe default

            retriever = CLIPRetriever(closet_emb)
            # reuse same torch model as dataset reranker uses, but point to closet_emb
            # We can access model from reco_ds.reranker.model if you stored it;
            # simplest: rebuild reranker using the same checkpointed model in memory:
            processed = DATA_DIR / "processed"
            model = SetEncoder(d_in=512, d_hidden=256)
            model.load_state_dict(torch.load(processed / "artifacts" / "compat_setencoder.pt", map_location="cpu"))
            model.to(device)
            model.eval()
            reranker = CompatibilityReranker(model, closet_emb, device)

            reco = OutfitRecommender(retriever=retriever, reranker=reranker, item_categories=closet_item_categories)

            # seed items for closet: we don't know exact closet indices for uploaded images,
            # so we treat them as "virtual" and only use query_emb for retrieval; reranker works on sets of indices.
            # For v1, we skip adding seed indices and just rank candidates by similarity + compat with top similar as pseudo-seed.
            # We'll approximate seed using top-3 retrieved closet items:
            pseudo_seed = candidate_indices[:3]
            seed_items = pseudo_seed
        else:
            # dataset mode: we don't have exact dataset indices of uploaded images either,
            # so we approximate seed by taking top-3 retrieved dataset items as pseudo-seed.
            pseudo_seed = candidate_indices[:3]
            seed_items = pseudo_seed

        # 4) Recommend using your pipeline (filter + rerank)
        results = reco.recommend(
            seed_items=seed_items,
            allowed_categories=allowed_categories if universe == "dataset" else None,
            top_k=int(top_k),
            candidate_k=int(candidate_k),
            max_per_category=int(max_per_cat),
        )

        st.caption(
            f"Mode: **{mode}** | Universe: **{universe}** | "
            f"Pseudo-seed size: {len(seed_items)}"
        )
        st.markdown("---")

        for idx, score in results:
            if universe == "dataset":
                item_id = idx_to_item_id.get(idx, str(idx))
                cat = item_categories_ds.get(idx, "unknown")
                st.markdown(f"**Item {item_id}**  \nCategory: `{cat}`  \nCompat score: `{score:.3f}`")
            else:
                closet_items = load_index()
                fname = closet_items[idx]["filename"] if idx < len(closet_items) else f"closet_{idx}"
                st.markdown(f"**Closet item: {fname}**  \nCompat score: `{score:.3f}`")
            st.markdown("---")
