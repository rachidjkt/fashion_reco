from __future__ import annotations

import numpy as np
import torch
from PIL import Image
import open_clip


class CLIPImageEmbedder:
    """
    Small wrapper around open_clip to embed PIL images into 512-d vectors.
    Uses normalization so dot-product ~= cosine similarity.
    """

    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.model.eval()

    @torch.inference_mode()
    def embed_pil(self, img: Image.Image) -> np.ndarray:
        x = self.preprocess(img).unsqueeze(0).to(self.device)  # (1,3,H,W)
        feat = self.model.encode_image(x)                      # (1,512) for ViT-B-32
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feat.squeeze(0).float().cpu().numpy()

    @torch.inference_mode()
    def embed_many(self, imgs: list[Image.Image], batch_size: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i:i + batch_size]
            xb = torch.stack([self.preprocess(im) for im in batch], dim=0).to(self.device)
            feat = self.model.encode_image(xb)
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            out.append(feat.float().cpu().numpy())
        return np.concatenate(out, axis=0)
