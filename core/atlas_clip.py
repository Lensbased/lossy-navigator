"""CLIP wrapper for the atlas — image embeddings + text query embeddings.

Uses openai/clip-vit-base-patch32 for speed/disk.  Single shared embedding
space lets us project text concepts and image samples on the same map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from PIL import Image


_MODEL = None
_PROCESSOR = None
_DEVICE = None
MODEL_NAME = "openai/clip-vit-base-patch32"


def _load():
    global _MODEL, _PROCESSOR, _DEVICE
    if _MODEL is not None:
        return _MODEL, _PROCESSOR, _DEVICE
    from transformers import CLIPModel, CLIPProcessor
    if torch.cuda.is_available():
        _DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        _DEVICE = "mps"
    else:
        _DEVICE = "cpu"
    _MODEL = CLIPModel.from_pretrained(MODEL_NAME).to(_DEVICE).eval()
    _PROCESSOR = CLIPProcessor.from_pretrained(MODEL_NAME)
    return _MODEL, _PROCESSOR, _DEVICE


def _to_tensor(feat):
    """get_*_features sometimes returns a tensor, sometimes a HF output object."""
    if hasattr(feat, "image_embeds"): return feat.image_embeds
    if hasattr(feat, "text_embeds"): return feat.text_embeds
    if hasattr(feat, "pooler_output"): return feat.pooler_output
    if hasattr(feat, "last_hidden_state"): return feat.last_hidden_state.mean(dim=1)
    return feat


@torch.inference_mode()
def embed_images(image_paths: Iterable[Path], batch: int = 16) -> np.ndarray:
    model, proc, dev = _load()
    paths = list(image_paths)
    out = None
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        inputs = proc(images=imgs, return_tensors="pt").to(dev)
        feat = _to_tensor(model.get_image_features(**inputs))
        emb = feat.float().cpu().numpy()
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        if out is None:
            out = np.zeros((len(paths), emb.shape[1]), dtype=np.float32)
        out[i:i + len(chunk)] = emb
    return out


@torch.inference_mode()
def embed_texts(texts: List[str], batch: int = 32) -> np.ndarray:
    model, proc, dev = _load()
    out = None
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        inputs = proc(text=chunk, return_tensors="pt", padding=True, truncation=True).to(dev)
        feat = _to_tensor(model.get_text_features(**inputs))
        emb = feat.float().cpu().numpy()
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        if out is None:
            out = np.zeros((len(texts), emb.shape[1]), dtype=np.float32)
        out[i:i + len(chunk)] = emb
    return out
