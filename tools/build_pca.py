"""Build PCA axes for the Mode-I navigator.

Strategy: encode a varied corpus of captions through T5, flatten each
conditioning tensor, run PCA on the resulting matrix.  The top K principal
components are 'semantic directions' in conditioning space.

For Mode I we use them as noise-space directions instead — heuristic
mapping by reshaping the axis to noise shape.  This is rough but produces
visible structured variation.

Usage:
    python tools/build_pca.py
        Builds tools/pca_axes.npz  (top 16 axes)

Run once.  Takes ~15 min on M3 Max.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.lossy_runtime import LossyRuntime  # noqa: E402

# small varied corpus — extend or replace as desired
CAPTIONS = [
    "a young woman with dark hair",
    "an old man with a white beard",
    "a child smiling",
    "a man with sunglasses",
    "a woman wearing a red hat",
    "a face in profile",
    "a face looking up",
    "a face looking down",
    "a woman with curly hair",
    "a man with short hair",
    "a face in soft sunlight",
    "a face in deep shadow",
    "a face by candlelight",
    "a woman with closed eyes",
    "a face laughing",
    "a face crying",
    "a face surprised",
    "a face calm and serene",
    "a young face with freckles",
    "a face with high cheekbones",
    "a face in side light from the left",
    "a face in side light from the right",
    "a face in the rain",
    "a face in golden hour light",
    "a portrait of a musician",
    "a portrait of a worker",
    "a portrait of a dancer",
    "a portrait of a soldier",
    "a face with wide open eyes",
    "a face with a small smile",
    "a face turning away",
    "a face very close up",
    "a face at a distance",
    "a face under harsh fluorescent light",
    "a face by firelight",
    "a face in moonlight",
    "a young man with thoughtful expression",
    "an elderly woman with kind eyes",
    "a teenager looking off-camera",
    "a person with weathered skin",
    "a person with smooth pale skin",
    "a face in a window reflection",
    "a face with a scar across the cheek",
    "a face in winter clothes",
    "a face partially obscured by a scarf",
    "a face with raindrops on the skin",
    "a face wearing a gas mask",
    "a face behind a veil",
    "a face with a tattoo on the temple",
    "a face with closed lips",
    "a face mid-laugh, mouth open",
    "a face shouting",
    "a face yawning",
    "a face whispering",
    "a face concentrating",
    "a face daydreaming",
    "a face just waking up",
    "a face about to sleep",
    "a face listening",
    "a face thinking",
]


def main(out_path: Path = ROOT / "tools" / "pca_axes.npz", n_axes: int = 16):
    rt = LossyRuntime(
        config_path="/Users/hitosteyerl/CC/ComfyUI/v8_facecrop.yaml",
        checkpoint_path="/Users/hitosteyerl/CC/ComfyUI/models/lossy/tracka_mmd-minsnr-step_4700000.pt",
    )

    print(f"[pca] encoding {len(CAPTIONS)} captions through T5 …")
    flat_dim = int(np.prod(rt.latent_shape))
    embed_matrix = []
    for i, c in enumerate(CAPTIONS):
        ctx, _ = rt.encode_prompt(c)
        # ctx is (1, S, d).  Reduce to a vector of size flat_dim by pooling + tile.
        v = ctx.float().mean(dim=1).flatten().cpu().numpy()  # (d,)
        # tile / truncate to exactly flat_dim
        rep = (flat_dim + v.size - 1) // v.size
        v = np.tile(v, rep)[:flat_dim]
        embed_matrix.append(v)
        print(f"  {i+1}/{len(CAPTIONS)}: {c[:50]}")

    M = np.stack(embed_matrix, axis=0)  # (N, flat_dim)
    M = M - M.mean(axis=0, keepdims=True)

    print(f"[pca] running SVD on {M.shape}…")
    # economy SVD
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    # explained variance
    total = (S ** 2).sum()
    ratio = (S ** 2) / total
    axes = Vt[:n_axes]            # (n_axes, flat_dim)

    # normalize each axis to unit length for predictable walk_range
    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)

    np.savez(out_path,
             axes=axes.astype(np.float32),
             explained_variance_ratio=ratio[:n_axes].astype(np.float32),
             latent_shape=np.array(rt.latent_shape, dtype=np.int32))
    print(f"[pca] saved {out_path} — {n_axes} axes, top variance ratios:")
    for i in range(min(5, n_axes)):
        print(f"  axis {i}: {ratio[i]:.4f}")


if __name__ == "__main__":
    main()
