"""Mode I — PCA direction walk in noise space.

Loads a precomputed set of PCA directions (top-K principal axes of a sample
of random noise vectors that produced 'good' Lossy outputs).  Walking along
one of these axes from the origin gives semantically coherent variation.

Run tools/build_pca.py once before this mode works.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch

from core.lossy_runtime import LossyRuntime
from core.rife import chain_to_video


PCA_FILE = Path(__file__).resolve().parents[2] / "tools" / "pca_axes.npz"


def axes_available() -> bool:
    return PCA_FILE.exists()


def list_axes():
    if not axes_available():
        return []
    d = np.load(PCA_FILE)
    return [(i, float(d["explained_variance_ratio"][i])) for i in range(d["axes"].shape[0])]


def render(
    runtime: LossyRuntime,
    prompt: str,
    seed: int,
    axis_index: int,
    walk_range: float,
    n_steps: int,
    out_dir: Path,
    sampling_steps: int = 50,
    guidance_scale: float = 5.0,
    frames_per_pair: int = 8,
    fps: int = 12,
    save_mp4: bool = True,
    progress_cb=None,
) -> List[Path]:
    """Walk along a PCA axis from -walk_range to +walk_range, centered on the seed's noise."""
    if not axes_available():
        raise RuntimeError(f"No PCA axes found at {PCA_FILE}.  Run tools/build_pca.py first.")

    out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(PCA_FILE)
    axis = torch.from_numpy(d["axes"][axis_index]).to(runtime.device, runtime.dtype)
    axis = axis.reshape(runtime.latent_shape)

    base_noise = runtime.make_noise(seed)
    ctx, mask = runtime.encode_prompt(prompt)

    paths: List[Path] = []
    for i in range(n_steps):
        t = -walk_range + (2 * walk_range) * (i / max(n_steps - 1, 1))
        n = base_noise + t * axis
        latent = runtime.sample(n, ctx, mask, num_steps=sampling_steps, guidance_scale=guidance_scale)
        img = runtime.decode_latent(latent)
        p = out_dir / f"pca_axis{axis_index}_{i:03d}_t{t:+.2f}.png"
        img.save(p)
        paths.append(p)
        print(f"[pca] {i+1}/{n_steps} t={t:+.2f} saved {p.name}")
        if progress_cb is not None:
            progress_cb(i + 1, n_steps, f"sampled {i+1}/{n_steps}")

    if save_mp4 and len(paths) >= 2:
        if progress_cb is not None:
            progress_cb(n_steps, n_steps, "encoding video …")
        mp4 = out_dir / f"pca_axis{axis_index}_walk.mp4"
        chain_to_video(paths, mp4, frames_per_pair=frames_per_pair, fps=fps)
        print(f"[pca] mp4: {mp4}")

    return paths
