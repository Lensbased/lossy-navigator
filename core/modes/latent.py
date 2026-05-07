"""Mode B — latent (noise) walk.

Slerp between two noise tensors (driven by two seeds), sample Lossy at N
points along the path, then RIFE between for video output.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from core.lossy_runtime import LossyRuntime, slerp
from core.rife import chain_to_video


def render(
    runtime: LossyRuntime,
    prompt: str,
    seed_a: int,
    seed_b: int,
    n_steps: int,
    out_dir: Path,
    sampling_steps: int = 50,
    guidance_scale: float = 5.0,
    frames_per_pair: int = 8,
    fps: int = 12,
    save_mp4: bool = True,
    progress_cb=None,
) -> List[Path]:
    """Generate n_steps stills along slerp(noise_a -> noise_b), optionally encode MP4.

    Returns list of saved still paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx, mask = runtime.encode_prompt(prompt)
    noise_a = runtime.make_noise(seed_a)
    noise_b = runtime.make_noise(seed_b)

    paths: List[Path] = []
    for i in range(n_steps):
        t = i / (n_steps - 1) if n_steps > 1 else 0.0
        n = slerp(t, noise_a, noise_b)
        latent = runtime.sample(n, ctx, mask, num_steps=sampling_steps, guidance_scale=guidance_scale)
        img = runtime.decode_latent(latent)
        p = out_dir / f"latent_{i:03d}_t{t:.3f}.png"
        img.save(p)
        paths.append(p)
        print(f"[latent] {i+1}/{n_steps} saved {p.name}")
        if progress_cb is not None:
            progress_cb(i + 1, n_steps, f"sampled {i+1}/{n_steps}")

    if save_mp4 and len(paths) >= 2:
        if progress_cb is not None:
            progress_cb(n_steps, n_steps, "encoding video …")
        mp4 = out_dir / "latent_walk.mp4"
        chain_to_video(paths, mp4, frames_per_pair=frames_per_pair, fps=fps)
        print(f"[latent] mp4: {mp4}")

    return paths
