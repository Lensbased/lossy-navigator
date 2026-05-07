"""Mode C — prompt walk.

Encode prompt A and prompt B as T5 conditioning, linearly interpolate the
embeddings, sample Lossy at N points along the blend.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from core.lossy_runtime import LossyRuntime, lerp, slerp
from core.rife import chain_to_video


def render(
    runtime: LossyRuntime,
    prompt_a: str,
    prompt_b: str,
    seed: int,
    n_steps: int,
    out_dir: Path,
    sampling_steps: int = 50,
    guidance_scale: float = 5.0,
    frames_per_pair: int = 8,
    fps: int = 12,
    blend_mode: str = "lerp",  # 'lerp' or 'slerp'
    save_mp4: bool = True,
    progress_cb=None,
) -> List[Path]:
    """Generate n_steps stills along the prompt-embedding blend a -> b.

    Same noise (seed) for all steps so only the conditioning varies.
    Returns list of saved still paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx_a, mask_a = runtime.encode_prompt(prompt_a)
    ctx_b, mask_b = runtime.encode_prompt(prompt_b)
    # masks should match shape; OR them (any token attended in either prompt is attended)
    mask = (mask_a + mask_b).clamp_max(1)
    noise = runtime.make_noise(seed)
    blend = slerp if blend_mode == "slerp" else lerp

    paths: List[Path] = []
    for i in range(n_steps):
        t = i / (n_steps - 1) if n_steps > 1 else 0.0
        ctx = blend(t, ctx_a, ctx_b)
        latent = runtime.sample(noise, ctx, mask, num_steps=sampling_steps, guidance_scale=guidance_scale)
        img = runtime.decode_latent(latent)
        p = out_dir / f"prompt_{i:03d}_t{t:.3f}.png"
        img.save(p)
        paths.append(p)
        print(f"[prompt] {i+1}/{n_steps} saved {p.name}")
        if progress_cb is not None:
            progress_cb(i + 1, n_steps, f"sampled {i+1}/{n_steps}")

    if save_mp4 and len(paths) >= 2:
        if progress_cb is not None:
            progress_cb(n_steps, n_steps, "encoding video …")
        mp4 = out_dir / "prompt_walk.mp4"
        chain_to_video(paths, mp4, frames_per_pair=frames_per_pair, fps=fps)
        print(f"[prompt] mp4: {mp4}")

    return paths
