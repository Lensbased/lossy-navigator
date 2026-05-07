"""Mode E — stills morph.

Load N pre-rendered stills.  For each consecutive pair:
    - VAE-encode both to latents (using Lossy's VAE)
    - SLERP K latents between them
    - Optional SDEdit-style refinement: add noise, denoise partial steps with Lossy
    - VAE-decode each latent
    - RIFE between decoded frames for pixel-level smoothness
    - Concatenate all into one MP4

This gives a "Lossy-flavored" morph between arbitrary input stills, not a
pure pixel crossfade.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
from PIL import Image

from core.lossy_runtime import LossyRuntime, slerp
from core.rife import chain_to_video


def _img_to_latent(rt: LossyRuntime, img: Image.Image) -> torch.Tensor:
    """VAE-encode a PIL image to a Lossy latent at the cfg's latent size."""
    target = int(rt.cfg.model.latent_size) * 8  # SDXL VAE downsamples 8x
    img = img.convert("RGB").resize((target, target), Image.LANCZOS)
    arr = torch.from_numpy(__import__("numpy").asarray(img)).float() / 255.0
    arr = arr.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0  # (1,3,H,W) [-1,1]
    arr = arr.to(device=rt.device, dtype=rt.dtype)
    with torch.inference_mode():
        latent_dist = rt.vae.vae.encode(arr).latent_dist.sample()
        # convert to "model space" (the inverse of decode_latent's pre-scaling)
        latent = (latent_dist - rt.vae_shift) * rt.vae_scale
    return latent.to(rt.dtype)


def render(
    runtime: LossyRuntime,
    image_paths: Iterable[Path],
    out_dir: Path,
    keyframes_per_pair: int = 4,
    refine: bool = False,
    refine_strength: float = 0.3,
    sampling_steps: int = 50,
    guidance_scale: float = 5.0,
    refine_prompt: str = "",
    frames_per_pair: int = 8,
    fps: int = 12,
    save_mp4: bool = True,
    progress_cb=None,
) -> List[Path]:
    """Walk N input stills via VAE-latent slerp, optional SDEdit refinement, then RIFE."""
    image_paths = [Path(p) for p in image_paths]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Encode all input stills to latents up front
    print(f"[stills_morph] VAE-encoding {len(image_paths)} stills …")
    latents = []
    for i, p in enumerate(image_paths):
        latents.append(_img_to_latent(runtime, Image.open(p)))
        if progress_cb is not None:
            progress_cb(i + 1, len(image_paths) * 2, f"encoding {i+1}/{len(image_paths)}")

    # Optional SDEdit refinement requires a prompt context (empty string OK)
    ctx, mask = (None, None)
    if refine:
        ctx, mask = runtime.encode_prompt(refine_prompt)

    # For each pair, slerp K latents
    saved: List[Path] = []
    n_pairs = len(latents) - 1
    keyframe_idx = 0
    total_keyframes = n_pairs * keyframes_per_pair + 1
    for pi in range(n_pairs):
        a, b = latents[pi], latents[pi + 1]
        steps = keyframes_per_pair + 1 if pi == 0 else keyframes_per_pair  # avoid duplicate seam
        offset = 0 if pi == 0 else 1 / keyframes_per_pair
        for k in range(steps):
            t = (k / keyframes_per_pair) if pi == 0 else (offset + k / keyframes_per_pair)
            t = min(t, 1.0)
            mixed = slerp(t, a, b)
            if refine and refine_strength > 0:
                # SDEdit: add noise to mixed, run partial denoise from t=refine_strength
                noise = torch.randn_like(mixed)
                noised = (1 - refine_strength) * mixed + refine_strength * noise
                # run a short denoise pass
                short_steps = max(int(sampling_steps * refine_strength), 4)
                mixed = runtime.sample(noised, ctx, mask,
                                       num_steps=short_steps, guidance_scale=guidance_scale)
            img = runtime.decode_latent(mixed)
            outp = out_dir / f"morph_{keyframe_idx:04d}_pair{pi}_t{t:.3f}.png"
            img.save(outp)
            saved.append(outp)
            keyframe_idx += 1
            if progress_cb is not None:
                progress_cb(len(image_paths) + keyframe_idx,
                            len(image_paths) + total_keyframes,
                            f"latent slerp {keyframe_idx}/{total_keyframes}")

    if save_mp4 and len(saved) >= 2:
        if progress_cb is not None:
            progress_cb(0, 1, "RIFE + ffmpeg …")
        mp4 = out_dir / "stills_morph.mp4"
        chain_to_video(saved, mp4, frames_per_pair=frames_per_pair, fps=fps,
                       progress_cb=progress_cb)
        print(f"[stills_morph] mp4: {mp4}")

    return saved
