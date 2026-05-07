"""LossyRuntime — minimal in-process wrapper around public-diffusion.

Loads model + VAE + T5 once and exposes:
    encode_prompt(text) -> (context, mask)
    make_noise(seed) -> tensor (B,C,H,W)
    sample(noise, context, mask, ...) -> latent
    decode_latent(latent) -> PIL.Image
    generate(prompt, seed, ...) -> PIL.Image  (convenience)

Plus interpolation helpers (slerp, lerp).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

# --- bootstrap public-diffusion src ---------------------------------------
PD_SRC = Path("/Users/hitosteyerl/CC/pd/public-diffusion/src")
if str(PD_SRC) not in sys.path:
    sys.path.insert(0, str(PD_SRC))

from pd.data.encode import T5Encoder, VAEEncoder  # noqa: E402
from pd.model.dit import build_model_from_cfg  # noqa: E402
from pd.utils.config import load_config  # noqa: E402


# ---- device + dtype helpers ----------------------------------------------
def _resolve_device(choice: str) -> str:
    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return choice


def _resolve_dtype(choice: str, cfg, device: str) -> torch.dtype:
    if choice == "fp16":
        return torch.float16
    if choice == "bf16":
        return torch.bfloat16
    if choice == "fp32":
        return torch.float32
    mp = str(getattr(getattr(cfg, "training", object()), "mixed_precision", "")).lower()
    if mp in {"fp16", "float16"}:
        return torch.float16
    if mp in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


# ---- our own euler sampler that accepts pre-generated noise --------------
def _euler_sample_init(model, init_noise, context, context_mask, num_steps, device, dtype, guidance_scale):
    x = init_noise.to(device=device, dtype=dtype)
    shape = x.shape
    dt = 1.0 / num_steps

    context = context.to(device=device, dtype=dtype)
    if context_mask is not None:
        context_mask = context_mask.to(device=device)

    use_cfg = guidance_scale > 1.0
    if use_cfg:
        uncond_context = torch.zeros_like(context)
        uncond_mask = torch.zeros_like(context_mask) if context_mask is not None else None

    timesteps = torch.linspace(1.0, dt, num_steps, device=device)
    with torch.autocast(device_type=device.split(":")[0], dtype=dtype):
        for t_val in timesteps:
            t = t_val.expand(shape[0])
            if use_cfg:
                v_cond = model(x, t, context, context_mask)
                v_uncond = model(x, t, uncond_context, uncond_mask)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = model(x, t, context, context_mask)
            x = x - dt * v
    return x


# ---- interpolation helpers -----------------------------------------------
def slerp(t: float, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between two tensors of same shape.
    t in [0,1].  Numerically stable for nearly-parallel a,b.
    """
    a_flat = a.flatten()
    b_flat = b.flatten()
    a_norm = a_flat / a_flat.norm()
    b_norm = b_flat / b_flat.norm()
    dot = (a_norm * b_norm).sum().clamp(-1.0, 1.0)
    if abs(dot.item()) > 0.9995:
        # nearly parallel — use linear
        return ((1 - t) * a + t * b)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    return (torch.sin((1 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b


def lerp(t: float, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (1 - t) * a + t * b


# ---- main class -----------------------------------------------------------
class LossyRuntime:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "auto",
        dtype: str = "auto",
        use_ema: bool = False,
    ):
        self.cfg = load_config(config_path)
        self.device = _resolve_device(device)
        self.dtype = _resolve_dtype(dtype, self.cfg, self.device)

        # -- model -------------------------------------------------------
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = None
        if isinstance(ckpt, dict):
            if use_ema and "ema" in ckpt:
                sd = ckpt["ema"]
            elif "model" in ckpt:
                sd = ckpt["model"]
            elif "ema" in ckpt:
                sd = ckpt["ema"]
        if sd is None:
            raise RuntimeError("checkpoint must have 'model' or 'ema' key")

        self.model = build_model_from_cfg(self.cfg)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[lossy_navigator] state_dict load: missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print(f"  first missing: {list(missing)[:5]}")
            if unexpected:
                print(f"  first unexpected: {list(unexpected)[:5]}")
        self.model.to(device=self.device, dtype=self.dtype).eval()

        # -- T5 ---------------------------------------------------------
        cache_dir = None
        if hasattr(self.cfg, "paths") and getattr(self.cfg.paths, "cache_dir", None):
            cache_dir = str(self.cfg.paths.cache_dir)
        self.t5 = T5Encoder(
            model_name=self.cfg.text_encoder.model_name,
            max_length=int(self.cfg.text_encoder.max_length),
            device=self.device,
            dtype=self.dtype,
            cache_dir=cache_dir,
        )

        # -- VAE --------------------------------------------------------
        self.vae = VAEEncoder(
            model_name=self.cfg.vae.model_name,
            device=self.device,
            dtype=self.dtype,
            cache_dir=cache_dir,
        )
        self.vae_scale = float(self.cfg.vae.scaling_factor)
        self.vae_shift = float(getattr(self.cfg.vae, "shift_factor", 0.0))

        # -- shape --------------------------------------------------------
        self.latent_shape = (
            1,
            int(self.cfg.model.latent_channels),
            int(self.cfg.model.latent_size),
            int(self.cfg.model.latent_size),
        )

        print(f"[lossy_navigator] loaded on {self.device}/{self.dtype}, latent={self.latent_shape}")

    # ---------------- API --------------------------------------------------
    @torch.inference_mode()
    def encode_prompt(self, prompt: str):
        embs_np, masks_np = self.t5.encode_captions([prompt])
        ctx = torch.from_numpy(embs_np[0]).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        mask = torch.from_numpy(masks_np[0]).unsqueeze(0).to(device=self.device)
        return ctx, mask

    def make_noise(self, seed: int) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        return torch.randn(self.latent_shape, generator=g, dtype=self.dtype).to(self.device)

    @torch.inference_mode()
    def sample(
        self,
        noise: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        num_steps: int = 100,
        guidance_scale: float = 5.0,
    ) -> torch.Tensor:
        return _euler_sample_init(
            model=self.model,
            init_noise=noise,
            context=context,
            context_mask=context_mask,
            num_steps=int(num_steps),
            device=self.device,
            dtype=self.dtype,
            guidance_scale=float(guidance_scale),
        )

    @torch.inference_mode()
    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        x = latent.float() / self.vae_scale + self.vae_shift
        decoded = self.vae.vae.decode(x.to(self.dtype)).sample  # (B,3,H,W) [-1,1]
        img = (decoded.float().clamp(-1, 1) + 1.0) / 2.0
        img = img[0].permute(1, 2, 0).cpu().numpy()
        img = (img * 255.0).round().astype(np.uint8)
        return Image.fromarray(img)

    def generate(self, prompt: str, seed: int, num_steps: int = 100, guidance_scale: float = 5.0) -> Image.Image:
        ctx, mask = self.encode_prompt(prompt)
        noise = self.make_noise(seed)
        latent = self.sample(noise, ctx, mask, num_steps, guidance_scale)
        return self.decode_latent(latent)
