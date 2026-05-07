# Project: lossy_navigator

## Purpose

Local Streamlit app for **exploring the Lossy (public-diffusion) model's latent
and conditioning space**. Six modes:

1. **Pixel** — RIFE-only frame interpolation between existing stills (no Lossy).
2. **Latent walk** — slerp between two noise seeds, sample Lossy at N points.
3. **Prompt walk** — interpolate between two prompt embeddings, same noise.
4. **PCA axis** — walk along precomputed semantic directions in T5 conditioning
   space (top-K principal axes from a 60-caption corpus).
5. **Stills morph (Lossy-aware)** — VAE-encode N stills, slerp latents through
   Lossy's VAE, optional SDEdit-style denoise refinement, RIFE-smooth → MP4.
6. **Atlas** — 3D rotatable map of 615 generated samples, KDE density,
   HDBSCAN-clustered, auto-labelled clusters and "holes" using a 379-concept
   CLIP-text vocabulary; concept query box projects any text into the same map.

Public repo (shared with Volker): https://github.com/Lensbased/lossy-navigator

## Tech Stack

- **Language:** Python 3.13
- **UI:** Streamlit (port 8502)
- **ML runtime:** PyTorch (MPS), in-process import of public-diffusion's
  `pd.model.dit`, `pd.data.encode`, `pd.training.rectified_flow`
- **Pixel interpolation:** rife-ncnn-vulkan binary (Vulkan-on-Mac)
- **Atlas:** OpenAI CLIP ViT-B/32, UMAP-learn, HDBSCAN, scipy KDE,
  Plotly Scatter3d
- **Video encode:** ffmpeg (system)
- **Reuses ComfyUI's `.venv`** (already has torch/transformers/diffusers)

## Architecture

```
lossy_navigator/
├── app.py                          # Streamlit, 6 tabs
├── core/
│   ├── lossy_runtime.py            # LossyRuntime: model + T5 + VAE + sampler
│   ├── rife.py                     # RIFE binary wrapper, chain_to_video()
│   ├── atlas_clip.py               # CLIP image+text embedder
│   └── modes/
│       ├── pixel.py
│       ├── latent.py
│       ├── prompt.py
│       ├── pca.py
│       ├── stills_morph.py
│       └── atlas.py                # plotly figure + concept query
├── tools/
│   ├── build_pca.py                # one-time: T5-encode 60 captions → PCA axes
│   ├── build_atlas_corpus.py       # one-time: generate 615 Lossy samples
│   └── build_atlas_index.py        # one-time: CLIP+UMAP+cluster+label
├── bin/rife-ncnn-vulkan-20221029-macos/   # downloaded binary, gitignored
└── output/<run_name>/              # all renders saved here
```

Desktop launcher: `~/Desktop/Start LOSSY Navigator.command`.

## Key Decisions

- **Reuse the ComfyUI venv** — avoided duplicate ~3 GB torch download.
- **Custom `_euler_sample_init()`** in lossy_runtime.py — public-diffusion's
  `euler_sample` always starts from `torch.randn`; we needed to inject
  controlled noise for slerp walks, so wrote a copy that takes init_noise.
- **CLIP ViT-B/32 over ViT-L/14** — 600 MB vs 1.7 GB; speed and disk over
  precision for atlas labelling.
- **3D atlas via UMAP n_components=3 + Scatter3d** — rotatable, scrollable,
  hover-able. KDE evaluated at samples drives marker size; 3D voxel grid for
  hole detection.
- **CLIP-text vocabulary lookup for cluster labels** — 379 hand-curated concept
  words; for each cluster centroid (in CLIP-image space), top-K nearest
  vocabulary words become the auto-label. Same for "holes" — find centroid of
  low-density voxels, query vocab.
- **Hot-reload disabled** in launcher — Streamlit's import cache breaks Lossy
  re-imports; full process restart needed for code changes.

## Challenges & Solutions

- **MPS bicubic+antialias not implemented for fp16** in
  `comfy/clip_model.py::clip_preprocess` — patched to cast to fp32 before
  interpolate, cast back. (Hit while testing DynamiCrafter fork — kept as
  prophylactic fix.)
- **CLIP `get_*_features()` returns object vs tensor across transformers
  versions** — `_to_tensor()` helper in `atlas_clip.py` handles both.
- **ComfyUI's `comfy/ldm/cogvideo/` got accidentally deleted** during disk
  cleanup — restored via `git checkout origin/master -- comfy/ldm/cogvideo/`.
  Lesson: only `models/`, `temp/`, `output/` are safe to delete; `comfy/` and
  `custom_nodes/` are source code.
- **Disk pressure** during build (5 GB free) — corpus ≈ 615 PNGs at 256×256
  optimized = ~150 MB; CLIP weights cached separately.
- **DynamiCrafter dead end** before this project: SD 2.1 text encoder
  (which DynamiCrafter wrapper hardcodes to download) is now gated/removed
  on HuggingFace. Workaround: download from `sd2-community/stable-diffusion-2-1`
  mirror to `models/clip/` so the existence check skips the download. We
  abandoned DynamiCrafter and moved on to this navigator.

## Lessons Learned

- **Fast smoke tests at every stage**: each new module got a 30-second
  end-to-end test in background before moving on. Caught the CLIP API
  mismatch and an output-size mismatch in <2 min each.
- **In-process Lossy beats subprocess-via-ComfyUI**: cleaner, faster,
  one-time model load.
- **Auto-labels via CLIP-text vocabulary** are surprisingly meaningful even
  for a noisy 4.7M-step facecrop checkpoint. Most clusters honestly labelled
  themselves "noise · motion blur · abstract" — confirming what we knew.
- **Provenance is preserved at the Lossy level**, broken at RIFE
  (Vimeo-90K) and CLIP (LAION-derived) layers — these are tools for
  exploration, not clean outputs.

## Related Projects

- **`pd`** — public-diffusion training pipeline (Lossy itself).
- **`pd/aesthetic`** — sibling MLP-based aesthetic scorer.
- **`pd/gov-scraper`** — sibling supplementary data scrapers.
- **`ComfyUI/custom_nodes/ComfyUI-LOSSY`** — sibling ComfyUI integration; the
  navigator is the standalone alternative.

## Status

- All 6 modes tested end-to-end ✓
- 615-sample 3D atlas built locally ✓
- PCA axes file built ✓
- Pushed to public repo (Lensbased/lossy-navigator) ✓
- Volker has independently rebuilt his own version

## How to use

```bash
# launch
~/Desktop/Start\ LOSSY\ Navigator.command

# rebuild PCA axes (~2 min, one-time)
cd /Users/hitosteyerl/CC/lossy_navigator
source /Users/hitosteyerl/CC/ComfyUI/.venv/bin/activate
python tools/build_pca.py

# rebuild atlas (~40-50 min corpus + ~2 min index, one-time)
python tools/build_atlas_corpus.py
python tools/build_atlas_index.py
```

Outputs land in `output/<mode>_<timestamp>/` with stills + MP4.

---

*Created: 2026-05-07*
*Last updated: 2026-05-07*
