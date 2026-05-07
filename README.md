# LOSSY Navigator

Local Streamlit app for exploring the LOSSY (formerly Public Diffusion) model's
latent + conditioning space.  Five modes:

- **Pixel** — RIFE-only frame interpolation between existing stills.
- **Latent walk** — slerp between two noise seeds, sample LOSSY at each step.
- **Prompt walk** — interpolate between two prompt embeddings, same noise.
- **PCA axis** — walk along precomputed semantic directions in T5 conditioning space.
- **Stills morph (Lossy-aware)** — VAE-encode N stills, slerp latents through
  Lossy's VAE, optional SDEdit refinement.

Each mode outputs N keyframe stills + an MP4 (RIFE-smoothed).

## Requirements

- Apple Silicon Mac with PyTorch + MPS, or CUDA box
- A working LOSSY checkpoint and config YAML
- A copy of the [public-diffusion](https://github.com/moellenhoff/public-diffusion)
  source on disk (the `feature/quantum-mmd-v6` branch — the one with RoPE in `dit.py`)

## Install

```bash
git clone https://github.com/Lensbased/lossy-navigator
cd lossy-navigator

# RIFE binary (macOS arm64)
mkdir -p bin && cd bin
curl -LO https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-macos.zip
unzip -q rife-ncnn-vulkan-20221029-macos.zip
cd ..

# Reuse an existing ComfyUI venv that already has torch/transformers/diffusers
# OR set up a fresh one:
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Edit `app.py` (top of file) to point `DEFAULT_CONFIG` and `DEFAULT_CHECKPOINT`
at your local Lossy YAML and `.pt` files. Also edit `core/lossy_runtime.py`'s
`PD_SRC` constant to your local public-diffusion `src/` path.

## Run

```bash
source .venv/bin/activate
streamlit run app.py --server.port 8502
```

Open <http://127.0.0.1:8502>.

## PCA mode

The PCA tab is locked until you build the axes file once:

```bash
python tools/build_pca.py
```

Takes ~2 minutes. Saves `tools/pca_axes.npz` (16 axes from 60-caption corpus).

## Output

Each render lands in `output/<mode>_<timestamp>/` with the keyframe stills and
the MP4. Folder opens in Finder when the desktop launcher boots the app.

## Provenance note

Lossy itself is provenance-clean (training data + model architecture).
**RIFE is not** — it's a small CNN trained on Vimeo-90K. Use the pixel /
RIFE-smoothed outputs as previews, not as final clean Lossy outputs.

## License

MIT for the navigator code. Lossy weights and public-diffusion source have
their own licenses — check those repos.
