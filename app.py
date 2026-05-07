"""LOSSY Navigator — local Streamlit app.

Four modes:
    Pixel:   RIFE between existing stills (no Lossy generation)
    Latent:  slerp between two noise seeds, sample Lossy at each step
    Prompt:  blend two prompt embeddings, sample Lossy at each step
    PCA:     walk along a precomputed semantic axis in conditioning space

Outputs save to output/<run_name>/ with stills + MP4.
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from core.lossy_runtime import LossyRuntime
from core.modes import latent as mode_latent
from core.modes import pca as mode_pca
from core.modes import pixel as mode_pixel
from core.modes import prompt as mode_prompt
from core.modes import stills_morph as mode_stills_morph
from core.lossy_runtime import slerp as _slerp_fn

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_CONFIG = "/Users/hitosteyerl/CC/ComfyUI/v8_facecrop.yaml"
DEFAULT_CHECKPOINT = "/Users/hitosteyerl/CC/ComfyUI/models/lossy/tracka_mmd-minsnr-step_4700000.pt"

st.set_page_config(page_title="LOSSY Navigator", layout="wide")


@st.cache_resource(show_spinner="Loading Lossy model …")
def get_runtime(config_path: str, checkpoint_path: str) -> LossyRuntime:
    return LossyRuntime(config_path=config_path, checkpoint_path=checkpoint_path)


# ---------------- sidebar -----------------------------------------------
st.sidebar.title("LOSSY")
st.sidebar.caption("latent / prompt / pixel / pca navigator")

with st.sidebar.expander("Model", expanded=False):
    config_path = st.text_input("config_path", DEFAULT_CONFIG)
    checkpoint_path = st.text_input("checkpoint_path", DEFAULT_CHECKPOINT)

with st.sidebar.expander("Generation defaults", expanded=False):
    sampling_steps = st.slider("sampling steps", 10, 200, 100, step=10)
    guidance_scale = st.slider("guidance_scale (CFG)", 1.0, 10.0, 5.0, step=0.5)
    frames_per_pair = st.slider("RIFE frames between keyframes", 2, 32, 8, step=1)
    fps = st.slider("video fps", 6, 30, 12)


def run_name(prefix: str) -> str:
    return f"{prefix}_{int(time.time())}"


def make_progress_cb(bar, log_box):
    """Returns a callback the mode functions call: cb(done, total, msg).
    Accumulates messages with timestamps in log_box.
    """
    history: list[str] = []
    def cb(done: int, total: int, msg: str = ""):
        frac = done / max(total, 1)
        bar.progress(min(max(frac, 0.0), 1.0))
        if msg:
            ts = time.strftime("%H:%M:%S")
            history.append(f"[{ts}] [{done}/{total}] {msg}")
            log_box.code("\n".join(history[-20:]), language="text")
    return cb


# ---------------- tabs ---------------------------------------------------
tab_pixel, tab_latent, tab_prompt, tab_pca, tab_stills = st.tabs(
    ["Pixel (RIFE only)", "Latent walk", "Prompt walk", "PCA axis", "Stills morph (Lossy-aware)"]
)

# ----- PIXEL ---------------------------------------------------------------
with tab_pixel:
    st.subheader("Pixel mode — RIFE between existing stills")
    st.caption("Drop ≥2 PNG/JPG files in order. RIFE blends pixels between consecutive pairs.")
    uploads = st.file_uploader(
        "stills (in order)",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg"],
        key="pixel_uploads",
    )
    if st.button("Render pixel walk", key="pixel_go"):
        if not uploads or len(uploads) < 2:
            st.error("Need at least 2 images.")
        else:
            run = run_name("pixel")
            run_dir = OUTPUT_DIR / run
            run_dir.mkdir()
            paths = []
            for i, u in enumerate(uploads):
                p = run_dir / f"input_{i:03d}_{u.name}"
                p.write_bytes(u.getbuffer())
                paths.append(p)
            bar = st.progress(0.0)
            status = st.empty()
            mp4 = mode_pixel.render(paths, run_dir / "pixel.mp4", frames_per_pair, fps,
                                    progress_cb=make_progress_cb(bar, status))
            bar.progress(1.0)
            pass
            st.success(f"saved {mp4}")
            st.video(str(mp4))


# ----- LATENT --------------------------------------------------------------
with tab_latent:
    st.subheader("Latent walk — slerp between two noise seeds")
    st.caption("Same prompt, two seeds.  Each frame is a real Lossy generation along the noise path.")
    prompt = st.text_area("prompt", "a face", key="latent_prompt", height=100)
    c1, c2, c3 = st.columns(3)
    seed_a = c1.number_input("seed A", value=42, step=1, key="latent_seed_a")
    seed_b = c2.number_input("seed B", value=99, step=1, key="latent_seed_b")
    n_keyframes = c3.slider("keyframes (Lossy generations)", 2, 16, 5, key="latent_n")
    # diagnostics
    with st.expander("vector data"):
        rt_dx = get_runtime(config_path, checkpoint_path)
        n_a = rt_dx.make_noise(int(seed_a))
        n_b = rt_dx.make_noise(int(seed_b))
        cos = (n_a.flatten() / n_a.flatten().norm()
               ).dot(n_b.flatten() / n_b.flatten().norm()).item()
        st.write(f"latent_shape: {tuple(n_a.shape)}  dtype: {n_a.dtype}  device: {n_a.device}")
        st.write(f"||noise_a||₂ = {n_a.float().norm().item():.3f}   ||noise_b||₂ = {n_b.float().norm().item():.3f}")
        st.write(f"cos(noise_a, noise_b) = {cos:+.4f}")
        st.write(f"slerp t values: {[round(i / max(int(n_keyframes)-1,1), 3) for i in range(int(n_keyframes))]}")

    if st.button("Render latent walk", key="latent_go"):
        run = run_name("latent")
        run_dir = OUTPUT_DIR / run
        rt = get_runtime(config_path, checkpoint_path)
        bar = st.progress(0.0)
        status = st.empty()
        paths = mode_latent.render(
            runtime=rt, prompt=prompt,
            seed_a=int(seed_a), seed_b=int(seed_b), n_steps=int(n_keyframes),
            out_dir=run_dir, sampling_steps=sampling_steps,
            guidance_scale=guidance_scale, frames_per_pair=frames_per_pair, fps=fps,
            progress_cb=make_progress_cb(bar, status),
        )
        bar.progress(1.0)
        mp4 = run_dir / "latent_walk.mp4"
        st.success(f"{len(paths)} stills + {mp4.name}")
        cols = st.columns(min(len(paths), 5))
        for i, p in enumerate(paths[:5]):
            cols[i].image(str(p), caption=p.name)
        st.video(str(mp4))


# ----- PROMPT --------------------------------------------------------------
with tab_prompt:
    st.subheader("Prompt walk — interpolate between two prompt embeddings")
    st.caption("Same noise seed, two prompts.  Each frame is a real Lossy generation along the conditioning blend.")
    prompt_a = st.text_area("prompt A (start)", "a young woman", key="prompt_a", height=80)
    prompt_b = st.text_area("prompt B (end)", "an old man", key="prompt_b", height=80)
    c1, c2, c3 = st.columns(3)
    seed = c1.number_input("seed (shared)", value=42, step=1, key="prompt_seed")
    n_keyframes = c2.slider("keyframes (Lossy generations)", 2, 16, 5, key="prompt_n")
    blend_mode = c3.selectbox("blend", ["lerp", "slerp"], key="prompt_blend")
    # diagnostics
    with st.expander("vector data"):
        rt_dx = get_runtime(config_path, checkpoint_path)
        ctx_a, _ = rt_dx.encode_prompt(prompt_a)
        ctx_b, _ = rt_dx.encode_prompt(prompt_b)
        a_pool = ctx_a.float().mean(dim=1).flatten()
        b_pool = ctx_b.float().mean(dim=1).flatten()
        cos = (a_pool / a_pool.norm()).dot(b_pool / b_pool.norm()).item()
        st.write(f"ctx_a shape: {tuple(ctx_a.shape)}  ctx_b shape: {tuple(ctx_b.shape)}")
        st.write(f"||ctx_a||₂ pooled = {a_pool.norm().item():.2f}   ||ctx_b||₂ pooled = {b_pool.norm().item():.2f}")
        st.write(f"cos(ctx_a, ctx_b) pooled = {cos:+.4f}")
        st.write(f"blend t values ({blend_mode}): {[round(i / max(int(n_keyframes)-1,1), 3) for i in range(int(n_keyframes))]}")

    if st.button("Render prompt walk", key="prompt_go"):
        run = run_name("prompt")
        run_dir = OUTPUT_DIR / run
        rt = get_runtime(config_path, checkpoint_path)
        bar = st.progress(0.0)
        status = st.empty()
        paths = mode_prompt.render(
            runtime=rt, prompt_a=prompt_a, prompt_b=prompt_b,
            seed=int(seed), n_steps=int(n_keyframes),
            out_dir=run_dir, sampling_steps=sampling_steps,
            guidance_scale=guidance_scale, frames_per_pair=frames_per_pair, fps=fps,
            blend_mode=blend_mode,
            progress_cb=make_progress_cb(bar, status),
        )
        bar.progress(1.0)
        mp4 = run_dir / "prompt_walk.mp4"
        st.success(f"{len(paths)} stills + {mp4.name}")
        cols = st.columns(min(len(paths), 5))
        for i, p in enumerate(paths[:5]):
            cols[i].image(str(p), caption=p.name)
        st.video(str(mp4))


# ----- PCA ----------------------------------------------------------------
with tab_pca:
    st.subheader("PCA axis walk — semantic direction in conditioning space")
    if not mode_pca.axes_available():
        st.warning(
            "No PCA axes file found.  Run **`python tools/build_pca.py`** once "
            "(~15 min) to generate `tools/pca_axes.npz`, then refresh this page."
        )
    else:
        axes_info = mode_pca.list_axes()
        prompt = st.text_area("prompt", "a face", key="pca_prompt", height=80)
        c1, c2, c3, c4 = st.columns(4)
        seed = c1.number_input("seed", value=42, step=1, key="pca_seed")
        axis_choice = c2.selectbox(
            "axis",
            options=[i for i, _ in axes_info],
            format_func=lambda i: f"axis {i}  (ev={axes_info[i][1]:.3f})",
            key="pca_axis",
        )
        walk_range = c3.slider("walk range ±", 0.5, 8.0, 3.0, step=0.5, key="pca_range")
        n_keyframes = c4.slider("keyframes", 3, 16, 7, key="pca_n")
        with st.expander("vector data"):
            ax_idx = int(axis_choice)
            d = __import__("numpy").load(mode_pca.PCA_FILE)
            ax = d["axes"][ax_idx]
            st.write(f"axis index: {ax_idx}  explained_variance_ratio: {axes_info[ax_idx][1]:.4f}")
            st.write(f"axis vector length (L2): {float((ax**2).sum()**0.5):.4f}")
            st.write(f"axis stats:  min={ax.min():.3f}  max={ax.max():.3f}  mean={ax.mean():.4f}  std={ax.std():.4f}")
            st.write(f"walk t values: {[round(-walk_range + (2*walk_range)*i/max(int(n_keyframes)-1,1), 2) for i in range(int(n_keyframes))]}")

        if st.button("Render PCA walk", key="pca_go"):
            run = run_name(f"pca_axis{axis_choice}")
            run_dir = OUTPUT_DIR / run
            rt = get_runtime(config_path, checkpoint_path)
            bar = st.progress(0.0)
            status = st.empty()
            paths = mode_pca.render(
                runtime=rt, prompt=prompt, seed=int(seed),
                axis_index=int(axis_choice), walk_range=float(walk_range),
                n_steps=int(n_keyframes), out_dir=run_dir,
                sampling_steps=sampling_steps, guidance_scale=guidance_scale,
                frames_per_pair=frames_per_pair, fps=fps,
                progress_cb=make_progress_cb(bar, status),
            )
            bar.progress(1.0)
            mp4 = run_dir / f"pca_axis{axis_choice}_walk.mp4"

# ----- STILLS MORPH (Lossy-aware) ---------------------------------------
with tab_stills:
    st.subheader("Stills morph — load N stills, slerp in Lossy's VAE-latent space, RIFE for smoothness")
    st.caption("Each input still gets VAE-encoded (Lossy's VAE), latents are spherically interpolated, "
               "then VAE-decoded. Optional SDEdit refinement runs Lossy partially on each interpolated latent for more 'Lossy presence'.")
    uploads = st.file_uploader(
        "stills (in order, ≥2)",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg"],
        key="stills_uploads",
    )
    c1, c2, c3 = st.columns(3)
    keyframes_per_pair = c1.slider("latent slerp keyframes per pair", 2, 16, 4, key="stills_kpp")
    refine = c2.checkbox("SDEdit refine (slower, more Lossy)", value=False, key="stills_refine")
    refine_strength = c3.slider("refine strength", 0.0, 0.9, 0.3, step=0.05, key="stills_strength",
                                disabled=not refine)
    refine_prompt = st.text_input("refine prompt (optional, used only if SDEdit on)", "",
                                  key="stills_refine_prompt")

    if st.button("Render stills morph", key="stills_go"):
        if not uploads or len(uploads) < 2:
            st.error("Need at least 2 images.")
        else:
            run = run_name("stills_morph")
            run_dir = OUTPUT_DIR / run
            run_dir.mkdir()
            paths_in = []
            for i, u in enumerate(uploads):
                p = run_dir / f"input_{i:03d}_{u.name}"
                p.write_bytes(u.getbuffer())
                paths_in.append(p)
            rt = get_runtime(config_path, checkpoint_path)
            bar = st.progress(0.0)
            status = st.empty()
            paths = mode_stills_morph.render(
                runtime=rt, image_paths=paths_in, out_dir=run_dir,
                keyframes_per_pair=int(keyframes_per_pair), refine=refine,
                refine_strength=float(refine_strength),
                refine_prompt=refine_prompt,
                sampling_steps=sampling_steps, guidance_scale=guidance_scale,
                frames_per_pair=frames_per_pair, fps=fps,
                progress_cb=make_progress_cb(bar, status),
            )
            bar.progress(1.0)
            mp4 = run_dir / "stills_morph.mp4"
            st.success(f"{len(paths)} latent keyframes + {mp4.name}")
            cols = st.columns(min(len(paths), 5))
            for i, p in enumerate(paths[:5]):
                cols[i].image(str(p), caption=p.name)
            st.video(str(mp4))
            st.success(f"{len(paths)} stills + {mp4.name}")
            cols = st.columns(min(len(paths), 5))
            for i, p in enumerate(paths[:5]):
                cols[i].image(str(p), caption=p.name)
            st.video(str(mp4))
