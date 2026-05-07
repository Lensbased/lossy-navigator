"""RIFE wrapper using rife-ncnn-vulkan binary.

Public API:
    interpolate_pair(img_a, img_b, n_inserted, out_dir) -> list[Path]
        Inserts n_inserted frames between two stills.  Returns ordered list of
        frame paths (including the two endpoints).

    chain_to_video(image_paths, out_mp4, frames_per_pair, fps)
        Walks consecutive pairs, RIFE-interps each, encodes one MP4.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RIFE_DIR = PROJECT_ROOT / "bin" / "rife-ncnn-vulkan-20221029-macos"
RIFE_BIN = RIFE_DIR / "rife-ncnn-vulkan"
RIFE_MODEL_DIR = RIFE_DIR / "rife-v4.6"


def _run_rife(in_dir: Path, out_dir: Path, num_frames: int, model_dir: Path = RIFE_MODEL_DIR):
    """Run RIFE on a folder of N keyframes to produce num_frames interpolated frames."""
    cmd = [
        str(RIFE_BIN),
        "-i", str(in_dir),
        "-o", str(out_dir),
        "-n", str(num_frames),
        "-m", str(model_dir),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"rife failed: {out.stderr}")


def interpolate_pair(img_a: Image.Image, img_b: Image.Image, n_inserted: int, out_dir: Path) -> List[Path]:
    """Insert n_inserted frames between img_a and img_b.

    Returns ordered list of length (n_inserted + 2).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_in = out_dir / "_in"
    tmp_in.mkdir(exist_ok=True)
    img_a.save(tmp_in / "0001.png")
    img_b.save(tmp_in / "0002.png")
    total_out = n_inserted + 2  # endpoints included
    _run_rife(tmp_in, out_dir, num_frames=total_out)
    shutil.rmtree(tmp_in)
    frames = sorted(out_dir.glob("*.png"))
    return frames


def chain_to_video(
    image_paths: Iterable[Path],
    out_mp4: Path,
    frames_per_pair: int = 16,
    fps: int = 12,
    progress_cb=None,
) -> Path:
    """Walk consecutive pairs of images, RIFE-interp each, encode one MP4.

    image_paths order matters.  frames_per_pair is the total frames between
    each pair INCLUDING endpoints (so 16 = 2 endpoints + 14 inserted).
    """
    image_paths = list(image_paths)
    if len(image_paths) < 2:
        raise ValueError("need at least 2 images")

    workdir = out_mp4.parent / f".{out_mp4.stem}_work"
    workdir.mkdir(parents=True, exist_ok=True)

    # Build sequential frames, dropping the first frame of each pair after the first
    # to avoid duplicate seam frames.
    frame_counter = 0
    final_dir = workdir / "frames"
    final_dir.mkdir(exist_ok=True)

    n_segments = len(image_paths) - 1
    for i in range(n_segments):
        if progress_cb is not None:
            progress_cb(i, n_segments, f"RIFE segment {i+1}/{n_segments}")
        seg_dir = workdir / f"seg_{i:03d}"
        seg_dir.mkdir(exist_ok=True)
        a = Image.open(image_paths[i]).convert("RGB")
        b = Image.open(image_paths[i + 1]).convert("RGB")
        # Resize both to same dims if mismatched
        if a.size != b.size:
            b = b.resize(a.size, Image.LANCZOS)
        seg_frames = interpolate_pair(a, b, n_inserted=frames_per_pair - 2, out_dir=seg_dir)
        # Drop first frame of all segments after the first to avoid duplicate seam
        if i > 0:
            seg_frames = seg_frames[1:]
        for sf in seg_frames:
            target = final_dir / f"f_{frame_counter:06d}.png"
            shutil.copy(sf, target)
            frame_counter += 1

    if progress_cb is not None:
        progress_cb(n_segments, n_segments, "encoding MP4 …")
    # ffmpeg encode
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(final_dir / "f_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(out_mp4),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {out.stderr}")

    shutil.rmtree(workdir)
    return out_mp4
