"""Mode A — pixel interpolation between existing stills via RIFE.

Takes a list of image paths, returns an MP4.
No Lossy generation involved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from core.rife import chain_to_video


def render(
    image_paths: Iterable[Path],
    out_mp4: Path,
    frames_per_pair: int = 16,
    fps: int = 12,
    progress_cb=None,
) -> Path:
    image_paths = [Path(p) for p in image_paths]
    return chain_to_video(image_paths, out_mp4, frames_per_pair=frames_per_pair, fps=fps, progress_cb=progress_cb)
