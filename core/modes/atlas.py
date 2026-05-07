"""Mode F — atlas browser (3D).

Loads output/atlas/index.npz + index_meta.json, provides:
    - available()
    - load_index() -> dict
    - build_figure() -> rotatable plotly Scatter3d (samples, clusters, holes)
    - query_concept(text) -> (xyz, density_score, nearest_cluster)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from core import atlas_clip


ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT / "output" / "atlas" / "index.npz"
META_PATH = ROOT / "output" / "atlas" / "index_meta.json"


def available() -> bool:
    return INDEX_PATH.exists() and META_PATH.exists()


def load_index() -> Optional[Dict]:
    if not available():
        return None
    data = dict(np.load(INDEX_PATH))
    meta = json.loads(META_PATH.read_text())
    return {
        "img_emb": data["img_emb"],
        "coords": data["coords"],                 # (N, 3)
        "sample_density": data.get("sample_density",
                                   np.ones(len(data["coords"]), dtype=np.float32)),
        "labels_int": data["cluster_labels_int"],
        "density": data["density"],               # (Gx, Gy, Gz)
        "density_x": data["density_x"],
        "density_y": data["density_y"],
        "density_z": data.get("density_z", None),
        "hole_centers": data["hole_centers"],     # (H, 3)
        **meta,
    }


def build_figure(idx: Dict, query_xyz: Optional[tuple] = None, query_text: str = ""):
    import plotly.graph_objects as go

    coords = idx["coords"]                  # (N, 3)
    labels_int = idx["labels_int"]
    cluster_labels = idx["cluster_labels"]
    hole_centers = idx["hole_centers"]
    hole_labels = idx["hole_labels"]
    prompts = idx["prompts"]
    seeds = idx["seeds"]
    sample_density = idx["sample_density"]

    fig = go.Figure()

    # Sample points — color by cluster, size by local density
    sd = sample_density
    sd_norm = (sd - sd.min()) / (sd.max() - sd.min() + 1e-8)
    sizes = 4 + sd_norm * 10                       # 4..14
    color = ["#aaaaaa" if l < 0 else f"hsl({(int(l)*47)%360},70%,55%)"
             for l in labels_int]
    text = [f"{prompts[i]}<br>seed={seeds[i]}<br>cluster={labels_int[i]}<br>density={sd[i]:.3f}"
            for i in range(len(coords))]
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode="markers",
        marker=dict(color=color, size=sizes,
                    line=dict(width=0.3, color="black"),
                    opacity=0.85),
        text=text, hovertemplate="%{text}<extra></extra>",
        name="samples",
    ))

    # Cluster labels at centroid
    for ci, lbls in enumerate(cluster_labels):
        m = labels_int == ci
        if not m.any() or not lbls:
            continue
        cx, cy, cz = coords[m].mean(axis=0)
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[cz], mode="text",
            text=["<b>" + " · ".join(lbls) + "</b>"],
            textfont=dict(size=12, color="white"),
            hoverinfo="skip", showlegend=False,
        ))

    # Hole markers
    if len(hole_centers) > 0:
        h_text = [f"HOLE near: {' · '.join(hole_labels[i])}" if i < len(hole_labels)
                  else "HOLE" for i in range(len(hole_centers))]
        fig.add_trace(go.Scatter3d(
            x=hole_centers[:, 0], y=hole_centers[:, 1], z=hole_centers[:, 2],
            mode="markers+text",
            marker=dict(symbol="x", size=8, color="red", line=dict(width=2)),
            text=["∅ " + " · ".join(hole_labels[i][:2]) if i < len(hole_labels) and hole_labels[i] else "∅"
                  for i in range(len(hole_centers))],
            textposition="top center",
            textfont=dict(size=10, color="red"),
            hovertext=h_text, hoverinfo="text",
            name="holes", showlegend=False,
        ))

    # Query point
    if query_xyz is not None:
        fig.add_trace(go.Scatter3d(
            x=[query_xyz[0]], y=[query_xyz[1]], z=[query_xyz[2]],
            mode="markers+text",
            marker=dict(symbol="diamond", size=12, color="yellow",
                        line=dict(width=2, color="black")),
            text=[f"  ❝{query_text}❞"], textposition="top center",
            textfont=dict(color="yellow", size=12),
            hoverinfo="skip", showlegend=False, name="query",
        ))

    fig.update_layout(
        height=750, margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            bgcolor="black",
        ),
        showlegend=False,
        title=dict(text=f"Atlas (3D) — {len(coords)} samples · {len(cluster_labels)} clusters · "
                        f"{len(hole_centers)} holes — drag to rotate, scroll to zoom",
                   x=0.01, font=dict(size=12)),
    )
    return fig


def query_concept(idx: Dict, text: str):
    """Encode a text concept via CLIP-text, project into the atlas via NN
    in CLIP-image space, return (xyz, density_score, nearest_cluster_label).
    """
    if not text.strip():
        return None, 0.0, None
    text_emb = atlas_clip.embed_texts([text])[0]
    sims = idx["img_emb"] @ text_emb
    top_k = 8
    top_idx = np.argsort(-sims)[:top_k]
    w = np.exp(sims[top_idx] - sims[top_idx].max())
    w = w / w.sum()
    xyz = (idx["coords"][top_idx] * w[:, None]).sum(axis=0)

    # density at xyz by nearest grid voxel lookup
    gx, gy, gz = idx["density_x"], idx["density_y"], idx["density_z"]
    xi = int(np.clip(np.searchsorted(gx, xyz[0]) - 1, 0, len(gx) - 2))
    yi = int(np.clip(np.searchsorted(gy, xyz[1]) - 1, 0, len(gy) - 2))
    zi = int(np.clip(np.searchsorted(gz, xyz[2]) - 1, 0, len(gz) - 2))
    density_score = float(idx["density"][xi, yi, zi])

    # nearest cluster label (vote among top-K samples)
    voted = idx["labels_int"][top_idx]
    voted = voted[voted >= 0]
    nearest_cluster = None
    if len(voted) > 0:
        cid = int(np.bincount(voted).argmax())
        if cid < len(idx["cluster_labels"]):
            nearest_cluster = idx["cluster_labels"][cid]

    return tuple(map(float, xyz)), density_score, nearest_cluster
