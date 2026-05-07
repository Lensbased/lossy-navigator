"""Mode F — atlas browser.

Loads output/atlas/index.npz + index_meta.json, provides:
    - status() / available()
    - load_index() -> dict
    - build_figure() -> plotly figure (KDE heatmap + samples + cluster labels)
    - query_concept(text) -> (xy, density_score, nearest_cluster)
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
        "coords": data["coords"],
        "labels_int": data["cluster_labels_int"],
        "density": data["density"],
        "density_x": data["density_x"],
        "density_y": data["density_y"],
        "hole_centers": data["hole_centers"],
        **meta,
    }


def build_figure(idx: Dict, query_xy: Optional[tuple] = None, query_text: str = ""):
    import plotly.graph_objects as go

    coords = idx["coords"]
    labels_int = idx["labels_int"]
    cluster_labels = idx["cluster_labels"]   # list[list[str]]
    hole_centers = idx["hole_centers"]
    hole_labels = idx["hole_labels"]
    prompts = idx["prompts"]
    seeds = idx["seeds"]

    fig = go.Figure()

    # KDE heatmap (background)
    fig.add_trace(go.Heatmap(
        x=idx["density_x"], y=idx["density_y"], z=idx["density"],
        colorscale="Viridis", showscale=False, opacity=0.55,
        hoverinfo="skip", name="density",
    ))

    # samples
    text = [f"{prompts[i]}<br>seed={seeds[i]}<br>cluster={labels_int[i]}"
            for i in range(len(coords))]
    color = ["#bbbbbb" if l < 0 else f"hsl({(int(l)*47)%360},70%,55%)" for l in labels_int]
    fig.add_trace(go.Scatter(
        x=coords[:, 0], y=coords[:, 1], mode="markers",
        marker=dict(color=color, size=7, line=dict(width=0.4, color="black")),
        text=text, hovertemplate="%{text}<extra></extra>", name="samples",
    ))

    # cluster labels (overlay text at centroid)
    for ci, lbls in enumerate(cluster_labels):
        m = labels_int == ci
        if not m.any() or not lbls:
            continue
        cx, cy = coords[m].mean(axis=0)
        fig.add_annotation(x=float(cx), y=float(cy), text="<b>" + " · ".join(lbls) + "</b>",
                           showarrow=False,
                           font=dict(size=11, color="white"),
                           bgcolor="rgba(0,0,0,0.65)", borderpad=3)

    # hole markers + labels
    for hi, hxy in enumerate(hole_centers):
        labs = hole_labels[hi] if hi < len(hole_labels) else []
        fig.add_trace(go.Scatter(
            x=[hxy[0]], y=[hxy[1]], mode="markers",
            marker=dict(symbol="x-thin", size=14, color="red", line=dict(width=2)),
            text=[f"HOLE near: {' · '.join(labs)}" if labs else "HOLE"],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False, name=f"hole_{hi}",
        ))
        if labs:
            fig.add_annotation(x=float(hxy[0]), y=float(hxy[1]), ay=-25, ax=0,
                               text="∅ " + " · ".join(labs[:2]),
                               showarrow=True, arrowsize=0.6, arrowwidth=1, arrowcolor="red",
                               font=dict(size=10, color="red"),
                               bgcolor="rgba(255,255,255,0.85)", borderpad=2)

    # query overlay
    if query_xy is not None:
        fig.add_trace(go.Scatter(
            x=[query_xy[0]], y=[query_xy[1]], mode="markers+text",
            marker=dict(symbol="star", size=24, color="yellow",
                        line=dict(width=2, color="black")),
            text=[f"  ❝{query_text}❞"], textposition="middle right",
            textfont=dict(color="black", size=12),
            hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        height=700, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="black",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x"),
        showlegend=False,
        title=dict(text=f"Atlas — {len(coords)} samples, {len(cluster_labels)} clusters, "
                        f"{len(hole_centers)} hole regions", x=0.01),
    )
    return fig


def query_concept(idx: Dict, text: str):
    """Encode a text concept via CLIP-text, project into the atlas via NN
    in CLIP-image space, return (xy, density_score, nearest_cluster_label).
    """
    if not text.strip():
        return None, 0.0, None
    text_emb = atlas_clip.embed_texts([text])[0]
    # nearest neighbor in image-embedding space
    sims = idx["img_emb"] @ text_emb
    top_k = 8
    top_idx = np.argsort(-sims)[:top_k]
    # weighted average of top-K coordinates, weights = softmax of sims
    w = sims[top_idx]
    w = np.exp(w - w.max())
    w = w / w.sum()
    xy = (idx["coords"][top_idx] * w[:, None]).sum(axis=0)

    # density at xy by bilinear lookup on the density grid
    gx, gy = idx["density_x"], idx["density_y"]
    xi = np.clip(np.searchsorted(gx, xy[0]) - 1, 0, len(gx) - 2)
    yi = np.clip(np.searchsorted(gy, xy[1]) - 1, 0, len(gy) - 2)
    density_score = float(idx["density"][yi, xi])

    # nearest cluster label
    labels_int = idx["labels_int"]
    cluster_labels = idx["cluster_labels"]
    voted = labels_int[top_idx]
    voted = voted[voted >= 0]
    nearest_cluster = None
    if len(voted) > 0:
        cid = int(np.bincount(voted).argmax())
        if cid < len(cluster_labels):
            nearest_cluster = cluster_labels[cid]

    return tuple(map(float, xy)), density_score, nearest_cluster
