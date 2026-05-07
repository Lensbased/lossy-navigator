"""Build the atlas index from a corpus directory.

Reads output/atlas/corpus/manifest.csv + sample PNGs, computes:
    - CLIP image embeddings (768-d for ViT-B/32 → 512 actually; ViT-L/14 → 768)
    - UMAP 2D projection
    - HDBSCAN clusters
    - Per-cluster auto-labels via CLIP-text ↔ centroid similarity
    - 2D KDE density grid
    - Detected "holes" (low-density regions) + their inferred labels

Saves output/atlas/index.npz + index_meta.json.
Run after build_atlas_corpus.py.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORPUS_DIR = ROOT / "output" / "atlas" / "corpus"
INDEX_PATH = ROOT / "output" / "atlas" / "index.npz"
META_PATH = ROOT / "output" / "atlas" / "index_meta.json"


# vocabulary used for auto-labelling clusters and holes
CONCEPT_VOCAB = [
    # subjects
    "face", "portrait", "person", "woman", "man", "child", "elderly", "young",
    "head", "eyes", "smile", "profile", "hair", "skin", "lips", "beard",
    # objects
    "drone", "bird", "dove", "plane", "boat", "car", "phone", "book", "suitcase",
    "cup", "table", "chair", "lamp", "candle", "flame", "fire", "smoke",
    # nature / landscape
    "forest", "tree", "mountain", "valley", "river", "lake", "ocean", "beach",
    "field", "meadow", "wildflower", "snow", "winter", "summer", "dawn", "dusk",
    "sky", "cloud", "fog", "mist", "storm", "rain", "moon", "sun",
    # urban / built
    "city", "street", "road", "building", "office", "hallway", "town", "village",
    "window", "door", "stained glass", "neon", "asphalt",
    # light / mood
    "candlelight", "sunlight", "moonlight", "shadow", "darkness", "golden hour",
    "soft light", "harsh light", "dramatic", "serene", "calm", "tense",
    "morning", "evening", "night",
    # texture / abstract
    "blur", "noise", "texture", "pattern", "abstract", "color", "monochrome",
    "warm tones", "cool tones", "muted", "vibrant",
    # emotions
    "joy", "sadness", "anger", "surprise", "fear", "love", "loneliness",
    "longing", "hope", "despair",
    # composition
    "close-up", "wide shot", "centered", "off-center", "symmetrical", "asymmetrical",
]


def load_manifest():
    rows = []
    with (CORPUS_DIR / "manifest.csv").open() as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def main(top_label_k: int = 3, hole_quantile: float = 0.10):
    print(f"[atlas-index] loading manifest from {CORPUS_DIR}")
    rows = load_manifest()
    if not rows:
        raise RuntimeError("Empty manifest. Run build_atlas_corpus.py first.")

    image_paths = [Path(r["path"]) for r in rows]
    prompts = [r["prompt"] for r in rows]
    seeds = [int(r["seed"]) for r in rows]

    # CLIP image embeddings
    from core import atlas_clip
    print(f"[atlas-index] CLIP-encoding {len(image_paths)} images …")
    img_emb = atlas_clip.embed_images(image_paths)            # (N, D), L2-normalized
    print(f"[atlas-index] image embeddings: {img_emb.shape}")

    # UMAP 2D
    import umap
    print("[atlas-index] UMAP …")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    coords = reducer.fit_transform(img_emb).astype(np.float32)  # (N, 2)

    # HDBSCAN clusters
    import hdbscan
    print("[atlas-index] HDBSCAN …")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=4, metric="euclidean")
    labels = clusterer.fit_predict(coords)                    # (N,) -1 = noise
    n_clusters = int(labels.max() + 1) if (labels >= 0).any() else 0
    print(f"[atlas-index] {n_clusters} clusters, {(labels==-1).sum()} noise points")

    # Per-cluster centroid in CLIP-space  + auto-label
    print(f"[atlas-index] CLIP-encoding vocabulary ({len(CONCEPT_VOCAB)} concepts) …")
    vocab_emb = atlas_clip.embed_texts(CONCEPT_VOCAB)         # (V, D)
    cluster_labels = []
    for c in range(n_clusters):
        mask = labels == c
        if not mask.any():
            cluster_labels.append([])
            continue
        centroid = img_emb[mask].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        sims = vocab_emb @ centroid                            # (V,)
        top = np.argsort(-sims)[:top_label_k]
        cluster_labels.append([CONCEPT_VOCAB[i] for i in top])

    # 2D KDE density grid (for heatmap + hole detection)
    from scipy.stats import gaussian_kde
    xy = coords.T
    kde = gaussian_kde(xy)
    grid_n = 80
    x_min, x_max = coords[:, 0].min() - 1, coords[:, 0].max() + 1
    y_min, y_max = coords[:, 1].min() - 1, coords[:, 1].max() + 1
    gx = np.linspace(x_min, x_max, grid_n)
    gy = np.linspace(y_min, y_max, grid_n)
    GX, GY = np.meshgrid(gx, gy)
    density = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape).astype(np.float32)

    # Detect "holes": low-density grid points within the convex hull of samples.
    # We label them by mapping each hole's grid coord back into CLIP-image-space:
    #   For each hole point, take K nearest sample points, average their CLIP
    #   embeddings, query vocab → top concept the hole is "near but missing".
    from scipy.spatial import cKDTree, ConvexHull
    hull = ConvexHull(coords)
    hull_path = coords[hull.vertices]
    # quick point-in-hull test via Delaunay
    from scipy.spatial import Delaunay
    in_hull = Delaunay(hull_path).find_simplex(np.column_stack([GX.ravel(), GY.ravel()])) >= 0
    flat_density = density.ravel()
    threshold = np.quantile(flat_density[in_hull], hole_quantile)
    hole_mask = (flat_density <= threshold) & in_hull
    hole_xy = np.column_stack([GX.ravel(), GY.ravel()])[hole_mask]
    print(f"[atlas-index] {len(hole_xy)} hole grid points (q={hole_quantile})")

    # cluster the holes themselves to avoid 200 labels — group nearby holes
    if len(hole_xy) > 0:
        hole_clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric="euclidean")
        hole_groups = hole_clusterer.fit_predict(hole_xy)
        n_hole_groups = int(hole_groups.max() + 1) if (hole_groups >= 0).any() else 0
    else:
        hole_groups = np.array([], dtype=int)
        n_hole_groups = 0

    print(f"[atlas-index] {n_hole_groups} hole regions")
    sample_tree = cKDTree(coords)
    hole_region_centers = []
    hole_region_labels = []
    for hg in range(n_hole_groups):
        m = hole_groups == hg
        if not m.any():
            continue
        center = hole_xy[m].mean(axis=0)
        # average CLIP-image embedding of K nearest samples to that hole center
        K = 8
        _, idxs = sample_tree.query(center, k=K)
        avg = img_emb[idxs].mean(axis=0)
        avg /= np.linalg.norm(avg) + 1e-8
        sims = vocab_emb @ avg
        top = np.argsort(-sims)[:top_label_k]
        hole_region_centers.append(center)
        hole_region_labels.append([CONCEPT_VOCAB[i] for i in top])

    # save
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        INDEX_PATH,
        img_emb=img_emb,
        coords=coords,
        cluster_labels_int=labels,
        density=density,
        density_x=gx,
        density_y=gy,
        hole_centers=np.array(hole_region_centers, dtype=np.float32) if hole_region_centers else np.zeros((0, 2), dtype=np.float32),
    )
    META_PATH.write_text(json.dumps({
        "n_samples": len(rows),
        "n_clusters": n_clusters,
        "cluster_labels": cluster_labels,                     # list[list[str]]
        "n_hole_regions": n_hole_groups,
        "hole_labels": hole_region_labels,                    # list[list[str]]
        "vocab": CONCEPT_VOCAB,
        "prompts": prompts,
        "seeds": seeds,
        "image_paths": [str(p) for p in image_paths],
    }, indent=2))

    print(f"[atlas-index] saved {INDEX_PATH}")
    print(f"[atlas-index] saved {META_PATH}")
    if cluster_labels:
        print("[atlas-index] cluster labels:")
        for i, ls in enumerate(cluster_labels):
            print(f"  cluster {i}: {ls}")
    if hole_region_labels:
        print("[atlas-index] hole region labels (concepts Lossy under-renders):")
        for i, ls in enumerate(hole_region_labels):
            print(f"  hole {i}: {ls}")


if __name__ == "__main__":
    main()
