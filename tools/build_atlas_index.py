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
    # people / faces
    "face", "portrait", "selfie", "person", "woman", "man", "child", "baby",
    "teenager", "elderly", "young", "old", "head", "eyes", "smile", "frown",
    "profile", "hair", "skin", "lips", "beard", "mustache", "freckles",
    "wrinkles", "scar", "tattoo", "glasses", "sunglasses", "mask", "veil",
    "scarf", "hat", "cap", "hood", "necklace", "earrings",
    # body / pose
    "hand", "hands", "arm", "shoulder", "torso", "legs", "feet", "back",
    "standing", "sitting", "lying down", "running", "walking", "dancing",
    "jumping", "kneeling", "praying", "fighting", "embracing", "waving",
    # animals
    "bird", "dove", "eagle", "owl", "crow", "fish", "cat", "dog", "horse",
    "wolf", "deer", "bear", "rabbit", "fox", "mouse", "snake", "butterfly",
    "bee", "spider", "insect", "whale", "dolphin", "shark",
    # objects
    "drone", "plane", "helicopter", "rocket", "satellite", "boat", "ship",
    "car", "truck", "bicycle", "motorcycle", "train", "bus",
    "phone", "computer", "screen", "camera", "microphone", "speaker",
    "book", "newspaper", "letter", "map", "globe", "clock", "watch",
    "suitcase", "bag", "wallet", "key", "lock", "knife", "gun", "tool",
    "cup", "mug", "bottle", "glass", "plate", "spoon", "fork",
    "table", "chair", "bed", "couch", "shelf", "desk", "lamp", "vase",
    "flag", "statue", "monument", "cross", "religious icon",
    # food / nature small
    "bread", "fruit", "apple", "orange", "vegetable", "leaf", "flower",
    "rose", "tulip", "grass", "moss", "fungus", "mushroom",
    # nature / landscape
    "forest", "jungle", "tree", "branch", "trunk", "roots", "pine", "oak",
    "mountain", "hill", "cliff", "valley", "canyon", "cave",
    "river", "stream", "waterfall", "lake", "pond", "ocean", "sea",
    "wave", "beach", "sand", "dune", "desert", "oasis",
    "field", "meadow", "wildflower", "wheat", "grass field",
    "snow", "ice", "glacier", "iceberg", "winter", "summer", "spring", "autumn",
    "dawn", "dusk", "sunrise", "sunset", "midnight",
    "sky", "cloud", "rainbow", "fog", "mist", "haze", "storm", "thunder",
    "lightning", "rain", "snowfall", "hail", "tornado",
    "moon", "full moon", "crescent moon", "sun", "star", "galaxy", "nebula",
    "aurora", "solar eclipse",
    # urban / built
    "city", "skyline", "street", "alley", "road", "highway", "bridge",
    "tunnel", "building", "skyscraper", "tower", "house", "cabin",
    "ruin", "factory", "office", "hallway", "corridor", "stairs",
    "town", "village", "market", "courtyard", "garden", "park",
    "window", "door", "wall", "fence", "gate", "stained glass",
    "neon", "billboard", "graffiti", "asphalt", "cobblestone", "brick",
    # interior
    "bedroom", "kitchen", "bathroom", "library", "studio", "warehouse",
    "church", "temple", "mosque", "synagogue", "cathedral",
    # light / mood
    "candlelight", "firelight", "sunlight", "moonlight", "starlight",
    "lamp light", "neon light", "torchlight", "headlight",
    "shadow", "silhouette", "darkness", "twilight", "golden hour",
    "blue hour", "soft light", "harsh light", "backlit", "rim light",
    "diffused light", "direct light", "spotlight",
    "dramatic", "serene", "calm", "tense", "ominous", "joyful", "melancholy",
    "morning", "midday", "evening", "night",
    # texture / abstract
    "blur", "motion blur", "out of focus", "noise", "grain", "static",
    "texture", "pattern", "geometry", "abstract", "color field",
    "color", "monochrome", "sepia", "grayscale", "high contrast",
    "warm tones", "cool tones", "muted", "vibrant", "saturated",
    "smoke", "steam", "dust", "particles", "bokeh", "lens flare",
    "reflection", "refraction", "translucent", "opaque", "glow",
    # emotions / atmosphere
    "joy", "sadness", "anger", "surprise", "fear", "love", "loneliness",
    "longing", "hope", "despair", "nostalgia", "wonder", "isolation",
    "intimacy", "tension", "freedom",
    # composition
    "close-up", "extreme close-up", "wide shot", "aerial view", "low angle",
    "high angle", "centered", "off-center", "symmetrical", "asymmetrical",
    "framed", "rule of thirds", "minimal", "cluttered",
    # styles / mediums (just visual cues)
    "painting", "sketch", "illustration", "documentary", "snapshot",
    "studio photo", "vintage photo", "polaroid", "x-ray", "infrared",
    "thermal", "schematic", "diagram",
    # weather / scenarios
    "war scene", "demonstration", "crowd", "stadium", "concert", "wedding",
    "funeral", "festival", "interview", "newscast",
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

    # UMAP 3D
    import umap
    print("[atlas-index] UMAP (3D) …")
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                        metric="cosine", random_state=42)
    coords = reducer.fit_transform(img_emb).astype(np.float32)  # (N, 3)

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

    # 3D KDE density grid + per-sample density score (used for marker sizing)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(coords.T)
    sample_density = kde(coords.T).astype(np.float32)
    grid_n = 25
    mins = coords.min(axis=0) - 1
    maxs = coords.max(axis=0) + 1
    gx = np.linspace(mins[0], maxs[0], grid_n)
    gy = np.linspace(mins[1], maxs[1], grid_n)
    gz = np.linspace(mins[2], maxs[2], grid_n)
    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
    grid_pts = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])
    density = kde(grid_pts.T).reshape(GX.shape).astype(np.float32)

    # Detect "holes": low-density grid voxels within the convex hull
    from scipy.spatial import cKDTree, ConvexHull, Delaunay
    hull = ConvexHull(coords)
    in_hull = Delaunay(coords[hull.vertices]).find_simplex(grid_pts) >= 0
    flat_density = density.ravel()
    threshold = np.quantile(flat_density[in_hull], hole_quantile)
    hole_mask = (flat_density <= threshold) & in_hull
    hole_xy = grid_pts[hole_mask]
    print(f"[atlas-index] {len(hole_xy)} hole voxels (q={hole_quantile})")

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
        sample_density=sample_density,
        cluster_labels_int=labels,
        density=density,
        density_x=gx,
        density_y=gy,
        density_z=gz,
        hole_centers=np.array(hole_region_centers, dtype=np.float32) if hole_region_centers else np.zeros((0, 3), dtype=np.float32),
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
