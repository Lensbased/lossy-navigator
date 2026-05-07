"""Generate a corpus of Lossy samples for the atlas.

Iterates a (caption × seed) grid, samples Lossy at each combination,
saves PNG + manifest CSV.

Usage:
    python tools/build_atlas_corpus.py [--n_seeds N] [--steps S]

Outputs:
    output/atlas/corpus/<id>.png   — sample images (256×256 thumbs)
    output/atlas/corpus/manifest.csv — id,prompt,seed,path

Defaults: 60 captions × 3 seeds = 180 samples.  ~30-45 min on M3 Max.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.lossy_runtime import LossyRuntime  # noqa: E402

# A varied corpus of captions across faces, scenes, objects, lighting moods.
CAPTIONS = [
    # faces
    "a young woman with dark hair", "an old man with a white beard",
    "a child smiling", "a man wearing sunglasses", "a face in profile",
    "a face looking up", "a face looking down", "a woman with curly hair",
    "a man with short hair", "a face in soft sunlight", "a face in deep shadow",
    "a face by candlelight", "a woman with closed eyes", "a face laughing",
    "a face crying", "a face surprised", "a face calm and serene",
    "a young face with freckles", "a face with high cheekbones",
    "a face in side light from the left", "a face in side light from the right",
    "a portrait of a musician", "a portrait of a worker", "a portrait of a dancer",
    "a portrait of a soldier", "a face whispering", "a face concentrating",
    "a face about to sleep", "a person with weathered skin",
    "a face partially obscured by a scarf",
    # scenes / landscape
    "a quiet forest at dawn", "a dense forest in summer", "a winter forest under snow",
    "a wide alpine valley", "a granite mountain peak", "a mirror-still alpine lake",
    "a meadow of wildflowers", "an empty desert at sunset", "a stormy ocean",
    "a calm beach at low tide", "a city street at night", "an empty office hallway",
    "a wheat field under a clear sky", "a cobblestone road in a small town",
    # objects / things
    "a single white dove flying", "a small black quadcopter drone",
    "an old leather suitcase", "a cup of black coffee on a table",
    "a vintage rotary telephone", "a stack of yellowing books",
    # abstract / lighting / atmosphere
    "soft morning fog over water", "golden hour light through trees",
    "rain falling on a window", "smoke drifting in still air",
    "neon reflections on wet asphalt", "moonlight on a snowy field",
    "sunlight cutting across a dusty room", "clouds passing over mountains",
    "a single flame in darkness", "sparks rising from a fire",
    "a beam of light through a stained-glass window",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--config", default="/Users/hitosteyerl/CC/ComfyUI/v8_facecrop.yaml")
    ap.add_argument("--checkpoint", default="/Users/hitosteyerl/CC/ComfyUI/models/lossy/tracka_mmd-minsnr-step_4700000.pt")
    ap.add_argument("--out_dir", default=str(ROOT / "output" / "atlas" / "corpus"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rt = LossyRuntime(config_path=args.config, checkpoint_path=args.checkpoint)

    seeds = [42 + 1000 * i for i in range(args.n_seeds)]
    total = len(CAPTIONS) * len(seeds)
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "prompt", "seed", "path"])
        idx = 0
        t0 = time.time()
        for cap in CAPTIONS:
            for seed in seeds:
                img = rt.generate(cap, seed=seed, num_steps=args.steps,
                                  guidance_scale=args.guidance)
                # save 256x256 thumb to keep disk small + atlas snappy
                img = img.resize((256, 256))
                p = out_dir / f"s{idx:04d}.png"
                img.save(p, optimize=True)
                w.writerow([idx, cap, seed, str(p)])
                f.flush()
                idx += 1
                elapsed = time.time() - t0
                eta = elapsed / idx * (total - idx)
                print(f"[atlas-corpus] {idx}/{total}  '{cap[:40]}'  seed={seed}  "
                      f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")

    print(f"[atlas-corpus] saved {idx} samples to {out_dir}")


if __name__ == "__main__":
    main()
