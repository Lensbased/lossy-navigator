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
    # --- faces / portraits (50)
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
    "a teenager looking at a phone", "an elderly woman in a red shawl",
    "a man with a long grey ponytail", "a woman with bleached short hair",
    "a baby asleep in a basket", "a couple holding each other close",
    "a face wearing a gas mask", "a face behind frosted glass",
    "a person with a tattoo on the neck", "a man yelling into a megaphone",
    "a woman speaking at a microphone", "a face under a hood, in shadow",
    "a clown laughing", "a worker with safety goggles",
    "a mountaineer with frostbitten cheeks", "a chef tasting food",
    "a doctor in surgical mask", "a soldier wiping sweat", "a child in a school uniform",
    "a face with running mascara",
    # --- bodies / pose (15)
    "a hand holding a flower", "two hands intertwined",
    "a runner mid-stride", "a dancer leaping",
    "a person crouching by a doorway", "a woman walking away in a long coat",
    "feet in muddy boots", "a man asleep on a couch",
    "two figures embracing in silhouette", "a praying figure in candlelight",
    "a back view of a hooded figure", "a person carrying a heavy bag",
    "a woman raising her arms",
    "a worker bent over a machine", "a child running through a field",
    # --- animals (15)
    "a single white dove flying", "an eagle perched on a branch",
    "a black crow on a fence", "a red fox in snow",
    "a stray cat on a roof", "a wet dog shaking water",
    "a horse galloping in dust", "a deer standing in mist",
    "a group of fish underwater", "a butterfly on a flower",
    "a hummingbird mid-flight", "a sleeping cat in sunlight",
    "an owl in moonlight", "a wolf howling in snow",
    "a goldfish in a glass bowl",
    # --- nature / landscape (40)
    "a quiet forest at dawn", "a dense forest in summer",
    "a winter forest under snow", "a moss-covered ancient tree",
    "a mossy stone in a stream", "a wide alpine valley",
    "a granite mountain peak", "a mirror-still alpine lake",
    "a meadow of wildflowers", "an empty desert at sunset",
    "a stormy ocean", "a calm beach at low tide",
    "a sand dune at sunset", "a coral reef underwater",
    "a frozen waterfall", "a bamboo forest", "a cherry tree in bloom",
    "wheat field under a clear sky", "tall pines in a fog bank",
    "a single tree on a hill", "a herd of sheep on a hillside",
    "an iceberg floating in a green sea", "a volcano erupting at night",
    "a bonfire on a beach", "a river delta from above",
    "a desert oasis with palm trees", "rolling hills with vineyards",
    "a snowy mountain pass", "a meandering river in a valley",
    "a forest clearing with sunbeams", "a cave entrance overgrown with vines",
    "a glacier under heavy clouds", "a salt flat at noon",
    "a tropical lagoon", "an arctic tundra at twilight",
    "a marsh with reflections of the sky",
    "a thunderstorm over the prairie", "low fog over a wet field at dawn",
    "the milky way over a desert", "northern lights over a frozen lake",
    # --- urban / architecture (35)
    "a city street at night", "an empty office hallway",
    "a wheat field next to a highway", "a cobblestone road in a small town",
    "an industrial harbor at dawn", "a lighthouse in a storm",
    "a brutalist concrete building", "a Gothic cathedral interior",
    "an underground subway platform", "a Tokyo-like neon alley",
    "a Berlin tenement courtyard", "a New York fire escape",
    "an abandoned warehouse interior", "a dusty library full of books",
    "a sunlit kitchen with wooden table", "a child's bedroom with toys",
    "a bathroom with steam on the mirror", "a graffiti-covered wall",
    "a market stall full of vegetables", "a rooftop bar at sunset",
    "a hospital corridor under fluorescent light",
    "a control room of a power plant", "a server room with blue LEDs",
    "a parking garage at night", "a basketball court at dusk",
    "a small fishing boat in harbor", "a freight train passing fields",
    "an airplane interior", "a refugee camp at dawn",
    "a barricade in a city street", "a peace demonstration with flags",
    "a stadium full of spectators", "a subway car interior at midnight",
    "a snow-covered village street", "a bombed building with smoke rising",
    # --- objects / things (25)
    "a small black quadcopter drone", "an old leather suitcase",
    "a cup of black coffee on a table", "a vintage rotary telephone",
    "a stack of yellowing books", "a polaroid on a windowsill",
    "an open notebook with handwriting", "a typewriter in low light",
    "a hand grenade on a wooden table", "a microscope on a lab bench",
    "a chess board mid-game", "a wedding ring on velvet",
    "an empty wine glass", "a flower in a glass vase",
    "a wooden mask hanging on a wall", "a Polaroid camera",
    "a globe of the world on a desk", "a stack of vinyl records",
    "a guitar leaning against a wall", "a violin on a chair",
    "a stethoscope on a bench", "a pile of seashells",
    "a worn leather boot", "a rotary clock on a station wall",
    "a spaceship console with many switches",
    # --- light / weather / atmosphere (25)
    "soft morning fog over water", "golden hour light through trees",
    "rain falling on a window", "smoke drifting in still air",
    "neon reflections on wet asphalt", "moonlight on a snowy field",
    "sunlight cutting across a dusty room", "clouds passing over mountains",
    "a single flame in darkness", "sparks rising from a fire",
    "a beam of light through a stained-glass window",
    "rain drops on a leaf", "lightning forking across a black sky",
    "snowflakes against a streetlight", "a rainbow over a wet road",
    "a desert mirage", "a halo around the moon",
    "auroras shimmering above a forest", "smoke from chimneys at sunrise",
    "fog rolling through a city street", "a swirl of dust in sunlight",
    "the silhouette of a tree at sunset", "the sun behind storm clouds",
    "stars reflected in a still lake", "a meteor streak across the sky",
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
