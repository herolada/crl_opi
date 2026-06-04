"""
Merge multiple YOLO datasets into a single dataset under ../data/.

Output classes:
  0 - adr   (from: datasets/adr  classes 0+1, datasets/adr_2  class 0)
  1 - drone  (from: datasets/drone class 0)
  2 - camo   (from: datasets/camo  class 0)

Usage:
  python process_datasets.py [--resize 640] [--balance]
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
DATASETS = REPO_ROOT / "datasets"
OUT = REPO_ROOT / "data"

# Final class list
CLASSES = ["adr", "drone", "camo"]

# Each entry: (source_dir, split_map, class_remap)
# split_map: dict of output_split -> list of source sub-paths (images dir)
# class_remap: dict of source_class_id -> output_class_id, omitted ids are dropped
SOURCES = [
    {
        "name": "adr",
        "root": DATASETS / "adr",
        "splits": {
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
        },
        "class_remap": {0: 0, 1: 0},  # adr + adr_sign -> 0
    },
    {
        "name": "adr_2",
        "root": DATASETS / "adr_2",
        "splits": {
            "train": "images/train",
            "val": "images/val",
        },
        "class_remap": {0: 0},
    },
    {
        "name": "drone",
        "root": DATASETS / "drone",
        "splits": {
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
        },
        "class_remap": {0: 1},
    },
    {
        "name": "camo",
        "root": DATASETS / "camo",
        "splits": {
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
        },
        "class_remap": {0: 2},
    },
]


def obb_to_aabb(coords: list[float]) -> tuple[float, float, float, float]:
    """Convert OBB (4 corners x1y1x2y2x3y3x4y4) to AABB (cx cy w h)."""
    xs = coords[0::2]
    ys = coords[1::2]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return cx, cy, w, h


def find_label_file(images_dir: Path, image_file: Path) -> Path | None:
    """Find the corresponding label file for an image.

    Handles two layouts:
      - <root>/<split>/images/ + <root>/<split>/labels/   (adr, drone, camo)
      - <root>/images/<split>/ + <root>/labels/<split>/   (adr_2)
    """
    stem = image_file.with_suffix(".txt").name
    candidates = [
        images_dir.parent / "labels" / stem,                          # layout 1
        images_dir.parent.parent / "labels" / images_dir.name / stem, # layout 2
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def transform_label(label_path: Path, class_remap: dict[int, int]) -> list[str]:
    """Read a label file, remap/filter classes, convert OBB→AABB. Returns output lines."""
    out_lines = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        src_class = int(parts[0])
        if src_class not in class_remap:
            continue
        dst_class = class_remap[src_class]
        values = [float(v) for v in parts[1:]]
        if len(values) == 4:
            cx, cy, w, h = values
        elif len(values) == 8:
            # OBB: 4 corner points -> axis-aligned bbox
            cx, cy, w, h = obb_to_aabb(values)
        else:
            # Unexpected format; skip
            continue
        out_lines.append(f"{dst_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out_lines


def resize_image(src: Path, dst: Path, max_size: int) -> None:
    from PIL import Image

    img = Image.open(src)
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    img.save(dst)


def process_split(
    source: dict,
    split: str,
    images_subpath: str,
    resize: int | None,
    stats: dict,
    class_images: dict[int, list[Path]] | None = None,
) -> None:
    """Copy and transform one source/split pair.

    If class_images is provided, appends each copied output image path to the
    list for its output class (used later for balancing).
    """
    images_dir = source["root"] / images_subpath
    if not images_dir.exists():
        return

    out_images = OUT / split / "images"
    out_labels = OUT / split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    prefix = source["name"]
    class_remap = source["class_remap"]
    out_classes = set(class_remap.values())

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue

        # Unique output name: <source>_<original_stem><ext>
        out_stem = f"{prefix}_{img_path.stem}"
        out_img = out_images / (out_stem + img_path.suffix)

        label_path = find_label_file(images_dir, img_path)
        label_lines = transform_label(label_path, class_remap) if label_path else []

        # Skip images with no kept annotations
        if not label_lines:
            stats["skipped"] += 1
            continue

        if resize:
            resize_image(img_path, out_img, resize)
        else:
            shutil.copy2(img_path, out_img)

        (out_labels / (out_stem + ".txt")).write_text("\n".join(label_lines) + "\n")
        stats["copied"] += 1

        if class_images is not None:
            for cls in out_classes:
                class_images[cls].append(out_img)


def balance_split(split: str, class_images: dict[int, list[Path]]) -> dict[int, dict]:
    """Oversample underrepresented classes using the already-tracked image lists.

    Target for each class = max_class_count, capped at 2x the class's original
    count so no class ends up more than 2:1 behind the largest.
    """
    images_dir = OUT / split / "images"
    labels_dir = OUT / split / "labels"

    max_count = max(len(imgs) for imgs in class_images.values())

    balance_stats = {}
    for cls, imgs in sorted(class_images.items()):
        original = len(imgs)
        target = original * (max_count//original) #min(max_count, original * 2)
        needed = target - original
        balance_stats[cls] = {"original": original, "target": target, "added": needed}
        if needed <= 0:
            continue

        extras = random.choices(imgs, k=needed)
        for i, src_img in enumerate(extras):
            new_stem = f"{src_img.stem}_bal{i:05d}"
            shutil.copy2(src_img, images_dir / (new_stem + src_img.suffix))
            shutil.copy2(labels_dir / (src_img.stem + ".txt"), labels_dir / (new_stem + ".txt"))

    return balance_stats


def write_yaml() -> None:
    cfg = {
        "path": str(OUT),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    (OUT / "data.yaml").write_text(yaml.dump(cfg, default_flow_style=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        metavar="MAX_SIZE",
        help="Resize images so the longest side is at most MAX_SIZE px (e.g. 640). "
             "Omit to keep original size (recommended — let YOLO handle resizing at train time).",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Oversample underrepresented classes in the train split. Each class is "
             "duplicated up to min(max_class_count, 2x_original) images.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for balancing.")
    args = parser.parse_args()
    random.seed(args.seed)

    if OUT.exists():
        print(f"Removing existing {OUT}")
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    train_class_images: dict[int, list[Path]] = defaultdict(list)

    total_stats: dict[str, dict] = {}
    for source in SOURCES:
        stats = {"copied": 0, "skipped": 0}
        for split, subpath in source["splits"].items():
            ci = train_class_images if split == "train" else None
            process_split(source, split, subpath, args.resize, stats, ci)
        total_stats[source["name"]] = stats
        print(f"  {source['name']:8s}  copied={stats['copied']}  skipped(no_labels)={stats['skipped']}")

    if args.balance:
        print("\nBalancing train split...")
        bal = balance_split("train", train_class_images)
        for cls, s in sorted(bal.items()):
            ratio = s["target"] / s["original"] if s["original"] else 0
            print(f"  class {cls} ({CLASSES[cls]:6s})  {s['original']:>6} -> {s['target']:>6}  (+{s['added']:>6}, {ratio:.2f}x)")

    write_yaml()

    print()
    for split in ("train", "val", "test"):
        n = len(list((OUT / split / "images").glob("*"))) if (OUT / split / "images").exists() else 0
        print(f"  {split:6s}: {n} images")

    print(f"\nDataset written to {OUT}")
    print(f"Classes: {CLASSES}")


if __name__ == "__main__":
    main()
