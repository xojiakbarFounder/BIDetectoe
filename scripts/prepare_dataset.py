#!/usr/bin/env python
"""Prepare, split, and validate YOLO-format datasets."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def _label_for(image: Path, image_root: Path, label_root: Path) -> Path:
    relative = image.relative_to(image_root).with_suffix(".txt")
    return label_root / relative


def _copy_pair(image: Path, label: Path, image_dst: Path, label_dst: Path) -> None:
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, image_dst)
    if label.exists():
        shutil.copy2(label, label_dst)


def _class_counts(label_files: list[Path]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for label_file in label_files:
        if not label_file.exists():
            continue
        for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if parts and parts[0].isdigit():
                counts[int(parts[0])] += 1
    return counts


def write_dataset_yaml(target: Path, class_names: list[str]) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    content = (
        "# YOLOv8 dataset config\n"
        f"path: {target.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        f"{names}\n"
    )
    (target / "dataset.yaml").write_text(content, encoding="utf-8")


def split_dataset(
    source_images: Path,
    source_labels: Path,
    target: Path,
    val_ratio: float,
    seed: int,
    copy: bool,
) -> None:
    images = _iter_images(source_images)
    random.Random(seed).shuffle(images)
    val_count = max(1, int(len(images) * val_ratio)) if images else 0
    val_set = set(images[:val_count])

    for image in images:
        split = "val" if image in val_set else "train"
        label = _label_for(image, source_images, source_labels)
        rel_image = image.relative_to(source_images)
        image_dst = target / "images" / split / rel_image
        label_dst = target / "labels" / split / rel_image.with_suffix(".txt")
        _copy_pair(image, label, image_dst, label_dst)
        if not copy:
            image.unlink()
            if label.exists():
                label.unlink()


def import_roboflow(roboflow_dir: Path, target: Path) -> None:
    mapping = {"train": "train", "valid": "val", "val": "val"}
    for source_split, target_split in mapping.items():
        image_root = roboflow_dir / source_split / "images"
        label_root = roboflow_dir / source_split / "labels"
        for image in _iter_images(image_root):
            label = _label_for(image, image_root, label_root)
            rel_image = image.relative_to(image_root)
            _copy_pair(
                image,
                label,
                target / "images" / target_split / rel_image,
                target / "labels" / target_split / rel_image.with_suffix(".txt"),
            )


def validate_dataset(target: Path) -> bool:
    ok = True
    all_labels: list[Path] = []

    for split in ("train", "val"):
        image_root = target / "images" / split
        label_root = target / "labels" / split
        images = _iter_images(image_root)
        missing = []
        empty = []

        for image in images:
            label = _label_for(image, image_root, label_root)
            all_labels.append(label)
            if not label.exists():
                missing.append(image)
                ok = False
            elif not label.read_text(encoding="utf-8", errors="ignore").strip():
                empty.append(label)

        print(f"{split}: {len(images)} images, {len(images) - len(missing)} labels")
        if missing:
            print(f"  missing labels: {len(missing)}")
            for item in missing[:10]:
                print(f"    - {item}")
        if empty:
            print(f"  empty labels: {len(empty)}")

    counts = _class_counts(all_labels)
    print("class instances:")
    if counts:
        for class_id, count in sorted(counts.items()):
            print(f"  class {class_id}: {count}")
    else:
        print("  none")

    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a YOLOv8 dataset.")
    parser.add_argument("--target", default="dataset", help="Output dataset directory.")
    parser.add_argument("--images", help="Source images directory to split.")
    parser.add_argument("--labels", help="Source YOLO labels directory to split.")
    parser.add_argument("--roboflow-dir", help="Roboflow YOLOv8 export directory.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--move", action="store_true", help="Move instead of copy.")
    parser.add_argument("--classes", default="person", help="Comma-separated class names.")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = (_project_root() / args.target).resolve()
    target.mkdir(parents=True, exist_ok=True)

    if not args.validate_only:
        if args.roboflow_dir:
            import_roboflow(Path(args.roboflow_dir).resolve(), target)
        elif args.images and args.labels:
            split_dataset(
                Path(args.images).resolve(),
                Path(args.labels).resolve(),
                target,
                args.val_ratio,
                args.seed,
                copy=not args.move,
            )
        else:
            print("No source provided. Validating existing dataset only.")

        class_names = [name.strip() for name in args.classes.split(",") if name.strip()]
        write_dataset_yaml(target, class_names or ["person"])

    return 0 if validate_dataset(target) else 1


if __name__ == "__main__":
    sys.exit(main())

