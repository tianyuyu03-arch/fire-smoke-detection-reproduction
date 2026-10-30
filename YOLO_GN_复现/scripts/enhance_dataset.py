"""Create CLAHE + contrast-stretched image copies and a matching YOLO data YAML."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


def enhance_image(image: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Apply per-channel min-max contrast stretching followed by CLAHE on luminance.

    Input and output use OpenCV's BGR uint8 layout. The paper reports the CLAHE+CS
    combination but its public repository does not specify all preprocessing
    parameters; these defaults are recorded as this reproduction's choices.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a three-channel uint8 BGR image")

    source = image.astype(np.float32)
    lows = source.min(axis=(0, 1), keepdims=True)
    highs = source.max(axis=(0, 1), keepdims=True)
    span = highs - lows
    stretched = np.where(span > 0, (source - lows) * (255.0 / np.maximum(span, 1.0)), source)
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(stretched, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def prepare_dataset(source_root: Path, output_root: Path, clip_limit: float, tile_grid: int) -> Path:
    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    total = 0
    for split in ("train", "valid", "test"):
        image_dir = source_root / split / "images"
        label_dir = source_root / split / "labels"
        output_images = output_root / split / "images"
        output_labels = output_root / split / "labels"
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in supported):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            enhanced = enhance_image(image, clip_limit=clip_limit, tile_grid=tile_grid)
            if not cv2.imwrite(str(output_images / image_path.name), enhanced):
                raise OSError(f"Could not write image: {output_images / image_path.name}")
            total += 1
        for label_path in label_dir.glob("*.txt"):
            shutil.copy2(label_path, output_labels / label_path.name)

    config = {
        "path": str(output_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 2,
        "names": {0: "Smoke", 1: "Fire"},
    }
    config_path = output_root.parent / "data_clahe_cs.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Enhanced {total} images; wrote {config_path}")
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "dataset")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "dataset_clahe_cs")
    parser.add_argument("--clip-limit", type=float, default=2.0)
    parser.add_argument("--tile-grid", type=int, default=8)
    args = parser.parse_args()
    prepare_dataset(args.source, args.output, args.clip_limit, args.tile_grid)


if __name__ == "__main__":
    main()
