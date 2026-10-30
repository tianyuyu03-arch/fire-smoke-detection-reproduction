"""Evaluate a trained checkpoint on the untouched test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code" / "ultralytics"))

from ultralytics import YOLO  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "data_clahe_cs.yaml")
    parser.add_argument("--device", default="3")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    model = YOLO(args.checkpoint, task="detect")
    model.val(data=str(args.data), split="test", imgsz=640, batch=16, device=args.device)


if __name__ == "__main__":
    main()
