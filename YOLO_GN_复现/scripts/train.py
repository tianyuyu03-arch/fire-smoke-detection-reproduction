"""Train the project-local YOLO11s-GN+P4-BiFPN implementation from scratch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "code" / "ultralytics"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FORK))

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data_clahe_cs.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="3")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="yolo11s_gn_bifpn_scratch")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset config does not exist: {args.data}")
    model = YOLO(ROOT / "models" / "yolo11s_gn_bifpn.yaml", task="detect")
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        pretrained=False,
        amp=False,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3.0,
        seed=args.seed,
        deterministic=True,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
