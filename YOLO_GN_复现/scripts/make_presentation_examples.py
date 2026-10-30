"""Create a few annotated test-set predictions for the group-meeting deck."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code" / "ultralytics"))

from ultralytics import YOLO  # noqa: E402


def main() -> None:
    test_root = ROOT / "dataset_clahe_cs" / "test"
    labels = sorted((test_root / "labels").glob("*.txt"))
    wanted = {"smoke_and_fire": {0, 1}, "smoke": {0}, "fire": {1}}
    selected: dict[str, Path] = {}
    for label in labels:
        classes = {
            int(line.split()[0])
            for line in label.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        for name, target in wanted.items():
            if name not in selected and target.issubset(classes):
                if name == "smoke_and_fire" or classes == target:
                    selected[name] = test_root / "images" / f"{label.stem}.jpg"
        if len(selected) == len(wanted):
            break
    if len(selected) != len(wanted):
        raise RuntimeError(f"Could not find all example classes: {selected}")

    weights = ROOT / "runs" / "yolo11s_gn_bifpn_clahe_cs_scratch_b40_300e" / "weights" / "best.pt"
    output = ROOT / "presentation" / "assets" / "predictions"
    model = YOLO(weights, task="detect")
    model.predict(
        source=[str(path) for path in selected.values()],
        imgsz=640,
        device=3,
        conf=0.25,
        save=True,
        project=str(output),
        name="test_examples",
        exist_ok=True,
        show_labels=True,
        show_conf=True,
    )
    print("Selected test images:")
    for name, path in selected.items():
        print(f"{name}: {path}")
    print(f"Annotated results: {output / 'test_examples'}")


if __name__ == "__main__":
    main()
