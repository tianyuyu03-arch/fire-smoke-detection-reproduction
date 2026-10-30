# YOLO11s-GN + P4 BiFPN reproduction workspace

This workspace contains the public author notebooks, the fixed Fire and Smoke dataset split, and an independent implementation of the method components that can be verified from the paper abstract and the public repository.

## Server training

The project is deployed at `./LOCAL_PROJECT_PATH复现`. The server uses the existing personal environment at `./LOCAL_PROJECT_PATH复现/.venv`; no system packages are changed.

```bash
cd ./LOCAL_PROJECT_PATH复现
source ./LOCAL_PROJECT_PATH复现/.venv/bin/activate
python scripts/patch_ultralytics.py code/ultralytics
python scripts/enhance_dataset.py
PYTHONPATH="$PWD/code/ultralytics:$PWD" python scripts/train.py \
  --data data_clahe_cs.yaml --epochs 300 --batch 40 --imgsz 640 --device 3
```

The training script always sets `pretrained=False`. It uses GPU 3 by default; check current shared-server usage before starting a long run.
The enhancement script writes `dataset_clahe_cs/` and `data_clahe_cs.yaml` without modifying the original dataset. To evaluate a completed run on the untouched test split, use `python scripts/validate.py runs/<run-name>/weights/best.pt --data data_clahe_cs.yaml --device 3`.

## Architecture and reproduction limits

- `models/yolo11s_gn_bifpn.yaml` replaces selected stride-2 convolutions in the backbone and neck with GhostConv and inserts a weighted P3/P4/P5 fusion node at P4.
- `models/ghost_bifpn.py` projects the three pyramid levels with GhostConv, aligns them to P4 resolution, and learns normalized nonnegative fusion weights.
- `scripts/patch_ultralytics.py` applies the YAML parser extension to the project-local Ultralytics copy under `code/`; it does not modify the shared environment's installed source.
- The author's public repository does not include the custom source/configuration that the notebooks reference. This implementation follows the published high-level description, but the exact private layer placement and preprocessing parameters could not be verified. Results must be reported as an independent, from-scratch implementation rather than an exact reproduction of the author's training run.
- The official author's notebook loads `yolo11s.pt`; this workspace follows the user's preference to train from random initialization, so its training protocol differs from that notebook.

The one-epoch runs in `runs/` are pipeline checks only. They are not paper results.
