# APConv / AEL-YOLO and Finaldata reproduction log

Date: 2026-10-22 (Asia/Shanghai)
Primary paper: Dong et al., “You only look once meets asymmetric pronged convolution: A unified framework for efficient fire and smoke detection,” EAAI 173 (2026), 114484. DOI: 10.1016/j.engappai.2026.114484.
Repository: https://github.com/123dsb-ux/Finaldata
Repository commit cloned: 5b3588189d3a7e6776b06492f8ba5c037c722e3c
Dataset link listed by the repository: Google Drive file `finaldata.zip` (about 1.9 GB).

## Local machine
- Windows, Python 3.11.9 available; no PyTorch in the current global Python.
- NVIDIA GeForce RTX 4060, 8 GB VRAM; desktop processes already use about 2.4 GB at inspection.
- D: drive free space at setup: about 248 GB.

## Initial repository inspection
- Contains the model YAML, modified `nn/tasks.py`, custom module files (`iAFF6.py`, `SAhead6.py`, `ECN.py`), `train.py`, and `val.py`.
- Does not contain a requirements file, dataset YAML, or the full `ultralytics` package.
- `train.py` expects a local `dataset.yaml`, sets `CUDA_LAUNCH_BLOCKING=1`, requests batch 16, 200 epochs, and seed 0.
- `val.py` loads `best.pt` and `dataset.yaml`; it requires path adaptation for this machine.
- These adaptation points will be tracked separately from the upstream commit.

## Current status
- Code repository cloned at the pinned commit above; custom modules integrated into local Ultralytics source.
- Dataset fully downloaded, extracted, and audited as recorded below.
- CUDA PyTorch installation is running. Training has not started; no reproduction metrics have been produced.

## 2026-10-22 update
- Dataset download completed: `dataset/finaldata.zip` (2,025,320,273 bytes); all 33,776 archive entries extracted without a reported CRC error.
- Exact split audit: train 14,244 images and labels; valid 2,641 images and labels; 1,747 train and 1,001 valid empty-label negatives. Image formats are JPG and PNG.
- YOLO box counts: train fire (ID 0) 25,868, smoke (ID 1) 12,985; valid fire 3,055, smoke 1,409. Class IDs were checked against example images.
- Added `dataset/dataset.yaml`, `run_train.py`, and `README_复现说明.md`.
- Integrated the repository's custom `nn/tasks.py`, `ECN.py`, `iAFF6.py`, and `SAhead6.py` into the local Ultralytics v8.3.40 checkout. Original upstream `tasks.py` is backed up in `notes/tasks.py.upstream-8.3.40`.
- CUDA PyTorch 2.5.1 + cu124 installation is still running. Model initialization, forward/backward smoke test, training, and validation remain pending. No reproduction metric has been claimed.
- Additional annotation audit found 13 degenerate boxes with zero width or height (in train split). The raw archive is kept unchanged; check whether the training loader filters these rows before deciding on any cleaned-label copy.
- User requested training from scratch. The local runner now sets `pretrained=False`; this intentionally differs from the author's public `train.py` (`pretrained=True`). No pretrained weights will be downloaded or loaded.
