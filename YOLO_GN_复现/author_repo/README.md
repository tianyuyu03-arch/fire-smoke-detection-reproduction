# 🔥 YOLO-GN: Fire and Smoke Detection with YOLOv11 + Ghost Modules + Enhancement 
# YOLO-Based Fire and Smoke Detection with Ghost Convolutions, Bidirectional Feature Pyramid
# Networks, and Image Enhancement

Christine Dewi, Dhananjay Thiruvady , Stephen Abednego Philemon , Nayyar Zaidi

This repository contains the official implementation of **YOLO-GN**, an enhanced YOLOv11-based model for robust fire and smoke detection in low-contrast environments.  
The method integrates **Ghost modules** and **image enhancement techniques** (CLAHE, Contrast Stretching, and DGC) to improve accuracy while maintaining lightweight efficiency.

---

## 📌 Features
- YOLOv11 backbone with **Ghost modules** for lightweight performance.
- **BiFPN**-based neck for multi-scale feature fusion.
- Integrated preprocessing pipeline: CLAHE, CS, and DGC.
- Pre-trained weights and dataset splits provided for reproducibility.
- Deployment guide with latency and quantization notes.

---

## LINK DATASET 

https://drive.google.com/drive/folders/1IrI2esm9fN2di8T9bnxS1gDZjyj6Z73B?usp=sharing 

<img width="1436" height="832" alt="Picture1" src="https://github.com/user-attachments/assets/36e34cd4-e91b-4e96-9b33-d0a9f2b06b43" />

<img width="1018" height="628" alt="image" src="https://github.com/user-attachments/assets/b78e8dd6-48fc-4fd1-928e-c0d249658bd8" />

<img width="1002" height="485" alt="image" src="https://github.com/user-attachments/assets/220557bf-8107-415f-bc37-c45a464abc93" />

## ⚙️ Installation

Clone this repository:
```bash
git clone https://github.com/ChristineDewi/YOLO_GN.git
cd YOLO_GN

conda create -n yologn python=3.10 -y
conda activate yologn
pip install -r requirements.txt
```
## 🚀 Training

Run YOLO-GN with:
```bash
python train.py --data configs/fire_smoke.yaml --img 640 --epochs 200 --batch 64
```

## Run evaluation:
```bash
python val.py --weights runs/train/exp/weights/best.pt --data configs/fire_smoke.yaml --img 640
```
## 🔬 Reproducibility

All scripts, configs, and dataset splits are released for full reproducibility.

Pre-trained weights are available in the link: https://drive.google.com/drive/folders/1MbsORBhgPD0UUepl1kToKxvTdG1T1JEz?usp=sharing 

## 📊 Results

<img width="997" height="252" alt="image" src="https://github.com/user-attachments/assets/5a3454b4-11eb-45f6-87bd-6adca8372c94" />

<img width="522" height="367" alt="image" src="https://github.com/user-attachments/assets/b719317d-3c1c-4eb9-aaad-9a491b0996fe" />

## 📜 Citation
```bash
If you use this work, please cite:

@article{YOLOGN2025,
  title   = {YOLO-GN: Lightweight Fire and Smoke Detection using YOLOv11 with Ghost Modules and Enhancement},
  author  = {Christine Dewi and ...},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2025}
}
```

## 🙌 Acknowledgements

Ultralytics YOLOv11 framework.

Open-source fire/smoke datasets (AIDER, D-Fire, Roboflow, etc.).


