"""Local reproduction entry point; keeps the author's model and train settings."""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_YAML = ROOT / "code" / "Finaldata" / "C2f-AFF+ECN+LSADetect.yaml"
DATA_YAML = ROOT / "dataset" / "dataset.yaml"

if __name__ == "__main__":
    model = YOLO(str(MODEL_YAML))
    model.train(
        data=str(DATA_YAML), epochs=200, patience=50, batch=16, imgsz=640,
        device=0, workers=8, project=str(ROOT / "runs"), name="author_config_200ep",
        exist_ok=False, pretrained=False, optimizer="SGD", seed=0, deterministic=True,
        close_mosaic=0, amp=False, lr0=0.01, lrf=0.01, momentum=0.937,
        weight_decay=0.0005, warmup_epochs=3.0, box=7.5, cls=0.5, dfl=1.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1,
        scale=0.5, fliplr=0.5, mosaic=1.0, mixup=0.0, save=True,
        cache=False, verbose=True,
    )