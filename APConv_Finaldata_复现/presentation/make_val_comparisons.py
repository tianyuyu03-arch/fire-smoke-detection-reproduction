"""Run the reproduced APConv model on validation images and render large GT/prediction boards."""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs/server_scratch_200ep_batch40/weights/best.pt"
VALID_IMAGES = ROOT / "dataset/finaldata/valid/images"
VALID_LABELS = ROOT / "dataset/finaldata/valid/labels"
OUTPUT = Path(__file__).resolve().parent / "images/reproduction"
CLASS_NAMES = {0: "Fire", 1: "Smoke"}
GT_COLOR = (20, 170, 245)
PRED_COLOR = (238, 72, 54)
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def read_labels(path: Path):
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c, x, y, w, h = line.split()[:5]
        boxes.append((int(float(c)), float(x), float(y), float(w), float(h)))
    return boxes


def image_for(stem: str) -> Path | None:
    return next((VALID_IMAGES / f"{stem}{ext}" for ext in EXTS if (VALID_IMAGES / f"{stem}{ext}").is_file()), None)


def candidates_by_group():
    groups = {"fire": [], "smoke": [], "mixed": [], "background": []}
    for label in sorted(VALID_LABELS.glob("*.txt")):
        boxes = read_labels(label)
        classes = {x[0] for x in boxes}
        group = "background" if not classes else ("mixed" if classes == {0, 1} else ("fire" if classes == {0} else "smoke"))
        image = image_for(label.stem)
        if image:
            groups[group].append({"group": group, "image": image, "labels": boxes,
                                  "min_area": min((x[3] * x[4] for x in boxes), default=0.0)})
    chosen = []
    rng = random.Random(20261024)
    for group, items in groups.items():
        if group == "background":
            rng.shuffle(items)
        else:
            items.sort(key=lambda x: x["min_area"])
        count = min(18, len(items))
        indices = sorted({round(i * (len(items) - 1) / max(1, count - 1)) for i in range(count)})
        chosen.extend(items[i] for i in indices)
    return chosen


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ab = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-9)


def evaluate(item, result):
    with Image.open(item["image"]) as im:
        width, height = im.size
    gt = []
    for c, x, y, w, h in item["labels"]:
        gt.append((c, ((x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height)))
    pred = []
    if result.boxes is not None:
        for c, conf, box in zip(result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.xyxy.cpu().tolist()):
            pred.append((int(c), float(conf), tuple(float(x) for x in box)))
    pairs = []
    for gi, (gc, gb) in enumerate(gt):
        for pi, (pc, conf, pb) in enumerate(pred):
            if gc == pc:
                overlap = iou(gb, pb)
                if overlap >= 0.5:
                    pairs.append((overlap, gi, pi, conf))
    matched_gt, matched_pred, matches = set(), set(), []
    for overlap, gi, pi, conf in sorted(pairs, reverse=True):
        if gi not in matched_gt and pi not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(pi)
            matches.append({"class": CLASS_NAMES[gt[gi][0]], "iou": round(overlap, 3), "confidence": round(conf, 3)})
    item.update({"gt": gt, "pred": pred, "matched": len(matches), "missed": len(gt) - len(matches),
                 "unmatched_pred": len(pred) - len(matches), "matches": matches})
    return item


def choose(items):
    output = []
    for group in ("fire", "smoke", "mixed"):
        pool = [x for x in items if x["group"] == group]
        if not pool:
            continue
        pool.sort(key=lambda x: (x["matched"] / max(1, len(x["gt"])), -x["unmatched_pred"]), reverse=True)
        output.extend([pool[0]])
        difficult = min(pool[1:], key=lambda x: (x["matched"] / max(1, len(x["gt"])), -x["unmatched_pred"])) if len(pool) > 1 else None
        if difficult:
            output.append(difficult)
    background = [x for x in items if x["group"] == "background"]
    if background:
        background.sort(key=lambda x: len(x["pred"]))
        output.extend([background[0]])
        if len(background) > 1:
            output.append(background[-1])
    return output


def font(size):
    for path in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def paint(source: Image.Image, boxes, title: str, color, size=(700, 480)):
    width, height = size
    header = 48
    out = Image.new("RGB", (width, header + height), "white")
    d = ImageDraw.Draw(out)
    d.text((16, 8), title, fill=(35, 45, 60), font=font(25))
    scale = min(width / source.width, height / source.height)
    rw, rh = round(source.width * scale), round(source.height * scale)
    shown = source.resize((rw, rh), Image.Resampling.LANCZOS)
    ox, oy = (width - rw) // 2, header + (height - rh) // 2
    out.paste(shown, (ox, oy))
    d = ImageDraw.Draw(out)
    for cls, box, conf in boxes:
        x1, y1, x2, y2 = box
        x1, x2 = ox + x1 * scale, ox + x2 * scale
        y1, y2 = oy + y1 * scale, oy + y2 * scale
        d.rectangle((x1, y1, x2, y2), outline=color, width=4)
        label = CLASS_NAMES[cls] + (f" {conf:.2f}" if conf is not None else "")
        b = d.textbbox((0, 0), label, font=font(19))
        tw, th = b[2] - b[0], b[3] - b[1]
        ty = max(header, int(y1) - th - 8)
        d.rectangle((int(x1), ty, int(x1) + tw + 10, ty + th + 7), fill=color)
        d.text((int(x1) + 5, ty + 2), label, fill="white", font=font(19))
    return out


def render(item, index):
    with Image.open(item["image"]) as im:
        source = im.convert("RGB")
    gt_boxes = [(c, box, None) for c, box in item["gt"]]
    pred_boxes = [(c, box, conf) for c, conf, box in item["pred"]]
    panels = [paint(source, gt_boxes, "Ground truth", GT_COLOR), paint(source, pred_boxes, "Model prediction", PRED_COLOR)]
    gap = 18
    footer = 72
    board = Image.new("RGB", (panels[0].width * 2 + gap * 3, panels[0].height + footer), (242, 245, 249))
    board.paste(panels[0], (gap, gap))
    board.paste(panels[1], (panels[0].width + gap * 2, gap))
    d = ImageDraw.Draw(board)
    note = f"GT {len(item['gt'])} | Pred {len(item['pred'])} | matched {item['matched']} | missed {item['missed']} | unmatched pred {item['unmatched_pred']}"
    d.text((gap + 6, board.height - 57), note, fill=(45, 55, 70), font=font(24))
    name = f"case_{index:02d}_{item['group']}.jpg"
    board.save(OUTPUT / name, quality=95, subsampling=0)
    return name


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidates = candidates_by_group()
    model = YOLO(str(WEIGHTS), task="detect")
    results = model.predict([str(x["image"]) for x in candidates], imgsz=640, conf=0.25,
                            device=0, batch=4, verbose=False, stream=True)
    evaluated = [evaluate(item, result) for item, result in zip(candidates, results)]
    selected = choose(evaluated)
    records = []
    for index, item in enumerate(selected, 1):
        filename = render(item, index)
        records.append({"file": filename, "image": item["image"].name, "group": item["group"],
                        "gt_count": len(item["gt"]), "pred_count": len(item["pred"]),
                        "matched": item["matched"], "missed": item["missed"],
                        "unmatched_pred": item["unmatched_pred"], "matches": item["matches"]})
    (OUTPUT / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
