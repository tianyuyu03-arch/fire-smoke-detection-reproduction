"""Run the reproduced YOLO-GN model on held-out test images and make GT/prediction boards."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code" / "ultralytics"))
from ultralytics import YOLO  # noqa: E402

WEIGHTS = ROOT / "runs/yolo11s_gn_bifpn_clahe_cs_scratch_b40_300e/weights/best.pt"
TEST = ROOT / "dataset_clahe_cs/test"
RAW_IMAGES = ROOT / "dataset/test/images"
OUTPUT = ROOT / "presentation/assets/comparisons"
CLASS_NAMES = {0: "Smoke", 1: "Fire"}
COLORS = {0: (35, 185, 255), 1: (185, 90, 255)}
PRED_COLOR = (255, 92, 58)


def read_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for row in path.read_text(encoding="utf-8").splitlines():
        if not row.strip():
            continue
        c, x, y, w, h = row.split()[:5]
        boxes.append((int(float(c)), float(x), float(y), float(w), float(h)))
    return boxes


def stratified_candidates() -> list[dict]:
    groups: dict[str, list[dict]] = {"smoke": [], "fire": [], "mixed": []}
    for label_path in sorted((TEST / "labels").glob("*.txt")):
        labels = read_label(label_path)
        if not labels:
            continue
        classes = {x[0] for x in labels}
        group = "mixed" if classes == {0, 1} else ("smoke" if classes == {0} else "fire")
        image = TEST / "images" / f"{label_path.stem}.jpg"
        raw = RAW_IMAGES / image.name
        if image.is_file() and raw.is_file():
            groups[group].append({"group": group, "name": image.name, "image": image, "raw": raw, "labels": labels,
                                  "min_area": min(x[3] * x[4] for x in labels)})

    selected = []
    for group, items in groups.items():
        items.sort(key=lambda x: x["min_area"])
        count = min(16, len(items))
        # Evenly sample the full range of object sizes, including difficult small targets.
        indices = sorted({round(i * (len(items) - 1) / max(1, count - 1)) for i in range(count)})
        selected.extend(items[i] for i in indices)
    return selected


def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def evaluate(item: dict, result) -> dict:
    with Image.open(item["raw"]) as im:
        width, height = im.size
    gt = []
    for c, x, y, w, h in item["labels"]:
        gt.append((c, ((x - w / 2) * width, (y - h / 2) * height,
                       (x + w / 2) * width, (y + h / 2) * height)))
    pred = []
    if result.boxes is not None:
        for cls, conf, xyxy in zip(result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.xyxy.cpu().tolist()):
            pred.append((int(cls), float(conf), tuple(float(v) for v in xyxy)))
    candidates = []
    for gi, (gc, gb) in enumerate(gt):
        for pi, (pc, conf, pb) in enumerate(pred):
            if gc == pc:
                iou = box_iou(gb, pb)
                if iou >= 0.5:
                    candidates.append((iou, gi, pi, conf))
    matched_gt, matched_pred, matches = set(), set(), []
    for iou, gi, pi, conf in sorted(candidates, reverse=True):
        if gi not in matched_gt and pi not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(pi)
            matches.append({"class": CLASS_NAMES[gt[gi][0]], "iou": round(iou, 3), "confidence": round(conf, 3)})
    item.update({"gt": gt, "pred": pred, "matched": len(matches), "missed": len(gt) - len(matches),
                 "false_positive": len(pred) - len(matches), "matches": matches,
                 "gt_count": len(gt), "pred_count": len(pred)})
    return item


def choose_examples(items: list[dict]) -> list[dict]:
    chosen = []
    for group, count in (("smoke", 3), ("fire", 3), ("mixed", 2)):
        pool = [x for x in items if x["group"] == group]
        # Take strong detections and representative misses/extra boxes so the pages are candid.
        pool.sort(key=lambda x: (x["matched"] / max(1, x["gt_count"]), -x["false_positive"],
                                sum(m["iou"] for m in x["matches"]) / max(1, len(x["matches"]))), reverse=True)
        picks = []
        if pool:
            picks.append(pool[0])
        if len(pool) > 1:
            picks.append(min(pool[1:], key=lambda x: (x["matched"] / max(1, x["gt_count"]), -x["false_positive"])))
        for item in pool:
            if len(picks) >= count:
                break
            if item not in picks:
                picks.append(item)
        chosen.extend(picks[:count])
    return chosen


def get_font(size: int):
    options = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]
    for path in options:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_panel(source: Image.Image, boxes, mode: str, size=(600, 390)) -> Image.Image:
    panel_w, panel_h = size
    header_h = 46
    canvas = Image.new("RGB", (panel_w, header_h + panel_h), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = get_font(24)
    draw.text((16, 8), mode, fill=(35, 45, 60), font=header_font)
    image = source.convert("RGB")
    scale = min(panel_w / image.width, panel_h / image.height)
    rw, rh = round(image.width * scale), round(image.height * scale)
    image = image.resize((rw, rh), Image.Resampling.LANCZOS)
    ox, oy = (panel_w - rw) // 2, header_h + (panel_h - rh) // 2
    canvas.paste(image, (ox, oy))
    draw = ImageDraw.Draw(canvas)
    line_w = 3
    label_font = get_font(18)
    for cls, xyxy, conf in boxes:
        x1, y1, x2, y2 = xyxy
        x1, x2 = ox + x1 * scale, ox + x2 * scale
        y1, y2 = oy + y1 * scale, oy + y2 * scale
        color = COLORS[cls] if mode == "Ground truth" else PRED_COLOR
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_w)
        tag = CLASS_NAMES[cls] + (f" {conf:.2f}" if conf is not None else "")
        bounds = draw.textbbox((0, 0), tag, font=label_font)
        tw, th = bounds[2] - bounds[0], bounds[3] - bounds[1]
        ty = max(header_h, int(y1) - th - 8)
        draw.rectangle((int(x1), ty, int(x1) + tw + 8, ty + th + 6), fill=color)
        draw.text((int(x1) + 4, ty + 2), tag, fill="white", font=label_font)
    return canvas


def render(item: dict, index: int) -> Path:
    with Image.open(item["raw"]) as im:
        original = im.convert("RGB")
    gt_boxes = [(c, b, None) for c, b in item["gt"]]
    pred_boxes = [(c, b, conf) for c, conf, b in item["pred"]]
    panels = [draw_panel(original, [], "Original"), draw_panel(original, gt_boxes, "Ground truth"),
              draw_panel(original, pred_boxes, "Model prediction")]
    gap = 12
    board = Image.new("RGB", (sum(p.width for p in panels) + gap * 4, panels[0].height + 78), (242, 245, 249))
    x = gap
    for panel in panels:
        board.paste(panel, (x, gap))
        x += panel.width + gap
    draw = ImageDraw.Draw(board)
    note = (f"GT {item['gt_count']}  |  Pred {item['pred_count']}  |  matched {item['matched']}  |  "
            f"missed {item['missed']}  |  extra {item['false_positive']}")
    draw.text((gap + 4, board.height - 54), note, fill=(45, 55, 70), font=get_font(23))
    out = OUTPUT / f"case_{index:02d}_{item['group']}.jpg"
    board.save(out, quality=94, subsampling=0)
    return out


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    candidates = stratified_candidates()
    model = YOLO(str(WEIGHTS), task="detect")
    results = model.predict([str(x["image"]) for x in candidates], imgsz=640, conf=0.25,
                            device=3, batch=8, verbose=False, stream=True)
    evaluated = [evaluate(item, result) for item, result in zip(candidates, results)]
    examples = choose_examples(evaluated)
    records = []
    for index, item in enumerate(examples, start=1):
        out = render(item, index)
        records.append({"file": out.name, "image": item["name"], "group": item["group"],
                        "gt_count": item["gt_count"], "pred_count": item["pred_count"],
                        "matched": item["matched"], "missed": item["missed"],
                        "false_positive": item["false_positive"], "matches": item["matches"]})
    (OUTPUT / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
