"""Apply the project-local BiFPN parser extension to an Ultralytics source tree."""

from __future__ import annotations

import argparse
from pathlib import Path


IMPORT_LINE = "from models.ghost_bifpn import BiFPNP4"
MODEL_MARKER = "class BaseModel(nn.Module):"
PARSER_MARKER = "        elif m is AIFI:\n"
PARSER_BRANCH = (
    "        elif m is BiFPNP4:\n"
    "            c1 = [ch[x] for x in f]\n"
    "            c2 = make_divisible(min(args[0], max_channels) * width, 8)\n"
    "            args = [c1, c2]\n"
)


def patch_tasks_file(tasks_file: Path) -> None:
    source = tasks_file.read_text(encoding="utf-8")
    if IMPORT_LINE not in source:
        if MODEL_MARKER not in source:
            raise ValueError(f"Cannot find model import insertion point in {tasks_file}")
        source = source.replace(MODEL_MARKER, f"{IMPORT_LINE}\n\n\n{MODEL_MARKER}", 1)
    if PARSER_BRANCH not in source:
        if PARSER_MARKER not in source:
            raise ValueError(f"Cannot find parse_model insertion point in {tasks_file}")
        source = source.replace(PARSER_MARKER, PARSER_BRANCH + PARSER_MARKER, 1)
    tasks_file.write_text(source, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ultralytics_root", type=Path, help="directory containing ultralytics/nn/tasks.py")
    args = parser.parse_args()
    tasks_file = args.ultralytics_root / "ultralytics" / "nn" / "tasks.py"
    patch_tasks_file(tasks_file)
    print(f"Patched {tasks_file}")


if __name__ == "__main__":
    main()
