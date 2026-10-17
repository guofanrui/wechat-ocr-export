#!/usr/bin/env python3
"""Cut WeChat's emoji panel screenshots into individual template tiles.

Writes emoji_templates/NNN.png plus a montage (build/emoji_montage.png) with the
index printed on each tile so the list can be labelled by eye.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent.parent
PANELS = HERE / "emoji_panel"
OUT = HERE / "emoji_templates"
BUILD = HERE / "build"
TILE = 72


def tiles_of(path: Path) -> list[tuple[int, int, int, int]]:
    img = Image.open(path).convert("RGB")
    a = np.asarray(img).astype(np.int16)
    # the panel background is white; anything else is an emoji or a label
    ink = (a.min(axis=2) < 244) | (a.max(axis=2) - a.min(axis=2) > 12)
    h, w = ink.shape
    rows = ink.sum(axis=1)
    bands, start = [], None
    for y in range(h):
        if rows[y] > 3 and start is None:
            start = y
        elif rows[y] <= 3 and start is not None:
            if y - start > 30:
                bands.append((start, y - 1))
            start = None
    if start is not None and h - start > 30:
        bands.append((start, h - 1))

    out = []
    for y0, y1 in bands:
        cols = ink[y0:y1 + 1].sum(axis=0)
        runs, s = [], None
        for x in range(w):
            if cols[x] > 2 and s is None:
                s = x
            elif cols[x] <= 2 and s is not None:
                if x - s > 24:
                    runs.append((s, x - 1))
                s = None
        if s is not None and w - s > 24:
            runs.append((s, w - 1))
        for x0, x1 in runs:
            out.append((x0, y0, x1 - x0 + 1, y1 - y0 + 1))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    records = []
    index = 0
    for panel in sorted(PANELS.glob("panel*.png")):
        img = Image.open(panel).convert("RGB")
        for box in tiles_of(panel):
            x, y, w, h = box
            tile = img.crop((x, y, x + w, y + h))
            tile = tile.resize((TILE, TILE), Image.LANCZOS)
            index += 1
            name = f"{index:03d}.png"
            tile.save(OUT / name)
            records.append({"index": index, "panel": panel.name, "box": box, "file": name})
    (OUT / "index.json").write_text(json.dumps(records, indent=1), encoding="utf-8")

    cols = 10
    rows = (index + cols - 1) // cols
    sheet = Image.new("RGB", (cols * TILE, rows * (TILE + 22)), "white")
    draw = ImageDraw.Draw(sheet)
    for i in range(index):
        r, c = divmod(i, cols)
        t = Image.open(OUT / f"{i + 1:03d}.png")
        sheet.paste(t, (c * TILE, r * (TILE + 22)))
        draw.text((c * TILE + 6, r * (TILE + 22) + TILE + 4), str(i + 1), fill="black")
    sheet.save(BUILD / "emoji_montage.png")
    print(f"提取 {index} 个表情模板 -> {OUT}")
    print(f"拼图 -> {BUILD / 'emoji_montage.png'} （{cols} 列 x {rows} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
