#!/usr/bin/env python3
"""Match a cropped emoji against the panel templates extracted from WeChat."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent.parent
TEMPLATES = HERE / "emoji_templates"
NAMES = HERE / "emoji_names.txt"
SIZE = 32


def _load_templates() -> tuple[list[str], np.ndarray, np.ndarray]:
    names = [ln.strip() for ln in NAMES.read_text(encoding="utf-8").splitlines() if ln.strip()]
    records = json.loads((TEMPLATES / "index.json").read_text(encoding="utf-8"))
    arrs, masks = [], []
    for rec in records:
        idx = rec["index"] - 1
        img = Image.open(TEMPLATES / rec["file"]).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
        a = np.asarray(img).astype(np.float32)
        arrs.append(a)
        masks.append(a.min(axis=2) < 240)
    stacked = np.stack(arrs)
    return names, stacked, np.stack(masks)


_NAMES: list[str] | None = None
_TPL: np.ndarray | None = None
_MSK: np.ndarray | None = None


def match(img: Image.Image, fill: tuple[int, int, int] | None = None) -> tuple[str, float]:
    """Return (name, distance) for the closest panel emoji."""
    global _NAMES, _TPL, _MSK
    if _TPL is None:
        try:
            _NAMES, _TPL, _MSK = _load_templates()
        except Exception:
            # Templates are built from your own WeChat emoji panel
            # (`make-emoji`); without them emoji are simply left unnamed.
            _NAMES, _TPL, _MSK = [], np.zeros((0, SIZE, SIZE, 3), np.float32), \
                np.zeros((0, SIZE, SIZE), bool)
    if len(_NAMES) == 0:
        return "表情", 1e9

    small = np.asarray(img.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)).astype(np.float32)
    if fill is None:
        flat = small.reshape(-1, 3).astype(np.uint32)
        packed = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
        vals, counts = np.unique(packed, return_counts=True)
        v = int(vals[counts.argmax()])
        fill = ((v >> 16) & 255, (v >> 8) & 255, v & 255)
    ink = np.abs(small - np.array(fill, dtype=np.float32)).max(axis=2) > 26

    best, best_d = "表情", 1e9
    for i in range(len(_NAMES)):
        # compare only where both sides have actual artwork: the panel tiles are
        # on white while a chat emoji sits on the bubble colour
        m = ink & _MSK[i]
        if m.sum() < 60:
            continue
        d = float(np.abs(small[m] - _TPL[i][m]).mean())
        if d < best_d:
            best, best_d = (_NAMES[i] if i < len(_NAMES) else f"表情{i+1:03d}"), d
    return best, best_d


def main() -> int:
    """Print a labelled montage so the name list can be checked by eye."""
    names, tpl, _ = _load_templates()
    cols = 10
    rows = (len(names) + cols - 1) // cols
    from PIL import ImageDraw
    sheet = Image.new("RGB", (cols * 78, rows * 92), "white")
    draw = ImageDraw.Draw(sheet)
    font = None
    for candidate in ("/System/Library/Fonts/Hiragino Sans GB.ttc",
                      "/System/Library/Fonts/STHeiti Medium.ttc",
                      "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"):
        try:
            from PIL import ImageFont
            font = ImageFont.truetype(candidate, 15)
            break
        except Exception:
            continue
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        tile = Image.open(TEMPLATES / f"{i + 1:03d}.png")
        sheet.paste(tile, (c * 78 + 3, r * 92 + 2))
        draw.text((c * 78 + 3, r * 92 + 74), f"{i + 1} {name}", fill="black", font=font)
    out = HERE / "build" / "emoji_named.png"
    sheet.save(out)
    print(f"已生成 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
