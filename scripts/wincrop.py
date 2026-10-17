#!/usr/bin/env python3
"""Crop a `screencapture -l` window shot down to the window content rectangle.

The raw capture includes the drop shadow, whose size changes with window focus,
so cropped output keeps pixel coordinates stable across a capture session.

Usage:  wincrop.py IN.png OUT.png
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def crop(path: Path, out: Path) -> tuple[int, int]:
    img = Image.open(path)
    if img.mode not in ("RGBA", "LA"):
        img.convert("RGB").save(out)
        return img.size
    rgba = img.convert("RGBA")
    alpha = np.asarray(rgba)[:, :, 3]
    ys, xs = np.where(alpha == 255)
    if xs.size:
        rgba = rgba.crop((int(xs.min()), int(ys.min()),
                          int(xs.max()) + 1, int(ys.max()) + 1))
    rgb = Image.new("RGB", rgba.size, (250, 250, 250))
    rgb.paste(rgba, mask=rgba.split()[3])
    rgb.save(out)
    return rgb.size


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: wincrop.py IN.png OUT.png")
    size = crop(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"{sys.argv[2]}: {size[0]}x{size[1]}")
