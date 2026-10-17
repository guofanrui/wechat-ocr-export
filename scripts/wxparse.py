#!/usr/bin/env python3
"""Turn a WeChat window screenshot + its OCR JSON into a structured message list.

Steps
  1. locate the message pane inside the window screenshot
  2. segment message bubbles by fill colour (incoming vs. own)
  3. attach OCR lines to bubbles; split the bubble text into paragraphs
  4. classify centred lines as timestamp separators or system notices
  5. flag leftover non-background regions as media (image / sticker / card)

Usage:
  wxparse.py --shot SHOT.png --ocr OCR.json --out PARSED.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import emoji_match           # panel templates, used to recognise emoji tiles
except Exception:                # pragma: no cover - templates are optional
    emoji_match = None

# WeChat light-theme fills, measured from real captures
BG = (250, 250, 250)          # message pane background
SIDEBAR = (238, 238, 240)     # chat-list sidebar (same tone as incoming bubbles)
IN_FILL = (238, 238, 240)     # incoming bubble
OUT_FILL = (176, 240, 167)    # own bubble (WeChat green as captured)
WHITE = (255, 255, 255)       # composer input box / card body

TIME_RE = re.compile(
    r"^("
    r"\d{1,2}:\d{2}"
    r"|今天\s*\d{1,2}:\d{2}"
    r"|昨天\s*\d{1,2}:\d{2}"
    r"|星期[一二三四五六日天]\s*\d{0,2}:?\d{0,2}"
    r"|\d{1,2}月\d{1,2}日\s*(\d{1,2}:\d{2})?"
    r"|\d{4}年\d{1,2}月\d{1,2}日\s*(\d{1,2}:\d{2})?"
    r"|[上下]午\s*\d{1,2}:\d{2}"
    r")$"
)

FILE_HINT = re.compile(
    r"(\.(pdf|docx?|xlsx?|pptx?|zip|rar|txt|md|csv|py|c|h|java|mp4|mov|mp3)\b|文件)", re.I)
LINK_HINT = re.compile(
    r"(https?://|www\.|链接|小程序|公众号|视频号|音乐|位置|名片|转账|红包)", re.I)
QUOTE_HINT = re.compile(r"^[「【\"']|引用|回复")


def load_rgb(path: Path) -> np.ndarray:
    """Load a window screenshot as just the window content.

    `screencapture -l` returns the window plus its drop shadow; the shadow size
    changes with focus, so the raw image size is not stable. The window content
    itself is the fully-opaque region, which is a fixed rectangle.
    """
    img = Image.open(path)
    if img.mode not in ("RGBA", "LA"):
        return np.asarray(img.convert("RGB"))

    rgba_img = img.convert("RGBA")
    arr = np.asarray(rgba_img)
    alpha_ch = arr[:, :, 3]
    ys, xs = np.where(alpha_ch == 255)
    if xs.size:
        rgba_img = rgba_img.crop((int(xs.min()), int(ys.min()),
                                  int(xs.max()) + 1, int(ys.max()) + 1))
        arr = np.asarray(rgba_img)

    if arr.shape[2] == 4:
        rgba = arr.astype(np.float32)
        alpha = rgba[:, :, 3:4] / 255.0
        rgb = rgba[:, :, :3] * alpha + np.array(BG, dtype=np.float32) * (1.0 - alpha)
        return rgb.astype(np.uint8)
    return np.asarray(rgba_img.convert("RGB"))


def mask_for(arr: np.ndarray, colour, tol: int) -> np.ndarray:
    target = np.array(colour, dtype=np.int16)
    return (np.abs(arr.astype(np.int16) - target) <= tol).all(axis=2)


def longest_run(row: np.ndarray) -> int:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8), [0]))))
    if edges.size < 2:
        return 0
    return int((edges[1::2] - edges[0::2]).max())


def group_paragraphs(lines: list[dict]) -> list[str]:
    """Merge consecutive OCR lines into paragraphs using vertical gaps."""
    if not lines:
        return []
    gaps = [lines[i + 1]["cy"] - lines[i]["cy"] for i in range(len(lines) - 1)]
    base = float(np.median(gaps)) if gaps else 0.0
    out: list[str] = []
    cur = [lines[0]["text"]]
    for i, g in enumerate(gaps):
        if base > 4 and g > base * 1.45:
            out.append("".join(cur))
            cur = []
        cur.append(lines[i + 1]["text"])
    out.append("".join(cur))
    return [p for p in out if p.strip()]


def reading_order(lines: list[dict]) -> list[dict]:
    """Order OCR lines the way they read: top to bottom, left to right.

    Vision splits one bubble line when a sticker sits in the middle of the text,
    and the two halves can differ by a couple of pixels in y. Sorting by y alone
    then swaps them. Group near-equal baselines into a row first.
    """
    rows: list[list[dict]] = []
    for ln in sorted(lines, key=lambda l: (l["y"], l["x"])):
        for row in rows:
            r0 = row[0]["y"]
            r1 = max(l["y"] + l["h"] for l in row)
            overlap = min(r1, ln["y"] + ln["h"]) - max(r0, ln["y"])
            if overlap > 0.5 * min(r1 - r0, ln["h"]):
                row.append(ln)
                break
        else:
            rows.append([ln])
    rows.sort(key=lambda r: min(l["y"] for l in r))
    out: list[dict] = []
    for row in rows:
        out.extend(sorted(row, key=lambda l: l["x"]))
    return out


def components(mask: np.ndarray, min_area: int) -> list[dict]:
    """Connected components via row runs + union-find.

    Equivalent to a flood fill but far faster in Python: the work scales with
    the number of horizontal runs rather than the number of pixels.
    """
    h, w = mask.shape
    rows: list[list[tuple[int, int]]] = [[] for _ in range(h)]
    index_of: dict[tuple[int, int], int] = {}
    for y in range(h):
        row = mask[y]
        if not row.any():
            continue
        edges = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8), [0]))))
        for s, e in zip(edges[0::2], edges[1::2]):
            index_of[(y, int(s))] = len(index_of)
            rows[y].append((int(s), int(e) - 1))

    if not index_of:
        return []

    parent = list(range(len(index_of)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for y in range(h - 1):
        cur, nxt = rows[y], rows[y + 1]
        if not cur or not nxt:
            continue
        j = 0
        for s, e in cur:
            while j < len(nxt) and nxt[j][1] < s - 1:
                j += 1
            k = j
            while k < len(nxt) and nxt[k][0] <= e + 1:
                union(index_of[(y, s)], index_of[(y + 1, nxt[k][0])])
                k += 1

    agg: dict[int, dict] = {}
    for y in range(h):
        for s, e in rows[y]:
            root = find(index_of[(y, s)])
            area = e - s + 1
            box = agg.get(root)
            if box is None:
                agg[root] = {"x0": s, "x1": e, "y0": y, "y1": y, "area": area}
            else:
                box["x0"] = min(box["x0"], s)
                box["x1"] = max(box["x1"], e)
                box["y1"] = y
                box["area"] += area

    return [
        {"x": b["x0"], "y": b["y0"], "w": b["x1"] - b["x0"] + 1,
         "h": b["y1"] - b["y0"] + 1, "area": b["area"]}
        for b in agg.values() if b["area"] >= min_area
    ]


def merge_media(blocks: list[dict]) -> list[dict]:
    """Merge media blocks that belong to one message (e.g. a sticker split in two)."""
    blocks = sorted(blocks, key=lambda b: b["y"])
    out: list[dict] = []
    for b in blocks:
        if out:
            prev = out[-1]
            px0, px1 = prev["x"], prev["x"] + prev["w"]
            bx0, bx1 = b["x"], b["x"] + b["w"]
            overlap = min(px1, bx1) - max(px0, bx0)
            gap = b["y"] - (prev["y"] + prev["h"])
            if overlap > 0.5 * min(prev["w"], b["w"]) and gap < 60:
                nx0, ny0 = min(px0, bx0), min(prev["y"], b["y"])
                nx1 = max(px1, bx1)
                ny1 = max(prev["y"] + prev["h"], b["y"] + b["h"])
                prev.update({"x": nx0, "y": ny0, "w": nx1 - nx0, "h": ny1 - ny0})
                continue
        out.append(dict(b))
    return out


def is_image_block(arr: np.ndarray, block: dict) -> bool:
    """True when a media block is a picture/sticker rather than a text-only card.

    A card is mostly a flat fill; an image has many distinct colours.
    """
    x0, y0 = block["x"], block["y"]
    x1, y1 = x0 + block["w"], y0 + block["h"]
    patch = arr[max(0, y0):y1, max(0, x0):x1]
    if patch.size == 0:
        return False
    packed = (patch[:, :, 0].astype(np.uint32) << 16) | \
             (patch[:, :, 1].astype(np.uint32) << 8) | patch[:, :, 2]
    values, counts = np.unique(packed, return_counts=True)
    return bool(counts.max() / counts.sum() < 0.62)


def _resolve_override(value: float, size: int) -> int:
    """Overrides are fractions of the image, or absolute pixels when > 1."""
    return int(value) if value > 1 else int(size * value)


EMOJI_TOKEN = "\ue000"


def line_darkness(arr: np.ndarray, ln: dict) -> float:
    """Luminance of the darkest pixels in an OCR line.

    WeChat draws ordinary message text near-black, but quotes and notices in a
    mid grey, so this separates the two reliably.
    """
    x0 = max(0, int(ln["x"]))
    y0 = max(0, int(ln["y"]))
    x1 = min(arr.shape[1], int(ln["x"] + ln["w"]) + 1)
    y1 = min(arr.shape[0], int(ln["y"] + ln["h"]) + 1)
    sub = arr[y0:y1, x0:x1]
    if sub.size == 0:
        return 0.0
    return float(np.percentile(sub.mean(axis=2), 3))


def find_quote_bars(arr: np.ndarray, pane_x0: int, pane_right: int,
                    msg_top: int, msg_bottom: int) -> list[tuple[int, int, int]]:
    """Locate the small vertical bar WeChat draws beside a quoted message.

    The bar is a run of neutral grey (~#E3E3E3, distinctly darker than both the
    bubble fill #EEEEF0 and the pane #FAFAFA) that is 2-6px wide and 20-260px
    tall, with pane background immediately to its left.
    """
    x0, x1 = pane_x0 + 40, pane_right - 30
    region = arr[msg_top:msg_bottom, x0:x1].astype(np.int16)
    lum = region.mean(axis=2)
    mx, mn = region.max(axis=2), region.min(axis=2)
    grey = (lum >= 214) & (lum <= 233) & ((mx - mn) <= 7)

    runs: list[list[int]] = []
    for cx in range(1, region.shape[1]):
        idx = np.flatnonzero(grey[:, cx])
        if idx.size == 0:
            continue
        # the bar must stand on the pane background, not inside a bubble
        left = region[:, cx - 1].mean(axis=1)
        start = prev = idx[0]
        for y in list(idx[1:]) + [10 ** 9]:
            if y == prev + 1:
                prev = y
                continue
            if 20 <= prev - start <= 260:
                mid = (start + prev) // 2
                if 244 <= left[mid] <= 256:
                    runs.append([x0 + cx, msg_top + int(start), msg_top + int(prev)])
            start = prev = y

    merged: list[list[int]] = []
    for x, y0, y1 in sorted(runs):
        for m in merged:
            if abs(x - m[0]) <= 5 and abs(y0 - m[1]) <= 20:
                m[1] = min(m[1], y0)
                m[2] = max(m[2], y1)
                break
        else:
            merged.append([x, y0, y1])
    return [(x, y0, y1) for x, y0, y1 in merged]
def detect_emojis(arr: np.ndarray, box: dict,
                  min_side: int = 18, max_side: int = 90) -> list[list[int]]:
    """Find WeChat's inline emoji inside a bubble by matching the fixed tiles.

    Emoji are drawn at a fixed size, so the reliable test is "does one of the
    panel tiles match here", not "where are the colourful pixels". A tile that is
    partly transparent (fireworks, party popper) would otherwise read as several
    fragments and get cut in half.
    """
    x0, y0 = box["x"], box["y"]
    x1, y1 = x0 + box["w"], y0 + box["h"]
    sub = arr[max(0, y0):y1, max(0, x0):x1]
    if sub.size == 0:
        return []
    sat = sub.astype(np.int16)
    sat = sat.max(axis=2) - sat.min(axis=2)
    mask = sat > 95
    mask[:, :8] = False
    mask[:, -8:] = False
    mask[:4] = False
    mask[-4:] = False

    parts = components(mask, min_area=60)
    parts = [p for p in parts if p["w"] >= 8 and p["h"] >= 8]
    clusters: list[dict] = []
    for p in sorted(parts, key=lambda q: (q["y"], q["x"])):
        for m in clusters:
            if (p["x"] <= m["x"] + m["w"] + 22 and p["x"] + p["w"] >= m["x"] - 22
                    and p["y"] <= m["y"] + m["h"] + 22 and p["y"] + p["h"] >= m["y"] - 22):
                m["x"] = min(m["x"], p["x"])
                m["y"] = min(m["y"], p["y"])
                m["w"] = max(m["x"] + m["w"], p["x"] + p["w"]) - m["x"]
                m["h"] = max(m["y"] + m["h"], p["y"] + p["h"]) - m["y"]
                break
        else:
            clusters.append(dict(p))

    best_matches: list[list[int]] = []
    for m in clusters:
        tile = min(max(m["h"], 26), 44)
        gx0 = max(0, m["x"] - 6)
        gx1 = min(sub.shape[1], m["x"] + m["w"] + 6)
        gy0 = max(0, m["y"] - 6)
        gy1 = min(sub.shape[0], m["y"] + m["h"] + 6)
        found: list[tuple[float, int, int]] = []
        for yy in range(gy0, max(gy0 + 1, gy1 - tile + 1), 2):
            for xx in range(gx0, max(gx0 + 1, gx1 - tile + 1), 2):
                patch = sub[yy:yy + tile, xx:xx + tile]
                if patch.shape[0] != tile or patch.shape[1] != tile:
                    continue
                try:
                    _, dist = emoji_match.match(Image.fromarray(patch))
                except Exception:
                    continue
                if dist <= 40:
                    found.append((dist, xx, yy))
        found.sort()
        for dist, xx, yy in found:                 # non-maximum suppression
            centre = (x0 + xx + tile / 2, y0 + yy + tile / 2)
            if any(abs(centre[0] - (r[0] + r[2] / 2)) < tile * 0.85
                   and abs(centre[1] - (r[1] + r[3] / 2)) < tile * 0.6
                   for r in best_matches):
                continue
            # Crop a fixed square centred on the match: the saturated part is
            # smaller than the emoji's full tile (white muzzle, soft edges), so
            # cropping by the match box used to clip the artwork.
            # centre the crop on the coloured cluster rather than the match box:
            # the cluster is the emoji's saturated body and sits centred in it
            # start from a generous box, then trim it to the artwork's own ink so
            # the crop is both complete and tight (no neighbouring text, no clip)
            # The saturated cluster IS the artwork's own body; a neighbouring
            # glyph is dark and unsaturated, so it never joins the cluster. Crop
            # around the cluster with a tight left margin (the glyph sits 2-3px
            # away) and a little more room on the other sides.
            mcx, mcy = x0 + m["x"], y0 + m["y"]
            bx0 = max(0, mcx - 2)
            by0 = max(0, mcy - 5)
            bw = min(arr.shape[1] - bx0, m["w"] + 6)
            bh = min(arr.shape[0] - by0, m["h"] + 10)
            best_matches.append([bx0, by0, bw, bh])
    best_matches.sort(key=lambda r: (r[1], r[0]))
    unique: list[list[int]] = []
    for r in best_matches:
        if any(abs(r[0] - u[0]) < 8 and abs(r[1] - u[1]) < 8 for u in unique):
            continue
        unique.append(r)
    return unique
def inject_emoji_tokens(bubble: dict, emojis: list[list[int]]) -> None:
    """Replace the characters OCR invented for an emoji with a template token.

    Vision often splits a bubble line at the sticker ("为什么最近这么冷淡" +
    "6o•有什么事情吗"), and the junk it produces for the emoji can land in
    either fragment. Placement uses the emoji's *centre*, because the artwork
    (50-60px) is much wider than a glyph (26-30px) and its left edge sits well
    inside the preceding character — measuring from the edge put every emoji
    one character early. Only ASCII junk under the emoji's centre is removed;
    a real Chinese character next to the sticker is never touched.
    """
    bubble["emojis"] = emojis
    # Vision sometimes reports the artwork itself as a separate "line" ("533",
    # "⑨"). Such a fragment lives entirely inside the emoji's box and is junk.
    for ln in bubble.get("lines", []):
        for ex0, ey0, ew, eh in emojis:
            if (ln["x"] >= ex0 - 2 and ln["x"] + ln["w"] <= ex0 + ew + 2
                    and ln["y"] >= ey0 - 4 and ln["y"] + ln["h"] <= ey0 + eh + 4):
                ln["text"] = ""
                break
    groups: dict[int, tuple[dict, float, list[list[int]]]] = {}
    for rect in emojis:
        ex0, ey0, ew, eh = rect
        centre = ex0 + ew / 2.0
        best = None
        for ln in bubble.get("lines", []):
            if not ln["text"]:
                continue
            if not (ln["y"] - 10 <= ey0 + eh / 2 <= ln["y"] + ln["h"] + 10):
                continue
            per = max(1.0, ln["w"] / len(ln["text"]))
            inside = ln["x"] - per <= centre <= ln["x"] + ln["w"] + per
            score = 1 if inside else 0
            if best is None or score > best[0]:
                best = (score, ln, per)
        if best is None:
            continue
        slot = groups.setdefault(id(best[1]), (best[1], best[2], []))
        slot[2].append(rect)

    for ln, per, rects in groups.values():
        text = ln["text"]
        # Pitch: the line holds n characters plus an emoji slot that is wider
        # than a glyph (~1.2x). Measure the artwork's *left edge* — that is where
        # the slot starts, and it survives the artwork overlapping its
        # neighbours, which is what put every emoji one character early.
        pitch = max(8.0, ln["w"] / max(1, len(text) + 0.2 * len(rects)))
        drop: set[int] = set()
        inserts: list[tuple[int, int]] = []
        for order, rect in enumerate(rects):
            ex0, ey0, ew, eh = rect
            centre = ex0 + ew / 2.0
            for i, ch in enumerate(text):
                c0 = ln["x"] + i * pitch
                cover = max(0.0, min(c0 + pitch, ex0 + ew) - max(c0, ex0)) / pitch
                junk = ord(ch) < 128 or ch in "•·。"
                if junk and cover >= 0.85 and abs(c0 + pitch / 2 - centre) <= 0.75 * pitch:
                    drop.add(i)
            idx = sum(1 for i, _ in enumerate(text)
                      if ln["x"] + (i + 0.5) * pitch < ex0)
            # A sticker at the end of a line: put the token after everything and
            # drop the one or two glyphs OCR invented for it ("明天见" read as
            # "明天见533").
            if ex0 + ew >= ln["x"] + ln["w"] - 6:
                idx = len(text)
                for i in range(len(text)):
                    c0 = ln["x"] + i * pitch
                    cover = max(0.0, min(c0 + pitch, ex0 + ew) - max(c0, ex0)) / pitch
                    if i >= len(text) - 3 and cover >= 0.5:
                        drop.add(i)
            # A sticker at the very end of a line often leaves Vision with one
            # invented Chinese glyph ("包饺子" read as "包饺子起"). If the
            # artwork sits past the text, that trailing glyph is the emoji.
            if idx >= len(text) - 1 and text and ord(text[-1]) >= 128:
                last0 = ln["x"] + (len(text) - 1) * pitch
                if min(last0 + pitch, ex0 + ew) - max(last0, ex0) >= 0.6 * pitch:
                    drop.add(len(text) - 1)
            inserts.append((idx, order))
        at: dict[int, list[str]] = {}
        for idx, _ in sorted(inserts, key=lambda t: t[0]):
            at.setdefault(idx, []).append(EMOJI_TOKEN)
        out: list[str] = []
        for i, ch in enumerate(text):
            out.extend(at.get(i, []))
            if i not in drop:
                out.append(ch)
        out.extend(at.get(len(text), []))
        ln["text"] = "".join(out)


def fill_ratio(arr: np.ndarray, box: dict, fill: tuple[int, int, int],
               tol: int = 6) -> float:
    """Share of a box that is exactly the bubble's own fill colour.

    A text bubble is painted almost entirely in its fill colour (only the glyphs
    are dark), so this sits around 0.85-0.98 for messages. A pasted picture —
    even one containing WeChat's own green bubbles — stays far below that. This
    is the primary test for "message vs. picture".
    """
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    sub = arr[int(y):int(y + h), int(x):int(x + w)].astype(np.int16)
    if sub.size == 0:
        return 0.0
    return float((np.abs(sub - np.array(fill, dtype=np.int16)).max(axis=2) <= tol).mean())


def looks_like_image(arr: np.ndarray, box: dict, lines: list[dict],
                     fill: tuple[int, int, int]) -> bool:
    """Tell a shared picture apart from a text bubble.

    A screenshot of another chat has a background close to WeChat's own bubble
    grey, so colour alone is not enough: text bubbles are wider than tall, keep a
    high share of flat fill, and use the normal font size, while a pasted
    picture is taller, busier and carries smaller text.
    """
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    if h <= 0:
        return False
    aspect = w / h
    if aspect < 0.9:
        return True
    if aspect > 3.4 or h < 120:
        return False
    sub = arr[y:y + h, x:x + w].astype(np.int16)
    fill_frac = float((np.abs(sub - np.array(fill, dtype=np.int16)).max(axis=2) <= 10).mean())
    # A region painted mostly in the bubble colour is a text bubble. Nothing
    # that was pasted into the chat is ever a flat block of WeChat green.
    if fill_frac >= 0.45:
        return False
    if float((np.abs(sub - np.array(OUT_FILL, dtype=np.int16)).max(axis=2) <= 10).mean()) > 0.45:
        return False
    if len(lines) == 0:
        return False
    heights = [ln["h"] for ln in lines] or [0]
    median_h = float(np.median(heights))
    return fill_frac < 0.70 and median_h < 17.0


def detect_image_messages(arr: np.ndarray, layout: dict, lines: list[dict],
                          used: set[int]) -> list[dict]:
    """Find pasted pictures (a shared photo or a screenshot) as whole regions.

    A picture is not a bubble: it has no bubble fill, its shape is not text-shaped
    and the text inside it is smaller. Detecting it up front keeps its inner text
    out of the message pipeline entirely, and lets a picture that the content
    mask split into pieces be reassembled before anything else looks at it.
    """
    h, w, _ = arr.shape
    pane_x0, pane_right = layout["pane_x0"], layout["pane_right"]
    mt, mb = layout["msg_top"], layout["msg_bottom"]

    # a chat screenshot's own background sits only ~12 levels away from the pane
    # background, so this detection needs a finer threshold than the bubble pass
    content = (np.abs(arr.astype(np.int16) - np.array(BG, dtype=np.int16)).max(axis=2) > 6)
    content[:, :pane_x0 + 6] = False
    content[:, pane_right - 6:] = False
    content[:mt] = False
    content[mb:] = False
    parts = components(content, min_area=6000)
    parts = [p for p in parts if p["w"] >= 140 and p["h"] >= 90]

    # pieces of one picture: overlapping in x, separated by only a small gap
    merged: list[dict] = []
    for p in sorted(parts, key=lambda q: q["y"]):
        for m in merged:
            overlap = min(p["x"] + p["w"], m["x"] + m["w"]) - max(p["x"], m["x"])
            gap = p["y"] - (m["y"] + m["h"])
            if overlap > 0.6 * min(p["w"], m["w"]) and -4 <= gap <= 60:
                m["x"] = min(m["x"], p["x"])
                m["w"] = max(m["x"] + m["w"], p["x"] + p["w"]) - m["x"]
                m["h"] = max(m["y"] + m["h"], p["y"] + p["h"]) - m["y"]
                m["area"] += p["area"]
                break
        else:
            merged.append(dict(p))

    mid = pane_x0 + (pane_right - pane_x0) / 2.0
    out: list[dict] = []
    for m in merged:
        if m["w"] < 180 or m["h"] < 110:
            continue
        # WeChat indents every message past the avatar column, so a region that
        # runs into the pane's left edge is layout, not a pasted picture.
        if m["x"] < pane_x0 + 60:
            continue
        # a pasted picture can be landscape; the fill/font tests below plus the
        # left-edge rule keep ordinary text bubbles out
        if m["w"] > 3.4 * m["h"]:
            continue
        inside = [ln for ln in lines
                  if m["x"] - 6 <= ln["cx"] <= m["x"] + m["w"] + 6
                  and m["y"] - 6 <= ln["cy"] <= m["y"] + m["h"] + 6]
        box = {"x": m["x"], "y": m["y"], "w": m["w"], "h": m["h"]}
        fills = (OUT_FILL, IN_FILL)
        # a green region is one of our own text bubbles, never a pasted picture
        patch = arr[m["y"]:m["y"] + m["h"], m["x"]:m["x"] + m["w"]].astype(np.int16)
        if patch.size and float(
                (np.abs(patch - np.array(OUT_FILL, dtype=np.int16)).max(axis=2) <= 10).mean()
        ) > 0.35:
            continue
        if not any(looks_like_image(arr, box, inside, f) for f in fills):
            continue
        for ln in inside:                      # keep the inner text out of play
            if ln in lines:
                used.add(lines.index(ln))
        out.append({"kind": "media", "side": "out" if (m["x"] + m["w"] / 2.0) > mid else "in",
                    "y": m["y"], "rect": [m["x"], m["y"], m["w"], m["h"]],
                    "label": "图片", "is_image": True, "text": "", "paragraphs": [],
                    "clipped_top": m["y"] <= mt + 8,
                    "clipped_bottom": m["y"] + m["h"] >= mb - 8})
    return out


def find_layout(arr: np.ndarray, ov: dict) -> dict:
    h, w, _ = arr.shape
    bg = mask_for(arr, BG, 3)
    sidebar = mask_for(arr, SIDEBAR, 4)
    n_bg, n_side = bg.sum(axis=0), sidebar.sum(axis=0)

    pane_x0 = None
    for x in range(int(w * 0.15), w):
        if n_bg[x] > 0.45 * h and n_bg[x] > n_side[x] * 1.2:
            pane_x0 = x
            break
    if pane_x0 is None:
        pane_x0 = int(w * 0.38)
    if ov.get("left") is not None:
        pane_x0 = int(w * ov["left"])

    # the window's right edge / scrollbar shows up as a tall dense column
    diff_all = np.abs(arr.astype(np.int16) - np.array(BG, dtype=np.int16)).max(axis=2)
    col_content = (diff_all > 12).mean(axis=0)
    pane_right = w - 4
    for x in range(w - 1, int(w * 0.55), -1):
        if col_content[x] > 0.5:
            pane_right = x - 3
            break
    pane_w = pane_right - pane_x0

    # content profile across the pane (ignore the window's right edge noise)
    core_l, core_r = pane_x0 + 8, pane_right - 8
    pane_core = arr[:, core_l:core_r]
    diff = np.abs(pane_core.astype(np.int16) - np.array(BG, dtype=np.int16)).max(axis=2)
    content_frac = (diff > 6).mean(axis=1)

    # composer: the message band ends at the input box's top border, a long
    # horizontal *neutral grey* line slightly darker than the pane background.
    # Bubble fills are also darker than the background, so require neutrality.
    core_i = pane_core.astype(np.int16)
    lo = core_i.min(axis=2)
    hi = core_i.max(axis=2)
    darker = ((np.array(BG, dtype=np.int16)[0] - lo) >= 4) & (lo >= 185) & ((hi - lo) <= 8)
    msg_bottom = h
    for y in range(int(h * 0.62), h):
        if longest_run(darker[y]) > pane_core.shape[1] * 0.6:
            msg_bottom = y - 6
            break
    # a wide flat image in the conversation can look like the composer border;
    # the composer is always low in the window, so reject implausible hits
    if msg_bottom < h * 0.72:
        msg_bottom = int(h * 0.85)
    if ov.get("bottom") is not None:
        msg_bottom = _resolve_override(ov["bottom"], h)

    # header: start at the first strong content row (first bubble) and walk up
    # through the clean gap, stopping at the title text block
    # the window title bar sits above a clean gap; message content starts below it
    body_start = int(h * 0.05)
    run = 0
    for y in range(int(h * 0.02), int(h * 0.35)):
        if content_frac[y] < 0.08:
            run += 1
        else:
            if run >= 10:
                body_start = y
                break
            run = 0

    # The message view starts right below the header's divider line, and bubbles
    # are clipped by that edge, so anything touching it is only partly visible.
    msg_top = 0
    for y in range(int(h * 0.02), int(h * 0.20)):
        if longest_run(darker[y]) > pane_core.shape[1] * 0.5:
            msg_top = y + 2
            break
    if msg_top == 0:
        first_strong = None
        for y in range(body_start, int(h * 0.5)):
            if content_frac[y] > 0.30:
                first_strong = y
                break
        if first_strong is None:
            first_strong = body_start
        msg_top = first_strong
        while msg_top - 1 > int(h * 0.02) and content_frac[msg_top - 1] < 0.08:
            msg_top -= 1
    if ov.get("top") is not None:
        msg_top = _resolve_override(ov["top"], h)

    return {
        "pane_x0": pane_x0, "pane_w": pane_w, "pane_right": pane_right,
        "core": [core_l, core_r],
        "msg_top": msg_top, "msg_bottom": msg_bottom,
        "avatar_l": pane_x0 + int(pane_w * 0.10),
        "avatar_r": pane_right - int(pane_w * 0.10),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", required=True)
    ap.add_argument("--ocr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--green-tol", type=int, default=18)
    ap.add_argument("--pane-left", type=float, default=None)
    ap.add_argument("--pane-top", type=float, default=None)
    ap.add_argument("--pane-bottom", type=float, default=None)
    ap.add_argument("--ocr-offset-x", type=int, default=0,
                    help="x offset of the OCR image inside the screenshot")
    ap.add_argument("--ocr-offset-y", type=int, default=0,
                    help="y offset of the OCR image inside the screenshot")
    ap.add_argument("--ocr-size", type=int, nargs=2, default=None,
                    help="size of the OCR image (defaults to the screenshot size)")
    args = ap.parse_args()

    shot = Path(args.shot)
    arr = load_rgb(shot)
    h, w, _ = arr.shape
    layout = find_layout(arr, {"left": args.pane_left, "top": args.pane_top,
                               "bottom": args.pane_bottom})
    pane_x0, pane_w = layout["pane_x0"], layout["pane_w"]
    msg_top, msg_bottom = layout["msg_top"], layout["msg_bottom"]

    raw = json.loads(Path(args.ocr).read_text(encoding="utf-8"))
    ocr = raw[0]["pages"][0]["lines"] if raw else []
    lines = []
    for ln in ocr:
        bx, by, bw, bh = ln["box"]
        ocr_w, ocr_h = args.ocr_size if args.ocr_size else (w, h)
        x = bx * ocr_w + args.ocr_offset_x
        y = (1.0 - by - bh) * ocr_h + args.ocr_offset_y
        wpx, hpx = bw * ocr_w, bh * ocr_h
        lines.append({"text": ln["text"], "conf": float(ln.get("confidence", 0)),
                      "x": x, "y": y, "w": wpx, "h": hpx,
                      "cx": x + wpx / 2, "cy": y + hpx / 2})
    # OCR loves to invent a glyph or two on the clipped edges (a bubble's
    # shadow, the composer's border). Text that is genuinely cut off here is
    # shown in full by the neighbouring screen.
    lines = [ln for ln in lines
             if msg_top - 4 <= ln["cy"] <= msg_bottom - 3]

    # ---- pasted pictures come first --------------------------------------
    # A screenshot of another chat has a background that matches WeChat's bubble
    # grey, so it has to be claimed before the bubble masks run, otherwise it is
    # mistaken for a text bubble and its inner text leaks into the transcript.
    # Pasted pictures are recognised through the bubble pass (a region the
    # colour test calls "incoming" but which starts far from the left edge).
    # The free-standing region scan proved too eager on long text bubbles, so it
    # stays out of the pipeline.
    pic_used: set[int] = set()
    picture_items: list[dict] = []
    picture_rects: list[tuple[int, int, int, int]] = []

    # ---- bubbles ----------------------------------------------------------
    in_mask = mask_for(arr, IN_FILL, 6)
    out_mask = mask_for(arr, OUT_FILL, args.green_tol)
    in_mask[:, :pane_x0] = False
    out_mask[:, :pane_x0] = False
    # The header divider and the composer sit right at the edges of the message
    # band and are the same tone as an incoming bubble; left in, they weld
    # themselves to the topmost/bottommost bubble and stretch it across the
    # whole pane.
    in_mask[:msg_top + 1] = False
    out_mask[:msg_top + 1] = False
    in_mask[msg_bottom:] = False
    out_mask[msg_bottom:] = False
    for x, y, w, h in picture_rects:
        in_mask[y:y + h, x:x + w] = False
        out_mask[y:y + h, x:x + w] = False

    min_area = int(max(900, pane_w * (msg_bottom - msg_top) * 0.0004))
    bubbles = []
    for side, mask in (("in", in_mask), ("out", out_mask)):
        for comp in components(mask, min_area=min_area):
            if comp["w"] < 22 or comp["h"] < 16:
                continue
            if comp["x"] < layout["avatar_l"] and comp["w"] < pane_w * 0.13:
                continue                      # incoming avatar thumbnail
            # keep partially clipped bubbles: one may run right up to the
            # composer border, or be cut by the window edge while scrolling
            top = max(comp["y"], msg_top)
            bottom = min(comp["y"] + comp["h"], msg_bottom)
            if bottom - top < comp["h"] * 0.35:
                continue
            comp["y"], comp["h"] = top, bottom - top
            comp["side"] = side
            bubbles.append(comp)
    bubbles.sort(key=lambda b: b["y"])

    used: set[int] = set()
    for b in bubbles:
        pad = 6
        own_idx = [i for i, ln in enumerate(lines) if i not in used
                   and b["x"] - pad <= ln["cx"] <= b["x"] + b["w"] + pad
                   and b["y"] - pad <= ln["cy"] <= b["y"] + b["h"] + pad]
        used.update(own_idx)
        own = reading_order([lines[i] for i in own_idx])
        b["lines"] = own
        fill = OUT_FILL if b["side"] == "out" else IN_FILL
        picture_shaped = (b["x"] >= pane_x0 + 60 and b["w"] >= 180 and b["h"] >= 110
                          and b["w"] <= 3.4 * b["h"])
        # An incoming bubble always hugs the left edge (~130px). A region the
        # colour test calls "incoming" but which starts far from that edge is a
        # pasted picture of the other side's chat, sent by the user. Our own
        # bubbles are right-aligned by design and must not match this rule.
        far_from_left = b["side"] == "in" and (b["x"] - pane_x0) > 250
        own_fill = fill_ratio(arr, b, fill)
        green_frac = fill_ratio(arr, b, OUT_FILL, tol=12)
        clipped = b["y"] <= msg_top + 8 or b["y"] + b["h"] >= msg_bottom - 8
        # Colour first: a message is painted almost entirely in its bubble
        # colour, a pasted picture is not — even when the picture itself shows
        # WeChat green. Geometry only breaks ties. Clipped bubbles are always
        # left as messages; the neighbouring screen shows them in full.
        if (picture_shaped and not clipped
                and max(own_fill, green_frac) < 0.45
                and (far_from_left or looks_like_image(arr, b, own, fill))):
            # a pasted screenshot: keep it as a picture, drop the text inside it
            b["as_image"] = True
            b["lines"] = []
            b["paragraphs"] = []
        else:
            emojis = detect_emojis(arr, b)
            if emojis:
                inject_emoji_tokens(b, emojis)
            b["paragraphs"] = group_paragraphs(own)
            if not b["paragraphs"] and emojis:
                # an emoji-only bubble has no OCR text to hang the token on
                b["paragraphs"] = [EMOJI_TOKEN * len(emojis)]

    # the pictures' inner text is already claimed, so it cannot leak further
    used |= pic_used

    # ---- centred lines: timestamps and system notices ---------------------
    # A quoted message is drawn in a smaller grey font, left-aligned with the
    # bubble and hanging directly under the message that quotes it (measured gap
    # ~16px above vs ~41px below), so it is attached to the bubble above.
    mid_x = pane_x0 + (layout["pane_right"] - pane_x0) / 2.0
    for bx, by0, by1 in find_quote_bars(arr, pane_x0, layout["pane_right"],
                                        msg_top, msg_bottom):
        # the quote belongs to the message just above it
        host = None
        for b in bubbles:
            bottom = b["y"] + b["h"]
            if bottom > by0 + 12 or by0 - bottom > 95:
                continue
            if host is None or bottom > host["y"] + host["h"]:
                host = b
        if host is None:
            continue
        to_right = bx < mid_x
        qlines = [i for i, ln in enumerate(lines)
                  if i not in used and len(ln["text"]) >= 1
                  and by0 - 10 <= ln["cy"] <= by1 + 12
                  and ln["cx"] >= pane_x0 + 40        # never the chat list
                  and ((ln["cx"] > bx + 2) if to_right else (ln["cx"] < bx - 2))]
        if not qlines:
            continue
        ordered = reading_order([lines[i] for i in qlines])
        qlines = [next(j for j in qlines if lines[j] is ln) for ln in ordered]
        host["quote"] = " ".join(lines[i]["text"] for i in qlines).strip()
        host["_quote_line_idx"] = list(qlines)
        used.update(qlines)
        host["_quote_rect"] = [bx - 6 if to_right else min(lines[i]["x"] for i in qlines) - 6,
                               int(by0) - 8,
                               (layout["pane_right"] - bx if to_right
                                else bx - min(lines[i]["x"] for i in qlines)) + 12,
                               int(by1 - by0) + 16]

    # Quotes are never emoji-scanned: a quote reproduces a real message, and the
    # renderer copies that message's own text and emoji instead.

    pane_mid = pane_x0 + pane_w / 2
    centred = []
    for i, ln in enumerate(lines):
        if i in used or not (msg_top < ln["cy"] < msg_bottom):
            continue
        # timestamp separators and system notices are short and centred; bubble
        # text lines are wide, so a width limit keeps them out of this bucket
        if (abs(ln["cx"] - pane_mid) < pane_w * 0.10
                and ln["w"] < pane_w * 0.55
                and ln["conf"] >= 0.3):
            centred.append((i, ln))
    for i, ln in centred:
        text = re.sub(r"\s+", " ", ln["text"]).strip()
        if TIME_RE.match(text):
            ln["kind"] = "timestamp"
        elif len(text) <= 60:
            ln["kind"] = "system"
        else:
            continue
        used.add(i)
        ln["text"] = text
        ln["rect"] = [round(ln["x"]), round(ln["y"]),
                      round(ln["w"]), round(ln["h"])]

    # ---- media blocks -----------------------------------------------------
    diff = np.abs(arr.astype(np.int16) - np.array(BG, dtype=np.int16)).max(axis=2)
    content = diff > 12
    content[:, :pane_x0 + 8] = False
    content[:, layout["pane_right"] - 6:] = False
    content[:msg_top] = False
    content[msg_bottom:] = False
    for b in bubbles:
        content[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]] = False

    media = [c for c in components(content, min_area=int(max(2500, pane_w * 4)))
             if c["w"] > 40 and c["h"] > 30]
    # A pasted picture is never painted in the bubble colour; a block that is
    # belongs to a text bubble (usually one clipped by the window edge).
    def mostly_bubble(block: dict) -> bool:
        return max(fill_ratio(arr, block, OUT_FILL, tol=12),
                   fill_ratio(arr, block, IN_FILL, tol=6)) >= 0.45

    media = [m for m in media if not mostly_bubble(m)]
    # drop avatar thumbnails: small blocks hugging either side of the pane
    media = [m for m in media
             if not (m["x"] < layout["avatar_l"] and m["w"] < pane_w * 0.14)
             and not (m["x"] + m["w"] > layout["pane_right"] - pane_w * 0.12
                      and m["w"] < pane_w * 0.14)]
    media = merge_media(media)
    for m in media:
        own = [ln for i, ln in enumerate(lines) if i not in used
               and m["x"] - 4 <= ln["cx"] <= m["x"] + m["w"] + 4
               and m["y"] - 4 <= ln["cy"] <= m["y"] + m["h"] + 4]
        for ln in own:
            used.add(lines.index(ln))
        own = reading_order(own)
        m["lines"] = own
        text = " / ".join(ln["text"] for ln in own)
        m["is_image"] = is_image_block(arr, m)
        if m["is_image"]:
            m["label"] = "图片/表情"
        elif not own:
            m["label"] = "图片/表情"
        elif FILE_HINT.search(text):
            m["label"] = "文件"
        elif LINK_HINT.search(text):
            m["label"] = "链接/卡片"
        elif QUOTE_HINT.search(text):
            m["label"] = "引用消息"
        else:
            m["label"] = "其他内容"
        m["side"] = "out" if (m["x"] + m["w"] / 2) > pane_mid else "in"

    # A date printed *inside* a shared picture is not a chat separator. If a
    # classified separator sits on top of a media block, hand it back.
    media_rects = [(m["x"], m["y"], m["x"] + m["w"], m["y"] + m["h"]) for m in media]
    kept_centred = []
    for i, ln in centred:
        r = ln.get("rect")
        host = None
        if r:
            for m, (a, b, c, d) in zip(media, media_rects):
                if (r[0] >= a - 4 and r[0] + r[2] <= c + 4
                        and r[1] >= b - 4 and r[1] + r[3] <= d + 4):
                    host = m
                    break
        if host is not None:
            used.discard(i)
            host["lines"].append(ln)
            host["lines"] = reading_order(host["lines"])
            host["text"] = " / ".join(l["text"] for l in host["lines"])
            continue
        kept_centred.append((i, ln))
    centred = kept_centred

    # ---- timeline ---------------------------------------------------------
    items = []
    items.extend(picture_items)
    for b in bubbles:
        rect = [b["x"], b["y"], b["w"], b["h"]]
        clipped_top = b["y"] <= msg_top + 8
        clipped_bottom = b["y"] + b["h"] >= msg_bottom - 8
        if b.get("as_image"):
            # colour is unreliable for a pasted picture (its own background can
            # match the other side's bubble grey), so use the geometry instead
            mid = pane_x0 + (layout["pane_right"] - pane_x0) / 2.0
            side = "out" if (b["x"] + b["w"] / 2.0) > mid else "in"
            items.append({"kind": "media", "side": side, "y": b["y"],
                          "rect": rect, "label": "图片", "is_image": True,
                          "text": "", "paragraphs": [],
                          "clipped_top": clipped_top,
                          "clipped_bottom": clipped_bottom})
            continue
        items.append({"kind": "message", "side": b["side"], "y": b["y"],
                      "rect": rect,
                      "quote": b.get("quote", ""),
                      "quote_emojis": b.get("quote_emojis", []),
                      "emojis": b.get("emojis", []),
                      "clipped_top": clipped_top,
                      "clipped_bottom": clipped_bottom,
                      "paragraphs": b["paragraphs"],
                      "conf_min": min([l["conf"] for l in b["lines"]], default=1.0)})
    for m in media:
        items.append({"kind": "media", "side": m["side"], "y": m["y"],
                      "rect": [m["x"], m["y"], m["w"], m["h"]],
                      "clipped_top": m["y"] <= msg_top + 8,
                      "clipped_bottom": m["y"] + m["h"] >= msg_bottom - 8,
                      "label": m["label"],
                      "is_image": m["is_image"],
                      "text": " / ".join(l["text"] for l in m["lines"]),
                      "paragraphs": group_paragraphs(m["lines"])})
    for i, ln in centred:
        items.append({"kind": ln["kind"], "y": ln["y"], "text": ln["text"],
                      "conf": ln["conf"]})
    items.sort(key=lambda it: it["y"])
    items = drop_fragments(items, pane_x0 + (layout["pane_right"] - pane_x0) / 2.0)

    result = {
        "shot": str(shot), "size": [w, h], "layout": layout, "items": items,
        "unassigned": [
            {"text": ln["text"], "conf": ln["conf"], "y": round(ln["cy"]),
             "x": round(ln["cx"])}
            for i, ln in enumerate(lines) if i not in used
        ],
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    kinds = {k: sum(1 for i in items if i["kind"] == k)
             for k in ("message", "media", "timestamp", "system")}
    print(f"{shot.name}: pane_x0={pane_x0} band=[{msg_top},{msg_bottom}] "
          f"messages={kinds['message']} media={kinds['media']} "
          f"timestamps={kinds['timestamp']} system={kinds['system']} "
          f"unassigned={len(result['unassigned'])}")
    for it in items:
        if it["kind"] == "message":
            body = " ⏎ ".join(it["paragraphs"])[:70]
            print(f"   [{it['side']:>3}] y={it['y']:>5} {body}")
        elif it["kind"] == "media":
            print(f"   [{it['side']:>3}] y={it['y']:>5} <{it['label']}> {it['text'][:50]}")
        else:
            print(f"   [ {it['kind'][:4]}] y={it['y']:>5} {it['text']}")
    return 0


def drop_fragments(items: list[dict], mid: float = 0.0) -> list[dict]:
    """Clean up detection debris around pasted pictures.

    A tall picture can break into a main body plus a thin leftover (its title
    bar, say), and a view-edge sliver can show up as a tiny bubble. A much
    smaller neighbour that touches a picture is folded back into it; slivers
    inside a picture and empty stubs are dropped.
    """
    media = [it for it in items if it["kind"] == "media"]
    # A quote is sometimes detected as a small bubble of its own. Its text always
    # opens with "name：", which a real message body does not, so fold such items
    # back into the quote of the message above.
    folded: list[dict] = []
    for it in items:
        if it["kind"] == "message":
            text = "".join(it.get("paragraphs") or []).strip()
            head = text.split("：", 1)[0]
            if "：" in text and 1 <= len(head) <= 12 and folded and \
                    folded[-1]["kind"] == "message":
                prev = folded[-1]
                prev["quote"] = (prev.get("quote") + " " + text).strip() \
                    if prev.get("quote") else text
                continue
        folded.append(it)
    items = folded
    media = [it for it in items if it["kind"] == "media"]

    def area(r):
        return r[2] * r[3]

    changed = True
    while changed:
        changed = False
        for small in sorted(media, key=lambda m: area(m["rect"])):
            for big in media:
                if big is small:
                    continue
                sr, br = small["rect"], big["rect"]
                if area(sr) * 4 > area(br):
                    continue                     # comparable size: a real picture
                overlap = min(sr[0] + sr[2], br[0] + br[2]) - max(sr[0], br[0])
                if overlap < 0.6 * min(sr[2], br[2]):
                    continue                     # must overlap horizontally
                gap = max(br[1], sr[1]) - min(br[1] + br[3], sr[1] + sr[3])
                if gap < -4 or gap > 40:
                    continue
                nx, ny = min(sr[0], br[0]), min(sr[1], br[1])
                big["rect"] = [nx, ny,
                               max(sr[0] + sr[2], br[0] + br[2]) - nx,
                               max(sr[1] + sr[3], br[1] + br[3]) - ny]
                media.remove(small)
                changed = True
                break
            if changed:
                break

    # A picture always hugs one side of the pane, so re-derive the side from the
    # merged rectangle; colour is useless here because a screenshot's own
    # background can match the other side's bubble grey.
    for m in media:
        x, y, w, h = m["rect"]
        m["y"] = y
        m["side"] = "out" if (x + w / 2.0) > mid else "in"

    keep: list[dict] = []
    for it in items:
        x, y, w, h = it.get("rect") or [0, 0, 0, 0]
        if it["kind"] != "media" and w and h:
            # Nothing sits on top of a picture in WeChat: a "message" overlapping
            # one is the picture's own content leaking into the bubble detector.
            for m in media:
                mx, my, mw, mh = m["rect"]
                ox = min(x + w, mx + mw) - max(x, mx)
                oy = min(y + h, my + mh) - max(y, my)
                if ox > 0 and oy > 0 and ox * oy > 0.2 * w * h:
                    it = None
                    break
            if it is None:
                continue
        if it["kind"] == "message":
            inside = any(x >= ox - 2 and y >= oy - 2 and x + w <= ox + ow + 2
                         and y + h <= oy + oh + 2
                         for ox, oy, ow, oh in (m["rect"] for m in media))
            empty = not "".join(it.get("paragraphs") or []).strip()
            if inside or (empty and not it.get("emojis") and h < 30):
                continue
            # A real bubble is at least one text line tall; a shorter sliver is
            # a bubble cut by the view edge, and its half-row reads as garbage.
            if h and h < 50:
                continue
            keep.append(it)
            continue
        if it["kind"] == "media":
            if it not in media:
                continue                         # folded into its bigger neighbour
            x, y, w, h = it["rect"]
            if any(other is not it
                   and x >= other["rect"][0] - 2 and y >= other["rect"][1] - 2
                   and x + w <= other["rect"][0] + other["rect"][2] + 2
                   and y + h <= other["rect"][1] + other["rect"][3] + 2
                   and area(it["rect"]) < 0.7 * area(other["rect"])
                   for other in media):
                continue
            keep.append(it)
            continue
        keep.append(it)
    return keep


if __name__ == "__main__":
    raise SystemExit(main())
