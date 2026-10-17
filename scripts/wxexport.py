#!/usr/bin/env python3
"""WeChat conversation capture + assembly + rendering pipeline.

Sub-commands
  capture    scroll the WeChat window screen by screen, saving crop + OCR + parse
  assemble   merge per-screen parses into one chronological message list
  render     write Markdown / HTML / cropped images

Everything runs locally; no network access is used anywhere in this file.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wxparse  # noqa: E402
import wincrop  # noqa: E402
import emoji_match  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
SHOTS, OCR, BUILD, IMAGES = HERE / "shots", HERE / "ocr", HERE / "build", HERE / "images"
TOOLS = HERE / "tools"
OUT = HERE / "outputs"
def _ocr_bin() -> Path:
    """Vision helper: $WXEXPORT_OCR, then the built one, then PATH."""
    for cand in (os.environ.get("WXEXPORT_OCR"), TOOLS / "ocrfast"):
        if cand and Path(cand).is_file():
            return Path(cand)
    found = shutil.which("ocrfast")
    return Path(found) if found else TOOLS / "ocrfast"


OCRBIN = _ocr_bin()
PEER = "对方"                  # set from --peer
ME = "我"
ME_ALIAS = "我"               # how your own account appears inside quotes


# --------------------------------------------------------------------------- io
def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    if "stdout" not in kw:
        kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    return subprocess.run(cmd, **kw)


def find_wechat_window() -> dict:
    res = sh([str(TOOLS / "wxwin"), "--json"])
    windows = json.loads(res.stdout)
    best = None
    for w in windows:
        if w.get("owner") in ("微信", "WeChat") and w["w"] > 300 and w["h"] > 300:
            if best is None or w["w"] * w["h"] > best["w"] * best["h"]:
                best = w
    if best is None:
        raise SystemExit("找不到微信窗口：请先打开微信并进入目标会话")
    return best


def capture_screen(win_id: int, index: int, tag: str = "a") -> tuple[Path, Path]:
    raw = SHOTS / f"raw_{tag}{index:03d}.png"
    shot = SHOTS / f"c{tag}{index:03d}.png"
    sh(["screencapture", "-x", "-l", str(win_id), str(raw)], check=True)
    wincrop.crop(raw, shot)
    # Every window size is normalised to one canonical canvas, so the layout
    # constants and the OCR coordinate space stay identical everywhere.
    img = Image.open(shot)
    if img.size != SHOT_SIZE:
        img.resize(SHOT_SIZE, Image.LANCZOS).save(shot)
    return raw, shot


SHOT_SIZE = (1632, 1948)          # every capture is normalised to this by wincrop


def run_ocr(shot: Path, index: int, tag: str = "a",
            dst: Path | None = None) -> tuple[Path, tuple[int, int], tuple[int, int]]:
    """OCR the whole normalised window, so OCR boxes and pixels share one space.

    Cropping to the message pane used to save a little time but produced a
    second coordinate system, and every downstream stage had to remember which
    one it was looking at. One space, no offsets: OCR coordinates are window
    coordinates, always. Text outside the pane (chat list, title bar) is dropped
    later by geometry.
    """
    dst = dst or OCR / f"c{tag}{index:03d}.json"
    if Image.open(shot).size != SHOT_SIZE:
        raise SystemExit(f"截图尺寸不是 {SHOT_SIZE}: {shot} {Image.open(shot).size}")
    with dst.open("w", encoding="utf-8") as fh:
        sh([str(OCRBIN), "--json", str(shot)], stdout=fh, check=True)
    return dst, (0, 0), SHOT_SIZE


def parse(shot: Path, ocr: Path, index: int, band: tuple[int, int] | None = None,
          ocr_offset: tuple[int, int] = (0, 0),
          ocr_size: tuple[int, int] = (0, 0)) -> dict:
    dst = BUILD / f"c{index:03d}.parsed.json"
    cmd = [sys.executable, str(HERE / "scripts/wxparse.py"),
           "--shot", str(shot), "--ocr", str(ocr), "--out", str(dst)]
    if band:
        cmd += ["--pane-top", str(band[0]), "--pane-bottom", str(band[1])]
    if ocr_offset != (0, 0):
        cmd += ["--ocr-offset-x", str(ocr_offset[0]),
                "--ocr-offset-y", str(ocr_offset[1])]
    if ocr_size != (0, 0):
        cmd += ["--ocr-size", str(ocr_size[0]), str(ocr_size[1])]
    res = sh(cmd)
    if res.returncode != 0:
        raise SystemExit(f"解析失败 {shot}:\n{res.stderr[-2000:]}")
    return json.loads(dst.read_text(encoding="utf-8"))


# ------------------------------------------------------------ scroll geometry
def row_profile(shot: Path) -> np.ndarray:
    """Fraction of non-background pixels per row — a scroll fingerprint."""
    arr = wxparse.load_rgb(shot)
    h, w, _ = arr.shape
    core = arr[:, int(w * 0.42):int(w * 0.95)].astype(np.int16)
    diff = np.abs(core - np.array(wxparse.BG, dtype=np.int16)).max(axis=2)
    return (diff > 12).mean(axis=1).astype(np.float32)


def estimate_shift(prev: np.ndarray, cur: np.ndarray,
                   msg_top: int, msg_bottom: int) -> tuple[int | None, float]:
    """Pixels the content moved down between two screens (i.e. scroll amount)."""
    n = len(cur)

    def cost(d: int) -> float | None:
        y0, y1 = msg_top + d, min(msg_bottom, n)
        if y1 - y0 < 250:
            return None
        return float(np.abs(cur[y0:y1] - prev[y0 - d:y1 - d]).mean())

    best, best_cost = None, None
    for d in list(range(0, 40, 2)) + list(range(120, 1600, 4)):
        c = cost(d)
        if c is not None and (best_cost is None or c < best_cost):
            best, best_cost = d, c
    if best is None:
        return None, 1.0
    for d in range(max(1, best - 5), best + 6):
        c = cost(d)
        if c is not None and c < best_cost:
            best, best_cost = d, c
    return best, best_cost


def anchor_shift(prev_screen: dict, cur_screen: dict,
                 min_len: int = 6, tol: int = 8) -> int | None:
    """Scroll offset measured from messages that appear identically in both.

    Only exact, long-enough text is used, and at least two anchors must agree on
    the same offset; fuzzy matches would happily pair two different "好的"s and
    skew the whole seam.
    """
    deltas: list[int] = []
    prev_items = [it for it in prev_screen["items"] if len(payload(it)) >= min_len]
    for a in cur_screen["items"]:
        pa = payload(a)
        if len(pa) < min_len:
            continue
        for b in prev_items:
            if a.get("side") == b.get("side") and payload(b) == pa:
                deltas.append(int(a["y"] - b["y"]))
                break
    if not deltas:
        return None
    if len(deltas) >= 2:
        median = float(np.median(deltas))
        if max(abs(d - median) for d in deltas) <= tol:
            return int(round(median))
        return None
    return deltas[0]


def annotate_segments(screens: list[dict], profiles: list[np.ndarray],
                      reached_top: bool) -> None:
    """Work out, per screen, which items are new (not fully shown before).

    This is a geometric decision based on the measured scroll offset, so a
    sticker with no OCR text cannot derail the ordering the way text matching
    could. It also makes duplicates impossible: an item is taken from the first
    screen that shows it *in full*.
    """
    for i, screen in enumerate(screens):
        mt = screen["layout"]["msg_top"]
        mb = screen["layout"]["msg_bottom"]
        layout_shift = None
        if i > 0 and profiles[i] is not None and profiles[i - 1] is not None:
            layout_shift, _ = estimate_shift(profiles[i - 1], profiles[i], mt, mb)
            screen["_profile_shift"] = layout_shift
            # identical messages are a much sharper ruler than the row profile,
            # so prefer them whenever at least two agree
            anchor = anchor_shift(screens[i - 1], screen)
            screen["_anchor_shift"] = anchor
            screen["_raw_shift"] = anchor if anchor is not None else layout_shift

    for screen in screens:
        screen["shift"] = screen.get("_raw_shift")

    for i, screen in enumerate(screens):
        mt = screen["layout"]["msg_top"]
        mb = screen["layout"]["msg_bottom"]
        layout_shift = screen.get("shift")
        screen["shift"] = layout_shift

        is_last = i == len(screens) - 1
        prev_items = screens[i - 1]["items"] if i > 0 else []
        keep: list[dict] = []
        for it in screen["items"]:
            rect0 = it.get("rect") or [0, it.get("y", 0), 0, it.get("h", 0)]
            tall = (rect0[3] if len(rect0) > 3 else 0) >= 0.55 * (mb - mt)
            if it.get("clipped_top") and not (is_last and reached_top) and not tall:
                continue                       # partial: a later screen has it whole
            rect = it.get("rect") or [0, it.get("y", 0), 0, it.get("h", 0)]
            item_h = rect[3] if len(rect) > 3 else 0
            if layout_shift is not None:
                y_prev = it["y"] - layout_shift
                if y_prev >= mt + 12 and y_prev + item_h <= mb - 12:
                    # Geometry alone can mis-judge when the offset estimate is
                    # off, so confirm the item really exists on the previous
                    # screen at that position before dropping it.
                    mine = payload(it)
                    for other in prev_items:
                        if other["kind"] != it["kind"]:
                            continue
                        if it["kind"] == "media":
                            # A pasted picture can be split differently on each
                            # screen and its colour-derived side can flip, so for
                            # pictures match on position alone.
                            pass
                        elif other.get("side") != it.get("side"):
                            continue
                        if abs(other["y"] - y_prev) > 30:
                            continue
                        if it["kind"] != "media":
                            theirs = payload(other)
                            if mine and theirs and mine != theirs:
                                continue       # same spot, different text => new
                            if (it.get("quote") or "") and not (other.get("quote") or ""):
                                # the same message, but this screen also caught
                                # the quote drawn underneath it
                                screens[i - 1].setdefault("_replace_rects", []).append(
                                    list(other.get("rect") or []))
                                y_prev = None
                                break
                        else:
                            # same picture seen again: prefer the fuller view
                            cur_area = rect[2] * rect[3]
                            oth_area = (other.get("rect") or [0, 0, 0, 0])[2] * \
                                (other.get("rect") or [0, 0, 0, 0])[3]
                            if cur_area > oth_area * 1.15:
                                screens[i - 1].setdefault("_replace_rects", []).append(
                                    list(other.get("rect") or []))
                            else:
                                y_prev = None
                                break
                        y_prev = None          # confirmed already captured
                        break
                    if y_prev is None:
                        continue
            keep.append(it)
        screen["new_items"] = list(reversed(keep))   # newest -> oldest

    stitch_partial_views(screens)


def item_text(item: dict) -> str:
    if item["kind"] == "message":
        return "".join(item.get("paragraphs") or [])
    return item.get("text", "")


def clock_of(text: str) -> str:
    """Time part of a separator label, e.g. '9月28日 12:17' -> '12:17'."""
    m = re.search(r"(\d{1,2}:\d{2})", text or "")
    return m.group(1) if m else ""


EMOJI_RE = re.compile("\ue000")


def split_emoji(text: str) -> list[tuple[str, str]]:
    """Split a text into ('text', s) / ('emoji', '') pieces, in order."""
    out: list[tuple[str, str]] = []
    pos = 0
    for m in EMOJI_RE.finditer(text):
        if m.start() > pos:
            out.append(("text", text[pos:m.start()]))
        out.append(("emoji", ""))
        pos = m.end()
    if pos < len(text):
        out.append(("text", text[pos:]))
    return out


def drop_repeated_stamps(items: list[dict]) -> list[dict]:
    """Remove a time separator picked up twice at a screen boundary."""
    out: list[dict] = []
    for it in items:
        if it["kind"] == "timestamp":
            window = out[-8:]
            for j, other in enumerate(window):
                if other["kind"] != "timestamp" or payload(other) != payload(it):
                    continue
                # WeChat prints a separator once; an identical one a few items
                # later with no different separator in between is a seam repeat
                between = [x for x in window[j + 1:] if x["kind"] == "timestamp"]
                if not between:
                    it = None
                break
            if it is None:
                continue
        out.append(it)
    return out


def collapse_same_geometry(items: list[dict]) -> list[dict]:
    """Drop the second copy of an item that two screens both contributed.

    With the edge margin widened, an item can now arrive from two consecutive
    screens; for pictures (no text to compare) the geometry is the only handle.
    """
    out: list[dict] = []
    for it in items:
        rect = it.get("rect") or [0, 0, 0, 0]
        dup = False
        for prev in out[-3:]:
            if prev["kind"] != it["kind"] or prev.get("side") != it.get("side"):
                continue
            pr = prev.get("rect") or [0, 0, 0, 0]
            if (abs(pr[0] - rect[0]) <= 14 and abs(pr[1] - rect[1]) <= 14
                    and abs(pr[2] - rect[2]) <= 0.3 * max(pr[2], rect[2])
                    and abs(pr[3] - rect[3]) <= 0.3 * max(pr[3], rect[3])
                    and not payload(it)):
                dup = True
                break
        if not dup:
            out.append(it)
    return out


def collapse_same_text(items: list[dict]) -> list[dict]:
    """Drop a text item that the widened edge margin let in twice."""
    out: list[dict] = []
    for it in items:
        rect = it.get("rect") or [0, 0, 0, 0]
        dup = False
        for prev in out[-2:]:
            pr = prev.get("rect") or [0, 0, 0, 0]
            if (prev["kind"] == it["kind"] and prev.get("side") == it.get("side")
                    and payload(prev) and payload(prev) == payload(it)
                    and abs(pr[1] - rect[1]) <= 24
                    and abs(pr[0] - rect[0]) <= 24):
                dup = True
                break
            # the same message captured twice usually differs by a character or
            # two of OCR, so fall back to a high similarity check
            pa, pb = payload(prev), payload(it)
            if (prev["kind"] == it["kind"] == "message"
                    and prev.get("side") == it.get("side")
                    and len(pa) >= 6 and len(pb) >= 6):
                ratio = difflib.SequenceMatcher(None, pa, pb, autojunk=False).ratio()
                # "新年快乐鸭米米" vs "新年快乐鸭养米": same message, one OCR
                # character apart, so also accept a shared 4-char prefix
                same_head = pa[:4] == pb[:4] and abs(len(pa) - len(pb)) <= 2
                if ratio >= 0.85 or same_head:
                    if len(pb) > len(pa):
                        out[-1] = it
                    dup = True
                    break
        if not dup:
            # two screens can both contribute the same bubble: identical size,
            # same speaker and near-identical text is the same message
            for prev in out[-6:]:
                if prev["kind"] != "message" or prev.get("side") != it.get("side"):
                    continue
                pr = prev.get("rect") or [0, 0, 0, 0]
                pa, pb = payload(prev), payload(it)
                if (abs(pr[2] - rect[2]) <= 8 and abs(pr[3] - rect[3]) <= 8
                        and len(pa) >= 6 and len(pb) >= 6):
                    ratio = difflib.SequenceMatcher(None, pa, pb, autojunk=False).ratio()
                    if ratio >= 0.7:
                        if len(pb) > len(pa):
                            out[out.index(prev)] = it
                        dup = True
                        break
        if not dup:
            out.append(it)
    return out


def union_views(items: list[dict]) -> list[dict]:
    """Combine the two halves of a message taller than the window.

    The geometric merge takes each message from the screen that shows it whole,
    but an extra-long message is never shown whole. Its top half and bottom half
    therefore arrive as two items: join them, keeping both halves' paragraph
    breaks.
    """
    out: list[dict] = []
    for it in items:
        if it["kind"] != "message":
            out.append(it)
            continue
        for back in range(1, min(4, len(out)) + 1):
            prev = out[-back]
            if not items_match(prev, it):
                continue
            paras = union_paragraphs(prev.get("paragraphs") or [],
                                     it.get("paragraphs") or [])
            total = len("".join(paras))
            if total <= max(len(payload(prev)), len(payload(it))):
                continue
            merged = dict(prev)
            merged["paragraphs"] = paras
            merged["clipped_top"] = bool(prev.get("clipped_top")) and bool(it.get("clipped_top"))
            merged["clipped_bottom"] = bool(prev.get("clipped_bottom")) and bool(it.get("clipped_bottom"))
            out[-back] = merged
            break
        else:
            out.append(it)
    return out


def repair_quotes(items: list[dict]) -> None:
    """Replace a quote's OCR text with the message it is quoting.

    A quote reproduces a real message, so that message's own text is the cleanest
    source. Matching stays conservative: same speaker, only messages just above,
    a prefix/substring match first, and otherwise a high fuzzy threshold.
    Anything unconvincing is dropped rather than guessed.
    """
    global ME_ALIAS
    if not ME_ALIAS:
        # Your own display name is whatever prefixes the quotes but is not the
        # peer's name — usually the most common one.
        tally: dict[str, int] = {}
        for it in items:
            prefix = (it.get("quote") or "").partition("：")[0].strip()
            if prefix and prefix not in (PEER, ME):
                tally[prefix] = tally.get(prefix, 0) + 1
        if tally:
            ME_ALIAS = max(tally, key=tally.get)
    for k, it in enumerate(items):
        if it["kind"] != "message":
            continue
        quote = (it.get("quote") or "").strip()
        if not quote:
            it["quote"] = ""
            continue
        prefix, sep, body = quote.partition("：")
        body = EMOJI_RE.sub("", body).strip()
        if len(body) < 5:
            it["quote"] = ""
            continue
        speaker = "out" if ME_ALIAS and ME_ALIAS in prefix else "in"
        best, best_r = None, 0.0
        # `items` is newest -> oldest here, so the messages shown above this one
        # are at *higher* indices
        for j in range(k + 1, min(len(items), k + 150)):
            other = items[j]
            if other["kind"] != "message" or other.get("side") != speaker:
                continue
            text = EMOJI_RE.sub("", payload(other)).strip()
            if len(text) < 5:
                continue
            if text.startswith(body) or body in text:
                best, best_r = text, 1.0
                break
            ratio = difflib.SequenceMatcher(None, body, text, autojunk=False).ratio()
            if ratio >= 0.8 and ratio > best_r:
                best, best_r = text, ratio
        if best is None:
            # no confident match: keep what the bar-anchored OCR read
            continue
        name = ME_ALIAS if speaker == "out" else PEER
        it["quote"] = f"{name}：{best}" if sep else best
        # copy the quoted message's own emoji so the quote shows them too
        src = next((o for o in items[k + 1:k + 150]
                    if o["kind"] == "message"
                    and EMOJI_RE.sub("", payload(o)).strip() == EMOJI_RE.sub("", best).strip()),
                   None)
        if src is not None and src.get("emojis"):
            it["quote_emojis"] = src["emojis"]
            it["quote_shot"] = src.get("shot")
def set_item_text(item: dict, text: str) -> None:
    if item["kind"] == "message":
        item["paragraphs"] = [text]
    else:
        item["text"] = text


def stitch_partial_views(screens: list[dict]) -> None:
    """Join a clipped bubble with the other half from the neighbouring screen.

    A message taller than the overlap between two screens is never fully visible
    anywhere, so its top half comes from one screen and its bottom half from
    another; the geometric offset tells us exactly where to look.
    """
    for k, screen in enumerate(screens):
        for it in screen["new_items"]:
            # Quotes stay with the screen that produced them: borrowing the
            # neighbour's quote here silently attached other messages' quotes.
            if not (it.get("clipped_top") or it.get("clipped_bottom")):
                continue
            text = item_text(it)
            if len(text) < 8:
                continue
            for other, target_y in neighbour_views(screens, k, it):
                cand = find_at(other, it, target_y, 26)
                if cand is None:
                    continue
                joined = union_paragraphs(it.get("paragraphs") or [],
                                          cand.get("paragraphs") or [])
                if joined and len("".join(joined)) > len(text):
                    it["paragraphs"] = joined
                    text = "".join(joined)


def neighbour_views(screens: list[dict], k: int, it: dict):
    """Yield (screen, expected y of `it` in that screen) for the adjacent screens."""
    if k > 0 and screens[k]["shift"] is not None:
        yield screens[k - 1], it["y"] - screens[k]["shift"]
    if k + 1 < len(screens) and screens[k + 1].get("shift") is not None:
        yield screens[k + 1], it["y"] + screens[k + 1]["shift"]


def find_at(screen: dict, it: dict, target_y: float, tol: float) -> dict | None:
    for o in screen["items"]:
        if o["kind"] != it["kind"]:
            continue
        if it.get("side") and o.get("side") and o["side"] != it["side"]:
            continue
        if abs(o["y"] - target_y) <= tol:
            return o
    return None


# ------------------------------------------------------------------- signatures
def norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def payload(item: dict) -> str:
    """The comparable text of an item (empty for an untagged sticker)."""
    if item["kind"] == "message":
        return norm("".join(item.get("paragraphs") or []))
    return norm(item.get("text", ""))


def sig(item: dict) -> str:
    return f"{item['kind']}|{item.get('side','')}|{payload(item)}"


def chrono(screen: dict) -> list[dict]:
    """Items of one screen ordered newest -> oldest (bottom of the window first)."""
    return sorted(screen["items"], key=lambda it: -it["y"])


# ---------------------------------------------------------------------- capture
def cmd_capture(args) -> int:
    sh(["open", "-a", "WeChat"])          # scroll events need WeChat in front
    time.sleep(1.2)
    win = find_wechat_window()
    win_x, win_y = float(win["x"]), float(win["y"])
    print(f"微信窗口 id={win['id']} bounds=({win_x:.0f},{win_y:.0f}) {win['w']}x{win['h']}")

    if args.from_bottom:
        # start from the newest message: drag the list all the way down
        cx = win_x + win["w"] * 0.7
        cy = win_y + win["h"] * 0.5
        print("先滚到最新消息处…")
        sh([str(TOOLS / "wxscroll"), f"{cx:.0f}", f"{cy:.0f}", "-40000",
            "--unit", "pixel"])
        time.sleep(2.5)

    screens: list[dict] = []
    profiles: list[np.ndarray | None] = []
    step_points = None
    cursor_x = cursor_y = 0.0
    index = 0
    retry = 0
    band: tuple[int, int] | None = None
    pending: list[tuple] = []          # (raw, shot, ocr) awaiting band calibration
    clock = datetime.now()             # newest resolved timestamp seen so far
    reached_top = False
    stop_before = (datetime.strptime(args.stop_before, "%Y-%m-%d")
                   if args.stop_before else None)

    while index < args.max_steps:
        raw, shot = capture_screen(win["id"], index, args.tag)
        ocr, used_off, used_size = run_ocr(shot, index, args.tag)
        screen = parse(shot, ocr, index, band, used_off, used_size)
        screen["shot"] = str(shot)
        screen["ocr"] = str(ocr)
        screen["ocr_offset"] = list(used_off)
        screen["ocr_size"] = list(used_size)
        for it in screen["items"]:
            it["shot"] = str(shot)

        if band is None and index == 2:
            # calibrate once: the window never moves during a session, so a
            # single band for every screen beats per-screen auto-detection
            prev_screens = list(screens)
            tops = [s["layout"]["msg_top"] for s in prev_screens + [screen]]
            botts = [s["layout"]["msg_bottom"] for s in prev_screens + [screen]]
            band = (int(np.median(tops)), int(np.median(botts)))
            print(f"固定消息区: top={band[0]} bottom={band[1]}")
            for i, s in enumerate(prev_screens):
                redo = parse(Path(s["shot"]), Path(s["ocr"]), i, band,
                             tuple(s.get("ocr_offset") or (0, 0)),
                             tuple(s.get("ocr_size") or (0, 0)))
                redo["shot"] = s["shot"]
                redo["ocr"] = s["ocr"]
                redo["ocr_offset"] = list(s.get("ocr_offset") or (0, 0))
                redo["ocr_size"] = list(s.get("ocr_size") or (0, 0))
                for it in redo["items"]:
                    it["shot"] = s["shot"]
                screens[i] = redo
            screen = parse(shot, ocr, index, band, used_off, used_size)
            screen["ocr_offset"] = list(used_off)
            screen["ocr_size"] = list(used_size)
            screen["ocr"] = str(ocr)
            for it in screen["items"]:
                it["shot"] = str(shot)

        lay = screen["layout"]
        band_top, band_bot = lay["msg_top"], lay["msg_bottom"]

        # advance the reverse clock to this screen's OLDEST separator; a
        # separator repeated in the overlap must not drag the year backwards
        stamps = [it for it in screen["items"] if it["kind"] == "timestamp"]
        resolved = [resolve_back(st["text"], clock) for st in stamps]
        # a separator that would jump the clock by months is really a date
        # printed inside a shared picture, not a chat separator
        resolved = [d for d in resolved
                    if d and (clock - d) <= timedelta(days=45)]
        if resolved and min(resolved) < clock:
            clock = min(resolved)
        reached = clock.date().isoformat() if stamps else ""
        if step_points is None:
            # content is Retina 2x, so image pixels / 2 = screen points
            step_points = max(80.0, (band_bot - band_top) * args.step_frac / 2.0)
            cursor_x = win_x + (lay["pane_x0"] + lay["pane_right"]) / 4.0
            cursor_y = win_y + (band_top + band_bot) / 4.0
            print(f"滚动步长 {step_points:.0f}pt，光标 ({cursor_x:.0f},{cursor_y:.0f})")

        if screens:
            prev_prof = profiles[-1] if profiles else None
            cur_prof = row_profile(shot)
            shift, _ = estimate_shift(prev_prof, cur_prof,
                                      screen["layout"]["msg_top"],
                                      screen["layout"]["msg_bottom"]) \
                if prev_prof is not None else (None, 1.0)
            band_h = screen["layout"]["msg_bottom"] - screen["layout"]["msg_top"]
            if shift is not None and shift < 25:
                print(f"[{index:03d}] 内容没有变化 —— 已到会话顶部，停止。")
                reached_top = True
                break
            if shift is not None and shift > band_h - 150 and retry < args.max_retry:
                retry += 1
                back = int(round(step_points * 0.55))
                step_points = max(200.0, step_points * 0.85)
                print(f"[{index:03d}] 滚动过头（重叠不足 {band_h - shift}px），"
                      f"回滚 {back}pt 重试 ({retry}/{args.max_retry})")
                sh([str(TOOLS / "wxscroll"), f"{cursor_x:.0f}", f"{cursor_y:.0f}",
                    str(-back), "--unit", "pixel"])
                time.sleep(args.wait + 0.3)
                continue

        screens.append(screen)
        profiles.append(row_profile(shot))
        print(f"[{index:03d}] items={len(screen['items'])} "
              f"到{reached or '?'} top={_short(screen['items'][0] if screen['items'] else None)}")

        index += 1
        retry = 0
        if stop_before and clock < stop_before:
            print(f"已翻到 {clock.date()}，早于切分点 {stop_before.date()}，停止。")
            break
        if index >= args.max_steps:
            break
        sh([str(TOOLS / "wxscroll"), f"{cursor_x:.0f}", f"{cursor_y:.0f}",
            str(int(round(step_points))), "--unit", "pixel"])
        time.sleep(args.wait)

    annotate_segments(screens, profiles, reached_top)
    (BUILD / f"screens_{args.tag}.json").write_text(
        json.dumps({"window": win, "screens": screens}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"已保存 {len(screens)} 屏到 build/screens_{args.tag}.json")
    return 0


def sigs_of(screen: dict) -> list[str]:
    return [sig(it) for it in screen["items"]]


def content_moved(prev: dict, cur: dict) -> bool:
    a, b = sigs_of(prev), sigs_of(cur)
    return a != b


def _short(item) -> str:
    if item is None:
        return "-"
    if item["kind"] == "message":
        return _clip(" ".join(item.get("paragraphs", [])))
    return _clip(item.get("text", item["kind"]))


def _clip(text: str, n: int = 28) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text[:n] + ("…" if len(text) > n else "")


# --------------------------------------------------------------------- assemble
def cmd_assemble(args) -> int:
    screens: list[dict] = []
    for tag in args.runs:
        path = BUILD / f"screens_{tag}.json"
        screens.extend(json.loads(path.read_text(encoding="utf-8"))["screens"])

    geometric = bool(screens) and all(s.get("new_items") is not None for s in screens)
    if not geometric and screens and all(Path(s["shot"]).is_file() for s in screens):
        # screens captured before the geometric pass existed: derive it now
        profiles = [row_profile(Path(s["shot"])) for s in screens]
        annotate_segments(screens, profiles, reached_top=False)
        geometric = True
    if geometric:
        # The capture step decided geometrically what each screen contributes,
        # so segments just concatenate: no text matching, no duplicate risk.
        merged: list[dict] = []
        for screen in screens:
            drop = [tuple(r) for r in screen.get("_replace_rects") or []]
            for it in screen["new_items"]:
                if drop and tuple(it.get("rect") or []) in drop:
                    continue                   # a fuller view replaced this one
                merged.append(dict(it))
        merged = drop_repeated_stamps(merged)
        merged = collapse_same_geometry(merged)
        merged = collapse_same_text(merged)
        merged = union_views(merged)
        repair_quotes(merged)
        print(f"几何合并：{len(screens)} 屏")
        merged.reverse()
        for it in merged:
            it.pop("y", None)
        (BUILD / "messages.json").write_text(
            json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
        counts = {}
        for it in merged:
            counts[it["kind"]] = counts.get(it["kind"], 0) + 1
        print(f"合并完成：{len(merged)} 项 {counts}")
        return 0

    merged = []
    for screen in screens:                      # capture order = newest first
        items = chrono(screen)
        # An item clipped by the top of the window is shown in full by the
        # previous screen, which was captured a moment earlier. `chrono` puts
        # the oldest item last, so drop from the tail.
        # The very first screen has no earlier capture to fall back on, so its
        # clipped tail stays: the next screen supplies the missing half.
        while merged and len(items) > 1 and items[-1].get("clipped_top"):
            items = items[:-1]
        if not merged:
            merged = [dict(it) for it in items]
            continue
        # The new screen's newest item was already seen near the tail of what we
        # have; everything from there on is re-observed *fully* (the tail of the
        # previous screen was clipped by the window top), so re-take it.
        idx, j = align(merged, items)
        if idx < 0:
            print(f"警告：{screen['shot']} 与已合并内容无重叠，可能漏内容")
            merged.extend(dict(it) for it in items)
        else:
            # Everything from the match onwards is re-observed by this screen.
            # Pair the two views up and union them: a message taller than the
            # window is clipped at a different offset on each screen, so the new
            # copy is often *not* a superset of the one we already had.
            merged = merged[:idx] + reconcile([dict(it) for it in merged[idx:]],
                                              [dict(it) for it in items[j:]])

    merged.reverse()                            # chronological, oldest first
    merged = collapse_adjacent(merged)
    for it in merged:
        it.pop("y", None)
    (BUILD / "messages.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    counts = {}
    for it in merged:
        counts[it["kind"]] = counts.get(it["kind"], 0) + 1
    print(f"合并完成：{len(merged)} 项 {counts}")
    return 0


def screen_ocr_geometry(shot: Path, ocr: Path) -> tuple[tuple[int, int], tuple[int, int]]:
    """OCR boxes are window coordinates — one space for every screen."""
    return (0, 0), Image.open(shot).size


def cmd_reparse(args) -> int:
    """Re-run the screen parser over already captured shots (after a parser fix)."""
    for tag in args.runs:
        path = BUILD / f"screens_{tag}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        screens = data["screens"]
        band = None
        for i, screen in enumerate(screens):
            shot = Path(screen["shot"])
            ocr = Path(screen.get("ocr") or (OCR / f"c{tag}{i:03d}.json"))
            if not shot.is_file():
                print(f"  {tag}{i:03d}: 缺少文件，跳过")
                continue
            if args.reocr or not ocr.is_file():
                run_ocr(shot, i, dst=ocr)
            off, size = screen_ocr_geometry(shot, ocr)
            # calibrate the band from a freshly parsed screen, never from the
            # stored layout: the parser's detection may have changed since
            fresh = parse(shot, ocr, 7000 + i, band, off, size)
            if band is None:
                band = (fresh["layout"]["msg_top"], fresh["layout"]["msg_bottom"])
            fresh.update({k: v for k, v in screen.items()
                          if k not in fresh and k != "items"})
            fresh["ocr_offset"] = list(off)
            fresh["ocr_size"] = list(size)
            fresh["shot"] = str(shot)
            fresh["ocr"] = str(ocr)
            for it in fresh["items"]:
                it["shot"] = str(shot)
            screens[i] = fresh
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{tag}: 重解析 {len(screens)} 屏")
    return 0


def overlap_len(merged: list[dict], items: list[dict]) -> int:
    """Largest n where the tail of `merged` equals the head of `items`."""
    a = [sig(it) for it in merged]
    b = [sig(it) for it in items]
    limit = min(len(a), len(b), 40)
    for n in range(limit, 0, -1):
        if a[-n:] == b[:n]:
            return n
    return 0


def rfind_match(merged: list[dict], target: dict) -> int:
    """Index of the last item in `merged` matching `target`, or -1.

    Matching is fuzzy so a bubble clipped at the window edge still lines up with
    its fully visible counterpart on the next screen.
    """
    if len(payload(target)) < 4:
        return -1                     # nothing distinctive to match on
    for i in range(len(merged) - 1, max(-1, len(merged) - 60), -1):
        if items_match(merged[i], target):
            return i
    return -1


def items_match(a: dict, b: dict, min_len: int = 4) -> bool:
    """True when two items are the same message (exactly, or partially)."""
    if a["kind"] != b["kind"]:
        return False
    if a.get("side") and b.get("side") and a["side"] != b["side"]:
        return False
    ta, tb = payload(a), payload(b)
    if len(ta) < min_len or len(tb) < min_len:
        return False
    if ta == tb:
        return True
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if short in long_ and len(short) / len(long_) >= 0.34:
        return True
    # WeChat renders consecutive messages from one person as a single green
    # block, so the same block can be clipped at a different offset on each
    # screen. Fall back to longest-common-substring similarity.
    if min(len(ta), len(tb)) < 12:
        return False
    match = difflib.SequenceMatcher(None, ta, tb, autojunk=False) \
        .find_longest_match(0, len(ta), 0, len(tb))
    ratio = match.size / min(len(ta), len(tb))
    if ratio >= 0.55:
        return True
    if match.size < 60:
        return False
    # A bubble taller than the window is seen as "top half" on one screen and
    # "bottom half" on the next, so the shared run can be under half of the
    # shorter view. The two views always share the same bubble geometry.
    ra, rb = a.get("rect") or [0, 0, 0, 0], b.get("rect") or [0, 0, 0, 0]
    same_geo = (abs(ra[0] - rb[0]) <= 10
                and abs(ra[2] - rb[2]) <= max(14, ra[2] * 0.08))
    return ratio >= 0.45 or (same_geo and ratio >= 0.30)


def any_overlap(prev: dict, cur: dict) -> bool:
    """True when the two screens share at least one message."""
    a = prev["items"]
    b = cur["items"]
    if not a or not b:
        return False
    for x in b[-6:]:
        for y in a[-45:]:
            if items_match(x, y):
                return True
    return False


def collapse_adjacent(items: list[dict]) -> list[dict]:
    """Merge items duplicated by a screen seam.

    A message taller than the window is seen clipped at a different offset on
    each screen, so the two copies are neither equal nor nested: they have to be
    joined on their common middle section.
    """
    out: list[dict] = []
    for it in items:
        # compare against a short recent window: a seam can leave a couple of
        # items between the two copies
        for back in range(1, 9):
            if back > len(out):
                break
            prev = out[-back]
            if not items_match(prev, it):
                continue
            if prev["kind"] == "message":
                a = "".join(prev.get("paragraphs") or [])
                b = "".join(it.get("paragraphs") or [])
                if a and b:
                    joined = union_texts(a, b)
                    if joined:
                        merged = dict(prev)
                        merged["paragraphs"] = [joined]
                        out[-back] = merged
                        it = None
                        break
                    # Same message captured with two different OCR readings:
                    # keep the longer, cleaner one instead of duplicating it.
                    out[-back] = prev if len(a) >= len(b) else it
                    it = None
                    break
            if len(sig(it)) > len(sig(prev)):
                out[-back] = it
            it = None
            break
        if it is None:
            continue
        out.append(it)
    return out


def join_text(a: str, b: str) -> str | None:
    """Join two partial views of the same message on their shared middle."""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    m = sm.find_longest_match(0, len(a), 0, len(b))
    if m.size < 12:
        return None
    tail_a = m.a + m.size >= len(a) - 2
    tail_b = m.b + m.size >= len(b) - 2
    if tail_a and tail_b:
        return a if len(a) >= len(b) else b
    if tail_a:                                # `a` stops where the shared part ends
        return a + b[m.b + m.size:]
    if tail_b:                                # `b` stops where the shared part ends
        return b + a[m.a + m.size:]
    return None


def union_texts(a: str, b: str) -> str | None:
    """Best-effort union of two partial views of the same message.

    Returns None when the two cannot be combined safely, in which case the
    caller keeps both items instead of discarding one.
    """
    if a == b:
        return a
    if a in b:
        return b
    if b in a:
        return a
    joined = join_text(a, b)
    if joined and len(joined) >= max(len(a), len(b)) - 8:
        return joined
    return None


def align(merged: list[dict], items: list[dict]) -> tuple[int, int]:
    """Find where the new screen's content starts inside `merged`.

    Returns (index in merged, index in items) of the first reliable match. The
    screen's first item can be a bubble clipped by the window edge, so a few
    leading items are tried before giving up.
    """
    for j in range(min(10, len(items))):
        idx = rfind_match(merged, items[j])
        if idx >= 0:
            return idx, j
    return -1, -1


def cut_paragraphs(paras: list[str], off: int) -> tuple[list[str], list[str], bool]:
    """Split a paragraph list at a character offset: (head, tail, mid)."""
    head: list[str] = []
    acc = 0
    for i, p in enumerate(paras):
        if acc + len(p) <= off:
            head.append(p)
            acc += len(p)
            continue
        cut = off - acc
        if cut > 0:
            head.append(p[:cut])
        tail = ([p[cut:]] if cut < len(p) else []) + list(paras[i + 1:])
        return [h for h in head if h.strip()], [t for t in tail if t.strip()], cut > 0
    return [h for h in head if h.strip()], [], False


def union_paragraphs(pa: list[str], pb: list[str]) -> list[str]:
    """Union two views of one message without losing paragraph breaks.

    A long message is observed twice — once clipped at the top of the window,
    once at the bottom — and the two views meet in the middle of a paragraph.
    Joining the raw text throws every break away, so match the cut instead and
    keep each view's own breaks.
    """
    pa = [p for p in pa if p.strip()]
    pb = [p for p in pb if p.strip()]
    A, B = "".join(pa), "".join(pb)
    if not A:
        return pb
    if not B:
        return pa
    if A == B:
        return pa
    if B in A:
        return pa
    if A in B:
        return pb
    for first, second in ((pa, pb), (pb, pa)):
        X, Y = "".join(first), "".join(second)
        # where does X start inside Y? OCR differs by a character or two, so
        # align on the longest common run rather than an exact prefix.
        m = difflib.SequenceMatcher(None, Y, X, autojunk=False) \
            .find_longest_match(0, len(Y), 0, len(X))
        if m.size < 16 or m.b > 4:
            continue
        k = m.a - m.b
        if k > 0:
            head, _, mid = cut_paragraphs(second, k)
            if mid and head:
                return head[:-1] + [head[-1] + first[0]] + first[1:]
            return head + first
    joined = union_texts(A, B)
    return [joined] if joined else []


def reconcile(tail: list[dict], new: list[dict]) -> list[dict]:
    """Pair the re-observed tail with the new screen's items, unioning pairs.

    `tail` and `new` are both ordered newest -> oldest. The new screen re-shows
    part of what we already had, plus older content, so each of its items is
    matched against the tail with a small forward search: a single OCR hiccup
    must not shift every following pairing by one position.
    """
    out: list[dict] = []
    ti = 0
    for it in new:
        hit = -1
        for cand in range(ti, min(ti + 6, len(tail))):
            if items_match(tail[cand], it):
                hit = cand
                break
        if hit < 0:
            out.append(it)
            continue
        out.extend(tail[ti:hit])          # unmatched leftovers keep their place
        out.extend(merge_items(tail[hit], it))
        ti = hit + 1
    # Anything the new screen never showed is older than its oldest item and
    # stays at the end. (With a forward search this is normally empty.)
    out.extend(tail[ti:])
    return out


def merge_items(a: dict, b: dict) -> list[dict]:
    """Combine two views of the same item into one (or two if not combinable)."""
    if a["kind"] != b["kind"]:
        return [a, b]
    if a["kind"] == "message":
        paras = union_paragraphs(a.get("paragraphs") or [], b.get("paragraphs") or [])
        if paras:
            merged = dict(a)
            merged["paragraphs"] = paras
            merged["clipped_top"] = bool(a.get("clipped_top")) and bool(b.get("clipped_top"))
            merged["clipped_bottom"] = bool(a.get("clipped_bottom")) and bool(b.get("clipped_bottom"))
            return [merged]
        # two OCR readings of one message: keep the fuller one
        return [a if len(payload(a)) >= len(payload(b)) else b]
    return [a if len(sig(a)) >= len(sig(b)) else b]


# ----------------------------------------------------------------------- render
MONTH_RE = re.compile(r"^(\d{1,2})月(\d{1,2})日(?:\s*(\d{1,2}):(\d{2}))?$")
YEAR_RE = re.compile(r"^(\d{4})年(\d{1,2})月(\d{1,2})日(?:\s*(\d{1,2}):(\d{2}))?$")
WEEK_RE = re.compile(r"^星期([一二三四五六日天])(?:\s*(\d{1,2}):(\d{2}))?$")
CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
TODAY_RE = re.compile(r"^今天\s*(\d{1,2}):(\d{2})$")
YEST_RE = re.compile(r"^昨天\s*(\d{1,2}):(\d{2})$")
AMPM_RE = re.compile(r"^([上下])午\s*(\d{1,2}):(\d{2})$")

WEEKDAY = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}

# characters Vision emits when it tries to read an emoji or sticker glyph
EMOJI_NOISE = re.compile(r"[\u2460-\u24ff\u3251-\u32bf]+")
CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
TRAILING_NOISE = re.compile(
    r"[\u3400-\u9fff\uf900-\ufaff]"                                   # ends with CJK
    r"[^\u3400-\u9fff\uf900-\ufaff"                                   # then a non-CJK
    r"。，！？；：、…—～《》〈〉「」『』（）()\[\]【】\"'“”‘’"                 # that is not
    r"?!.,;:]"                                                        # punctuation
    r"$"
)


def clean_text(text: str) -> str:
    """Remove glyphs Vision produces when it reads a small emoji image."""
    out = EMOJI_NOISE.sub("", text or "")
    if TRAILING_NOISE.search(out):
        out = out[:-1]
    return out


def resolve_time(label: str, ref: datetime) -> datetime | None:
    label = label.strip()
    m = MONTH_RE.match(label)
    if m:
        mon, day = int(m.group(1)), int(m.group(2))
        hh, mm = int(m.group(3) or 0), int(m.group(4) or 0)
        try:
            when = ref.replace(month=mon, day=day, hour=hh, minute=mm,
                               second=0, microsecond=0)
        except ValueError:
            return None
        if when > ref + timedelta(days=1):
            when = when.replace(year=ref.year - 1)
        return when
    m = WEEK_RE.match(label)
    if m:
        target = WEEKDAY[m.group(1)]
        hh, mm = int(m.group(2) or 0), int(m.group(3) or 0)
        day = ref.date()
        while day.weekday() != target:
            day -= timedelta(days=1)
        return datetime(day.year, day.month, day.day, hh, mm)
    for rx, delta in ((TODAY_RE, 0), (YEST_RE, 1)):
        m = rx.match(label)
        if m:
            d = (ref - timedelta(days=delta)).date()
            return datetime(d.year, d.month, d.day, int(m.group(1)), int(m.group(2)))
    m = CLOCK_RE.match(label)
    if m:
        return ref.replace(hour=int(m.group(1)), minute=int(m.group(2)),
                           second=0, microsecond=0)
    m = AMPM_RE.match(label)
    if m:
        hh = int(m.group(2)) % 12 + (12 if m.group(1) == "上" else 0)
        return ref.replace(hour=hh, minute=int(m.group(3)), second=0, microsecond=0)
    return None


def resolve_back(label: str, ref: datetime, slack_days: int = 3) -> datetime | None:
    """Resolve a WeChat timestamp to the latest datetime not after `ref`.

    Scrolling back in time is monotonic, so "10月3日" seen below a February
    timestamp must mean the *previous* October, not the coming one. `slack_days`
    allows for a separator repeated in the overlap between two screens.
    """
    label = label.strip()
    cutoff = ref + timedelta(days=slack_days)
    m = YEAR_RE.match(label)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                            int(m.group(4) or 0), int(m.group(5) or 0))
        except ValueError:
            return None
    m = MONTH_RE.match(label)
    if m:
        mon, day = int(m.group(1)), int(m.group(2))
        hh, mm = int(m.group(3) or 0), int(m.group(4) or 0)
        best = None
        for year in (ref.year + 1, ref.year, ref.year - 1, ref.year - 2):
            try:
                cand = ref.replace(year=year, month=mon, day=day,
                                   hour=hh, minute=mm, second=0, microsecond=0)
            except ValueError:
                continue
            if cand <= cutoff and (best is None or cand > best):
                best = cand
        return best
    m = WEEK_RE.match(label)
    if m:
        target = WEEKDAY[m.group(1)]
        hh, mm = int(m.group(2) or 0), int(m.group(3) or 0)
        day = ref.date()          # a weekday label can never point into the future
        for _ in range(8):
            if day.weekday() == target:
                break
            day -= timedelta(days=1)
        return datetime(day.year, day.month, day.day, hh, mm)
    m = YEST_RE.match(label)
    if m:
        d = (ref - timedelta(days=1)).date()
        return datetime(d.year, d.month, d.day, int(m.group(1)), int(m.group(2)))
    m = TODAY_RE.match(label)
    if m:
        return ref.replace(hour=int(m.group(1)), minute=int(m.group(2)),
                           second=0, microsecond=0)
    m = CLOCK_RE.match(label)
    if m:
        cand = ref.replace(hour=int(m.group(1)), minute=int(m.group(2)),
                           second=0, microsecond=0)
        return cand if cand <= cutoff else cand - timedelta(days=1)
    m = AMPM_RE.match(label)
    if m:
        hh = int(m.group(2)) % 12 + (12 if m.group(1) == "上" else 0)
        cand = ref.replace(hour=hh, minute=int(m.group(3)), second=0, microsecond=0)
        return cand if cand <= cutoff else cand - timedelta(days=1)
    return None


def date_candidates(label: str, around: datetime) -> list[datetime]:
    """All plausible datetimes a WeChat separator could mean."""
    label = label.strip()
    out: list[datetime] = []
    m = YEAR_RE.match(label)
    if m:
        try:
            return [datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                             int(m.group(4) or 0), int(m.group(5) or 0))]
        except ValueError:
            return []
    m = MONTH_RE.match(label)
    if m:
        mon, day = int(m.group(1)), int(m.group(2))
        hh, mm = int(m.group(3) or 0), int(m.group(4) or 0)
        for year in range(around.year - 1, around.year + 2):
            try:
                out.append(datetime(year, mon, day, hh, mm))
            except ValueError:
                continue
        return out
    m = WEEK_RE.match(label)
    if m:
        target = WEEKDAY[m.group(1)]
        hh, mm = int(m.group(2) or 0), int(m.group(3) or 0)
        day = around.date()
        while day.weekday() != target:
            day -= timedelta(days=1)
        for delta in (0, -7, 7):
            d = day + timedelta(days=delta)
            out.append(datetime(d.year, d.month, d.day, hh, mm))
        return out
    for rx, delta in ((TODAY_RE, 0), (YEST_RE, 1)):
        m = rx.match(label)
        if m:
            d = (around - timedelta(days=delta)).date()
            return [datetime(d.year, d.month, d.day, int(m.group(1)), int(m.group(2)))]
    m = CLOCK_RE.match(label)
    if m:
        base = around.replace(hour=int(m.group(1)), minute=int(m.group(2)),
                              second=0, microsecond=0)
        return [base + timedelta(days=d) for d in (0, -1, 1)]
    m = AMPM_RE.match(label)
    if m:
        hh = int(m.group(2)) % 12 + (12 if m.group(1) == "上" else 0)
        base = around.replace(hour=hh, minute=int(m.group(3)), second=0, microsecond=0)
        return [base + timedelta(days=d) for d in (0, -1, 1)]
    return out


def resolve_forward(label: str, last: datetime | None, now: datetime) -> datetime | None:
    """Resolve a separator to the earliest plausible time at or after `last`.

    Going forward means a single mis-parse can only push one separator later, it
    cannot drag every following date back a year the way a backward pass can.
    """
    cands = date_candidates(label, last or now)
    if not cands:
        return None
    cands = [c for c in cands if c <= now + timedelta(days=1)]
    if not cands:
        return None
    if last is None:
        return min(cands)
    ok = [c for c in cands if c >= last - timedelta(days=1)]
    return min(ok) if ok else min(cands)


def cmd_render(args) -> int:
    items = json.loads((BUILD / "messages.json").read_text(encoding="utf-8"))
    ref = datetime.now()
    out_dir = Path(args.out_dir) if args.out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    # Separators are resolved against two kinds of anchor: explicit dates such as
    # "2025年8月26日" are certain, and the newest message is "about now". Bare
    # labels between anchors are resolved forward, the tail backwards, so neither
    # end can drag a year.
    stamps = [i for i, it in enumerate(items) if it["kind"] == "timestamp"]
    anchors: list[tuple[int, datetime]] = []
    for k in stamps:
        m = YEAR_RE.match(items[k]["text"].strip())
        if not m:
            continue
        try:
            anchors.append((k, datetime(int(m.group(1)), int(m.group(2)),
                                        int(m.group(3)), int(m.group(4) or 0),
                                        int(m.group(5) or 0))))
        except ValueError:
            continue

    for pos, k in enumerate(stamps):
        it = items[k]
        explicit = next((d for a, d in anchors if a == k), None)
        if explicit is not None:
            it["_dt"] = explicit
            continue
        lower = next((d for a, d in reversed(anchors) if a < k), None)
        upper = next((d for a, d in anchors if a > k), None)
        if upper is None:
            # tail of the conversation: anchor on "the newest message is recent"
            it["_dt"] = resolve_back(it["text"], ref)
        else:
            it["_dt"] = resolve_forward(it["text"], lower, upper)

    # A date printed inside a shared picture can be picked up as a separator and
    # land far from its neighbours; such an outlier is noise, not a real break.
    resolved = [(i, items[i]["_dt"]) for i in stamps if items[i].get("_dt")]
    for n, (i, when) in enumerate(resolved):
        prev_dt = resolved[n - 1][1] if n > 0 else None
        next_dt = resolved[n + 1][1] if n + 1 < len(resolved) else None
        far_prev = prev_dt is None or abs((when - prev_dt).days) > 45
        far_next = next_dt is None or abs((next_dt - when).days) > 45
        if far_prev and far_next:
            items[i]["_dt"] = None

    # attach a wall-clock stamp to every item, carried forward from the last
    # timestamp separator that WeChat itself renders
    current: datetime | None = None
    seq: list[dict] = []
    img_i = 0
    emoji_i = 0
    for it in items:
        if it["kind"] == "timestamp":
            if it.get("_dt"):
                current = it["_dt"]
            seq.append({**it, "when": current})
            continue
        entry = {**it, "when": current}
        if it["kind"] == "message" and (it.get("emojis") or it.get("quote_emojis")):
            files = {}
            labels = {}
            # quote emojis first, then the body's: the counters line up with the
            # order the tokens appear in the rendered text
            ordered = list(it.get("quote_emojis") or []) + list(it.get("emojis") or [])
            for idx, rect in enumerate(ordered):
                name = f"emoji_{emoji_i:04d}.png"
                dst = images_out / name
                shot_for = it.get("quote_shot") if idx < len(it.get("quote_emojis") or []) \
                    else it.get("shot")
                if _crop_rect(shot_for, rect, dst, pad=0):
                    files[str(idx)] = name
                    try:
                        label, dist = emoji_match.match(Image.open(dst))
                        # a system emoji (car, flag, ...) is not in WeChat's
                        # panel, so an unconvincing best match is left unnamed
                        # a system emoji is not in WeChat's panel, and a weak
                        # match tends to invent a name ("👍" -> "吐"), so require
                        # a close match before showing one
                        labels[str(idx)] = label if dist <= 42 else "表情"
                    except Exception:
                        labels[str(idx)] = "表情"
                    emoji_i += 1
            entry["emoji_files"] = files
            entry["emoji_names"] = labels
        if it["kind"] == "media":
            img_i += 1
            name = f"img{img_i:04d}.png"
            if _crop_media(it, images_out / name):
                entry["image"] = name
            else:
                img_i -= 1
        seq.append(entry)

    md_path = out_dir / f"微信-{PEER}-聊天记录.md"
    html_path = out_dir / f"微信-{PEER}-聊天记录.html"
    md_path.write_text(render_markdown(seq), encoding="utf-8")
    html_path.write_text(render_html(seq), encoding="utf-8")
    print(f"已生成：\n  {md_path}\n  {html_path}\n  图片 {img_i} 张 -> {images_out}")
    print(f"消息 {sum(1 for s in seq if s['kind']=='message')} 条，"
          f"媒体 {sum(1 for s in seq if s['kind']=='media')} 条，"
          f"时间分隔 {sum(1 for s in seq if s['kind']=='timestamp')} 个")
    return 0


def _crop_rect(shot_path, rect, dst: Path, pad: int = 6) -> bool:
    shot = Path(shot_path or "")
    if not shot.is_file():
        return False
    x, y, w, h = rect
    img = Image.open(shot).convert("RGB")
    if x + w > img.width or y + h > img.height:
        return False                    # rect came from a different screenshot
    box = (max(0, x - pad), max(0, y - pad),
           min(img.width, x + w + pad), min(img.height, y + h + pad))
    img.crop(box).save(dst)
    return True


def _crop_media(item: dict, dst: Path) -> bool:
    return _crop_rect(item.get("shot") or item.get("source_shot"),
                      item["rect"], dst, pad=6)


def who(side: str) -> str:
    return ME if side == "out" else PEER


def para_markdown(text: str, files: dict, counter: list, labels: dict | None = None) -> str:
    """Render one paragraph, swapping emoji tokens for the cropped pictures."""
    out = []
    for kind, value in split_emoji(text):
        if kind == "text":
            out.append(clean_text(value))
        else:
            key = str(counter[0])
            name = (files or {}).get(key)
            label = (labels or {}).get(key, "表情")
            counter[0] += 1
            out.append(f"![{label}](images/{name})" if name else f"〔{label}〕")
    return "".join(out)


def para_html(text: str, files: dict, counter: list, labels: dict | None = None) -> str:
    out = []
    for kind, value in split_emoji(text):
        if kind == "text":
            out.append(_esc(clean_text(value)))
        else:
            key = str(counter[0])
            name = (files or {}).get(key)
            label = (labels or {}).get(key, "表情")
            counter[0] += 1
            out.append(f'<img class="emoji" src="images/{name}" alt="{_esc(label)}" '
                       f'title="{_esc(label)}">'
                       if name else f'<span class="tag">〔{_esc(label)}〕</span>')
    return "".join(out)


def render_markdown(seq: list[dict]) -> str:
    lines = [f"# 微信聊天记录 · {PEER}", "",
             "> 由本机微信客户端截图 + OCR 识别整理，仅保存在本地。", ""]
    day = None
    for it in seq:
        when: datetime | None = it.get("when")
        cur_day = when.date() if when else None
        if cur_day != day:
            day = cur_day
            title = (day.strftime("%Y-%m-%d") + " " + _weekday_cn(day)) if day else "时间未知"
            lines += ["", f"## {title}", ""]
        if it["kind"] == "timestamp":
            # The separator only drives the date heading and each message's own
            # time label, so it is not repeated in the body.
            continue
        elif it["kind"] == "system":
            lines += [f"> {it['text']}", ""]
        elif it["kind"] == "message":
            stamp = when.strftime("%H:%M") if when else ""
            lines.append(f"**{who(it['side'])}**{(' · ' + stamp) if stamp else ''}：")
            files = it.get("emoji_files") or {}
            labels = it.get("emoji_names") or {}
            n_quote = len(it.get("quote_emojis") or [])
            counter = [n_quote]
            body = [para_markdown(p, files, counter, labels)
                    for p in (it.get("paragraphs") or [])]
            body = [p for p in body if p.strip()]
            if not body and files:                 # a message that is only emoji
                body = [f"![{labels.get(str(k), '表情')}](images/{name})"
                        for k, name in sorted(files.items(), key=lambda kv: int(kv[0]))]
            body = body or ["〔表情〕"]
            if len(body) > 1 and all(p.startswith("![") and p.endswith(")") for p in body):
                body = ["".join(body)]             # 👍👍👍 belongs on one line
            lines.extend(body)
            if it.get("quote"):
                # WeChat shows the quoted message underneath the message that
                # quotes it, so keep the same order here
                qcounter = [0]
                lines += [f"> {para_markdown(it['quote'], files, qcounter, labels)}"]
            lines.append("")
        elif it["kind"] == "media":
            stamp = when.strftime("%H:%M") if when else ""
            if it.get("image"):
                lines.append(f"**{who(it['side'])}**{(' · ' + stamp) if stamp else ''}：")
                lines.append(f"![{it['label']}](images/{it['image']})")
            else:
                alt = clean_text(it.get("text", "")) or it["label"]
                lines.append(f"**{who(it['side'])}**{(' · ' + stamp) if stamp else ''}："
                             f"〔{alt}〕")
            lines.append("")
    return "\n".join(lines) + "\n"


def _weekday_cn(d) -> str:
    return "星期" + "一二三四五六日"[d.weekday()]


def render_html(seq: list[dict]) -> str:
    rows = []
    day = None
    for it in seq:
        when: datetime | None = it.get("when")
        cur_day = when.date() if when else None
        if cur_day != day:
            day = cur_day
            title = (day.strftime("%Y-%m-%d") + " " + _weekday_cn(day)) if day else "时间未知"
            rows.append(f'<div class="day">{title}</div>')
        if it["kind"] == "timestamp":
            continue                       # date heading + per-message time suffice
        elif it["kind"] == "system":
            rows.append(f'<div class="stamp">{_esc(it["text"])}</div>')
        elif it["kind"] == "message":
            side = "me" if it["side"] == "out" else "peer"
            files = it.get("emoji_files") or {}
            labels = it.get("emoji_names") or {}
            n_quote = len(it.get("quote_emojis") or [])
            counter = [n_quote]
            paras = [para_html(p, files, counter, labels)
                     for p in (it.get("paragraphs") or [])]
            paras = [p for p in paras if p.strip()]
            if not paras and files:                # a message that is only emoji
                paras = [f'<img class="emoji" src="images/{name}" '
                         f'alt="{_esc(labels.get(str(k), "表情"))}">'
                         for k, name in sorted(files.items(), key=lambda kv: int(kv[0]))]
            paras = paras or ["〔表情〕"]
            if len(paras) > 1 and all(p.startswith('<img class="emoji"') for p in paras):
                paras = ["".join(paras)]
            body = ""
            body += "<br>".join(paras)
            if it.get("quote"):
                qcounter = [0]
                body += ('<div class="quote">'
                         + para_html(it["quote"], files, qcounter, labels) + '</div>')
            rows.append(_bubble(side, who(it["side"]), when, body))
        elif it["kind"] == "media":
            side = "me" if it["side"] == "out" else "peer"
            if it.get("image"):
                body = (f'<img src="images/{it["image"]}" '
                        f'alt="{_esc(it["label"])}">')
            else:
                alt = clean_text(it.get("text", "")) or it["label"]
                body = f'<span class="tag">〔{_esc(alt)}〕</span>'
            rows.append(_bubble(side, who(it["side"]), when, body))

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>微信聊天记录 · {_esc(PEER)}</title>
<style>
 :root {{ color-scheme: light; }}
 body {{ margin:0; background:#f2f2f2; font:16px/1.7 -apple-system,"PingFang SC",sans-serif; color:#1a1a1b; }}
 header {{ position:sticky; top:0; background:#fff; border-bottom:1px solid #e6e6e8;
           padding:14px 20px; font-weight:600; }}
 main {{ max-width:820px; margin:0 auto; padding:20px 16px 80px; }}
 .day, .stamp {{ text-align:center; color:#9f9fa5; font-size:13px; margin:22px 0 12px; }}
 .row {{ display:flex; margin:10px 0; gap:10px; }}
 .row.me {{ flex-direction:row-reverse; }}
 .who {{ color:#9f9fa5; font-size:12px; margin-bottom:4px; }}
 .col {{ display:flex; flex-direction:column; max-width:78%; }}
 .bubble {{ padding:10px 14px; border-radius:10px; background:#eeeef0;
            white-space:pre-wrap; word-break:break-word; }}
 .row.me .col {{ align-items:flex-end; }}
 .me .bubble {{ background:#b0f0a7; }}
 .bubble img {{ max-width:100%; border-radius:8px; display:block; }}
 .quote {{ border-left:3px solid #d8d8dc; padding-left:8px; margin:6px 0 0;
           color:#9f9fa5; font-size:0.88em; }}
 /* our own bubbles are right-aligned, so WeChat draws their quote bar on the right */
 .row.me .quote {{ border-left:none; border-right:3px solid #d8d8dc;
                   padding-left:0; padding-right:8px; text-align:right; }}
 .bubble img.emoji {{ display:inline-block; height:1.32em; width:auto;
                      vertical-align:-0.25em; margin:0 1px; border-radius:0; }}
 .tag {{ color:#6f6f73; }}
</style></head><body>
<header>微信聊天记录 · {_esc(PEER)}　<span style="font-weight:400;color:#9f9fa5">
（本机截图 + OCR 整理，仅存本地）</span></header>
<main>
{chr(10).join(rows)}
</main></body></html>
"""


def _bubble(side: str, name: str, when, body: str) -> str:
    stamp = when.strftime("%H:%M") if when else ""
    return (f'<div class="row {side}"><div class="col">'
            f'<div class="who">{_esc(name)}{(" · " + stamp) if stamp else ""}</div>'
            f'<div class="bubble">{body}</div></div></div>')


def _esc(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ------------------------------------------------------------------------- main
def cmd_doctor(args) -> int:
    """Check everything `capture` needs, one line per item, with fixes."""
    ok = True

    def check(good: bool, label: str, hint: str = "") -> None:
        nonlocal ok
        print(f"{'✓' if good else '✗'} {label}")
        if not good:
            ok = False
            if hint:
                print(f"    → {hint}")

    def info(label: str) -> None:
        print(f"· {label}")

    ver = platform.mac_ver()[0]
    check(bool(ver), f"macOS {ver}" if ver else "macOS",
          "这个工具依赖 screencapture / Vision，只能在 macOS 上跑")

    for mod in ("numpy", "PIL"):
        try:
            m = __import__(mod)
            info(f"{mod} {getattr(m, '__version__', '?')}")
        except Exception:
            check(False, f"{mod} 已安装", "python3 -m pip install -r requirements.txt")

    for tool, why in (("clang", "编译 wxwin / wxscroll"),
                      ("swiftc", "编译 ocrfast")):
        found = shutil.which(tool)
        check(bool(found), f"{tool} 可用", f"安装 Xcode Command Line Tools（{why}）；见 docs/troubleshooting.md")

    for name in ("wxwin", "wxscroll"):
        path = TOOLS / name
        check(path.is_file() and os.access(path, os.X_OK), f"tools/{name} 已编译", "在仓库根目录执行 make")

    check(OCRBIN.is_file() and os.access(OCRBIN, os.X_OK),
          f"OCR 程序 {OCRBIN}",
          "执行 make；若 swiftc 报 SDK 版本不匹配，见 docs/troubleshooting.md")

    try:
        win = find_wechat_window()
        info(f"微信窗口 {win['w']}×{win['h']} @ ({win['x']:.0f},{win['y']:.0f})")
    except SystemExit:
        win = None
        check(False, "找到微信窗口", "先打开微信，并点进要导出的会话")

    if win:
        probe = BUILD / "doctor_shot.png"
        BUILD.mkdir(exist_ok=True)
        res = sh(["screencapture", "-x", "-l", str(win["id"]), str(probe)])
        blank = True
        if res.returncode == 0 and probe.is_file():
            try:
                grey = Image.open(probe).convert("L")
                blank = grey.getextrema()[1] < 8      # all black = no permission
            except Exception:
                blank = True
        check(not blank, "屏幕录制权限",
              "系统设置 → 隐私与安全性 → 屏幕录制，勾选你的终端/IDE，然后重启它")
        probe.unlink(missing_ok=True)

    scroll = TOOLS / "wxscroll"
    if scroll.is_file():
        res = sh([str(scroll), "--check"])
        check(res.returncode == 0, "辅助功能权限",
              "系统设置 → 隐私与安全性 → 辅助功能，勾选你的终端/IDE，然后重启它")

    tpl = HERE / "emoji_templates"
    if tpl.is_dir() and any(tpl.glob("*.png")):
        info(f"表情模板 {len(list(tpl.glob('*.png')))} 个")
    else:
        info("表情模板未生成（可选）：见 README「微信表情」一节")

    print()
    print("全部通过，可以开始采集。" if ok else "有项目未通过，按上面的提示处理后重跑 doctor。")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export a WeChat conversation with screenshots + macOS Vision OCR.")
    ap.add_argument("--peer", default=os.environ.get("WXEXPORT_PEER", "对方"),
                    help="名字：会话对方的显示名，用于输出标题（也可设 WXEXPORT_PEER）")
    ap.add_argument("--me", default=os.environ.get("WXEXPORT_ME", ""),
                    help="你自己在微信里的昵称；引用里靠它区分谁说的")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture")
    c.add_argument("--max-steps", type=int, default=200)
    c.add_argument("--wait", type=float, default=0.9)
    c.add_argument("--from-bottom", action="store_true")
    c.add_argument("--stop-before", default=None,
                   help="stop once the oldest captured separator predates YYYY-MM-DD")
    c.add_argument("--tag", default="a", help="suffix for build/screens_<tag>.json")
    c.add_argument("--step-frac", type=float, default=0.5,
                   help="fraction of the visible message height to scroll per screen")
    c.add_argument("--max-retry", type=int, default=2)
    c.set_defaults(func=cmd_capture)

    a = sub.add_parser("assemble")
    a.add_argument("--runs", nargs="+", default=["a"],
                   help="screen sets to concatenate, newest run first")
    a.set_defaults(func=cmd_assemble)

    rp = sub.add_parser("reparse")
    rp.add_argument("--runs", nargs="+", default=["a"])
    rp.add_argument("--reocr", action="store_true",
                    help="re-run OCR on the whole normalised screenshot first")
    rp.set_defaults(func=cmd_reparse)

    r = sub.add_parser("render")
    r.add_argument("--out-dir", default=None)
    r.set_defaults(func=cmd_render)

    d = sub.add_parser("doctor")
    d.set_defaults(func=cmd_doctor)

    args = ap.parse_args()
    global PEER, ME_ALIAS
    PEER = args.peer
    ME_ALIAS = args.me
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
