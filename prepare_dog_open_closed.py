#!/usr/bin/env python3
"""Extract a same-dog closed/open pair from Wikimedia Commons Yawn.gif.

Only selects candidates automatically; inspect contact_sheet.png before use.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

SOURCE_URL = "https://upload.wikimedia.org/wikipedia/commons/8/83/Yawn.gif"
SOURCE_PAGE = "https://commons.wikimedia.org/wiki/File:Yawn.gif"


def fetch_gif(gif_path: Path) -> Path:
    if gif_path.is_file():
        return gif_path
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "LivePortraitResearchFrameExtractor/0.1 (educational-use)"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read()
    if len(data) < 1024:
        raise RuntimeError("Downloaded GIF was unexpectedly small")
    with Image.open(io.BytesIO(data)) as im:
        if im.format != "GIF" or getattr(im, "n_frames", 1) < 2:
            raise RuntimeError("Downloaded file is not a multi-frame GIF")
    gif_path.write_bytes(data)
    return gif_path


def crop_frame(frame: Image.Image, center_x: float, center_y: float, size: int) -> Image.Image:
    w, h = frame.size
    side = min(w, h)
    left = max(0, min(w - side, int(round(center_x * w - side / 2))))
    top = max(0, min(h - side, int(round(center_y * h - side / 2))))
    return frame.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.LANCZOS)


def score_mouth_proxy(image: Image.Image) -> dict:
    """Weak image-color proxy, NOT a verified mouth detector."""
    a = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = a.shape[:2]
    # Heuristic ROI for a roughly centered dog face; user should verify contact sheet.
    roi = a[int(h * .50):int(h * .81), int(w * .30):int(w * .70)]
    gray = (.299 * roi[..., 0] + .587 * roi[..., 1] + .114 * roi[..., 2])
    dark_fraction = float(np.mean(gray < 82))
    tongue_fraction = float(np.mean((roi[..., 0] > 75) & (roi[..., 0] > 1.28 * roi[..., 1]) & (roi[..., 0] > 1.27 * roi[..., 2])))
    return {
        "dark_fraction": round(dark_fraction, 5),
        "pink_fraction": round(tongue_fraction, 5),
        "proxy_score": round(dark_fraction + 0.70 * tongue_fraction, 6),
    }


def create_sheet(images: list[Image.Image], scores: list[dict], closed: int, opened: int, out: Path) -> None:
    size = 200
    pad = 28
    columns = 3
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * size, rows * (size + pad)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, img in enumerate(images):
        x = (i % columns) * size
        y = (i // columns) * (size + pad)
        sheet.paste(img.resize((size, size), Image.Resampling.LANCZOS), (x, y))
        tags = []
        if i == closed:
            tags.append("CLOSED?")
            draw.rectangle((x + 1, y + 1, x + size - 2, y + size - 2), outline="blue", width=4)
        if i == opened:
            tags.append("OPEN?")
            draw.rectangle((x + 5, y + 5, x + size - 6, y + size - 6), outline="green", width=4)
        draw.text((x + 6, y + size + 4), f"frame {i}  {','.join(tags)}", fill="black")
    sheet.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create same-dog closed/open image candidates from a real six-frame dog yawning GIF")
    ap.add_argument("--output-dir", default="build/onnx_animal_v062/yawn_pair")
    ap.add_argument("--gif", default=None, help="Optional local Yawn.gif (skip download)")
    ap.add_argument("--closed-index", type=int, default=None, help="Override automatically chosen closed frame index")
    ap.add_argument("--open-index", type=int, default=None, help="Override automatically chosen open frame index")
    ap.add_argument("--crop-center-x", type=float, default=.5)
    ap.add_argument("--crop-center-y", type=float, default=.5)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()
    if not (0 <= args.crop_center_x <= 1 and 0 <= args.crop_center_y <= 1):
        ap.error("crop centers must lie in [0,1]")
    if args.size < 128 or args.size > 2048:
        ap.error("size must be 128..2048")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    original = Path(args.gif) if args.gif else out / "Yawn.gif"
    try:
        original = fetch_gif(original)
    except Exception as ex:
        print(f"Download failed: {ex}\nOpen {SOURCE_PAGE} in your browser, download the original GIF, then rerun with --gif PATH", file=sys.stderr)
        raise SystemExit(2) from ex

    with Image.open(original) as gif:
        frames = []
        for i in range(getattr(gif, "n_frames", 1)):
            gif.seek(i)
            # .convert gets the compositor's frame, not just the palette delta
            frame = gif.convert("RGB")
            frames.append(crop_frame(frame, args.crop_center_x, args.crop_center_y, args.size))
    if len(frames) < 2:
        raise RuntimeError("Need two or more frames")
    scores = [score_mouth_proxy(f) for f in frames]
    closed = int(np.argmin([s["proxy_score"] for s in scores])) if args.closed_index is None else args.closed_index
    opened = int(np.argmax([s["proxy_score"] for s in scores])) if args.open_index is None else args.open_index
    if not (0 <= closed < len(frames) and 0 <= opened < len(frames)) or closed == opened:
        raise ValueError(f"Invalid selection: closed={closed}, open={opened}; choose distinct indexes 0..{len(frames)-1}")
    for i, frame in enumerate(frames):
        frame.save(out / f"frame_{i:02d}.png")
    frames[closed].save(out / "dog_closed.png")
    frames[opened].save(out / "dog_open.png")
    create_sheet(frames, scores, closed, opened, out / "contact_sheet.png")
    manifest = {
        "source_page": SOURCE_PAGE,
        "source_file": str(original),
        "author": "SecretDisc",
        "license": "CC BY-SA 3.0 (or GFDL as offered by the author)",
        "note": "Auto labels from a weak proxy; visually confirm the mouth state and head alignment. Do not present as guaranteed ground truth.",
        "frame_count": len(frames),
        "closed_frame": closed,
        "open_frame": opened,
        "frame_scores": scores,
        "crop_center": [args.crop_center_x, args.crop_center_y],
    }
    (out / "selection.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Created:", (out / "dog_closed.png").resolve())
    print("Created:", (out / "dog_open.png").resolve())
    print("Preview:", (out / "contact_sheet.png").resolve())
    print(f"Automatic candidates: closed=frame {closed}, open=frame {opened}.")
    print("IMPORTANT: inspect contact_sheet.png; if wrong, rerun with --closed-index N --open-index M")


if __name__ == "__main__":
    main()
