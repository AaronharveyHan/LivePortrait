#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare real official LivePortrait animal demo inputs for V0.6 ONNX.

Uses existing repository assets (no network access required):
  assets/examples/source/s25.jpg       (official Shiba Inu portrait)
  assets/examples/driving/d19.mp4      (official corresponding driver video)

Produces three 256x256 PNG files and a labeled comparison contact sheet.
No automatic animal landmark crop is claimed or performed.
"""
import argparse
from pathlib import Path
import json
import sys


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parent,
                   help='LivePortrait repository root; default = this script directory')
    p.add_argument('--source', type=Path,
                   help='Override official source image path (absolute or relative to repo)')
    p.add_argument('--driving', type=Path,
                   help='Override official driving video path (absolute or relative to repo)')
    p.add_argument('--out-dir', type=Path,
                   help='Directory for generated 256x256 inputs (default: build/onnx_animal_v06/official_dog_demo)')
    p.add_argument('--target-frame', type=int, default=-1,
                   help='Driving frame index to animate toward; -1 = ~0.5 second')
    p.add_argument('--max-frames', type=int, default=240,
                   help='Safety bound on video decoding, default 240 frames')
    p.add_argument('--overwrite', action='store_true', help='Replace prior outputs within the demo directory')
    return p.parse_args()


def resolve(repo, value, default):
    p = Path(value) if value is not None else Path(default)
    return p if p.is_absolute() else repo / p


def square_256(bgr, cv2):
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError('Expected non-empty 3-channel source/video image')
    h, w = bgr.shape[:2]
    if min(h, w) <= 0:
        raise ValueError('Invalid image dimensions')
    # Center-square crop only, NOT a detector-aligned crop.
    if h > w:
        top = (h - w) // 2
        bgr = bgr[top:top + w, :]
    elif w > h:
        left = (w - h) // 2
        bgr = bgr[:, left:left + h]
    side = bgr.shape[0]
    mode = cv2.INTER_AREA if side > 256 else cv2.INTER_LINEAR
    return cv2.resize(bgr, (256, 256), interpolation=mode)


def read_video(video_path, cv2, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError('Cannot open sample MP4: {}. Check OpenCV codec support.'.format(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or fps > 240:
        fps = 25.0
    frames = []
    try:
        for i in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(square_256(frame, cv2))
    finally:
        cap.release()
    if len(frames) < 2:
        raise ValueError('Driving video must decode >=2 frames. Found {}'.format(len(frames)))
    return frames, fps


def contact_sheet(images, cv2):
    import numpy as np
    pad = 14
    cell_w = 280
    cell_h = 290
    columns = 3
    rows = (len(images) + columns - 1) // columns
    canvas = np.full((rows * cell_h + pad, columns * cell_w + pad, 3), 244, dtype=np.uint8)
    for i, (label, bgr) in enumerate(images):
        x = pad + (i % columns) * cell_w
        y = pad + (i // columns) * cell_h
        canvas[y:y + 256, x:x + 256] = bgr
        cv2.putText(canvas, label[:34], (x, y + 274), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (24, 24, 24), 1, cv2.LINE_AA)
    return canvas


def main():
    args = cli()
    if args.target_frame < -1 or args.max_frames < 2:
        raise ValueError('--target-frame must be -1 or >=0; --max-frames must be >=2')
    repo = args.repo.resolve()
    source = resolve(repo, args.source, 'assets/examples/source/s25.jpg')
    driving = resolve(repo, args.driving, 'assets/examples/driving/d19.mp4')
    out = resolve(repo, args.out_dir, 'build/onnx_animal_v06/official_dog_demo')
    for file in (source, driving):
        if not file.is_file():
            raise FileNotFoundError('Missing official example: {}\n'
                                    'Check that the repository has assets/examples or use --source / --driving'.format(file))
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('Use liveportrait-onnx environment with opencv-python-headless and numpy') from exc
    src_image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if src_image is None:
        raise RuntimeError('Cannot decode source image: {}'.format(source))
    src_image = square_256(src_image, cv2)
    video_frames, fps = read_video(driving, cv2, args.max_frames)
    count = len(video_frames)
    target_index = (min(count - 1, max(1, int(round(fps * 0.5))))
                    if args.target_frame == -1 else args.target_frame)
    if target_index < 1 or target_index >= count:
        raise ValueError('Target frame must be between 1 and {} for selected sample video'.format(count - 1))
    candidate_indices = sorted(set([0, target_index, max(1, count // 4),
                                    max(1, count // 2), max(1, count * 3 // 4), count - 1]))
    generated_names = ['dog_source.png', 'driver_first.png', 'driver_target.png',
                       'contact_sheet.jpg', 'selection.json']
    generated_names += ['candidate_{:04d}.png'.format(i) for i in candidate_indices]
    if not args.overwrite:
        existing = [out / name for name in generated_names if (out / name).exists()]
        if existing:
            raise FileExistsError('{} already exists; rerun with --overwrite to replace demo files'.format(existing[0]))
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / 'dog_source.png'), src_image)
    cv2.imwrite(str(out / 'driver_first.png'), video_frames[0])
    cv2.imwrite(str(out / 'driver_target.png'), video_frames[target_index])
    for i in candidate_indices:
        cv2.imwrite(str(out / 'candidate_{:04d}.png'.format(i)), video_frames[i])
    panel = [('SOURCE s25 dog', src_image)] + [
        ('DRIVING frame {}{}'.format(i, ' [TARGET]' if i == target_index else ''), video_frames[i])
        for i in candidate_indices]
    cv2.imwrite(str(out / 'contact_sheet.jpg'), contact_sheet(panel, cv2))
    diff = float(np.mean(np.abs(video_frames[target_index].astype(np.float32) - video_frames[0].astype(np.float32))))
    metadata = {
        'origin': 'official LivePortrait app_animals.py example: source s25.jpg + driving d19.mp4',
        'source_path': str(source), 'driving_path': str(driving),
        'fps': fps, 'decoded_frames': count, 'first_frame_index': 0,
        'target_frame_index': target_index, 'mean_frame_pixel_difference': diff,
        'candidate_indices': candidate_indices,
        'preprocess': 'center-square 256x256; no animal detector or landmark alignment',
        'note': 'Actual realism is NOT guaranteed; inspect contact_sheet.jpg and result.png',
    }
    (out / 'selection.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('[OK] Three ready-to-use images written to:', out)
    print('[OK] dog_source.png, driver_first.png, driver_target.png')
    print('[OK] contact_sheet.jpg (visual candidate selection)')
    print('[INFO] {} driving frames, fps={:.2f}, target index={}, pixel diff={:.2f}'.format(
        count, fps, target_index, diff))
    print('NEXT COMMAND:')
    print('python scripts/run_animal_onnx_frame.py --source "{}" --driving-first "{}" '
          '--driving-target "{}" --threads 4 --output "{}" --report-json "{}"'.format(
              out / 'dog_source.png', out / 'driver_first.png', out / 'driver_target.png',
              out / 'result.png', out / 'result.json'))
    if diff < 1.0:
        print('[WARN] First/target frames look similar. Pick a different candidate via --target-frame N.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print('[ERROR]', exc, file=sys.stderr)
        sys.exit(2)
