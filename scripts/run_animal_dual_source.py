#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V0.6.2: two same-dog photo mouth-state experiment, NOT audio lip sync.

Requires cropped CLOSED and OPEN images of the SAME dog. It reconstructs each
state with ONNX (default) or uses original images for fast inspection, registers
upper faces with a conservative translation, then blends ONLY the mouth ROI.
No neural interpolation, inferred jaw motion or temporal coherence is claimed.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.animal_onnx_validation import inspect_model_files
from src.utils.animal_dual_source import validate_roi


def parse_alphas(value):
    try:
        alphas = tuple(float(v.strip()) for v in value.split(','))
    except ValueError as exc:
        raise ValueError('--alphas must be comma-separated numbers') from exc
    if not 2 <= len(alphas) <= 11 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in alphas):
        raise ValueError('--alphas needs 2..11 finite values in [0,1]')
    if len(set(alphas)) != len(alphas) or tuple(sorted(alphas)) != alphas:
        raise ValueError('--alphas must be strictly ascending with no duplicates')
    if len({round(v * 1000) for v in alphas}) != len(alphas):
        raise ValueError('--alphas collide at 0.001 file-name precision')
    if alphas[0] != 0 or alphas[-1] != 1:
        raise ValueError('--alphas must start with 0 and end with 1')
    return alphas


def parse_roi(value):
    try:
        roi = tuple(float(v.strip()) for v in value.split(','))
    except ValueError as exc:
        raise ValueError('--roi requires four comma-separated floats') from exc
    return validate_roi(roi)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-closed', type=Path)
    p.add_argument('--source-open', type=Path)
    identity = p.add_mutually_exclusive_group()
    identity.add_argument('--same-dog', action='store_true', help='User asserts same dog, not auto-verified')
    identity.add_argument('--different-dog-demo', action='store_true', help='Diagnostic ONLY; invalid for natural mouth evaluation')
    p.add_argument('--input-mode', choices=('onnx', 'original'), default='onnx',
                   help='onnx=F/M/W/G reconstruction of both sources; original=fast blend visual test')
    p.add_argument('--alignment', choices=('translation', 'none'), default='translation')
    p.add_argument('--max-shift', type=float, default=20)
    p.add_argument('--roi', default='0.29,0.49,0.74,0.83',
                   help='Normalized x0,y0,x1,y1 mouth ROI; must visually confirm for your dog')
    p.add_argument('--feather-px', type=float, default=24,
                   help='Mouth mask feathering in 512px output coordinates')
    p.add_argument('--alphas', default='0,0.25,0.5,0.75,1')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--onnx-dir', type=Path, default=ROOT / 'build/onnx_animal_v04')
    p.add_argument('--weights-root', type=Path, default=ROOT / 'pretrained_weights')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'build/onnx_animal_v062')
    p.add_argument('--overwrite', action='store_true', help='Replace only named V0.6.2 output files')
    p.add_argument('--plan', action='store_true', help='Print preflight without loading any models')
    return p.parse_args(argv)


def validate_args(args):
    alphas = parse_alphas(args.alphas)
    roi = parse_roi(args.roi)
    if args.threads < 1:
        raise ValueError('--threads must be >=1')
    if not math.isfinite(args.max_shift) or not 0 <= args.max_shift <= 64:
        raise ValueError('--max-shift must be 0..64')
    if not math.isfinite(args.feather_px) or not 1 <= args.feather_px <= 128:
        raise ValueError('--feather-px must be 1..128')
    if args.plan:
        return alphas, roi
    if not (args.same_dog or args.different_dog_demo):
        raise ValueError('Must explicitly choose --same-dog or --different-dog-demo')
    if not args.source_closed or not args.source_open:
        raise ValueError('Both --source-closed and --source-open are required')
    for path in (args.source_closed, args.source_open):
        if not path.is_file():
            raise FileNotFoundError('Missing image: {}'.format(path))
    if args.source_closed.resolve() == args.source_open.resolve():
        raise ValueError('Two distinct source photo paths are required')
    if not args.output_dir.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('--output-dir must be inside repository to avoid accidental external overwrite')
    if args.output_dir.resolve() in (args.source_closed.resolve().parent, args.source_open.resolve().parent):
        raise ValueError('Output directory cannot be an input photo directory')
    return alphas, roi


def planned_files(output_dir, alphas):
    names = ['closed_reference.png', 'open_reference_aligned.png', 'mouth_mask.png',
             'contact_sheet.png', 'report.json']
    names += ['mouth_{:03d}.png'.format(round(a * 1000)) for a in alphas]
    return [output_dir / name for name in names]


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _save_png(path, img):
    from PIL import Image
    Image.fromarray(img, 'RGB').save(path)


def _sheet(images):
    from PIL import Image, ImageDraw, ImageOps
    cell = 256
    canvas = Image.new('RGB', (len(images) * cell, cell + 34), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (alpha, arr) in enumerate(images):
        thumbnail = ImageOps.fit(Image.fromarray(arr, 'RGB'), (cell, cell))
        canvas.paste(thumbnail, (i * cell, 0))
        draw.text((i * cell + 12, cell + 9), 'alpha={:.2f}'.format(alpha), fill='black')
    return canvas


def plan(args, alphas, roi):
    files = []
    for code in ('f','m','w','g'):
        item = inspect_model_files(code, args.weights_root, args.onnx_dir)
        files.append({'model': code.upper(), 'onnx_ready': item['ready_benchmark'],
                      'issues': [x for x in item['issues'] if 'checkpoint' not in x]})
    return {'schema_version': 1, 'stage': 'v0.6.2_dual_source',
            'input_mode': args.input_mode, 'source_closed': str(args.source_closed),
            'source_open': str(args.source_open), 'alphas': alphas, 'roi': roi,
            'registration': args.alignment,
            'onnx_files': files, 'models_used': args.input_mode == 'onnx',
            'note': 'Photo registration and image-space mouth blending; not audio-driven lip sync.'}


def run(args, alphas, roi):
    import numpy as np
    import cv2
    from src.utils.animal_dual_source import (
        estimate_translation, align_reconstruction, mouth_mask,
        blend_mouth, render_reference, region_change_metrics,
    )
    from scripts.run_animal_onnx_frame import read_rgb, open_sessions
    for path in planned_files(args.output_dir, alphas):
        if path.exists() and not args.overwrite:
            raise FileExistsError('Output exists: {} (use --overwrite)'.format(path))
    closed, pc = read_rgb(args.source_closed)
    opened, po = read_rgb(args.source_open)
    if args.alignment == 'translation':
        registration = estimate_translation(closed, opened, max_shift=args.max_shift)
    else:
        registration = {'mode': 'none', 'score': None, 'dx_256': 0.0, 'dy_256': 0.0}
    t0 = time.perf_counter()
    if args.input_mode == 'onnx':
        for code in ('f','m','w','g'):
            meta = inspect_model_files(code, args.weights_root, args.onnx_dir)
            if not meta['ready_benchmark']:
                raise FileNotFoundError('{} ONNX missing: {}'.format(code, meta['issues']))
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError('ONNX mode requires onnxruntime') from exc
        sessions = open_sessions(args, ort)
        init_ms = (time.perf_counter() - t0) * 1000
        cache = {}
        c, c_stats = render_reference(sessions, closed, cache)
        o, o_stats = render_reference(sessions, opened, cache)
    else:
        init_ms = 0.0
        c = cv2.resize(closed, (512, 512), interpolation=cv2.INTER_LINEAR)
        o = cv2.resize(opened, (512, 512), interpolation=cv2.INTER_LINEAR)
        c_stats = o_stats = {'cache_hit': False, 'stages_ms': {}}
    aligned = align_reconstruction(o, registration)
    mask = mouth_mask(512, 512, roi, args.feather_px)
    frames = [(a, blend_mouth(c, aligned, a, mask)) for a in alphas]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_png(output_dir / 'closed_reference.png', c)
    _save_png(output_dir / 'open_reference_aligned.png', aligned)
    _save_png(output_dir / 'mouth_mask.png', np.repeat(np.rint(mask * 255).astype(np.uint8)[:,:,None],3,axis=2))
    for alpha, rgb in frames:
        _save_png(output_dir / 'mouth_{:03d}.png'.format(round(alpha * 1000)), rgb)
    _sheet(frames).save(output_dir / 'contact_sheet.png')
    report = {
        'schema_version': 1, 'stage': 'v0.6.2_dual_source',
        'identity_asserted_same_dog': bool(args.same_dog),
        'diagnostic_different_dog': bool(args.different_dog_demo),
        'mode': args.input_mode, 'threads': args.threads,
        'inputs': {
            'closed': {'path': str(args.source_closed), 'sha256': _hash_file(args.source_closed), 'resize': pc},
            'open': {'path': str(args.source_open), 'sha256': _hash_file(args.source_open), 'resize': po},
        },
        'alignment': registration, 'mouth_roi': list(roi), 'feather_px': args.feather_px,
        'alphas': list(alphas), 'ort_session_init_ms': init_ms,
        'model_cache': {'closed': c_stats, 'open': o_stats},
        'frames': [{'alpha': a, 'path': str(output_dir / 'mouth_{:03d}.png'.format(round(a * 1000))),
                    'metrics': region_change_metrics(c, img, mask)} for a, img in frames],
        'limitations': [
            'NOT a phoneme or audio lip-sync system',
            'Mouth blends are RGB composites, not 3-D jaw articulation',
            'Intermediate alphas can ghost tongues/teeth',
            'Same-dog claim is user-provided, not automatically verified',
            'No video, automatic animal crop, pose/deformable registration or temporal continuity',
        ],
    }
    (output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return report


def main(argv=None):
    args = parse_args(argv)
    alphas, roi = validate_args(args)
    if args.plan:
        print(json.dumps(plan(args, alphas, roi), indent=2, ensure_ascii=False))
        return 0
    run(args, alphas, roi)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, FileNotFoundError, FileExistsError) as exc:
        print('[ERROR]', exc, file=sys.stderr)
        sys.exit(2)
