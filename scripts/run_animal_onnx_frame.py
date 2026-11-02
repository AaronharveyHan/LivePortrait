#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V0.6 isolated CPU ONNX animal v1.1 single-frame pipeline.

No crop, no stitching S, no pasteback, no video encoding. Real motion needs TWO
DRIVING frames: anchor (first) and target. The `--synthetic` smoke test checks
execution only; its output is not a meaningful animal animation.

  python scripts/run_animal_onnx_frame.py --plan
  python scripts/run_animal_onnx_frame.py --synthetic --threads 4
  python scripts/run_animal_onnx_frame.py --source dog.png --driving-first first.png --driving-target later.png
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.animal_onnx_contract import get_contract
from src.utils.animal_onnx_validation import inspect_model_files, validate_ort_io

MODEL_NAMES = ('f', 'm', 'w', 'g')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--plan', action='store_true', help='No torch, numpy, ORT, Pillow or SciPy needed')
    p.add_argument('--synthetic', action='store_true', help='Use deterministic random RGB; numeric smoke only')
    p.add_argument('--source', type=Path, help='Pre-cropped source animal RGB photo')
    p.add_argument('--driving-first', type=Path, help='Pre-cropped first driving frame (motion anchor)')
    p.add_argument('--driving-target', type=Path, help='Pre-cropped target driving frame')
    p.add_argument('--onnx-dir', type=Path, default=ROOT / 'build/onnx_animal_v04')
    p.add_argument('--weights-root', type=Path, default=ROOT / 'pretrained_weights')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--driving-multiplier', type=float, default=1.0)
    p.add_argument('--output', type=Path, default=ROOT / 'build/onnx_animal_v06/frame.png')
    p.add_argument('--report-json', type=Path, default=None)
    p.add_argument('--overwrite', action='store_true', help='Allow replacing the PNG/report')
    return p.parse_args(argv)


def validate_args(args):
    if args.threads < 1:
        raise ValueError('--threads must be >= 1')
    if not math.isfinite(args.driving_multiplier) or args.driving_multiplier < 0:
        raise ValueError('--driving-multiplier must be finite and >= 0')
    if args.plan:
        return
    if args.synthetic:
        if any(p is not None for p in (args.source, args.driving_first, args.driving_target)):
            raise ValueError('--synthetic cannot be combined with image paths')
    elif not all(p is not None for p in (args.source, args.driving_first, args.driving_target)):
        raise ValueError('Require ALL: --source, --driving-first, --driving-target (or --synthetic)')
    if args.output.suffix.lower() != '.png':
        raise ValueError('--output must be a .png path')
    if args.report_json is not None and args.report_json.suffix.lower() != '.json':
        raise ValueError('--report-json must be a .json path')
    if args.report_json is not None and args.report_json.resolve() == args.output.resolve():
        raise ValueError('Report path and PNG path must differ')
    for path in (args.output, args.report_json):
        if path is not None and path.exists() and not args.overwrite:
            raise FileExistsError('{} exists; add --overwrite to replace'.format(path))
    if not args.synthetic:
        for p in (args.source, args.driving_first, args.driving_target):
            if not p.is_file():
                raise FileNotFoundError('Missing input photo: {}'.format(p))


def plan(args):
    items = []
    for code in MODEL_NAMES:
        entry = inspect_model_files(code, args.weights_root, args.onnx_dir)
        # ONNX-only inference requires graphs/manifests; checkpoints optional.
        items.append({'model': code.upper(), 'onnx_ready': entry['ready_benchmark'],
                      'onnx_file': entry['onnx']['path'],
                      'checkpoint_exists': entry['checkpoint']['nonempty'],
                      'manifest_valid': not any('manifest' in issue for issue in entry['issues']),
                      'issues': [v for v in entry['issues'] if 'checkpoint' not in v]})
    return {'schema_version': 1, 'stage': 'v0.6_single_frame',
            'scope': 'animal_v1.1_fixed_batch1_fp32',
            'pipeline': 'F(source) + M(source, first, target) -> relative_motion -> W -> G',
            'supported': {'crop': False, 'stitching': False, 'pasteback': False,
                          'video_encoding': False},
            'model_files': items,
            'note': 'Metadata check, not a graph execution or visual parity check.'}


def open_sessions(args, ort):
    from src.utils.animal_onnx_validation import validate_ort_io
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sessions = {}
    for code in MODEL_NAMES:
        entry = inspect_model_files(code, args.weights_root, args.onnx_dir)
        if not entry['ready_benchmark']:
            problems = [p for p in entry['issues'] if 'checkpoint' not in p]
            raise FileNotFoundError('{} ONNX not ready: {}'.format(code.upper(), '; '.join(problems)))
        session = ort.InferenceSession(entry['onnx']['path'], sess_options=opts,
                                       providers=['CPUExecutionProvider'])
        if session.get_providers() != ['CPUExecutionProvider']:
            raise RuntimeError('Expected CPUExecutionProvider for {}'.format(code.upper()))
        validate_ort_io(session, get_contract(code))
        sessions[code] = session
    return sessions


def read_rgb(path):
    """For parity with legacy cv2.resize, prefer cv2. Fall back to Pillow bilinear."""
    import numpy as np
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raw is None:
            raise ValueError('Cannot decode image: {}'.format(path))
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (256, 256):
            rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(rgb), 'opencv-bilinear'
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError('Image input needs Pillow or opencv-python-headless. '
                           'Install: python -m pip install pillow') from exc
    with Image.open(path) as img:
        rgb = img.convert('RGB')
        if rgb.size != (256, 256):
            rgb = rgb.resize((256, 256), resample=Image.Resampling.BILINEAR)
        return np.asarray(rgb, dtype=np.uint8).copy(), 'pillow-bilinear'


def load_inputs(args):
    import numpy as np
    if args.synthetic:
        rng = np.random.default_rng(args.seed)
        frames = [rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
                  for _ in range(3)]
        return (*frames, 'synthetic-random')
    images = [read_rgb(path) for path in (args.source, args.driving_first, args.driving_target)]
    return (*(item[0] for item in images), ','.join(item[1] for item in images))


def _rss_mib():
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _save_png(path, array):
    try:
        from PIL import Image
    except ImportError:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError('Saving PNG requires pillow or OpenCV') from exc
        if not cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR)):
            raise OSError('cv2.imwrite failed: {}'.format(path))
    else:
        Image.fromarray(array, 'RGB').save(path)


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    if args.plan:
        print(json.dumps(plan(args), indent=2, ensure_ascii=False))
        return 0
    # Fail fast before loading potentially multi-GB ONNX sessions.
    for code in MODEL_NAMES:
        entry = inspect_model_files(code, args.weights_root, args.onnx_dir)
        if not entry['ready_benchmark']:
            issues = [v for v in entry['issues'] if 'checkpoint' not in v]
            raise FileNotFoundError('{} missing ONNX prerequisite: {}'.format(code.upper(), '; '.join(issues)))
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError('Install numpy and onnxruntime in the CPU ONNX environment') from exc
    from src.utils.animal_onnx_chain import run_one_frame
    frames = load_inputs(args)
    rss_before = _rss_mib()
    started = time.perf_counter()
    sessions = open_sessions(args, ort)
    load_ms = (time.perf_counter() - started) * 1000.0
    rss_loaded = _rss_mib()
    result = run_one_frame(sessions, frames[0], frames[1], frames[2],
                           driving_multiplier=args.driving_multiplier)
    rss_done = _rss_mib()
    report = {'schema_version': 1, 'stage': 'v0.6_single_frame',
              'mode': 'synthetic_smoke' if args.synthetic else 'real_pre_cropped_images',
              'source': [str(x) for x in (args.source, args.driving_first, args.driving_target)] if not args.synthetic else None,
              'image_preprocess': frames[3], 'threads': args.threads,
              'ort_session_init_ms': load_ms,
              'rss_mib': {'before_sessions': rss_before,
                          'after_sessions': rss_loaded, 'after_inference': rss_done},
              'result': result['metrics'],
              'note': 'Experimental no-crop, no-S, no-pasteback CPU single-frame chain. '
                      'Synthetic smoke images are NOT meaningful animal animation.'}
    # Write files only after the full chain returned valid outputs.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _save_png(args.output, result['rgb'])
    report['output_png'] = str(args.output)
    if args.report_json is not None:
        serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(serialized, encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, FileNotFoundError, FileExistsError) as exc:
        print('[ERROR]', exc, file=sys.stderr)
        sys.exit(2)
