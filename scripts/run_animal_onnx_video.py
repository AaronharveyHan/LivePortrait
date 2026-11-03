#!/usr/bin/env python3
# coding: utf-8
"""V0.7.0: stream driving-video expression/head pose to an animal MP4 on CPU.

The driving face should already be tracked/cropped or fairly stationary.
For full-size source/driver imagery, center-crop is a geometric crop only;
there is NO automatic face detection. No audio mux or real-time guarantee.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', type=Path, help='Dog source photo; head-centric crop recommended')
    p.add_argument('--driving', type=Path, help='Pre-cropped stable human face MP4')
    p.add_argument('--output', type=Path, default=ROOT/'build/onnx_animal_v070/dog_driven.mp4')
    p.add_argument('--report', type=Path, help='JSON report (default: output name with .json suffix)')
    p.add_argument('--comparison', type=Path, help='Optional MP4: driver | dog source | generated')
    p.add_argument('--onnx-dir', type=Path, default=ROOT/'build/onnx_animal_v04')
    p.add_argument('--weights-root', type=Path, default=ROOT/'pretrained_weights')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--max-frames', type=int, default=60, help='Default 60 frames; 0=all video frames')
    p.add_argument('--fps', type=float, help='Optional FPS override, otherwise original video FPS')
    p.add_argument('--source-crop', choices=('center-crop', 'stretch'), default='center-crop')
    p.add_argument('--driving-crop', choices=('center-crop', 'stretch'), default='center-crop')
    p.add_argument('--motion-mode', choices=('full', 'expression-only', 'pose-only'), default='full')
    p.add_argument('--driving-multiplier', type=float, default=1.0)
    p.add_argument('--progress-every', type=int, default=10)
    p.add_argument('--overwrite', action='store_true', help='Only replace final output files after successful encoding')
    p.add_argument('--plan', action='store_true', help='Metadata-only preflight, no ONNX/PyTorch/video decode')
    return p


def output_paths(args):
    return [args.output, args.report or args.output.with_suffix('.json')] + ([args.comparison] if args.comparison else [])


def validate_args(args):
    from src.utils.animal_onnx_video import check_motion_mode, valid_gain
    check_motion_mode(args.motion_mode)
    valid_gain(args.driving_multiplier)
    if not 1 <= args.threads <= 128:
        raise ValueError('--threads must be in 1..128')
    if args.max_frames < 0:
        raise ValueError('--max-frames must be >= 0')
    if args.progress_every < 1:
        raise ValueError('--progress-every must be >=1')
    if args.fps is not None and (not math.isfinite(args.fps) or not 1 <= args.fps <= 120):
        raise ValueError('--fps must be finite 1..120')
    out = output_paths(args)
    if args.output.suffix.lower() != '.mp4' or (args.comparison and args.comparison.suffix.lower() != '.mp4'):
        raise ValueError('Video outputs must have .mp4 suffix')
    if out[1].suffix.lower() != '.json':
        raise ValueError('Report path must have .json suffix')
    if len({p.resolve() for p in out}) != len(out):
        raise ValueError('Video/comparison/report paths must differ')
    if args.plan:
        return
    if not args.source or not args.driving:
        raise ValueError('Both --source and --driving are required')
    for path in (args.source, args.driving):
        if not path.is_file():
            raise FileNotFoundError('Missing source/driver: {}'.format(path))
    ins = {args.source.resolve(), args.driving.resolve()}
    if any(p.resolve() in ins for p in out):
        raise ValueError('Output must not overwrite source or driving video')
    if any(p.exists() and not args.overwrite for p in out):
        raise FileExistsError('Output exists (use --overwrite): {}'.format(next(p for p in out if p.exists())))


def model_preflight(args):
    from src.utils.animal_onnx_validation import inspect_model_files
    models = []
    for code in ('f', 'm', 'w', 'g'):
        entry = inspect_model_files(code, args.weights_root, args.onnx_dir)
        issues = [x for x in entry['issues'] if 'checkpoint' not in x]
        models.append({'model': code.upper(), 'onnx_ready': bool(entry['ready_benchmark']), 'issues': issues})
    return models


def plan(args):
    return {'stage': 'v0.7.0_animal_expression_video', 'schema_version': 1,
            'source': str(args.source), 'driving': str(args.driving),
            'output': str(args.output), 'comparison': str(args.comparison),
            'motion_mode': args.motion_mode, 'max_frames': args.max_frames,
            'source_crop': args.source_crop, 'driving_crop': args.driving_crop,
            'threads': args.threads, 'models': model_preflight(args),
            'cached': ['F(source)', 'M(source)', 'M(driving_first)', 'ConvexHull multiplier'],
            'per_frame': ['M(driving_target)', 'W', 'G', 'stream encode'],
            'limits': ['No automatic face detection/crop tracking', 'No audio mux',
                       'Not real-time; no specific animal mouth/eye semantic control']}


def sampled_rss_mib():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / (1024*1024), 2)
    except ImportError:
        return None


def checked_fps(meta, override):
    if override is not None:
        return float(override), 'user_override'
    value = meta.get('fps') if isinstance(meta, dict) else None
    try:
        value = float(value)
    except (ValueError, TypeError):
        value = float('nan')
    if not math.isfinite(value) or not 1 <= value <= 120:
        return 25.0, 'fallback_25fps_invalid_video_metadata'
    return value, 'video_metadata'


def run(args, *, reader_factory=None, writer_factory=None, session_factory=None):
    """Injectable IO factories facilitate no-model unit/integration tests."""
    from src.utils.animal_onnx_video import (CachedAnimalDriver, crop_rgb256,
                                             ImageioFfmpegWriter)
    from src.utils.animal_stream_writer import AnimalStreamWriter
    from scripts.run_animal_onnx_frame import open_sessions
    import numpy as np

    paths = output_paths(args)
    models = model_preflight(args) if session_factory is None else []
    if session_factory is None and not all(x['onnx_ready'] for x in models):
        raise FileNotFoundError('Four ONNX models required: {}'.format(models))
    if reader_factory is None:
        import imageio
        reader_factory = lambda path: imageio.get_reader(str(path), 'ffmpeg')
    if writer_factory is None:
        writer_factory = ImageioFfmpegWriter
    from PIL import Image
    with Image.open(args.source) as im:
        source_prepared = crop_rgb256(np.asarray(im.convert('RGB'), dtype=np.uint8), args.source_crop)
    reader = reader_factory(args.driving)
    frames = None
    begin = time.perf_counter()
    rss_samples = []
    frame_count = 0
    sink = None
    try:
        meta = reader.get_meta_data()
        fps, fps_origin = checked_fps(meta, args.fps)
        from src.utils.animal_video_reader import iter_video_frames_rgb
        # Existing generator owns reader.close() upon exhaustion/close.
        frames = iter_video_frames_rgb(args.driving, resize=None,
                          max_frames=args.max_frames or None, reader_factory=lambda _: reader)
        try:
            first = crop_rgb256(next(frames), args.driving_crop)
        except StopIteration as exc:
            raise ValueError('Driving video has no decodable frames') from exc
        rss_samples.append(sampled_rss_mib())
        stamp = time.perf_counter()
        if session_factory is None:
            import onnxruntime as ort
            sessions = open_sessions(args, ort)
        else:
            sessions = session_factory(args)
        ort_init_ms = (time.perf_counter()-stamp)*1000
        stamp = time.perf_counter()
        driver = CachedAnimalDriver(sessions, source_prepared, first,
                                    mode=args.motion_mode, driving_multiplier=args.driving_multiplier)
        cache_init_ms = (time.perf_counter()-stamp)*1000
        rss_samples.append(sampled_rss_mib())

        # Stage MP4/JSON in temporary sibling dir: no partial final artifacts.
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.v070_', dir=str(paths[0].parent)) as td:
            tmp = Path(td)
            tmp_video = tmp/'driven.mp4'
            tmp_comparison = tmp/'comparison.mp4' if args.comparison else None
            sink = AnimalStreamWriter(str(tmp_video), str(tmp_comparison) if tmp_comparison else None,
                                      fps, source_prepared, writer_factory=writer_factory)
            start_render = time.perf_counter()
            try:
                def write_frame(rgb256, is_first):
                    nonlocal frame_count
                    image, metrics = driver.render(rgb256, is_first=is_first)
                    # Existing writer rescales first and driving input only when comparison is enabled.
                    sink.write(image, driving_frame=rgb256)
                    frame_count += 1
                    if frame_count % args.progress_every == 0:
                        print('frames={} W_ms={:.1f} G_ms={:.1f}'.format(
                            frame_count, metrics['W'], metrics['G']), file=sys.stderr, flush=True)
                    rss_samples.append(sampled_rss_mib())
                    # Keep bounded metrics; no growing list of frame timings/frames.
                    if len(rss_samples) > 2:
                        recent = rss_samples.pop()
                        old_max = rss_samples[0]
                        if old_max is None or (recent is not None and recent > old_max):
                            rss_samples[0] = recent

                write_frame(first, True)
                for raw in frames:
                    write_frame(crop_rgb256(raw, args.driving_crop), False)
            finally:
                sink.close()
                sink = None
            render_ms = (time.perf_counter()-start_render)*1000
            rss_samples.append(sampled_rss_mib())
            report = {'schema_version': 1, 'stage': 'v0.7.0_animal_expression_video',
                      'source': str(args.source), 'driving': str(args.driving),
                      'output': str(args.output), 'comparison': str(args.comparison) if args.comparison else None,
                      'fps': fps, 'fps_source': fps_origin, 'frame_count': frame_count,
                      'source_crop': args.source_crop, 'driving_crop': args.driving_crop,
                      'max_frames': args.max_frames, 'threads': args.threads,
                      'ort_session_load_ms': ort_init_ms, 'source_and_anchor_cache_init_ms': cache_init_ms,
                      'video_generation_ms': render_ms, 'overall_elapsed_ms': (time.perf_counter()-begin)*1000,
                      'average_wall_seconds_per_frame': render_ms/frame_count/1000,
                      'effective_generation_fps': frame_count/(render_ms/1000) if render_ms>0 else None,
                      'rss_sampled_peak_mib': max((v for v in rss_samples if v is not None), default=None),
                      'rss_note': 'Sampled process RSS, not guaranteed OS peak; psutil optional',
                      'model_metrics': driver.summary(),
                      'limitations': ['CPU and fixed-shape animal v1.1 ONNX only',
                                      'No automatic face crop, landmarks, or tracking',
                                      'No stitching S or pasteback; output 512x512 RGB',
                                      'No audio mux; MP4 is silent',
                                      'Expression-only freezes R/t but learned exp is not semantically separated',
                                      'Source photo should have head-centric framing; pre-cropped driver recommended']}
            (tmp/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n',encoding='utf-8')
            dests = [(tmp_video, paths[0]), (tmp/'report.json', paths[1])]
            if tmp_comparison:
                dests.append((tmp_comparison, args.comparison))
            for _, dest in dests:
                dest.parent.mkdir(parents=True, exist_ok=True)
            # Overwrite only after all artifacts exist and both encoders close.
            for src, dest in dests:
                os.replace(src, dest)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return report
    finally:
        if frames is not None:
            frames.close()
        else:
            reader.close()
        if sink is not None:
            sink.close()


def main(argv=None):
    args = make_parser().parse_args(argv)
    validate_args(args)
    if args.plan:
        print(json.dumps(plan(args), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    run(args)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, FileNotFoundError, FileExistsError, ImportError) as exc:
        print('[ERROR]', exc, file=sys.stderr)
        sys.exit(2)
