#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LivePortrait animal ONNX V0.5: preflight, CPU parity, and CPU latency.

Examples:
  python scripts/validate_animal_onnx.py --mode preflight --model all
  python scripts/validate_animal_onnx.py --mode benchmark --model f --repeat 5
  python scripts/validate_animal_onnx.py --mode compare --model f --output-json build/f_parity.json

Preflight requires standard-library Python only. It does NOT validate ONNX
operators or checkpoint integrity. Benchmark requires onnxruntime + NumPy but
no PyTorch/weights. Compare also requires CPU PyTorch, PyYAML and weights.
It never modifies inference, model files or checkpoints.
"""

import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.animal_onnx_contract import get_contract, selected_models, validate_model_config
from src.utils.animal_onnx_validation import (
    CHECKPOINTS, compare_arrays, inspect_model_files, inventory,
    latency_summary, validate_ort_io,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=('preflight', 'benchmark', 'compare'), default='preflight')
    p.add_argument('--model', choices=('f', 'm', 'w', 'g', 'all'), default='f', type=str.lower)
    p.add_argument('--weights-root', type=Path, default=ROOT / 'pretrained_weights')
    p.add_argument('--onnx-dir', type=Path, default=ROOT / 'build/onnx_animal_v04')
    p.add_argument('--models-yaml', type=Path, default=ROOT / 'src/config/models.yaml')
    p.add_argument('--output-json', type=Path, help='Optional path for a machine-readable report')
    p.add_argument('--strict', action='store_true',
                   help='Preflight exits nonzero when selected models are not compare-ready')
    p.add_argument('--experimental-w', action='store_true',
                   help='Opt in to executing W (5D volumetric GridSample)')
    p.add_argument('--threads', type=int, default=2, help='ORT CPU intra-op thread count')
    p.add_argument('--warmup', type=int, default=1, help='ORT untimed warmup iterations')
    p.add_argument('--repeat', type=int, default=5, help='ORT timed iterations')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--atol', type=float, default=1e-3)
    p.add_argument('--rtol', type=float, default=1e-3)
    return p.parse_args(argv)


def validate_args(args):
    if args.threads < 1 or args.warmup < 0 or args.repeat < 1:
        raise ValueError('Require --threads >= 1, --warmup >= 0, --repeat >= 1')
    import math
    if not all(math.isfinite(x) and x >= 0 for x in (args.atol, args.rtol)):
        raise ValueError('Require finite, nonnegative --atol and --rtol')
    if args.mode != 'preflight' and 'w' in selected_models(args.model) and not args.experimental_w:
        raise ValueError('W uses 5D GridSample; use --experimental-w for benchmark/compare')


def _load_ort():
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError('This mode needs NumPy and onnxruntime. '
                           'Use --mode preflight without them.') from exc
    return np, ort


def _inputs_for_model(code, contract, np, seed):
    rng = np.random.default_rng(seed + ord(code))
    feeds = {}
    for spec in contract.inputs:
        if spec.name in ('source_image', 'image'):
            value = rng.random(spec.shape, dtype=np.float32)
        else:
            value = (rng.standard_normal(spec.shape).astype(np.float32) * 0.05)
        feeds[spec.name] = np.ascontiguousarray(value, dtype=np.float32)
    return feeds


def _session(onnx_path, ort, threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(onnx_path), sess_options=opts,
                                   providers=['CPUExecutionProvider'])
    if 'CPUExecutionProvider' not in session.get_providers():
        raise RuntimeError('ORT session was not initialized with CPUExecutionProvider')
    return session


def run_onnx(session, contract, feed, *, warmup, repeat, benchmark):
    validate_ort_io(session, contract)
    names = [s.name for s in contract.outputs]
    for _ in range(warmup if benchmark else 0):
        session.run(names, feed)
    timings = []
    last = None
    for _ in range(repeat if benchmark else 1):
        start = time.perf_counter()
        last = session.run(names, feed)
        if benchmark:
            timings.append((time.perf_counter() - start) * 1000)
    if len(last) != len(contract.outputs):
        raise ValueError('Wrong number of ORT outputs for {}'.format(contract.code))
    for spec, tensor in zip(contract.outputs, last):
        if tuple(tensor.shape) != spec.shape or str(tensor.dtype) != 'float32':
            raise ValueError('ORT output {} does not satisfy shape/dtype contract'.format(spec.name))
    return last, (latency_summary(timings) if benchmark else None)


def _torch_outputs(code, contract, feed, args):
    """Only load one PyTorch module, then release it before ORT session creation."""
    try:
        import torch
        import yaml
        from scripts.export_animal_onnx import _create_model, _make_adapter, _normalize_outputs
    except ImportError as exc:
        raise RuntimeError('Parity needs CPU PyTorch and PyYAML installed separately.') from exc
    if not args.models_yaml.is_file():
        raise FileNotFoundError('Missing architecture YAML: {}'.format(args.models_yaml))
    with args.models_yaml.open(encoding='utf-8') as fh:
        config = yaml.safe_load(fh)
    validate_model_config(config)
    path = args.weights_root / CHECKPOINTS[code]
    model = _create_model(code, config, path, torch)
    adapter = _make_adapter(code, model, torch).eval()
    try:
        tensor_inputs = tuple(torch.from_numpy(feed[s.name]) for s in contract.inputs)
        with torch.inference_mode():
            outputs = _normalize_outputs(adapter(*tensor_inputs))
        if len(outputs) != len(contract.outputs):
            raise ValueError('PyTorch output count differs from contract')
        detached = []
        for spec, t in zip(contract.outputs, outputs):
            arr = t.detach().float().cpu().numpy().copy()
            if tuple(arr.shape) != spec.shape:
                raise ValueError('PyTorch output {} shape differs from contract'.format(spec.name))
            detached.append(arr)
        return detached
    finally:
        del adapter, model
        gc.collect()


def execute_one(code, args, np, ort):
    contract = get_contract(code)
    inventory_row = inspect_model_files(code, args.weights_root, args.onnx_dir)
    available = (inventory_row['ready_compare'] if args.mode == 'compare'
                 else inventory_row['ready_benchmark'])
    if not available:
        relevant = (inventory_row['issues'] if args.mode == 'compare'
                    else [x for x in inventory_row['issues'] if 'checkpoint' not in x])
        raise FileNotFoundError('{} not ready: {}. Run --mode preflight --model {}.'.format(
            code.upper(), '; '.join(relevant), code))
    feed = _inputs_for_model(code, contract, np, args.seed)
    reference = _torch_outputs(code, contract, feed, args) if args.mode == 'compare' else None
    session = _session(inventory_row['onnx']['path'], ort, args.threads)
    try:
        actual, stats = run_onnx(session, contract, feed, warmup=args.warmup,
                                 repeat=args.repeat, benchmark=args.mode == 'benchmark')
        result = {'model': code.upper(), 'mode': args.mode,
                  'provider': 'CPUExecutionProvider', 'verified_contract': True,
                  'onnx_path': inventory_row['onnx']['path']}
        if stats is not None:
            result['latency'] = stats
        if reference is not None:
            metrics = {spec.name: compare_arrays(a, b, atol=args.atol, rtol=args.rtol)
                       for spec, a, b in zip(contract.outputs, reference, actual)}
            result['parity'] = metrics
            result['passed'] = all(m['passed'] for m in metrics.values())
            result['tolerances'] = {'atol': args.atol, 'rtol': args.rtol}
        else:
            result['passed'] = True
        return result
    finally:
        del session
        gc.collect()


def _emit(report, output_path):
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    print(serialized, end='')
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding='utf-8')


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    if args.mode == 'preflight':
        report = inventory(args.model, args.weights_root, args.onnx_dir)
        _emit(report, args.output_json)
        return 2 if args.strict and not all(m['ready_compare'] for m in report['models']) else 0

    # Validate paths BEFORE importing the potentially heavy runtime packages.
    for code in selected_models(args.model):
        entry = inspect_model_files(code, args.weights_root, args.onnx_dir)
        required = entry['ready_compare'] if args.mode == 'compare' else entry['ready_benchmark']
        if not required:
            raise FileNotFoundError('{} missing prerequisite: {}'.format(code.upper(),
                                     '; '.join(entry['issues'] if args.mode == 'compare' else
                                               [i for i in entry['issues'] if 'checkpoint' not in i])))

    np, ort = _load_ort()
    rows = []
    for code in selected_models(args.model):
        rows.append(execute_one(code, args, np, ort))
    report = {'schema_version': 1, 'scope': 'animal_v1.1_fixed_batch1_fp32',
              'mode': args.mode, 'models': rows,
              'note': 'CPU model-only validation; not a video/GT1030 benchmark.'}
    _emit(report, args.output_json)
    return 0 if all(row['passed'] for row in rows) else 3


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        print('[ERROR]', exc, file=sys.stderr)
        sys.exit(2)
