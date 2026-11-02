# coding: utf-8
"""CPU-only ONNX validation helpers for animal v1.1.

Inventory and manifest checks use only Python's standard library.
NumPy is imported lazily only for numerical parity metrics. No model loading,
PyTorch, CUDA or ONNX Runtime imports occur in this module.
"""

import json
import math
from pathlib import Path

from .animal_onnx_contract import get_contract, selected_models


CHECKPOINTS = {
    'f': 'liveportrait_animals/base_models_v1.1/appearance_feature_extractor.pth',
    'm': 'liveportrait_animals/base_models_v1.1/motion_extractor.pth',
    'w': 'liveportrait_animals/base_models_v1.1/warping_module.pth',
    'g': 'liveportrait_animals/base_models_v1.1/spade_generator.pth',
}


def _file_metadata(path):
    path = Path(path)
    found = path.is_file()
    size = path.stat().st_size if found else 0
    return {'path': str(path), 'exists': found, 'bytes': size, 'nonempty': found and size > 0}


def manifest_issues(manifest, contract, onnx_name):
    """Return human-readable issues. A manifest is metadata, not proof of provenance."""
    if not isinstance(manifest, dict):
        return ['manifest must contain a JSON object']
    issues = []
    expected = contract.as_dict()
    for key in ('model', 'model_type', 'inputs', 'outputs'):
        if manifest.get(key) != expected[key]:
            issues.append('manifest {} differs from V0.4 contract'.format(key))
    if manifest.get('animal_version') != 'v1.1':
        issues.append('manifest animal_version must be v1.1')
    if manifest.get('onnx_file') != onnx_name:
        issues.append('manifest onnx_file does not match requested graph')
    opset = manifest.get('opset_requested')
    if type(opset) is not int or opset < (20 if contract.code == 'w' else 16):
        issues.append('manifest opset_requested is invalid for {}'.format(contract.code.upper()))
    # V0.4 only records the checkpoint filename, not the content digest.
    # Never claim this manifest cryptographically binds the checkpoint.
    return issues


def inspect_model_files(code, weights_root, onnx_dir):
    """Gather shallow checks without loading weights, ONNX, NumPy or PyTorch."""
    contract = get_contract(code)
    weights = _file_metadata(Path(weights_root) / CHECKPOINTS[code])
    graph_path = Path(onnx_dir) / 'animal_{}_v1.1.onnx'.format(code)
    graph = _file_metadata(graph_path)
    manifest_path = graph_path.with_suffix('.json')
    manifest_file = _file_metadata(manifest_path)
    issues = []
    if not weights['nonempty']:
        issues.append('missing or empty animal v1.1 checkpoint')
    if not graph['nonempty']:
        issues.append('missing or empty ONNX graph')
    if not manifest_file['nonempty']:
        issues.append('missing or empty V0.4 export manifest')
    elif manifest_file['bytes'] > 2 * 1024 * 1024:
        issues.append('manifest exceeds 2 MiB safety limit')
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            issues.extend(manifest_issues(manifest, contract, graph_path.name))
        except (OSError, ValueError, UnicodeError) as exc:
            issues.append('invalid JSON manifest: {}'.format(type(exc).__name__))
    graph_issues = [i for i in issues if 'checkpoint' not in i]
    return {
        'model': code.upper(), 'experimental': contract.experimental,
        'checkpoint': weights, 'onnx': graph, 'manifest': manifest_file,
        'ready_benchmark': not graph_issues,
        'ready_compare': not issues,
        'issues': issues,
        'note': 'File presence/manifest metadata only; no graph or numerical validation performed.',
    }


def inventory(model, weights_root, onnx_dir):
    return {
        'schema_version': 1,
        'mode': 'preflight',
        'scope': 'animal_v1.1_fixed_batch1_fp32',
        'models': [inspect_model_files(code, weights_root, onnx_dir)
                   for code in selected_models(model)],
    }


def validate_ort_io(session, contract):
    """Reject swapped tensors or dynamic/wrong I/O sizes before running ORT."""
    for label, actual, expected in (
        ('input', session.get_inputs(), contract.inputs),
        ('output', session.get_outputs(), contract.outputs),
    ):
        if len(actual) != len(expected):
            raise ValueError('{}: expected {} {}(s), found {}'.format(
                contract.code.upper(), len(expected), label, len(actual)))
        for node, spec in zip(actual, expected):
            if node.name != spec.name:
                raise ValueError('{} {} name mismatch: expected {}, got {}'.format(
                    contract.code.upper(), label, spec.name, node.name))
            if tuple(node.shape) != spec.shape:
                raise ValueError('{} {} {} shape mismatch: expected {}, got {}'.format(
                    contract.code.upper(), label, spec.name, spec.shape, node.shape))
            if getattr(node, 'type', None) != 'tensor(float)':
                raise ValueError('{} {} {} must be float32 tensor'.format(
                    contract.code.upper(), label, spec.name))


def latency_summary(samples_ms):
    """Deterministic wall-time statistics; does not imply CUDA benchmarking."""
    samples = sorted(float(v) for v in samples_ms)
    if not samples or not all(math.isfinite(v) and v >= 0 for v in samples):
        raise ValueError('latency samples must be finite, nonnegative and nonempty')

    def percentile(p):
        at = (len(samples) - 1) * p / 100.0
        lo, hi = math.floor(at), math.ceil(at)
        return samples[lo] + (samples[hi] - samples[lo]) * (at - lo)

    return {
        'count': len(samples), 'min_ms': samples[0],
        'mean_ms': sum(samples) / len(samples),
        'p50_ms': percentile(50), 'p95_ms': percentile(95),
        'max_ms': samples[-1],
    }


def compare_arrays(reference, candidate, *, atol=1e-3, rtol=1e-3):
    """Elementwise PyTorch-vs-ORT comparison, with robust nonfinite detection.

    The real comparison requires NumPy, but `preflight` and `--help` do not.
    Uses the standard `abs(a-b) <= atol + rtol*abs(a)` condition.
    """
    if not all(math.isfinite(x) and x >= 0 for x in (atol, rtol)):
        raise ValueError('atol/rtol must be finite nonnegative numbers')
    import numpy as np
    a = np.asarray(reference)
    b = np.asarray(candidate)
    if a.shape != b.shape:
        raise ValueError('parity shape mismatch: reference {} vs candidate {}'.format(
            a.shape, b.shape))
    if a.size == 0:
        raise ValueError('empty arrays are not valid model outputs')
    if a.dtype != np.dtype('float32') or b.dtype != np.dtype('float32'):
        raise ValueError('parity tensors must both be float32, got {} / {}'.format(a.dtype, b.dtype))
    finite = np.isfinite(a) & np.isfinite(b)
    count_nonfinite = int(a.size - np.count_nonzero(finite))
    if count_nonfinite:
        # Treat every nonfinite pair as a mismatch, even NaN vs NaN.
        # Count any additional finite-but-out-of-tolerance elements accurately.
        x = a[finite].astype(np.float64)
        y = b[finite].astype(np.float64)
        finite_bad = int(np.count_nonzero(np.abs(x - y) > (atol + rtol * np.abs(x))))
        return {
            'shape': list(a.shape), 'elements': int(a.size),
            'passed': False, 'mismatches': count_nonfinite + finite_bad,
            'nonfinite_elements': count_nonfinite,
            'max_abs_error': None, 'mean_abs_error': None,
            'max_relative_error': None,
        }
    # Promote for measurement, so near-float32-overflow inputs do not hide errors.
    x = a.astype(np.float64)
    y = b.astype(np.float64)
    diff = np.abs(x - y)
    allowed = atol + rtol * np.abs(x)
    bad = diff > allowed
    return {
        'shape': list(a.shape), 'elements': int(a.size),
        'passed': not bool(np.any(bad)), 'mismatches': int(np.count_nonzero(bad)),
        'nonfinite_elements': 0,
        'max_abs_error': float(np.max(diff)),
        'mean_abs_error': float(np.mean(diff)),
        'max_relative_error': float(np.max(diff / np.maximum(np.abs(x), 1e-6))),
    }
