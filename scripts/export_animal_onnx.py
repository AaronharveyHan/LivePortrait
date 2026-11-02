#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V0.4 experimental animal-v1.1 ONNX export; no pipeline/backend changes.

Examples:
  python scripts/export_animal_onnx.py --plan --model all
  python scripts/export_animal_onnx.py --model f --validate-runtime
  python scripts/export_animal_onnx.py --model w --experimental-w --opset 20

--plan uses only the standard library. Actual export needs CPU PyTorch,
PyYAML, ONNX and onnxscript; --validate-runtime also needs onnxruntime.
"""

import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.animal_onnx_contract import (
    MOTION_OUTPUT_NAMES, get_contract, selected_models,
    validate_model_config, check_shapes,
)

CHECKPOINTS = {
    'f': 'liveportrait_animals/base_models_v1.1/appearance_feature_extractor.pth',
    'm': 'liveportrait_animals/base_models_v1.1/motion_extractor.pth',
    'w': 'liveportrait_animals/base_models_v1.1/warping_module.pth',
    'g': 'liveportrait_animals/base_models_v1.1/spade_generator.pth',
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='f', choices=('f', 'm', 'w', 'g', 'all'), type=str.lower,
                   help='F/M/W/G or all; default F to avoid excessive CPU memory usage')
    p.add_argument('--plan', action='store_true',
                   help='Show I/O contract + missing weights; no torch/onnx/CUDA required')
    p.add_argument('--weights-root', type=Path, default=ROOT / 'pretrained_weights')
    p.add_argument('--models-yaml', type=Path, default=ROOT / 'src/config/models.yaml')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'build/onnx_animal_v04')
    p.add_argument('--opset', type=int, default=20)
    p.add_argument('--experimental-w', action='store_true',
                   help='Explicitly allow W export (requires volumetric GridSample support)')
    p.add_argument('--exporter', choices=('dynamo', 'legacy'), default='dynamo')
    p.add_argument('--validate-runtime', action='store_true',
                   help='Run ONNX Runtime CPU and compare outputs (slow)')
    p.add_argument('--skip-onnx-check', action='store_true',
                   help='Skip ONNX structural checker, not recommended')
    p.add_argument('--overwrite', action='store_true', help='Allow replacing existing ONNX files')
    return p.parse_args(argv)


def checkpoint_path(weights_root, name):
    return weights_root / CHECKPOINTS[name]


def plan_payload(args):
    rows = []
    for name in selected_models(args.model):
        path = checkpoint_path(args.weights_root, name)
        row = get_contract(name).as_dict()
        row['checkpoint_relative'] = CHECKPOINTS[name]
        row['checkpoint_exists'] = path.is_file()
        rows.append(row)
    return {
        'schema_version': 1,
        'scope': 'animal_v1.1_fixed_batch1_fp32',
        'opset_requested': args.opset,
        'exporter': args.exporter,
        'plan_only': bool(args.plan),
        'models': rows,
        'note': ('W requires an ONNX exporter/runtime supporting 5D GridSample, '
                 'and remains experimental; the CLI never edits LivePortrait inference.'),
    }


def validate_export_selection(args):
    if args.opset < 16:
        raise ValueError('opset must be >=16; use 20 or later for experimental W')
    for name in selected_models(args.model):
        if name == 'w':
            if not args.experimental_w:
                raise ValueError('W uses 5D GridSample: pass --experimental-w to opt in')
            if args.opset < 20:
                raise ValueError('W uses volumetric 5D GridSample: ONNX opset 20 or later is required')
            if args.exporter == 'legacy':
                raise ValueError('W requires --exporter dynamo; legacy 5D GridSample may be invalid')


def _load_export_deps():
    try:
        import torch
        import yaml
        import onnx
    except ImportError as exc:
        raise RuntimeError('Export dependencies missing: install CPU torch, pyyaml, '
                           'onnx and onnxscript. --plan requires none of these.') from exc
    return torch, yaml, onnx


def _create_model(name, config, checkpoint, torch):
    """Load only one model. Importing here avoids XPose and face-detector startup."""
    params = config['model_params']
    if name == 'f':
        from src.modules.appearance_feature_extractor import AppearanceFeatureExtractor
        model = AppearanceFeatureExtractor(**params['appearance_feature_extractor_params'])
    elif name == 'm':
        from src.modules.motion_extractor import MotionExtractor
        model = MotionExtractor(**params['motion_extractor_params'])
    elif name == 'w':
        from src.modules.warping_network import WarpingNetwork
        model = WarpingNetwork(**params['warping_module_params'])
    elif name == 'g':
        from src.modules.spade_generator import SPADEDecoder
        model = SPADEDecoder(**params['spade_generator_params'])
    else:
        raise ValueError(name)
    # Repository helper.load_model() also expects a plain state dict here.
    try:
        weights = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
    except TypeError:  # older supported torch versions
        weights = torch.load(str(checkpoint), map_location='cpu')
    model.load_state_dict(weights, strict=True)
    return model.eval().float()


def _make_adapter(name, model, torch):
    if name == 'm':
        class MotionAdapter(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, image):
                info = self.inner(image)
                return tuple(info[key] for key in MOTION_OUTPUT_NAMES)
        return MotionAdapter(model)

    if name == 'w':
        class WarpingAdapter(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, feature_3d, kp_driving, kp_source):
                # Actual WarpingNetwork.forward positional order: driving, source.
                return self.inner(feature_3d, kp_driving, kp_source)['out']
        return WarpingAdapter(model)
    return model


def _sample_inputs(name, contract, torch):
    torch.manual_seed(17)
    inputs = []
    for tensor in contract.inputs:
        if tensor.name in ('source_image', 'image'):
            value = torch.rand(tensor.shape, dtype=torch.float32)
        elif tensor.name.startswith('kp_'):
            value = torch.randn(tensor.shape, dtype=torch.float32) * 0.05
        else:
            value = torch.randn(tensor.shape, dtype=torch.float32) * 0.05
        inputs.append(value)
    return tuple(inputs)


def _normalize_outputs(output):
    return tuple(output) if isinstance(output, (tuple, list)) else (output,)


def _validate_ort(path, contract, inputs, torch_outputs):
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('--validate-runtime needs onnxruntime and numpy') from exc
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    expected_names = [t.name for t in contract.inputs]
    actual_names = [t.name for t in session.get_inputs()]
    if expected_names != actual_names:
        raise RuntimeError('ORT input mismatch: expected {}, found {}'.format(
            expected_names, actual_names))
    feed = {spec.name: tensor.detach().cpu().numpy()
            for spec, tensor in zip(contract.inputs, inputs)}
    got = session.run([o.name for o in contract.outputs], feed)
    check_shapes(got, contract)
    for spec, torch_output, ort_output in zip(contract.outputs, torch_outputs, got):
        np.testing.assert_allclose(torch_output.detach().cpu().numpy(), ort_output,
                                   rtol=1e-3, atol=1e-3,
                                   err_msg='ORT vs PyTorch mismatch: {}'.format(spec.name))
    print('[OK] ONNX Runtime CPU parity:', contract.code.upper())


def export_one(name, args, config, torch, onnx):
    contract = get_contract(name)
    source = checkpoint_path(args.weights_root, name)
    if not source.is_file():
        raise FileNotFoundError('Missing checkpoint: {}. Download animal v1.1 weights first.'.format(source))
    dest = args.output_dir / ('animal_{}_v1.1.onnx'.format(name))
    if dest.exists() and not args.overwrite:
        raise FileExistsError('{} exists; use --overwrite to replace it'.format(dest))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = None
    adapter = None
    inputs = None
    try:
        model = _create_model(name, config, source, torch)
        adapter = _make_adapter(name, model, torch).eval()
        inputs = _sample_inputs(name, contract, torch)
        with torch.inference_mode():
            actual = _normalize_outputs(adapter(*inputs))
        check_shapes(actual, contract)
        print('[OK] PyTorch output shapes:', name.upper())

        # Modern exporter is intentionally explicit; PyTorch 2.9+ defaults
        # dynamo=True but older releases differ. No half precision on CPU.
        print('[EXPORT]', name.upper(), '->', dest)
        try:
            torch.onnx.export(
                adapter, inputs, str(dest),
                input_names=[s.name for s in contract.inputs],
                output_names=[s.name for s in contract.outputs],
                opset_version=args.opset,
                dynamo=(args.exporter == 'dynamo'),
                external_data=True,
            )
        except Exception:
            # A failure may leave a partial ONNX file; do not present success.
            if dest.is_file():
                dest.unlink()
            raise
        if not args.skip_onnx_check:
            onnx.checker.check_model(str(dest))  # file path supports large graphs
            print('[OK] ONNX checker:', dest.name)
        if args.validate_runtime:
            _validate_ort(dest, contract, inputs, actual)
        manifest = contract.as_dict()
        manifest.update({'schema_version': 1, 'animal_version': 'v1.1',
                         'opset_requested': args.opset, 'torch_version': torch.__version__,
                         'exporter': args.exporter,
                         'checkpoint_filename': source.name,
                         'onnx_file': dest.name,
                         'onnx_checker_passed': not args.skip_onnx_check,
                         'runtime_cpu_parity_passed': bool(args.validate_runtime)})
        output_manifest = dest.with_suffix('.json')
        output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                   encoding='utf-8')
        print('[OK] Exported:', dest)
    finally:
        del adapter, model, inputs
        gc.collect()


def main(argv=None):
    args = parse_args(argv)
    if args.plan:
        print(json.dumps(plan_payload(args), ensure_ascii=False, indent=2))
        return 0
    validate_export_selection(args)
    names = selected_models(args.model)
    missing = [str(checkpoint_path(args.weights_root, name))
               for name in names if not checkpoint_path(args.weights_root, name).is_file()]
    if missing:
        print('Missing weights (nothing exported):\n  ' + '\n  '.join(missing), file=sys.stderr)
        return 2
    torch, yaml, onnx = _load_export_deps()
    if args.models_yaml.is_file():
        with args.models_yaml.open(encoding='utf-8') as fh:
            config = yaml.safe_load(fh)
    else:
        raise FileNotFoundError('Missing model configuration: {}'.format(args.models_yaml))
    validate_model_config(config)
    for name in names:
        export_one(name, args, config, torch, onnx)
    print('[DONE] Export request complete. Original LivePortrait pipeline unchanged.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, FileNotFoundError, FileExistsError) as exc:
        print('[ERROR]', exc, file=sys.stderr)
        sys.exit(1)
