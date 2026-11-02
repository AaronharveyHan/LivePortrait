# coding: utf-8
"""Pure-Python ONNX I/O contracts for LivePortrait animal F/M/W/G.

This module intentionally does not import torch, ONNX, CUDA or checkpoint code.
V0.4 contracts fix batch=1 and image=256x256; they are not generic shape
inference for arbitrarily changed LivePortrait model configurations.
"""

from dataclasses import dataclass
from typing import Tuple


MODEL_NAMES = ('f', 'm', 'w', 'g')
MOTION_OUTPUT_NAMES = ('pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp')

# GitHub animal v1.1 (src/config/models.yaml) architectural invariants.
# Validate before export, rather than silently produce a mislabeled graph.
REQUIRED_MODEL_PARAMS = {
    'appearance_feature_extractor_params': {
        'image_channel': 3, 'num_down_blocks': 2,
        'max_features': 512, 'reshape_channel': 32, 'reshape_depth': 16,
    },
    'motion_extractor_params': {'num_kp': 21},
    'warping_module_params': {
        'num_kp': 21, 'block_expansion': 64, 'num_down_blocks': 2,
        'reshape_channel': 32,
    },
    'spade_generator_params': {
        'upscale': 2, 'block_expansion': 64,
        'num_down_blocks': 2, 'max_features': 512,
    },
}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: str = 'float32'

    def as_dict(self):
        return {'name': self.name, 'shape': list(self.shape), 'dtype': self.dtype}


@dataclass(frozen=True)
class ModelContract:
    code: str
    model_type: str
    checkpoint_field: str
    inputs: Tuple[TensorSpec, ...]
    outputs: Tuple[TensorSpec, ...]
    experimental: bool = False

    def as_dict(self):
        return {
            'model': self.code.upper(),
            'model_type': self.model_type,
            'checkpoint_field': self.checkpoint_field,
            'experimental': self.experimental,
            'inputs': [i.as_dict() for i in self.inputs],
            'outputs': [o.as_dict() for o in self.outputs],
        }


def get_contract(name):
    """Return the fixed V0.4 batch-one animal v1.1 I/O contract."""
    if not isinstance(name, str):
        raise ValueError('Model must be F, M, W or G')
    key = name.lower()
    if key == 'f':
        return ModelContract(
            'f', 'appearance_feature_extractor', 'checkpoint_F_animal',
            (TensorSpec('source_image', (1, 3, 256, 256)),),
            (TensorSpec('feature_3d', (1, 32, 16, 64, 64)),),
        )
    if key == 'm':
        return ModelContract(
            'm', 'motion_extractor', 'checkpoint_M_animal',
            (TensorSpec('image', (1, 3, 256, 256)),),
            (
                TensorSpec('pitch', (1, 66)),
                TensorSpec('yaw', (1, 66)),
                TensorSpec('roll', (1, 66)),
                TensorSpec('t', (1, 3)),
                TensorSpec('exp', (1, 63)),
                TensorSpec('scale', (1, 1)),
                TensorSpec('kp', (1, 63)),
            ),
        )
    if key == 'w':
        return ModelContract(
            'w', 'warping_module', 'checkpoint_W_animal',
            (
                TensorSpec('feature_3d', (1, 32, 16, 64, 64)),
                TensorSpec('kp_driving', (1, 21, 3)),
                TensorSpec('kp_source', (1, 21, 3)),
            ),
            (TensorSpec('warped_feature', (1, 256, 64, 64)),),
            experimental=True,
        )
    if key == 'g':
        return ModelContract(
            'g', 'spade_generator', 'checkpoint_G_animal',
            (TensorSpec('warped_feature', (1, 256, 64, 64)),),
            (TensorSpec('output_image', (1, 3, 512, 512)),),
        )
    raise ValueError('Unknown model: {!r}; choose F, M, W or G'.format(name))


def selected_models(name):
    """Return a stable ordered tuple for CLI selection."""
    if not isinstance(name, str):
        raise ValueError('Model name must be a string')
    name = name.lower()
    if name == 'all':
        return MODEL_NAMES
    if name in MODEL_NAMES:
        return (name,)
    raise ValueError('Unknown model selection: {!r}'.format(name))


def validate_model_config(config):
    """Fail closed if the checked-in YAML drifts from the documented v1.1 graph."""
    try:
        params = config['model_params']
    except (TypeError, KeyError) as exc:
        raise ValueError('Missing model_params in src/config/models.yaml') from exc
    failures = []
    for section, fields in REQUIRED_MODEL_PARAMS.items():
        actual = params.get(section, {}) or {}
        for name, expected in fields.items():
            if actual.get(name) != expected:
                failures.append('{}.{}, expected {}, found {}'.format(
                    section, name, expected, actual.get(name)))
    if failures:
        raise ValueError('ONNX contract mismatch with models.yaml: ' + '; '.join(failures))


def check_shapes(outputs, contract):
    """Compare actual tensor-like output shapes with a ModelContract."""
    if len(outputs) != len(contract.outputs):
        raise ValueError('{}: expected {} outputs, got {}'.format(
            contract.code.upper(), len(contract.outputs), len(outputs)))
    for actual, expected in zip(outputs, contract.outputs):
        shape = tuple(int(dim) for dim in actual.shape)
        if shape != expected.shape:
            raise ValueError('{}: expected shape {}, got {}'.format(
                expected.name, expected.shape, shape))
