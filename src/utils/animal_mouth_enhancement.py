# coding: utf-8
"""Experimental selective movement of implicit animal-v1.1 keypoints.

The 21 keypoints are learned LATENT coordinates, NOT documented anatomical
landmarks. There is deliberately NO automatic mouth/jaw preset. Choose indices
only after an empirical ablation, and check the resulting images for distortion.

This module is stdlib-only until apply_mouth_enhancement() is called.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MouthEnhanceConfig:
    enabled: bool = False
    mouth_kp_indices: tuple = ()
    lower_mouth_kp_indices: tuple = ()
    x_scale: float = 1.0
    y_scale: float = 1.0
    z_scale: float = 1.0
    lower_y_bias: float = 0.0

    def as_dict(self):
        return {
            'enabled': self.enabled,
            'mouth_kp_indices': list(self.mouth_kp_indices),
            'lower_mouth_kp_indices': list(self.lower_mouth_kp_indices),
            'x_scale': float(self.x_scale),
            'y_scale': float(self.y_scale),
            'z_scale': float(self.z_scale),
            'lower_y_bias': float(self.lower_y_bias),
            'indices_are_anatomical_landmarks': False,
        }


def parse_index_csv(value):
    """Parse zero-based index list; empty string selects no points."""
    if not isinstance(value, str):
        raise ValueError('keypoint index list must be a string')
    if not value.strip():
        return ()
    tokens = value.split(',')
    if any(not token.strip() for token in tokens):
        raise ValueError('keypoint indices must be comma-separated integers without blanks')
    try:
        result = tuple(int(token.strip()) for token in tokens)
    except ValueError as exc:
        raise ValueError('keypoint indices must be comma-separated integers') from exc
    return result


def validate_mouth_config(cfg, num_kp=21):
    """Reject ambiguous or invalid selections; never guess anatomical indices."""
    if not isinstance(cfg, MouthEnhanceConfig):
        raise ValueError('mouth_cfg must be a MouthEnhanceConfig')
    if type(cfg.enabled) is not bool:
        raise ValueError('mouth enhancement enabled must be boolean')
    for field in ('mouth_kp_indices', 'lower_mouth_kp_indices'):
        indices = getattr(cfg, field)
        if not isinstance(indices, tuple):
            raise ValueError('{} must be a tuple'.format(field))
        if any(type(i) is not int or not 0 <= i < num_kp for i in indices):
            raise ValueError('{} must contain integers within 0..{}'.format(field, num_kp - 1))
        if len(set(indices)) != len(indices):
            raise ValueError('{} contains duplicate indices'.format(field))
    if cfg.lower_mouth_kp_indices and not set(cfg.lower_mouth_kp_indices).issubset(cfg.mouth_kp_indices):
        raise ValueError('lower-mouth indices must be a subset of mouth indices')
    for field in ('x_scale', 'y_scale', 'z_scale'):
        value = getattr(cfg, field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('{} must be a finite number'.format(field))
        if not 0 <= value <= 3:
            raise ValueError('{} must be within 0..3 (experimental safety bound)'.format(field))
    bias = cfg.lower_y_bias
    if isinstance(bias, bool) or not isinstance(bias, (int, float)) or not math.isfinite(bias):
        raise ValueError('lower_y_bias must be finite')
    if abs(bias) > 0.05:
        raise ValueError('lower_y_bias must be within -0.05..0.05 (experimental safety bound)')
    if bias != 0 and not cfg.lower_mouth_kp_indices:
        raise ValueError('nonzero lower_y_bias requires --mouth-lower-kp-indices')
    if cfg.enabled:
        if not cfg.mouth_kp_indices:
            raise ValueError('mouth enhancement requires --mouth-kp-indices; no automatic landmark preset')
    elif (cfg.mouth_kp_indices or cfg.lower_mouth_kp_indices or
          (cfg.x_scale, cfg.y_scale, cfg.z_scale) != (1, 1, 1) or bias != 0):
        raise ValueError('disabled mouth enhancement has nondefault parameters')
    return cfg


def apply_mouth_enhancement(kp_source, kp_driving, cfg):
    """Scale selected point DELTAS around source, then apply optional signed Y bias.

    Inputs must be FP32 [1,21,3]. Nonselected points remain exactly unchanged;
    neither input is mutated. Signed Y bias is not guaranteed to open a jaw.
    """
    import numpy as np
    from .animal_onnx_chain import require_tensor

    validate_mouth_config(cfg)
    source = require_tensor(kp_source, (1, 21, 3), label='mouth source kp')
    driving = require_tensor(kp_driving, (1, 21, 3), label='mouth driving kp')
    if not cfg.enabled:
        return driving
    out = driving.copy()
    indices = list(cfg.mouth_kp_indices)
    delta = driving[:, indices, :] - source[:, indices, :]
    scales = np.asarray([cfg.x_scale, cfg.y_scale, cfg.z_scale], dtype=np.float32)
    out[:, indices, :] = source[:, indices, :] + delta * scales[None, None, :]
    if cfg.lower_mouth_kp_indices:
        lower = list(cfg.lower_mouth_kp_indices)
        out[:, lower, 1] += np.float32(cfg.lower_y_bias)
    return require_tensor(np.ascontiguousarray(out, dtype=np.float32), (1, 21, 3), label='mouth enhanced kp')
