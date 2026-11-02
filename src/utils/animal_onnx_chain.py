# coding: utf-8
"""Animal v1.1 F/M/W/G ONNX frame chain. NumPy, ORT, SciPy imported lazily.

Important: reproduces LivePortraitPipelineAnimal.execute with flag_stitching=False,
flag_do_crop=False, flag_pasteback=False, relative driving motion. The first
DRIVING frame anchors relative motion; a second frame is necessary for movement.
No model weights, torch, detectors, videos or external service are needed here.
"""

import math
import time

from .animal_onnx_contract import get_contract
from .animal_onnx_validation import validate_ort_io

MODELS = ('f', 'm', 'w', 'g')


def _np():
    import numpy as np
    return np


def require_tensor(value, shape, *, label):
    np = _np()
    arr = np.asarray(value)
    if arr.shape != tuple(shape):
        raise ValueError('{} must have shape {}, got {}'.format(label, tuple(shape), arr.shape))
    if arr.dtype != np.dtype('float32'):
        raise ValueError('{} must be float32, got {}'.format(label, arr.dtype))
    if not np.all(np.isfinite(arr)):
        raise ValueError('{} has NaN/Inf'.format(label))
    return np.ascontiguousarray(arr)


def prepare_rgb(image):
    """RGB uint8 HWC 256x256 -> FP32 NCHW 0..1 (same as prepare_source)."""
    np = _np()
    array = np.asarray(image)
    if array.shape != (256, 256, 3) or array.dtype != np.dtype('uint8'):
        raise ValueError('Input must be RGB uint8 HWC 256x256, got {} {}'.format(array.shape, array.dtype))
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)


def pose_to_degree(logits):
    """LivePortrait src/utils/camera.py headpose_pred_to_degree (66 bins)."""
    np = _np()
    x = require_tensor(logits, (1, 66), label='headpose logits')
    z = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(z).astype(np.float32)
    weights = ex / ex.sum(axis=1, keepdims=True)
    idx = np.arange(66, dtype=np.float32)[None]
    return (np.sum(weights * idx, axis=1, dtype=np.float32) * np.float32(3) - np.float32(97.5)).astype(np.float32)


def rotation_matrix(pitch, yaw, roll):
    """Return row-vector right-multiplication rotation R=(Rz @ Ry @ Rx)^T."""
    np = _np()
    angles = []
    for label, val in (('pitch', pitch), ('yaw', yaw), ('roll', roll)):
        x = require_tensor(val, (1,), label=label)
        angles.append(np.float32(x[0] / 180.0 * np.pi))
    x, y, z = angles
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return np.ascontiguousarray((rz @ ry @ rx).T[None], dtype=np.float32)


def motion_info(raw):
    """Raw MotionAdapter output dict -> canonical kp / exp and degree rotation."""
    np = _np()
    for key in ('pitch', 'yaw', 'roll'):
        require_tensor(raw[key], (1, 66), label='M.' + key)
    kp = require_tensor(raw['kp'], (1, 63), label='M.kp').reshape(1, 21, 3)
    exp = require_tensor(raw['exp'], (1, 63), label='M.exp').reshape(1, 21, 3)
    scale = require_tensor(raw['scale'], (1, 1), label='M.scale')
    t = require_tensor(raw['t'], (1, 3), label='M.t')
    pitch, yaw, roll = (pose_to_degree(raw[k]) for k in ('pitch', 'yaw', 'roll'))
    rotation = rotation_matrix(pitch, yaw, roll)
    return {'kp': kp, 'exp': exp, 'scale': scale, 't': t, 'R': rotation}


def source_keypoints(source):
    """Original wrapper.transform_keypoint: source KPs, including t.xy only."""
    np = _np()
    kp = np.matmul(source['kp'], source['R'])
    out = (kp + source['exp']) * source['scale'][..., None]
    out = out.copy()
    out[..., :2] += source['t'][:, None, :2]
    return require_tensor(np.ascontiguousarray(out, dtype=np.float32), (1, 21, 3), label='x_s')


def driving_keypoints(source, driving):
    """Animal pipeline: uses source canonical KP and SCALE; driving R/exp/t."""
    np = _np()
    out = (np.matmul(source['kp'], driving['R']) + driving['exp']) * source['scale'][..., None]
    out = out.copy()
    out[..., :2] += driving['t'][:, None, :2]
    return require_tensor(np.ascontiguousarray(out, dtype=np.float32), (1, 21, 3), label='x_d')


def motion_multiplier(source_kp, first_driving_kp):
    """Match src/utils/helper.py: sqrt(ConvexHull(x_s).volume / hull(x_d0).volume).

    `ConvexHull.volume` is the 3-D hull volume (despite the upstream variable
    name `source_area`). SciPy is intentionally required for exact behavior.
    """
    import numpy as np
    try:
        from scipy.spatial import ConvexHull
        from scipy.spatial import QhullError
    except ImportError as exc:
        raise RuntimeError('Relative motion needs scipy: python -m pip install scipy') from exc
    a = require_tensor(source_kp, (1, 21, 3), label='source keypoints')
    b = require_tensor(first_driving_kp, (1, 21, 3), label='first driving keypoints')
    try:
        volume_source = float(ConvexHull(a[0]).volume)
        volume_driving = float(ConvexHull(b[0]).volume)
    except QhullError as exc:
        raise ValueError('Degenerate 3D keypoints, ConvexHull cannot form a volume') from exc
    if not all(math.isfinite(x) and x > 0 for x in (volume_source, volume_driving)):
        raise ValueError('Nonpositive or nonfinite ConvexHull volume')
    return float(np.sqrt(volume_source) / np.sqrt(volume_driving))


def relative_target(source, driving_first, driving_target, *, driving_multiplier=1.0):
    """Animal relative motion: x_s + (x_di-x_d0)*hull_multiplier * driving_multiplier.

    Does not use stitching/retargeting S. For a first==target frame the output
    keypoints equal source keypoints (and this is expected).
    """
    np = _np()
    if not math.isfinite(driving_multiplier) or driving_multiplier < 0:
        raise ValueError('driving_multiplier must be finite and nonnegative')
    x_s = source_keypoints(source)
    x_d0 = driving_keypoints(source, driving_first)
    x_di = driving_keypoints(source, driving_target)
    mult = motion_multiplier(x_s, x_d0)
    x_d = x_s + (x_di - x_d0) * np.float32(mult)
    x_d = x_s + (x_d - x_s) * np.float32(driving_multiplier)
    return x_s, require_tensor(np.ascontiguousarray(x_d, dtype=np.float32), (1, 21, 3), label='x_d_target'), mult


def output_rgb(pred):
    np = _np()
    raw = require_tensor(pred, (1, 3, 512, 512), label='G.output_image')
    nhwc = np.transpose(raw, (0, 2, 3, 1))
    return np.clip(np.clip(nhwc, 0, 1) * 255, 0, 255).astype(np.uint8)[0]


def execute_onnx(session, code, *inputs):
    """Strict F/M/W/G ORT boundary; can be tested with fake sessions."""
    np = _np()
    contract = get_contract(code)
    validate_ort_io(session, contract)
    if len(inputs) != len(contract.inputs):
        raise ValueError('Wrong input count for {}'.format(code.upper()))
    feeds = {spec.name: require_tensor(value, spec.shape, label=code.upper()+'.'+spec.name)
             for value, spec in zip(inputs, contract.inputs)}
    output_names = [spec.name for spec in contract.outputs]
    got = session.run(output_names, feeds)
    if len(got) != len(output_names):
        raise ValueError('Wrong output count for {}'.format(code.upper()))
    return {spec.name: require_tensor(v, spec.shape, label=code.upper()+'.'+spec.name)
            for spec, v in zip(contract.outputs, got)}


def run_one_frame(sessions, source_rgb, first_rgb, target_rgb, *, driving_multiplier=1.0, mouth_cfg=None):
    """F(source) + M(source, first, target) + W + G, return RGB image and metrics."""
    np = _np()
    from .animal_mouth_enhancement import (
        MouthEnhanceConfig, validate_mouth_config, apply_mouth_enhancement,
    )
    if mouth_cfg is None:
        mouth_cfg = MouthEnhanceConfig()
    validate_mouth_config(mouth_cfg)
    missing = set(MODELS) - set(sessions)
    if missing:
        raise ValueError('Missing ONNX sessions: {}'.format(','.join(sorted(missing))))
    src = prepare_rgb(source_rgb)
    first = prepare_rgb(first_rgb)
    target = prepare_rgb(target_rgb)
    stages_ms = {}
    start_all = time.perf_counter()

    def timed(name, fn):
        tick = time.perf_counter()
        ret = fn()
        stages_ms[name] = (time.perf_counter() - tick) * 1000.0
        return ret

    fs = timed('F_source', lambda: execute_onnx(sessions['f'], 'f', src)['feature_3d'])
    m_source = timed('M_source', lambda: motion_info(execute_onnx(sessions['m'], 'm', src)))
    m_first = timed('M_first', lambda: motion_info(execute_onnx(sessions['m'], 'm', first)))
    m_target = timed('M_target', lambda: motion_info(execute_onnx(sessions['m'], 'm', target)))
    x_s, x_d, multiplier = timed('motion_postprocess', lambda: relative_target(
        m_source, m_first, m_target, driving_multiplier=driving_multiplier))
    driving_before_mouth = None
    if mouth_cfg.enabled:
        driving_before_mouth = x_d.copy()
        x_d = timed('mouth_enhancement', lambda: apply_mouth_enhancement(x_s, x_d, mouth_cfg))
    warped = timed('W', lambda: execute_onnx(sessions['w'], 'w', fs, x_d, x_s)['warped_feature'])
    generated = timed('G', lambda: execute_onnx(sessions['g'], 'g', warped)['output_image'])
    rgb = timed('RGB_output', lambda: output_rgb(generated))
    total_ms = (time.perf_counter() - start_all) * 1000.0
    metrics = {
            'provider': 'CPUExecutionProvider', 'model_version': 'animal_v1.1',
            'batch': 1, 'dtype': 'float32',
            'cropping': False, 'stitching': False, 'pasteback': False,
            'relative_motion': True, 'driving_multiplier': float(driving_multiplier),
            'convex_hull_multiplier': multiplier,
            'stages_ms': stages_ms, 'total_inference_ms': total_ms,
            'image_size': list(rgb.shape),
    }
    if mouth_cfg.enabled:
        metrics['mouth_enhancement'] = mouth_cfg.as_dict()
    keypoints = {'source': x_s, 'driving': x_d}
    if driving_before_mouth is not None:
        keypoints['driving_before_mouth'] = driving_before_mouth
    return {'rgb': rgb, 'metrics': metrics, 'keypoints': keypoints}
