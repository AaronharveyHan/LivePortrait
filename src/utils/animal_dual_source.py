# coding: utf-8
"""V0.6.2 exploratory closed/open state blending (not anatomical lip rigging).

Inputs must be cropped/aligned photos of the SAME animal. This is an image-space
experiment: reusing two reference appearances cannot reconstruct a physical jaw.
No model weights, GPU code, network, inference entrypoints or training are changed.
"""

import hashlib
import math
import time


def _np():
    import numpy as np
    return np


def checked_image(image, *, size, name):
    np = _np()
    a = np.asarray(image)
    if a.shape != (size, size, 3) or a.dtype != np.dtype('uint8'):
        raise ValueError('{} must be RGB uint8 {}x{}x3'.format(name, size, size))
    return np.ascontiguousarray(a)


def image_digest(image):
    a = checked_image(image, size=image.shape[0], name='image')
    if a.shape[0] != a.shape[1]:
        raise ValueError('Digest requires square RGB image')
    return hashlib.sha256(a.tobytes()).hexdigest()


def validate_roi(roi):
    if len(roi) != 4 or not all(math.isfinite(v) for v in roi):
        raise ValueError('ROI needs four finite normalized coordinates')
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError('ROI must satisfy 0<=x0<x1<=1 and 0<=y0<y1<=1')
    if (x1 - x0) < 0.06 or (y1 - y0) < 0.06:
        raise ValueError('ROI too small for safe feathering')
    return tuple(float(x) for x in roi)


def mouth_mask(width, height, roi, feather_px):
    """Soft rectangular mouth mask, 0 outside, 1 at sufficiently deep interior."""
    np = _np()
    x0, y0, x1, y1 = validate_roi(roi)
    if not (isinstance(feather_px, (int, float)) and math.isfinite(feather_px)
            and feather_px >= 1 and feather_px <= min(width, height) / 4):
        raise ValueError('feather_px invalid')
    if width < 64 or height < 64:
        raise ValueError('Image resolution too small')
    xs = (np.arange(width, dtype=np.float32) + 0.5)[None, :]
    ys = (np.arange(height, dtype=np.float32) + 0.5)[:, None]
    dist = np.minimum(np.minimum(xs - x0 * width, x1 * width - xs),
                      np.minimum(ys - y0 * height, y1 * height - ys))
    v = np.clip(dist / float(feather_px), 0, 1)
    return np.ascontiguousarray((v * v * (3 - 2 * v)).astype(np.float32))


def blend_mouth(closed, open_aligned, alpha, mask):
    """Return closed face with only an alpha-scaled open-state mouth appearance.

    alpha=1 does NOT mean the whole open image; only its mouth region is used.
    Intermediate alphas may introduce ghosted teeth/tongues; inspect critically.
    """
    np = _np()
    c = checked_image(closed, size=512, name='closed reconstruction')
    o = checked_image(open_aligned, size=512, name='registered open reconstruction')
    if not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must be finite in [0,1]')
    m = np.asarray(mask)
    if m.shape != (512, 512) or not np.all(np.isfinite(m)) or np.min(m) < 0 or np.max(m) > 1:
        raise ValueError('invalid 512x512 finite [0,1] mouth mask')
    if alpha == 0:
        return c.copy()
    if alpha == 1 and np.array_equal(c, o):
        return c.copy()
    wt = (m * np.float32(alpha))[:, :, None]
    out = c.astype(np.float32) + (o.astype(np.float32) - c.astype(np.float32)) * wt
    return np.ascontiguousarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))


def estimate_translation(closed_rgb256, open_rgb256, *, max_shift=20):
    """Find conservative upper-face-only translation; transform is for WARP_INVERSE_MAP.

    This is registration assistance, not face recognition or guaranteed alignment.
    Reject low-confidence/misaligned photos rather than silently distorting them.
    """
    np = _np()
    c = checked_image(closed_rgb256, size=256, name='closed input')
    o = checked_image(open_rgb256, size=256, name='open input')
    if not isinstance(max_shift, (int, float)) or not math.isfinite(max_shift) or max_shift < 0 or max_shift > 64:
        raise ValueError('max_shift must be between 0 and 64 pixels')
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError('Registration requires opencv-python-headless') from exc
    gray_c = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray_o = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    upper_face = np.zeros((256, 256), np.uint8)
    upper_face[38:133, 38:217] = 255  # avoid mouth; require similarly cropped head pose
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    try:
        score, matrix = cv2.findTransformECC(gray_c, gray_o, warp,
                                              cv2.MOTION_TRANSLATION, criteria,
                                              inputMask=upper_face, gaussFiltSize=5)
    except cv2.error as exc:
        raise ValueError('Upper-face registration failed; use better aligned photos or --alignment none') from exc
    dx, dy = float(matrix[0, 2]), float(matrix[1, 2])
    if not (math.isfinite(score) and math.isfinite(dx) and math.isfinite(dy)):
        raise ValueError('Registration has non-finite values')
    if abs(dx) > max_shift or abs(dy) > max_shift or score < 0.40:
        raise ValueError('Registration rejected: score={:.3f}, shift=({:.1f},{:.1f}) 256px'.format(score, dx, dy))
    return {'mode': 'translation', 'score': float(score), 'dx_256': dx, 'dy_256': dy}


def align_reconstruction(open_rgb512, registration):
    np = _np()
    moving = checked_image(open_rgb512, size=512, name='open reconstruction')
    if registration['mode'] == 'none':
        return moving.copy()
    if registration['mode'] != 'translation':
        raise ValueError('unknown alignment mode')
    import cv2
    matrix = np.array([[1, 0, 2 * registration['dx_256']],
                       [0, 1, 2 * registration['dy_256']]], dtype=np.float32)
    return cv2.warpAffine(moving, matrix, (512, 512),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT_101)


def render_reference(sessions, rgb256, cache, *, cache_key=None):
    """Run each F/M/W/G exactly once per unique source; keep only RGB output.

    For a no-motion reference, x_driving = x_source. The F/M output is cached
    for that source within this process. Two different states each need W/G.
    """
    from . import animal_onnx_chain as ch
    np = _np()
    rgb256 = checked_image(rgb256, size=256, name='source')
    if cache_key is None:
        cache_key = image_digest(rgb256)
    if cache_key in cache:
        return cache[cache_key]['rgb'].copy(), {'cache_hit': True, 'stages_ms': {}}
    if set(ch.MODELS) - set(sessions):
        raise ValueError('Four ONNX sessions F/M/W/G are required')
    times = {}

    def measured(name, fn):
        start = time.perf_counter()
        result = fn()
        times[name] = (time.perf_counter() - start) * 1000
        return result

    tensor = ch.prepare_rgb(rgb256)
    feat = measured('F', lambda: ch.execute_onnx(sessions['f'], 'f', tensor)['feature_3d'])
    motion = measured('M', lambda: ch.motion_info(ch.execute_onnx(sessions['m'], 'm', tensor)))
    kp = measured('source_keypoints', lambda: ch.source_keypoints(motion))
    warped = measured('W', lambda: ch.execute_onnx(sessions['w'], 'w', feat, kp, kp)['warped_feature'])
    output = measured('G', lambda: ch.execute_onnx(sessions['g'], 'g', warped)['output_image'])
    rgb = measured('RGB', lambda: ch.output_rgb(output))
    cache[cache_key] = {'rgb': rgb, 'feature': feat, 'motion': motion, 'keypoints': kp}
    return rgb.copy(), {'cache_hit': False, 'stages_ms': times}


def region_change_metrics(closed, blended, mask):
    """Descriptive change only; NOT a realism/lip-sync quality score."""
    np = _np()
    c = checked_image(closed, size=512, name='closed')
    b = checked_image(blended, size=512, name='blended')
    delta = np.abs(c.astype(np.float32) - b.astype(np.float32)).mean(axis=2)
    inside = mask > 0
    outside = ~inside
    return {'mouth_region_mean_abs': float(delta[inside].mean()) if np.any(inside) else 0,
            'outside_region_changed_pixels': int(np.count_nonzero(delta[outside] > 0)),
            'mean_abs_all': float(delta.mean())}
