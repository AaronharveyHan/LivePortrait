# coding: utf-8
"""V0.7.0: cached, streaming CPU animal expression transfer via F/M/W/G.

No GUI, automatic face crop, RGB crossfade, custom mouth ROI, or learned
semantic mouth channels. The driving clip must have a reasonably stable face.
"""

import math
import time


def check_motion_mode(mode):
    if mode not in ('full', 'expression-only', 'pose-only'):
        raise ValueError('motion mode must be full, expression-only or pose-only')
    return mode


def valid_gain(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or not 0 <= value <= 3):
        raise ValueError('driving multiplier must be finite in [0,3]')
    return float(value)


def select_motion(first, target, mode):
    """Compose whole implicit expression/pose fields; NOT phoneme isolation.

    The relative target is always computed against the actual first frame.
    freeze R/t for expression-only, freeze exp for pose-only. No semantic
    identification of eye, tongue, or mouth keypoints is claimed.
    """
    check_motion_mode(mode)
    if mode == 'full':
        return target
    hybrid = dict(target)
    if mode == 'expression-only':
        hybrid['R'] = first['R']
        hybrid['t'] = first['t']
    else:
        hybrid['exp'] = first['exp']
    return hybrid


class CachedAnimalDriver:
    """One session bundle, one source F/M and one first driving M per video.

    The returned images are always new arrays, with no history accumulated.
    Cache only feature, source info/keypoints and anchor driving keypoints.
    """

    def __init__(self, sessions, source_rgb256, first_rgb256, *,
                 mode='full', driving_multiplier=1.0):
        from . import animal_onnx_chain as ch
        import numpy as np

        self.mode = check_motion_mode(mode)
        self.gain = valid_gain(driving_multiplier)
        missing = set(ch.MODELS) - set(sessions)
        if missing:
            raise ValueError('Missing ONNX sessions: {}'.format(', '.join(sorted(missing))))
        self.sessions = sessions
        self.stages_init_ms = {}

        def stage(name, operation):
            start = time.perf_counter()
            result = operation()
            self.stages_init_ms[name] = (time.perf_counter() - start) * 1000
            return result

        source = ch.prepare_rgb(source_rgb256)
        anchor = ch.prepare_rgb(first_rgb256)
        self.feature = stage('F_source', lambda: ch.execute_onnx(sessions['f'], 'f', source)['feature_3d'])
        self.source_info = stage('M_source', lambda: ch.motion_info(
            ch.execute_onnx(sessions['m'], 'm', source)))
        self.first_info = stage('M_first', lambda: ch.motion_info(
            ch.execute_onnx(sessions['m'], 'm', anchor)))
        self.x_source = ch.source_keypoints(self.source_info)
        self.x_first_driving = ch.driving_keypoints(self.source_info, self.first_info)
        self.hull_multiplier = stage('convex_hull_multiplier', lambda: ch.motion_multiplier(
            self.x_source, self.x_first_driving))
        self.frames = 0
        self.stage_sums_ms = {'M_target': 0.0, 'motion_postprocess': 0.0,
                              'W': 0.0, 'G': 0.0, 'RGB_output': 0.0}
        self.stage_max_ms = dict.fromkeys(self.stage_sums_ms, 0.0)
        self.init_cache_bytes = int(sum(arr.nbytes for arr in
            (self.feature, self.x_source, self.x_first_driving)))
        self._np = np

    def render(self, rgb256, *, is_first=False):
        from . import animal_onnx_chain as ch
        np = self._np
        times = {}

        def stage(name, operation):
            start = time.perf_counter()
            result = operation()
            times[name] = (time.perf_counter() - start) * 1000
            return result

        # Reuse anchor M instead of running it twice on the first frame.
        driving = self.first_info if is_first else stage('M_target', lambda: ch.motion_info(
            ch.execute_onnx(self.sessions['m'], 'm', ch.prepare_rgb(rgb256))))
        if is_first:
            times['M_target'] = 0.0
        adjusted = select_motion(self.first_info, driving, self.mode)

        def keypoints():
            x_current = ch.driving_keypoints(self.source_info, adjusted)
            x_target = (self.x_source + (x_current - self.x_first_driving) *
                        np.float32(self.hull_multiplier * self.gain))
            return ch.require_tensor(np.ascontiguousarray(x_target, dtype=np.float32),
                                     (1, 21, 3), label='video x_driving')

        x_target = stage('motion_postprocess', keypoints)
        warped = stage('W', lambda: ch.execute_onnx(self.sessions['w'], 'w',
                         self.feature, x_target, self.x_source)['warped_feature'])
        pred = stage('G', lambda: ch.execute_onnx(self.sessions['g'], 'g', warped)['output_image'])
        rgb = stage('RGB_output', lambda: ch.output_rgb(pred))
        for name in self.stage_sums_ms:
            value = times[name]
            self.stage_sums_ms[name] += value
            self.stage_max_ms[name] = max(self.stage_max_ms[name], value)
        self.frames += 1
        return rgb, times

    def summary(self):
        n = self.frames
        return {
            'frames': n, 'mode': self.mode, 'driving_multiplier': self.gain,
            'hull_multiplier': float(self.hull_multiplier),
            'model_init_stages_ms': dict(self.stages_init_ms),
            'cached_numpy_bytes': self.init_cache_bytes,
            'per_frame_mean_ms': {k: v/n if n else None for k, v in self.stage_sums_ms.items()},
            'per_frame_max_ms': dict(self.stage_max_ms),
            'F_calls_expected': 1, 'M_calls_expected': n + 1,
            'W_calls_expected': n, 'G_calls_expected': n,
            'note': 'Counts assume only this pipeline uses the sessions. No crop/tracker, no S/stitching.'
        }


def crop_rgb256(image, mode='center-crop'):
    """Deterministic frame preprocessing, not face detection or tracking."""
    import cv2
    import numpy as np
    a = np.asarray(image)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8 or min(a.shape[:2]) < 16:
        raise ValueError('Frame must be RGB uint8 HxWx3 with both dimensions >=16')
    if mode not in ('center-crop', 'stretch'):
        raise ValueError('crop mode must be center-crop or stretch')
    h, w = a.shape[:2]
    if mode == 'center-crop':
        dim = min(h, w)
        y0, x0 = (h-dim)//2, (w-dim)//2
        a = a[y0:y0+dim, x0:x0+dim]
    return np.ascontiguousarray(cv2.resize(a, (256, 256), interpolation=cv2.INTER_LINEAR))


class ImageioFfmpegWriter:
    """Small writer adapter avoiding heavyweight src.utils.video imports."""

    def __init__(self, *, wfp, fps, ffmpeg_params=None):
        import imageio
        self.writer = imageio.get_writer(str(wfp), format='FFMPEG', mode='I', fps=float(fps),
                                         codec='libx264', pixelformat='yuv420p',
                                         macro_block_size=2, ffmpeg_params=ffmpeg_params or ['-crf', '18'])

    def write(self, rgb):
        if self.writer is None:
            raise RuntimeError('video writer is closed')
        self.writer.append_data(rgb)

    def close(self):
        writer, self.writer = self.writer, None
        if writer is not None:
            writer.close()
