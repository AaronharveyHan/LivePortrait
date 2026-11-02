# coding: utf-8
"""V0.6 animal ONNX chain tests: CPU NumPy + optional SciPy / PyTorch."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np
except ImportError:
    np = None

from src.utils.animal_onnx_contract import get_contract
from src.utils import animal_onnx_chain as ch


class FakeSession:
    def __init__(self, code, fn):
        self.contract = get_contract(code)
        self.fn = fn
        self.calls = 0
        self.last_feed = None

    def get_inputs(self):
        return [SimpleNamespace(name=s.name, shape=list(s.shape), type='tensor(float)')
                for s in self.contract.inputs]

    def get_outputs(self):
        return [SimpleNamespace(name=s.name, shape=list(s.shape), type='tensor(float)')
                for s in self.contract.outputs]

    def run(self, names, feed):
        self.calls += 1
        self.last_feed = feed
        assert names == [s.name for s in self.contract.outputs]
        return self.fn(feed)


@unittest.skipUnless(np is not None, 'NumPy is required')
class AnimalOnnxChainTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(33)

    def m_raw(self, shift=0):
        rng = np.random.default_rng(87)
        kp = (rng.normal(size=(1, 21, 3)) * 0.3).astype(np.float32)
        pitch = np.zeros((1, 66), np.float32)
        yaw = np.zeros((1, 66), np.float32)
        roll = np.zeros((1, 66), np.float32)
        for v in (pitch, yaw, roll):
            v[0, 32] = 10
        yaw[0, min(65, 33 + shift)] = 12
        return {
            'pitch': pitch, 'yaw': yaw, 'roll': roll,
            'kp': kp.reshape(1, -1),
            'exp': (rng.normal(size=(1, 21, 3)) * 0.005 + 0.003 * shift).astype(np.float32).reshape(1, -1),
            't': np.array([[0.02 * shift, 0.01 * shift, 6.0]], np.float32),
            'scale': np.array([[1.1]], np.float32),
        }

    def test_source_image_contract(self):
        rgb = self.rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        out = ch.prepare_rgb(rgb)
        self.assertEqual(out.shape, (1, 3, 256, 256))
        self.assertEqual(out.dtype, np.float32)
        self.assertAlmostEqual(float(out[0, 1, 4, 5]), int(rgb[4, 5, 1])/255, places=7)

    def test_bad_image_shape(self):
        with self.assertRaisesRegex(ValueError, '256x256'):
            ch.prepare_rgb(np.zeros((128, 256, 3), np.uint8))

    def test_bad_image_dtype(self):
        with self.assertRaisesRegex(ValueError, 'RGB uint8'):
            ch.prepare_rgb(np.zeros((256, 256, 3), np.float32))

    def test_motion_source_shape(self):
        info = ch.motion_info(self.m_raw())
        self.assertEqual(info['kp'].shape, (1, 21, 3))
        self.assertEqual(info['R'].shape, (1, 3, 3))
        self.assertEqual(ch.source_keypoints(info).shape, (1, 21, 3))

    def test_headpose_peak(self):
        logits = np.full((1, 66), -100.0, np.float32)
        logits[0, 32] = 100.0
        self.assertAlmostEqual(float(ch.pose_to_degree(logits)[0]), 32*3-97.5, places=4)

    def test_rotation_identity(self):
        x = np.zeros(1, np.float32)
        np.testing.assert_allclose(ch.rotation_matrix(x,x,x), np.eye(3, dtype=np.float32)[None], atol=1e-7)

    def test_rotation_right_multiplied(self):
        x = np.array([90], np.float32)
        z = np.zeros(1, np.float32)
        got = ch.rotation_matrix(x, z, z)[0]
        np.testing.assert_allclose(got, np.array([[1,0,0],[0,0,1],[0,-1,0]], np.float32), atol=1e-6)

    def test_source_ignores_tz(self):
        info = ch.motion_info(self.m_raw())
        original = ch.source_keypoints(info)
        changed = dict(info, t=info['t'].copy())
        changed['t'][0, 2] = 9999
        np.testing.assert_array_equal(original, ch.source_keypoints(changed))

    def test_driving_uses_source_canonical_and_scale(self):
        source = ch.motion_info(self.m_raw(0))
        driving = ch.motion_info(self.m_raw(1))
        x = ch.driving_keypoints(source, driving)
        np.testing.assert_allclose(x[..., 0], ((source['kp'] @ driving['R'] + driving['exp']) * source['scale'][..., None])[...,0] + driving['t'][0,0], atol=1e-6)

    def test_wrong_output_contract(self):
        c = get_contract('m')
        def wrong(_):
            return [np.zeros((1, 66), np.float32)] * 7
        s = FakeSession('m', wrong)
        with self.assertRaisesRegex(ValueError, 'M.t'):
            ch.execute_onnx(s, 'm', np.zeros(c.inputs[0].shape, np.float32))

    def test_nonfinite_runtime_rejected(self):
        c = get_contract('f')
        def bad(_):
            data = np.zeros(c.outputs[0].shape, np.float32)
            data.flat[0] = np.nan
            return [data]
        s = FakeSession('f', bad)
        with self.assertRaisesRegex(ValueError, 'NaN/Inf'):
            ch.execute_onnx(s, 'f', np.zeros(c.inputs[0].shape, np.float32))

    def test_bad_float64_rejected(self):
        with self.assertRaisesRegex(ValueError, 'float32'):
            ch.require_tensor(np.zeros((1,3), np.float64), (1,3), label='x')

    def test_rgb_clip_and_no_rounding(self):
        arr = np.zeros((1,3,512,512), np.float32)
        arr[0,0] = 0.5
        arr[0,1] = 1.1
        arr[0,2] = -0.1
        got = ch.output_rgb(arr)
        self.assertEqual(got[10,10].tolist(), [127,255,0])

    def test_chain_full_fake_sessions(self):
        a = np.zeros((256,256,3), dtype=np.uint8)
        b = np.ones((256,256,3), dtype=np.uint8)
        c = np.full((256,256,3), 2, dtype=np.uint8)
        m_calls = [self.m_raw(0), self.m_raw(1), self.m_raw(2)]
        def f_fn(_): return [np.ones((1,32,16,64,64), np.float32)]
        def m_fn(_):
            d = m_calls.pop(0)
            return [d[o.name] for o in get_contract('m').outputs]
        def w_fn(feed):
            self.assertEqual(feed['kp_source'].shape, (1,21,3))
            self.assertEqual(feed['kp_driving'].shape, (1,21,3))
            self.assertFalse(np.array_equal(feed['kp_source'], feed['kp_driving']))
            return [np.full((1,256,64,64), 0.5, np.float32)]
        def g_fn(feed):
            self.assertAlmostEqual(float(feed['warped_feature'][0,0,0,0]), 0.5)
            return [np.full((1,3,512,512), 0.5, np.float32)]
        ss = {'f': FakeSession('f', f_fn), 'm': FakeSession('m', m_fn),
              'w': FakeSession('w', w_fn), 'g': FakeSession('g', g_fn)}
        with patch.object(ch, 'motion_multiplier', return_value=1.0):
            result = ch.run_one_frame(ss, a,b,c)
        self.assertEqual(result['rgb'].shape, (512,512,3))
        self.assertEqual(int(result['rgb'][0,0,0]), 127)
        self.assertEqual(ss['f'].calls, 1)
        self.assertEqual(ss['m'].calls, 3)
        self.assertEqual(ss['w'].calls, 1)
        self.assertEqual(ss['g'].calls, 1)
        self.assertTrue(0 < result['metrics']['total_inference_ms'])
        self.assertFalse(result['metrics']['stitching'])

    def test_invalid_multiplier_rejected(self):
        source = ch.motion_info(self.m_raw())
        for v in (-1.0, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                ch.relative_target(source,source,source,driving_multiplier=v)

    def test_source_first_equals_target_identity(self):
        source = ch.motion_info(self.m_raw(0))
        driver = ch.motion_info(self.m_raw(1))
        with patch.object(ch, 'motion_multiplier', return_value=1.2):
            x_s, x_d, ratio = ch.relative_target(source,driver,driver)
        np.testing.assert_allclose(x_s,x_d,rtol=0,atol=1e-6)
        self.assertEqual(ratio,1.2)

    @unittest.skipUnless(importlib.util.find_spec('scipy') is not None, 'SciPy optional')
    def test_true_3d_convex_hull_volume(self):
        a = (self.rng.normal(size=(1,21,3)) * 0.5).astype(np.float32)
        b = a*2
        ratio = ch.motion_multiplier(a,b)
        self.assertAlmostEqual(ratio, 1/2**1.5, places=5)

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'torch optional')
    def test_torch_headpose_rotation_numerical_parity(self):
        import torch
        rng = np.random.default_rng(100)
        a = rng.normal(size=(1,66)).astype(np.float32)
        degree_np = ch.pose_to_degree(a)
        degree_t = ((torch.softmax(torch.from_numpy(a),1)*torch.arange(66,dtype=torch.float32)).sum(1)*3-97.5).numpy()
        np.testing.assert_allclose(degree_np,degree_t,atol=2e-5)
        deg = np.array([11.2], np.float32)
        deg2 = np.array([-9.2], np.float32)
        deg3 = np.array([3.3], np.float32)
        if importlib.util.find_spec('src.utils.camera') is None:
            self.skipTest('Upstream camera.py not present in this isolated test fixture')
        from src.utils.camera import get_rotation_matrix
        reference = get_rotation_matrix(torch.from_numpy(deg),torch.from_numpy(deg2),torch.from_numpy(deg3)).numpy()
        np.testing.assert_allclose(ch.rotation_matrix(deg,deg2,deg3),reference,atol=2e-6)


if __name__ == '__main__':
    unittest.main()
