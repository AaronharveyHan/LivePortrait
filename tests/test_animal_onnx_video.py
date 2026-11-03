# coding: utf-8
"""V0.7.0 stream/caching tests; no model weights, network or GPU."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.utils import animal_onnx_chain as ch
from src.utils import animal_onnx_video as vid
from src.utils.animal_onnx_contract import get_contract


class FakeSession:
    def __init__(self, code, fn):
        self.contract = get_contract(code)
        self.fn = fn
        self.calls = 0
        self.feeds = []

    def get_inputs(self):
        return [SimpleNamespace(name=s.name, shape=list(s.shape), type='tensor(float)')
                for s in self.contract.inputs]

    def get_outputs(self):
        return [SimpleNamespace(name=s.name, shape=list(s.shape), type='tensor(float)')
                for s in self.contract.outputs]

    def run(self, names, feed):
        self.calls += 1
        self.feeds.append(feed)
        return self.fn(feed)


def make_motion(k=0):
    rng = np.random.default_rng(31)
    pts = rng.normal(size=(1, 21, 3)).astype(np.float32) * 0.20
    logits = np.full((1, 66), -5, dtype=np.float32)
    logits[0, 32] = 8
    logits_yaw = logits.copy()
    logits_yaw[0, 33 if k > 0 else 32] = 12
    return {
        'kp': pts.reshape((1, 63)),
        'exp': (rng.normal(size=(1, 21, 3))*0.01 + k*0.01).astype(np.float32).reshape((1, 63)),
        'scale': np.ones((1, 1), np.float32),
        't': np.array([[k*0.03, 0, 0]], np.float32),
        'pitch': logits, 'yaw': logits_yaw, 'roll': logits,
    }


def make_sessions():
    ss = {}
    ss['f'] = FakeSession('f', lambda _: [np.full((1, 32, 16, 64, 64), .2, np.float32)])

    def m_fn(feed):
        k = float(feed['image'][0,0,0,0])  # 0..1
        m = make_motion(k)
        return [m[s.name] for s in get_contract('m').outputs]
    ss['m'] = FakeSession('m', m_fn)
    ss['w'] = FakeSession('w', lambda _: [np.full((1, 256, 64, 64), .2, np.float32)])
    ss['g'] = FakeSession('g', lambda _: [np.full((1, 3, 512, 512), .3, np.float32)])
    return ss


class V070CachedDriverTests(unittest.TestCase):
    def setUp(self):
        self.source = np.full((256,256,3), 10, np.uint8)
        self.first = np.full((256,256,3), 20, np.uint8)
        self.target = np.full((256,256,3), 40, np.uint8)

    def test_cache_sessions_and_source(self):
        ss=make_sessions()
        with patch.object(ch, 'motion_multiplier', return_value=1.5) as mm:
            video=vid.CachedAnimalDriver(ss,self.source,self.first)
            first,t0=video.render(self.first,is_first=True)
            later,t1=video.render(self.target)
            later2,t2=video.render(self.target)
        self.assertEqual(ss['f'].calls,1)
        self.assertEqual(ss['m'].calls,4) # source, anchor, two targets
        self.assertEqual(ss['w'].calls,3)
        self.assertEqual(ss['g'].calls,3)
        self.assertEqual(mm.call_count,1)
        self.assertEqual(t0['M_target'],0)
        self.assertEqual(first.shape,(512,512,3))
        self.assertEqual(int(later[0,0,0]),76)
        self.assertEqual(video.summary()['frames'],3)
        self.assertGreater(video.init_cache_bytes,0)

    def test_first_frame_motion_is_source_zero(self):
        ss=make_sessions()
        with patch.object(ch, 'motion_multiplier', return_value=1):
            video=vid.CachedAnimalDriver(ss,self.source,self.first)
            video.render(self.first,is_first=True)
        np.testing.assert_allclose(ss['w'].feeds[0]['kp_driving'], ss['w'].feeds[0]['kp_source'])

    def test_expression_only_freezes_pose(self):
        ss=make_sessions()
        with patch.object(ch, 'motion_multiplier', return_value=1):
            video=vid.CachedAnimalDriver(ss,self.source,self.first,mode='expression-only')
            video.render(self.target)
            baseline=vid.select_motion(video.first_info, ch.motion_info(make_motion(0.5)), 'expression-only')
        np.testing.assert_array_equal(baseline['R'],video.first_info['R'])
        np.testing.assert_array_equal(baseline['t'],video.first_info['t'])
        self.assertIsNot(baseline['exp'],video.first_info['exp'])

    def test_pose_only_freezes_expression(self):
        a=ch.motion_info(make_motion(0))
        b=ch.motion_info(make_motion(1))
        p=vid.select_motion(a,b,'pose-only')
        np.testing.assert_array_equal(p['exp'],a['exp'])
        self.assertIs(p['R'],b['R'])

    def test_parity_with_v06_relative_keypoints(self):
        # Cached chain must match the established 3-image V0.6 model mapping.
        ss_cached=make_sessions()
        ss_old=make_sessions()
        with patch.object(ch, 'motion_multiplier', return_value=1.1):
            cached=vid.CachedAnimalDriver(ss_cached,self.source,self.first,mode='full',
                                          driving_multiplier=1.2)
            cached.render(self.target)
            baseline=ch.run_one_frame(ss_old,self.source,self.first,self.target,
                                      driving_multiplier=1.2)
        np.testing.assert_allclose(ss_cached['w'].feeds[0]['kp_driving'],
                                   baseline['keypoints']['driving'],atol=1e-6)
        np.testing.assert_allclose(ss_cached['w'].feeds[0]['kp_source'],
                                   baseline['keypoints']['source'],atol=1e-6)

    def test_no_source_mutation(self):
        ss=make_sessions()
        original=self.source.copy()
        with patch.object(ch, 'motion_multiplier',return_value=1):
            video=vid.CachedAnimalDriver(ss,self.source,self.first)
            video.render(self.target)
        np.testing.assert_array_equal(self.source,original)

    def test_cropping_center_preserves_rgb(self):
        im=np.zeros((32,64,3),np.uint8)
        im[:,16:48]=[10,20,30]
        out=vid.crop_rgb256(im)
        self.assertEqual(out.shape,(256,256,3))
        np.testing.assert_array_equal(out[100,100],[10,20,30])

    def test_stretch_differs_from_center(self):
        im=np.zeros((32,64,3),np.uint8)
        im[:,:16]=[255,0,0]
        self.assertNotEqual(vid.crop_rgb256(im,'stretch').sum(),vid.crop_rgb256(im).sum())

    def test_invalid_modes_and_gains(self):
        for x in (-1,float('nan'),float('inf'),3.1,True):
            with self.subTest(x=x),self.assertRaises(ValueError):
                vid.valid_gain(x)
        for x in ('mouth','none',None):
            with self.subTest(mode=x),self.assertRaises(ValueError):
                vid.check_motion_mode(x)

    def test_bad_input_rejected(self):
        with self.assertRaises(ValueError):
            vid.crop_rgb256(np.zeros((12,12,3),dtype=np.uint8))
        with self.assertRaises(ValueError):
            vid.crop_rgb256(np.zeros((16,16,3),dtype=np.float32))


if __name__=='__main__': unittest.main()
