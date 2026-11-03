# coding: utf-8
"""V0.6.2 dual-source mouth blending: no actual weights or GPU required."""

import unittest
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.utils import animal_dual_source as ds
from src.utils.animal_onnx_contract import get_contract


class FakeSession:
    def __init__(self, code, fn):
        self.contract = get_contract(code)
        self.fn = fn
        self.calls = 0

    def get_inputs(self):
        return [SimpleNamespace(name=s.name, shape=list(s.shape), type='tensor(float)')
                for s in self.contract.inputs]

    def get_outputs(self):
        return [SimpleNamespace(name=s.name, shape=list(s.shape), type='tensor(float)')
                for s in self.contract.outputs]

    def run(self, names, feeds):
        self.calls += 1
        assert names == [s.name for s in self.contract.outputs]
        return self.fn(feeds)


class DualSourceTests(unittest.TestCase):
    def setUp(self):
        self.closed = np.full((512, 512, 3), 50, np.uint8)
        self.opened = np.full((512, 512, 3), 200, np.uint8)
        self.roi = (.29,.49,.74,.83)
        self.mask = ds.mouth_mask(512,512,self.roi,24)

    def test_mask_empty_outside(self):
        self.assertEqual(float(self.mask[10,10]), 0)
        self.assertEqual(float(self.mask[-1,-1]), 0)
        self.assertEqual(float(self.mask[320,265]), 1)

    def test_mask_smooth_edge(self):
        values = self.mask[300, :]
        self.assertTrue(np.any((values > 0) & (values < 1)))

    def test_mask_reject_bounds_and_nan(self):
        for roi in ((0,0,0,1),(.5,.1,.4,.8),(.1,0,float('nan'),1),(0,0,.01,1)):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                ds.mouth_mask(512,512,roi,24)

    def test_mask_reject_feather(self):
        for feather in (0,-1,130,float('inf')):
            with self.subTest(feather=feather), self.assertRaises(ValueError):
                ds.mouth_mask(512,512,self.roi,feather)

    def test_alpha_zero_exact(self):
        got = ds.blend_mouth(self.closed,self.opened,0,self.mask)
        np.testing.assert_array_equal(got,self.closed)

    def test_alpha_one_preserves_outside_and_uses_center(self):
        got = ds.blend_mouth(self.closed,self.opened,1,self.mask)
        self.assertEqual(int(got[320,265,0]),200)
        self.assertEqual(int(got[30,30,0]),50)

    def test_alpha_mid(self):
        got = ds.blend_mouth(self.closed,self.opened,.5,self.mask)
        self.assertEqual(int(got[320,265,0]),125)
        self.assertEqual(int(got[30,30,0]),50)

    def test_alpha_invalid(self):
        for v in (-.1,1.2,float('nan'),float('inf')):
            with self.subTest(v=v), self.assertRaises(ValueError):
                ds.blend_mouth(self.closed,self.opened,v,self.mask)

    def test_invalid_shape_and_mask(self):
        with self.assertRaises(ValueError):
            ds.blend_mouth(self.closed[:128],self.opened,1,self.mask)
        with self.assertRaises(ValueError):
            ds.blend_mouth(self.closed,self.opened,1,np.zeros((256,256)))
        bad = self.mask.copy();bad[0,0]=np.nan
        with self.assertRaises(ValueError):
            ds.blend_mouth(self.closed,self.opened,1,bad)

    def test_region_change_outside_zero(self):
        got=ds.blend_mouth(self.closed,self.opened,.6,self.mask)
        metrics=ds.region_change_metrics(self.closed,got,self.mask)
        self.assertEqual(metrics['outside_region_changed_pixels'],0)
        self.assertGreater(metrics['mouth_region_mean_abs'],0)

    def test_reconstruction_none_alignment(self):
        got=ds.align_reconstruction(self.opened, {'mode':'none'})
        np.testing.assert_array_equal(got,self.opened)
        got[0,0,0] = 4
        self.assertEqual(self.opened[0,0,0],200)

    def test_registration_small_translation(self):
        import cv2
        src=np.zeros((256,256,3),np.uint8)
        for y in range(30,136,20):
            cv2.circle(src,(100+(y%43),y),14,(220,150,65),-1)
        cv2.rectangle(src,(90,70),(190,122),(170,90,195),-1)
        moving=cv2.warpAffine(src,np.float32([[1,0,4],[0,1,3]]),(256,256))
        meta=ds.estimate_translation(src,moving,max_shift=20)
        self.assertEqual(meta['mode'],'translation')
        self.assertAlmostEqual(abs(meta['dx_256']),4,delta=1)
        self.assertAlmostEqual(abs(meta['dy_256']),3,delta=1)

    def test_registration_bad_maxshift(self):
        img=np.zeros((256,256,3),np.uint8)
        with self.assertRaises(ValueError):
            ds.estimate_translation(img,img,max_shift=-1)

    def test_render_reference_cached_F_M_W_G(self):
        rng=np.random.default_rng(13)
        im1=rng.integers(0,255,(256,256,3),dtype=np.uint8)
        im2=rng.integers(0,255,(256,256,3),dtype=np.uint8)
        refkp=rng.normal(size=(1,63)).astype(np.float32)
        def f(_):return [np.zeros((1,32,16,64,64),np.float32)]
        def m(_):
            result={'pitch':np.zeros((1,66),np.float32),'yaw':np.zeros((1,66),np.float32),
                    'roll':np.zeros((1,66),np.float32),'t':np.zeros((1,3),np.float32),
                    'exp':np.zeros((1,63),np.float32),'scale':np.ones((1,1),np.float32),'kp':refkp}
            return [result[c.name] for c in get_contract('m').outputs]
        def w(feed):
            np.testing.assert_array_equal(feed['kp_source'],feed['kp_driving'])
            return [np.zeros((1,256,64,64),np.float32)]
        def g(_):return [np.zeros((1,3,512,512),np.float32)]
        ss={'f':FakeSession('f',f),'m':FakeSession('m',m),
            'w':FakeSession('w',w),'g':FakeSession('g',g)}
        cache={}
        a,stat1=ds.render_reference(ss,im1,cache)
        b,stat2=ds.render_reference(ss,im1,cache)
        c,stat3=ds.render_reference(ss,im2,cache)
        self.assertFalse(stat1['cache_hit'])
        self.assertTrue(stat2['cache_hit'])
        self.assertFalse(stat3['cache_hit'])
        self.assertEqual(len(cache),2)
        self.assertEqual([ss[k].calls for k in ('f','m','w','g')],[2,2,2,2])
        self.assertEqual(a.shape,(512,512,3))
        np.testing.assert_array_equal(a,b)

if __name__=='__main__': unittest.main()
