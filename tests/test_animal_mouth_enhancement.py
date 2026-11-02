# coding: utf-8
"""V0.6.1 keypoint-selective motion tests, no ONNX models required."""
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
from src.utils.animal_mouth_enhancement import (
    MouthEnhanceConfig as Cfg,
    apply_mouth_enhancement as enhance,
    parse_index_csv,
    validate_mouth_config as validate,
)
from src.utils import animal_onnx_chain as chain


class MouthEnhancementTests(unittest.TestCase):
    def setUp(self):
        self.src = np.zeros((1, 21, 3), np.float32)
        self.dst = np.full((1, 21, 3), 0.01, np.float32)

    def test_disabled_noop_and_no_mutation(self):
        original = self.dst.copy()
        out = enhance(self.src, self.dst, Cfg())
        np.testing.assert_array_equal(out, original)
        np.testing.assert_array_equal(self.dst, original)

    def test_only_selected_axes_and_points_change(self):
        cfg = Cfg(enabled=True, mouth_kp_indices=(3, 5), x_scale=1.0, y_scale=1.25, z_scale=0.5)
        out = enhance(self.src, self.dst, cfg)
        np.testing.assert_array_equal(out[:, [0, 2, 4, 6, 20], :], self.dst[:, [0, 2, 4, 6, 20], :])
        np.testing.assert_array_equal(out[:, [3, 5], 0], self.dst[:, [3, 5], 0])
        np.testing.assert_allclose(out[0, [3, 5], 1], 0.0125, atol=1e-7)
        np.testing.assert_allclose(out[0, [3, 5], 2], 0.005, atol=1e-7)

    def test_lower_y_bias_only_lower_subset(self):
        cfg = Cfg(enabled=True, mouth_kp_indices=(1, 2, 3), lower_mouth_kp_indices=(2,), lower_y_bias=-0.005)
        out = enhance(self.src, self.dst, cfg)
        self.assertAlmostEqual(float(out[0, 2, 1]), 0.005, places=6)
        self.assertAlmostEqual(float(out[0, 1, 1]), 0.01, places=6)
        self.assertAlmostEqual(float(out[0, 3, 1]), 0.01, places=6)

    def test_zero_delta_only_bias_can_move(self):
        cfg = Cfg(enabled=True, mouth_kp_indices=(2,), y_scale=1.8)
        np.testing.assert_array_equal(enhance(self.src, self.src, cfg), self.src)
        with_bias = Cfg(enabled=True, mouth_kp_indices=(2,), lower_mouth_kp_indices=(2,), lower_y_bias=0.002)
        got = enhance(self.src, self.src, with_bias)
        self.assertAlmostEqual(float(got[0,2,1]), 0.002, places=6)

    def test_outside_kp_range(self):
        for bad in (-1, 21, True, 1.5):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, '0..20'):
                validate(Cfg(enabled=True, mouth_kp_indices=(bad,)))

    def test_duplicate_or_subset_rejected(self):
        cases = [Cfg(enabled=True, mouth_kp_indices=(1,1)),
                 Cfg(enabled=True, mouth_kp_indices=(1,), lower_mouth_kp_indices=(2,)),
                 Cfg(enabled=True, mouth_kp_indices=(1,), lower_mouth_kp_indices=(1,1))]
        for c in cases:
            with self.subTest(c=c), self.assertRaises(ValueError): validate(c)

    def test_need_explicit_mouth_indices(self):
        with self.assertRaisesRegex(ValueError, 'mouth-kp-indices'):
            validate(Cfg(enabled=True, y_scale=1.2))

    def test_bad_scalars(self):
        for name in ('x_scale', 'y_scale', 'z_scale'):
            for value in (float('nan'), float('inf'), -0.1, 4.0):
                with self.subTest(field=name,value=value), self.assertRaises(ValueError):
                    validate(Cfg(enabled=True, mouth_kp_indices=(0,), **{name:value}))
        for value in (float('nan'), float('inf'), -0.06, 0.06):
            with self.subTest(bias=value), self.assertRaises(ValueError):
                validate(Cfg(enabled=True, mouth_kp_indices=(0,), lower_mouth_kp_indices=(0,),lower_y_bias=value))

    def test_bias_needs_lower_indices(self):
        with self.assertRaisesRegex(ValueError, 'lower-kp-indices'):
            validate(Cfg(enabled=True, mouth_kp_indices=(0,),lower_y_bias=0.01))

    def test_zero_and_csv_parse(self):
        self.assertEqual(parse_index_csv(''), ())
        self.assertEqual(parse_index_csv(' 2 , 8 '), (2,8))
        for raw in ('1,,2','1,','abc','0.5'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):parse_index_csv(raw)

    def test_float_dtype_and_shape_rejection(self):
        c=Cfg(enabled=True,mouth_kp_indices=(1,),y_scale=1.2)
        with self.assertRaisesRegex(ValueError,'float32'):
            enhance(self.src.astype(np.float64), self.dst, c)
        with self.assertRaisesRegex(ValueError,'shape'):
            enhance(self.src[:,:,:2], self.dst, c)

    def test_config_manifest_is_json_primitive(self):
        import json
        c=Cfg(enabled=True,mouth_kp_indices=(1,3),lower_mouth_kp_indices=(3,),lower_y_bias=0.005)
        blob=json.loads(json.dumps(c.as_dict()))
        self.assertEqual(blob['mouth_kp_indices'],[1,3])
        self.assertFalse(blob['indices_are_anatomical_landmarks'])

    def test_full_chain_mouth_applied_before_w(self):
        source = self.src.copy()
        before = self.dst.copy()
        feeds = []
        def mocked_run(session, code, *inputs):
            if code == 'f': return {'feature_3d':np.zeros((1,32,16,64,64),np.float32)}
            if code == 'm': return {}
            if code == 'w':
                feeds.append(inputs[1].copy())
                return {'warped_feature':np.zeros((1,256,64,64),np.float32)}
            if code == 'g': return {'output_image':np.zeros((1,3,512,512),np.float32)}
            raise AssertionError(code)
        sessions={k:object() for k in ('f','m','w','g')}
        image=np.zeros((256,256,3),np.uint8)
        cfg=Cfg(enabled=True,mouth_kp_indices=(2,),y_scale=1.5)
        with patch.object(chain,'execute_onnx', side_effect=mocked_run), \
             patch.object(chain,'motion_info',return_value={}), \
             patch.object(chain,'relative_target',return_value=(source,before,1.0)):
            out=chain.run_one_frame(sessions,image,image,image,mouth_cfg=cfg)
        self.assertEqual(len(feeds),1)
        self.assertEqual(feeds[0][0,0,1],before[0,0,1])
        self.assertAlmostEqual(float(feeds[0][0,2,1]),0.015,places=6)
        np.testing.assert_array_equal(out['keypoints']['driving_before_mouth'],before)
        self.assertIn('mouth_enhancement',out['metrics']['stages_ms'])
        self.assertEqual(out['metrics']['mouth_enhancement']['mouth_kp_indices'],[2])

    def test_backward_compatible_noop_default_chain(self):
        source=self.src.copy()
        before=self.dst.copy()
        feed=[]
        def mock_exec(session,code,*inputs):
            if code=='f':return {'feature_3d':np.zeros((1,32,16,64,64),np.float32)}
            if code=='m':return {}
            if code=='w':
                feed.append(inputs[1].copy())
                return {'warped_feature':np.zeros((1,256,64,64),np.float32)}
            return {'output_image':np.zeros((1,3,512,512),np.float32)}
        sessions={k:object() for k in ('f','m','w','g')}
        image=np.zeros((256,256,3),np.uint8)
        with patch.object(chain,'execute_onnx',side_effect=mock_exec), \
             patch.object(chain,'motion_info',return_value={}), \
             patch.object(chain,'relative_target',return_value=(source,before,1.0)):
            out=chain.run_one_frame(sessions,image,image,image)
        np.testing.assert_array_equal(feed[0],before)
        self.assertNotIn('mouth_enhancement',out['metrics']['stages_ms'])
        self.assertNotIn('driving_before_mouth',out['keypoints'])


if __name__ == '__main__':
    unittest.main()
