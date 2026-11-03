# coding: utf-8
"""V0.6.3 geometric mouth warp tests: no model files or GPU."""
import copy
import json
import math
import unittest

import numpy as np
from src.utils import animal_mouth_geometry as geo
from src.utils.animal_mouth_annotator import annotation_html


class GeometricMouthTests(unittest.TestCase):
    def setUp(self):
        self.spec=geo.create_template((.29,.49,.74,.83))
        self.spec['verified']=True
        # Controlled movement within ROI, not claims about dog landmark locations.
        self.spec['open']['lower_lip'][1] += .027
        self.spec['open']['chin'][1] += .016
        yy,xx=np.indices((512,512))
        a=np.zeros((512,512,3),dtype=np.uint8)
        a[:,:,0]=(xx%256).astype(np.uint8)
        a[:,:,1]=(yy%256).astype(np.uint8)
        a[:,:,2]=75
        self.closed=a
        self.opened=a.copy()
        self.opened[260:385,200:340]=[80,190,150]

    def test_template_unverified(self):
        template=geo.create_template((.29,.49,.74,.83))
        self.assertFalse(template['verified'])
        self.assertEqual(template['closed'],template['open'])
        with self.assertRaisesRegex(ValueError,'confirmed'):
            geo.validate_landmarks(template)

    def test_landmarks_verified(self):
        roi,names,c,o=geo.validate_landmarks(self.spec)
        self.assertEqual(len(names),5)
        self.assertEqual(c.shape,(5,2))
        self.assertEqual(o.shape,(5,2))

    def test_mismatched_names_rejected(self):
        other=copy.deepcopy(self.spec);del other['open']['chin']
        with self.assertRaisesRegex(ValueError,'names'):
            geo.validate_landmarks(other)

    def test_point_outside_rejected(self):
        other=copy.deepcopy(self.spec);other['open']['chin']=[.99,.85]
        with self.assertRaisesRegex(ValueError,'inside'):
            geo.validate_landmarks(other)

    def test_duplicate_rejected(self):
        other=copy.deepcopy(self.spec);other['open']['chin']=other['open']['upper_lip'][:]
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            geo.validate_landmarks(other)

    def test_nonfinite_rejected(self):
        other=copy.deepcopy(self.spec);other['open']['chin'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):
            geo.validate_landmarks(other)

    def test_bad_roi_rejected(self):
        for roi in ((0,0,1,1),(.5,.5,.4,.7),(.1,.1,.2,.2),(float('nan'),.1,.6,.9)):
            with self.subTest(roi=roi),self.assertRaises(ValueError):
                geo.validate_roi(roi)

    def test_large_displacement_rejected(self):
        other=copy.deepcopy(self.spec)
        other['open']['left_corner']=[.73,.82]
        with self.assertRaises(ValueError):
            geo.validate_landmarks(other)

    def test_invalid_image_rejected(self):
        for bad in (self.closed[0:10],self.closed.astype(np.float32)):
            with self.subTest(shape=bad.shape),self.assertRaises(ValueError):
                geo.blend_geometry(bad,self.opened,.5,self.spec)

    def test_invalid_alpha_and_policy(self):
        for v in (float('nan'),-1,1.1,float('inf')):
            with self.subTest(value=v),self.assertRaises(ValueError):
                geo.blend_geometry(self.closed,self.opened,v,self.spec)
        with self.assertRaisesRegex(ValueError,'policy'):
            geo.blend_geometry(self.closed,self.opened,.5,self.spec,policy='bad')

    def test_alpha_zero_exact(self):
        for policy in ('soft','dominant'):
            got=geo.blend_geometry(self.closed,self.opened,0,self.spec,policy=policy)
            np.testing.assert_array_equal(got,self.closed)
            self.assertIsNot(got,self.closed)

    def test_outside_roi_exact_preservation(self):
        roi=self.spec['roi'];x0,y0,x1,y1=roi
        mask=np.zeros((512,512),bool)
        mask[int(y0*512):int(math.ceil(y1*512)),int(x0*512):int(math.ceil(x1*512))]=True
        for policy in ('soft','dominant'):
            for a in (0.25,.5,.75,1):
                with self.subTest(policy=policy,alpha=a):
                    got=geo.blend_geometry(self.closed,self.opened,a,self.spec,policy=policy)
                    self.assertEqual(np.count_nonzero(got[~mask]!=self.closed[~mask]),0)
                    self.assertEqual(geo.mouth_geometry_metrics(self.closed,got,roi)['outside_region_changed_pixels'],0)

    def test_full_open_keeps_mouth_not_full_face(self):
        got=geo.blend_geometry(self.closed,self.opened,1,self.spec,policy='soft')
        np.testing.assert_array_equal(got[10,10],self.closed[10,10])
        self.assertGreater(np.abs(got[325,275].astype(int)-self.closed[325,275].astype(int)).sum(),0)

    def test_geometry_changes_with_alpha(self):
        a=geo.blend_geometry(self.closed,self.opened,.25,self.spec,policy='dominant')
        b=geo.blend_geometry(self.closed,self.opened,.75,self.spec,policy='dominant')
        self.assertGreater(int(np.sum(a!=b)),0)

    def test_dominant_is_not_simple_crossfade(self):
        soft=geo.blend_geometry(self.closed,self.opened,.25,self.spec,policy='soft')
        dom=geo.blend_geometry(self.closed,self.opened,.25,self.spec,policy='dominant')
        self.assertFalse(np.array_equal(soft,dom))

    def test_geometry_refuses_fold(self):
        s=copy.deepcopy(self.spec)
        s['open']['left_corner'],s['open']['right_corner']=s['open']['right_corner'],s['open']['left_corner']
        with self.assertRaisesRegex(ValueError,'fold'):
            geo.blend_geometry(self.closed,self.opened,.5,s)

    def test_guides(self):
        a,b=geo.guides(self.closed,self.opened,self.spec)
        self.assertEqual(a.size,(512,512))
        self.assertNotEqual(a.tobytes(),b.tobytes())

    def test_offline_annotator(self):
        html=annotation_html(self.closed,self.opened,geo.create_template((.29,.49,.74,.83)))
        self.assertIn('data:image/png;base64,',html)
        self.assertIn('mouth_landmarks_verified.json',html)
        self.assertIn('verified:true',html)
        self.assertIn('left_corner',html)
        self.assertNotIn('https://',html)


if __name__=='__main__':unittest.main()
