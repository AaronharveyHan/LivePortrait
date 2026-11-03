# coding: utf-8
"""V0.6.4 mesh, texture layering and validation; pure CPU, no ONNX weights."""
import copy
import math
import unittest

import numpy as np

from src.utils import animal_mouth_geometry as geo
from src.utils import animal_mouth_multiref as mr


class MultiReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        yy,xx=np.indices((512,512))
        cls.base=np.empty((512,512,3),np.uint8)
        cls.base[:,:,0]=(xx%256).astype(np.uint8)
        cls.base[:,:,1]=(yy%256).astype(np.uint8)
        cls.base[:,:,2]=80
        cls.spec=geo.create_template((.29,.49,.74,.83))
        cls.spec['verified']=True
        cls.spec['open']['lower_lip'][1]+=.025
        cls.spec['open']['chin'][1]+=.012
        cls.roi,cls.names,cls.cp,cls.op=geo.validate_landmarks(cls.spec)
        cls.opened=cls.base.copy()
        cls.opened[277:370,225:330]=[12,200,140]
        cls.mid=cls.base.copy()
        cls.mid[282:345,220:322]=[30,172,113]
        cls.frames=[{'alpha':0.,'rgb':cls.base,'points':cls.cp},
                    {'alpha':.4,'rgb':cls.mid,'points':mr.intermediate_proxy_points(.4,cls.cp,cls.op)},
                    {'alpha':1.,'rgb':cls.opened,'points':cls.op}]
        cls.meshes=mr.prepare_meshes(cls.frames,cls.roi)

    def test_frame_spec(self):
        a,path=mr.parse_frame_spec('0.333=C:\\demo\\dog.png')
        self.assertAlmostEqual(a,.333)
        self.assertEqual(str(path),'C:\\demo\\dog.png')

    def test_frame_spec_reject_bad(self):
        for value in ('bad','0.3','nan=a.png','0=a.png','1=a.png','-0.3=a.png','0.2=','abc=a'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                mr.parse_frame_spec(value)

    def test_reference_alphas(self):
        self.assertEqual(mr.validate_reference_alphas((0,.3,.8,1)),(0.,.3,.8,1.))
        for seq in ((0,1,0.5),(0,0.02,1),(0.1,1),(0,1.1),(0,float('nan'),1)):
            with self.subTest(seq=seq),self.assertRaises(ValueError):
                mr.validate_reference_alphas(seq)

    def test_target_validation(self):
        self.assertEqual(mr.validate_target_alphas((0,.25,1)),(0,.25,1))
        for seq in ((0,0.25,0.25,1),(0,0.5),(0,0.5,0.5004,1),(0,float('inf'),1)):
            with self.subTest(seq=seq),self.assertRaises(ValueError):
                mr.validate_target_alphas(seq)

    def test_proxy_middle_is_mathematical_only(self):
        x=mr.intermediate_proxy_points(.4,self.cp,self.op)
        np.testing.assert_allclose(x,self.cp*.6+self.op*.4)

    def test_manual_normalized_points(self):
        vals={k:list(self.spec['closed'][k]) for k in self.names}
        got=mr.validate_manual_points(vals,self.names,self.roi)
        self.assertEqual(got.shape,(5,2))

    def test_manual_reject_invalid(self):
        vals={k:list(self.spec['closed'][k]) for k in self.names}
        for mutation in ('extra','nan','outside','dup'):
            bad=copy.deepcopy(vals)
            if mutation=='extra': bad['foo']=[.5,.5]
            if mutation=='nan':bad['chin'][0]=float('nan')
            if mutation=='outside':bad['chin'][0]=0.99
            if mutation=='dup':bad['chin']=bad['upper_lip'][:]
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                mr.validate_manual_points(bad,self.names,self.roi)

    def test_meshes_are_prepared_for_all_intervals(self):
        self.assertEqual(len(self.meshes),2)
        for a,b,tri in self.meshes:
            self.assertEqual(a.shape,b.shape)
            self.assertGreater(len(tri),10)

    def test_reject_fold(self):
        bad=self.op.copy()
        i=list(self.names).index('left_corner')
        j=list(self.names).index('right_corner')
        bad[i],bad[j]=bad[j].copy(),bad[i].copy()
        with self.assertRaises(ValueError):
            mr.pair_mesh(self.cp,bad,self.roi)

    def test_bad_image(self):
        frames=copy.copy(self.frames)
        frames[1]={'alpha':.4,'rgb':self.mid[:100],'points':self.frames[1]['points']}
        with self.assertRaises(ValueError):
            mr.prepare_meshes(frames,self.roi)

    def test_exact_closed_not_aliased(self):
        a,details=mr.render_layered(self.base,self.frames,0,self.roi,self.names,self.meshes)
        np.testing.assert_array_equal(a,self.base)
        self.assertIsNot(a,self.base)
        self.assertTrue(details['exact_anchor'])

    def test_exact_mid_preserves_mouth_texture(self):
        from src.utils.animal_dual_source import mouth_mask
        a,details=mr.render_layered(self.base,self.frames,.4,self.roi,self.names,self.meshes)
        mask=mouth_mask(512,512,self.roi,24)
        iy,ix=np.unravel_index(np.argmax(mask),mask.shape)
        np.testing.assert_array_equal(a[iy,ix],self.mid[iy,ix])
        self.assertTrue(details['exact_anchor'])

    def test_exact_open_uses_open_mouth_only(self):
        a,details=mr.render_layered(self.base,self.frames,1,self.roi,self.names,self.meshes)
        self.assertEqual(mr.outside_changed(self.base,a,self.roi),0)
        self.assertTrue(details['exact_anchor'])
        self.assertFalse(np.array_equal(a,self.base))

    def test_midpoint_neighbor_selection(self):
        out,details=mr.render_layered(self.base,self.frames,.15,self.roi,self.names,self.meshes)
        self.assertEqual(details['interior_source'],0.)
        self.assertFalse(details['exact_anchor'])
        out2,details2=mr.render_layered(self.base,self.frames,.3,self.roi,self.names,self.meshes)
        self.assertEqual(details2['interior_source'],.4)
        self.assertFalse(np.array_equal(out,out2))

    def test_flow_and_nearest_preserve_outside(self):
        for method in mr.METHODS:
            with self.subTest(method=method):
                out,_=mr.render_layered(self.base,self.frames,.6,self.roi,self.names,self.meshes,method=method)
                self.assertEqual(mr.outside_changed(self.base,out,self.roi),0)
                self.assertEqual(out.dtype,np.uint8)
                self.assertEqual(out.shape,(512,512,3))

    def test_invalid_policy_and_alpha(self):
        for kwargs in ({'method':'bad'}, {'alpha':float('nan')}, {'alpha':-0.1}):
            d={'alpha':.2,'method':'layered_nearest'}
            d.update(kwargs)
            with self.assertRaises(ValueError):
                mr.render_layered(self.base,self.frames,d['alpha'],self.roi,self.names,self.meshes,method=d['method'])

    def test_interior_mask_stays_inside_roi(self):
        from src.utils.animal_mouth_multiref import _mouth_interior_mask
        mask=_mouth_interior_mask(self.roi,self.frames[1]['points'],self.names)
        self.assertEqual(mask.shape,(512,512))
        self.assertEqual(mask[0,0],0)
        self.assertTrue(np.any(mask>0))

    def test_region_mae_zero_symmetric(self):
        self.assertEqual(mr.region_mae(self.base,self.base,self.roi),0)
        self.assertAlmostEqual(mr.region_mae(self.base,self.mid,self.roi),mr.region_mae(self.mid,self.base,self.roi))

    def test_mid_image_changes_inside_roi(self):
        a,_=mr.render_layered(self.base,self.frames,.4,self.roi,self.names,self.meshes)
        self.assertGreater(mr.region_mae(self.base,a,self.roi),0)


if __name__=='__main__':unittest.main()
