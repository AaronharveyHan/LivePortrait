# coding: utf-8
"""V0.6.6 donor cavity and anchored tongue, deterministic CPU tests."""
import unittest
import numpy as np
from src.utils import animal_mouth_geometry as geo
from src.utils import animal_mouth_multiref as multi
from src.utils import animal_mouth_v065 as old
from src.utils import animal_mouth_v066 as new

class V066ImageTests(unittest.TestCase):
    def setUp(self):
        self.roi=(.26,.45,.79,.94)
        template=geo.create_template(self.roi)
        template['verified']=True
        template['open']['lower_lip'][1]+=.035
        template['open']['chin'][1]+=.022
        self.roi,self.names,cp,op=geo.validate_landmarks(template)
        middle=(cp+op)/2
        img=np.full((512,512,3),[100,110,70],np.uint8)
        img[323:410,214:335]=[28,37,40]
        tongue=img.copy()
        tongue[356:393,252:283]=[222,134,154]
        self.frames=[{'alpha':0.,'rgb':img,'points':cp},
                     {'alpha':.5,'rgb':tongue.copy(),'points':middle},
                     {'alpha':1.,'rgb':tongue,'points':op}]
        self.polys={0.:np.empty((0,2),np.float32),
                    .5:np.asarray([[252,354],[282,354],[285,393],[250,393]],np.float32),
                    1.:np.asarray([[252,354],[282,354],[285,393],[250,393]],np.float32)}
        self.pos=[0,.5,1]

    def test_multiframe_donor_fraction(self):
        cavities,masks,confidence,notes=new.prepare_multiref_cavity(self.frames,self.polys,self.roi,self.names)
        self.assertEqual(len(cavities),3)
        self.assertEqual(len(masks),3)
        self.assertEqual(notes[0]['repair_pixels'],0)
        self.assertGreater(notes[1]['repair_pixels'],0)
        self.assertGreater(notes[1]['donor_coverage_fraction'],0)
        self.assertGreater(float(np.max(confidence[1])),0)
        self.assertTrue(any(x['status']=='considered' for x in notes[1]['candidate_donors']))

    def test_inpaint_fallback_zero_donor_strength(self):
        cavities,_,confidence,notes=new.prepare_multiref_cavity(self.frames,self.polys,self.roi,self.names,donor_strength=0)
        old_cav,_=old.prepare_cavity_frames(self.frames,self.polys)
        for a,b in zip(cavities,old_cav):np.testing.assert_array_equal(a['rgb'],b['rgb'])
        self.assertEqual(notes[1]['donor_coverage_fraction'],0)
        self.assertEqual(np.count_nonzero(confidence[1]),0)

    def test_no_change_outside_repair_support(self):
        import cv2
        cavities,masks,_,_=new.prepare_multiref_cavity(self.frames,self.polys,self.roi,self.names)
        for f,m,cf in zip(self.frames,masks,cavities):
            support=cv2.dilate(m,np.ones((5,5),np.uint8))>0
            np.testing.assert_array_equal(cf['rgb'][~support],f['rgb'][~support])

    def test_donor_strength_changes_repair(self):
        a,_,_,_=new.prepare_multiref_cavity(self.frames,self.polys,self.roi,self.names,donor_strength=0)
        b,_,_,_=new.prepare_multiref_cavity(self.frames,self.polys,self.roi,self.names,donor_strength=1)
        self.assertTrue(np.any(a[1]['rgb']!=b[1]['rgb']))

    def test_root_fixed_when_tongue_changes_length(self):
        mask=old.polygon_mask(self.polys[1.])
        source=self.frames[1]
        for a in (.2,.5,1.):
            rgb,alpha,info=new._root_and_overlay(source['rgb'],mask,source['points'],source['points'],self.names,a,self.roi)
            self.assertEqual(rgb.shape,(512,512,3))
            self.assertGreater(int(np.count_nonzero(alpha)),0)
            self.assertTrue(np.isfinite(alpha).all())
            if a==.2:ref=info['root_xy']
            else:np.testing.assert_allclose(info['root_xy'],ref,atol=1e-8)

    def test_zero_tongue_returns_no_alpha(self):
        mask=old.polygon_mask(self.polys[1.])
        _,alpha,info=new._root_and_overlay(self.frames[1]['rgb'],mask,self.frames[1]['points'],self.frames[1]['points'],self.names,0,self.roi)
        self.assertIsNone(info)
        self.assertEqual(np.count_nonzero(alpha),0)

    def test_root_feather_suppresses_root_edge(self):
        mask=old.polygon_mask(self.polys[1.])
        _,a,_=new._root_and_overlay(self.frames[1]['rgb'],mask,self.frames[1]['points'],self.frames[1]['points'],self.names,1,self.roi,root_feather_px=3)
        _,b,_=new._root_and_overlay(self.frames[1]['rgb'],mask,self.frames[1]['points'],self.frames[1]['points'],self.names,1,self.roi,root_feather_px=25)
        self.assertNotEqual(float(a.sum()),float(b.sum()))

    def test_render_keeps_outside_roi(self):
        cavities,masks,_,_=new.prepare_multiref_cavity(self.frames,self.polys,self.roi,self.names)
        mesh=multi.prepare_meshes(cavities,self.roi)
        for q in (0,.15,.7,1):
            with self.subTest(q=q):
                base,out,alpha,audit=new.render_rooted(self.frames[0]['rgb'],self.frames,cavities,mesh,masks,.5,q,self.pos,self.roi,self.names)
                self.assertEqual(out.dtype,np.uint8)
                self.assertEqual(alpha.shape,(512,512))
                self.assertEqual(audit['outside_roi_changed_pixels'],0)
                self.assertEqual(multi.outside_changed(self.frames[0]['rgb'],base,self.roi),0)

    def test_jaw_is_independent_of_tongue_amount(self):
        cavities,masks,_,_=new.prepare_multiref_cavity(self.frames,self.polys,self.roi,self.names)
        mesh=multi.prepare_meshes(cavities,self.roi)
        a,_,_,_=new.render_rooted(self.frames[0]['rgb'],self.frames,cavities,mesh,masks,.5,0,self.pos,self.roi,self.names)
        b,_,_,_=new.render_rooted(self.frames[0]['rgb'],self.frames,cavities,mesh,masks,.5,1,self.pos,self.roi,self.names)
        np.testing.assert_array_equal(a,b)

    def test_invalid_parameters(self):
        for kw in ({'donor_strength':-1},{'dilate_px':2},{'inpaint_radius':0}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):
                new.prepare_multiref_cavity(self.frames,self.polys,self.roi,self.names,**kw)
        mask=old.polygon_mask(self.polys[1.])
        for kw in ({'root_depth':float('nan')},{'root_feather_px':0}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):
                new._root_and_overlay(self.frames[1]['rgb'],mask,self.frames[1]['points'],self.frames[1]['points'],self.names,.5,self.roi,**kw)

if __name__=='__main__':unittest.main()
