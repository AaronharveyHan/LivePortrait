# coding: utf-8
"""V0.6.5 finite geometry, verified polygon, separated tongue, continuity tests."""
import copy
import unittest

import numpy as np
from src.utils import animal_mouth_geometry as geo
from src.utils import animal_mouth_multiref as multi
from src.utils import animal_mouth_v065 as v


class JawTests(unittest.TestCase):
    def setUp(self):
        self.names=('left_corner','upper_lip','right_corner','lower_lip','chin')
        self.p0=np.asarray([[200,280],[250,275],[300,280],[250,300],[250,315]],float)
        self.p1=np.asarray([[200,280],[250,275],[300,280],[250,335],[250,352]],float)
        self.m=.2*self.p0+.8*self.p1
        self.frames=[{'alpha':0.,'points':self.p0}, {'alpha':.333333,'points':self.m}, {'alpha':1.,'points':self.p1}]

    def test_aperture_increases(self):
        self.assertLess(v.jaw_aperture(self.p0,self.names),v.jaw_aperture(self.p1,self.names))

    def test_normalization_not_equals_time(self):
        pos,raw=v.calibrate_openings(self.frames,self.names)
        self.assertAlmostEqual(pos[1],.8)
        self.assertNotAlmostEqual(pos[1],.333333)
        self.assertEqual(len(raw),3)

    def test_geometry_maps_to_time(self):
        pos,_=v.calibrate_openings(self.frames,self.names)
        self.assertAlmostEqual(v.opening_to_time(pos[1],pos,self.frames),.333333)
        self.assertAlmostEqual(v.opening_to_time(.0,pos,self.frames),0)
        self.assertAlmostEqual(v.opening_to_time(1.,pos,self.frames),1)

    def test_time_diagnostic_fallback(self):
        pos,_=v.calibrate_openings(self.frames,self.names,mode='time')
        self.assertEqual(pos,[0.,.333333,1.])

    def test_nonmonotonic_refused(self):
        bad=list(self.frames)
        bad[1]={**bad[1],'points':self.p1*1.1-self.p0*.1}
        with self.assertRaisesRegex(ValueError,'non-monotone'):
            v.calibrate_openings(bad,self.names)

    def test_geometry_peak_uses_aperture_apex_not_last_photo(self):
        p2=self.p1.copy();p2[3,1]+=30;p2[4,1]+=30
        p3=self.p1.copy();p3[3,1]+=10;p3[4,1]+=10
        fs=[{'alpha':0.,'points':self.p0},
            {'alpha':.333333,'points':self.m},
            {'alpha':.666667,'points':p2},
            {'alpha':1.,'points':p3}]
        with self.assertRaisesRegex(ValueError,'non-monotone'):
            v.calibrate_openings(fs,self.names,mode='geometry')
        pos,raw=v.calibrate_openings(fs,self.names,mode='geometry_peak')
        self.assertEqual(len(pos),4)
        self.assertEqual(pos[0],0.)
        self.assertEqual(pos[2],1.)
        self.assertLess(pos[3],1.)
        self.assertAlmostEqual(v.opening_to_time(1.,pos,fs),.666667)
        self.assertAlmostEqual(v.opening_to_time(pos[1],pos,fs),.333333)
        self.assertAlmostEqual(v.opening_to_time(.5,pos,fs),
                               float(np.interp(.5,pos[:3],[0.,.333333,.666667])))

    def test_geometry_peak_rejects_second_rise(self):
        p2=self.p1.copy();p2[3,1]+=30;p2[4,1]+=30
        p3=self.p1.copy();p3[3,1]+=15;p3[4,1]+=15
        p4=self.p1.copy();p4[3,1]+=20;p4[4,1]+=20
        fs=[{'alpha':a,'points':p} for a,p in
            [(0.,self.p0),(.25,self.m),(.5,p2),(.75,p3),(1.,p4)]]
        with self.assertRaisesRegex(ValueError,'second opening rise'):
            v.calibrate_openings(fs,self.names,mode='geometry_peak')

    def test_opening_to_time_rejects_invalid_positions(self):
        for pos in ([0.,.9,.8], [0.,1.,.5, .7], [0.,.5,.5]):
            fs=[{'alpha':float(i)/(len(pos)-1)} for i in range(len(pos))]
            with self.subTest(pos=pos),self.assertRaises(ValueError):
                v.opening_to_time(.5,pos,fs)

    def test_invalid_opening_points(self):
        bad=self.p0.copy();bad[3]=bad[1]
        with self.assertRaisesRegex(ValueError,'Lower lip'):
            v.jaw_aperture(bad,self.names)
        bad=self.p0.copy();bad[0]=bad[2]
        with self.assertRaisesRegex(ValueError,'too close'):
            v.jaw_aperture(bad,self.names)

    def test_default_curve_delay(self):
        self.assertEqual(v.default_tongue_amount(.25),0)
        self.assertEqual(v.default_tongue_amount(1),1)
        self.assertTrue(0<v.default_tongue_amount(.5)<v.default_tongue_amount(.75)<1)

    def test_bad_opening_raised(self):
        for x in (-1,2,float('nan'),True):
            with self.subTest(x=x),self.assertRaises(ValueError):
                v.opening_to_time(x,[0,.8,1],self.frames)


class TongueTests(unittest.TestCase):
    def setUp(self):
        self.roi=(.29,.49,.74,.83)
        self.names=('left_corner','upper_lip','right_corner','lower_lip','chin')
        spec=geo.create_template(self.roi)
        c={name:xy.copy() for name,xy in spec['closed'].items()}
        mid={name:xy.copy() for name,xy in c.items()}
        opened={name:xy.copy() for name,xy in c.items()}
        mid['lower_lip'][1]+=.019
        mid['chin'][1]+=.013
        opened['lower_lip'][1]+=.04
        opened['chin'][1]+=.026
        def pts(points):return np.asarray([points[n] for n in self.names],float)*512
        self.frames=[]
        for a,point,color in [(0,c,[10,55,70]),(.45,mid,[70,80,100]),(1,opened,[120,110,130])]:
            image=np.empty((512,512,3),np.uint8);image[:]=[85,100,40]
            image[320:355,245:300]=color
            self.frames.append({'alpha':a,'points':pts(point),'rgb':image})
        self.roi=tuple(self.roi)
        self.polygons={0:np.empty((0,2),np.float32),
                       .45:np.asarray([[.48,.67],[.52,.65],[.55,.70],[.53,.73],[.49,.72]],np.float32)*512,
                       1:np.asarray([[.48,.63],[.54,.63],[.56,.74],[.53,.79],[.48,.77]],np.float32)*512}
        self.manifest={'schema_version':1,'coordinate_system':'normalized_xy_on_512px_aligned_reference',
                       'verified':True,'source_sha256':{'0.0':'abc','0.45':'def','1.0':'ghi'},
                       'polygons':{str(a):(p/512).tolist() for a,p in self.polygons.items()}}
        self.hashes=self.manifest['source_sha256']

    def test_good_mask_and_explicit_absent(self):
        polys=v.validate_tongue_manifest(self.manifest,self.frames,self.hashes,self.roi)
        self.assertEqual(len(polys[0]),0)
        self.assertGreater(np.count_nonzero(v.polygon_mask(polys[1])),100)

    def test_unverified_refused(self):
        x=copy.deepcopy(self.manifest);x['verified']=False
        with self.assertRaisesRegex(ValueError,'verified'):
            v.validate_tongue_manifest(x,self.frames,self.hashes,self.roi)

    def test_hash_guard(self):
        with self.assertRaisesRegex(ValueError,'hashes'):
            v.validate_tongue_manifest(self.manifest,self.frames,{**self.hashes,'1.0':'different'},self.roi)

    def test_missing_frame_refused(self):
        x=copy.deepcopy(self.manifest);del x['polygons']['0.45']
        with self.assertRaisesRegex(ValueError,'every'):
            v.validate_tongue_manifest(x,self.frames,self.hashes,self.roi)

    def test_outside_roi_refused(self):
        x=copy.deepcopy(self.manifest);x['polygons']['1'][0]=[.99,.5]
        with self.assertRaisesRegex(ValueError,'outside'):
            v.validate_tongue_manifest(x,self.frames,self.hashes,self.roi)

    def test_nonfinite_refused(self):
        x=copy.deepcopy(self.manifest);x['polygons']['1'][0]=[float('nan'),.5]
        with self.assertRaisesRegex(ValueError,'finite'):
            v.validate_tongue_manifest(x,self.frames,self.hashes,self.roi)

    def test_bad_polygon_refused(self):
        x=copy.deepcopy(self.manifest);x['polygons']['1']=[[.5,.6],[.51,.61]]
        with self.assertRaisesRegex(ValueError,'3..60'):
            v.validate_tongue_manifest(x,self.frames,self.hashes,self.roi)

    def test_all_absent_refused(self):
        x=copy.deepcopy(self.manifest)
        for key in x['polygons']:x['polygons'][key]=[]
        with self.assertRaisesRegex(ValueError,'All tongue'):
            v.validate_tongue_manifest(x,self.frames,self.hashes,self.roi)

    def test_inpaint_modifies_only_polygon_neighborhood(self):
        cavity,masks=v.prepare_cavity_frames(self.frames,self.polygons)
        self.assertEqual(len(cavity),3)
        np.testing.assert_array_equal(cavity[0]['rgb'],self.frames[0]['rgb'])
        np.testing.assert_array_equal(cavity[-1]['rgb'][0,0],self.frames[-1]['rgb'][0,0])
        self.assertEqual(int(masks[0].sum()),0)

    def test_separate_tongue_amount_zero_one(self):
        cavity,masks=v.prepare_cavity_frames(self.frames,self.polygons)
        cm=multi.prepare_meshes(cavity,self.roi)
        pos,_=v.calibrate_openings(self.frames,self.names)
        a,ma,_=v.render_independent(self.frames[0]['rgb'],self.frames,cavity,cm,self.polygons,masks,
                                    .9,0,pos,self.roi,self.names)
        b,mb,audit=v.render_independent(self.frames[0]['rgb'],self.frames,cavity,cm,self.polygons,masks,
                                    .9,1,pos,self.roi,self.names)
        self.assertEqual(np.count_nonzero(ma),0)
        self.assertGreater(np.count_nonzero(mb>.05),100)
        self.assertFalse(np.array_equal(a,b))
        self.assertEqual(audit['outside_roi_changed_pixels'],0)
        self.assertEqual(multi.outside_changed(self.frames[0]['rgb'],b,self.roi),0)

    def test_same_opening_different_tongue(self):
        cavity,masks=v.prepare_cavity_frames(self.frames,self.polygons)
        cm=multi.prepare_meshes(cavity,self.roi)
        pos,_=v.calibrate_openings(self.frames,self.names)
        results=[v.render_independent(self.frames[0]['rgb'],self.frames,cavity,cm,self.polygons,masks,
                                    .65,q,pos,self.roi,self.names)[0] for q in (0,.25,.75)]
        self.assertFalse(np.array_equal(results[0],results[1]))
        self.assertFalse(np.array_equal(results[1],results[2]))

    def test_invalid_tongue_amount(self):
        cavity,masks=v.prepare_cavity_frames(self.frames,self.polygons)
        cm=multi.prepare_meshes(cavity,self.roi)
        for q in (-1,2,float('inf'),True):
            with self.subTest(q=q),self.assertRaises(ValueError):
                v.render_independent(self.frames[0]['rgb'],self.frames,cavity,cm,self.polygons,masks,
                                     .5,q,[0,.5,1],self.roi,self.names)

    def test_continuity_audit(self):
        a=np.zeros((512,512,3),np.uint8)
        b=a.copy();b[260:370,190:300]=255
        c=v.continuity_report([a,a,b],[0,.2,.5],[0,.2,.9],self.roi,
                              max_opening_step=.3,max_tongue_step=.4,max_roi_mae=5)
        self.assertFalse(c['all_pass'])
        self.assertTrue(c['steps'][0]['pass'])
        self.assertFalse(c['steps'][1]['pass'])

    def test_continuity_no_false_guarantee(self):
        z=np.zeros((512,512,3),np.uint8)
        c=v.continuity_report([z,z,z],[0,.2,.4],[0,.1,.2],self.roi)
        self.assertTrue(c['all_pass'])
        self.assertIn('does not prove natural',c['note'])

    def test_offline_annotator_requires_each_frame_confirmed(self):
        from src.utils.animal_tongue_annotator import tongue_annotation_html
        html=tongue_annotation_html(self.frames,self.hashes,self.roi)
        self.assertIn('data:image/png;base64,',html)
        self.assertIn('confirmed.some(x=>!x)',html)
        self.assertIn('mouth_tongue_polygons_verified.json',html)
        self.assertNotIn('https://',html)
        self.assertNotIn('http://',html)


if __name__=='__main__':unittest.main()
