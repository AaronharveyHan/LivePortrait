# coding: utf-8
"""V0.6.4 argument, file safety, complete offline integration tests."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts import run_animal_mouth_multiref as cli
from src.utils import animal_mouth_geometry as geo


class MultiRefCLITests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory(prefix='v064_',dir=cli.ROOT)
        self.addCleanup(self.t.cleanup)
        self.root=Path(self.t.name)
        self.ref=self.root/'v062'
        self.ref.mkdir()
        self.out=self.root/'out'
        self.spec=geo.create_template((.29,.49,.74,.83))
        self.spec['verified']=True
        self.spec['open']['lower_lip'][1]+=.025
        self.spec['open']['chin'][1]+=.012
        closed=np.zeros((512,512,3),np.uint8)
        closed[:]=[125,150,70]
        closed[275:325,218:310]=[32,45,48]
        opened=closed.copy()
        opened[270:375,218:310]=[15,40,110]
        mid=closed.copy()
        mid[278:350,218:310]=[24,40,93]
        hold=closed.copy()
        hold[280:345,218:310]=[24,40,88]
        paths=[self.ref/'closed_reference.png',self.ref/'open_reference_aligned.png',self.root/'middle.png',self.root/'holdout.png']
        for path,arr in zip(paths,[closed,opened,mid,hold]):
            Image.fromarray(arr,'RGB').save(path)
        self.spec['reference_sha256']={'closed':cli.sha(paths[0]),'open':cli.sha(paths[1])}
        self.labels=self.root/'landmarks.json'
        self.labels.write_text(json.dumps(self.spec),encoding='utf-8')
        (self.ref/'report.json').write_text('{"mode":"original"}',encoding='utf-8')
        self.argv=['--reference-dir',str(self.ref),'--landmarks',str(self.labels),'--mid','0.4='+str(paths[2]),
                   '--mid-aligned','--same-dog','--output-dir',str(self.out), '--method','layered_nearest']

    def test_plan_no_images_loaded(self):
        args=cli.parser().parse_args(self.argv+['--plan'])
        alphas,mids,hold=cli.validate_args(args)
        self.assertEqual(len(mids),1)
        self.assertIsNone(hold)

    def test_requires_same_dog(self):
        args=cli.parser().parse_args([v for v in self.argv if v!='--same-dog'])
        with self.assertRaisesRegex(ValueError,'same-dog'):
            cli.validate_args(args)

    def test_missing_mid_refused(self):
        args=cli.parser().parse_args(self.argv+['--mid','0.6='+str(self.root/'missing.png')])
        with self.assertRaises(FileNotFoundError):
            cli.validate_args(args)

    def test_duplicate_alpha_refused(self):
        args=cli.parser().parse_args(self.argv+['--mid','0.4='+str(self.root/'middle.png')])
        with self.assertRaises(ValueError):
            cli.validate_args(args)

    def test_holdout_alpha_overlap_refused(self):
        args=cli.parser().parse_args(self.argv+['--holdout','0.4='+str(self.root/'holdout.png')])
        with self.assertRaisesRegex(ValueError,'Holdout'):
            cli.validate_args(args)

    def test_holdout_same_photo_refused(self):
        args=cli.parser().parse_args(self.argv+['--holdout','0.7='+str(self.root/'middle.png')])
        with self.assertRaises(ValueError):
            cli.validate_args(args)

    def test_output_may_not_equal_source_dir(self):
        args=cli.parser().parse_args(self.argv+['--output-dir',str(self.ref)])
        with self.assertRaises(ValueError):
            cli.validate_args(args)

    def test_bad_mid_points_manifest(self):
        manifest=self.root/'mid_points.json'
        manifest.write_text(json.dumps({'schema_version':1,'verified':False,'coordinate_system':'normalized_xy_on_512px_aligned_reference','frames':{'.4':{}}}),encoding='utf-8')
        args=cli.parser().parse_args(self.argv+['--mid-points',str(manifest)])
        a,m,h=cli.validate_args(args)
        with self.assertRaises(ValueError):
            cli.run(args,a,m,h)

    def test_complete_nearest_and_holdout(self):
        args=cli.parser().parse_args(self.argv+['--holdout','0.7='+str(self.root/'holdout.png')])
        a,m,h=cli.validate_args(args)
        report=cli.run(args,a,m,h)
        self.assertTrue(report['outside_roi_changed_all_zero'])
        self.assertTrue((self.out/'comparison_sheet.png').is_file())
        self.assertTrue((self.out/'mid_01_aligned.png').is_file())
        self.assertEqual(report['reference_alphas'],[0,.4,1])
        self.assertIn('v063_soft',report['holdout_evaluation']['roi_mae'])
        self.assertIn('layered_nearest',report['holdout_evaluation']['roi_mae'])
        self.assertEqual(report['middle_frames'][0]['landmark_source'],'endpoint_linear_proxy_UNVERIFIED')
        with self.assertRaises(FileExistsError):
            cli.run(args,a,m,h)

    def test_wrong_reference_sha_refused(self):
        spec=json.loads(self.labels.read_text(encoding='utf-8'))
        spec['reference_sha256']['closed']='bad'
        self.labels.write_text(json.dumps(spec),encoding='utf-8')
        args=cli.parser().parse_args(self.argv)
        a,m,h=cli.validate_args(args)
        with self.assertRaisesRegex(ValueError,'do not match'):
            cli.run(args,a,m,h)

    def test_raw_mid_with_onnx_reference_refused(self):
        (self.ref/'report.json').write_text('{"mode":"onnx"}',encoding='utf-8')
        args=cli.parser().parse_args([v for v in self.argv if v!='--mid-aligned'])
        a,m,h=cli.validate_args(args)
        with self.assertRaisesRegex(ValueError,'ONNX-reconstructed'):
            cli.run(args,a,m,h)


if __name__=='__main__':unittest.main()

class MultiRefUIStandaloneTests(unittest.TestCase):
    def setUp(self):
        self.base=MultiRefCLITests(methodName='test_plan_no_images_loaded')
        self.base.setUp()
        self.addCleanup(self.base.tearDown if hasattr(self.base, 'tearDown') else lambda: None)
        self.addCleanup(self.base.doCleanups)

    def test_offline_html_requires_real_clicks(self):
        from src.utils.animal_mouth_multi_annotator import annotation_html
        from src.utils.animal_mouth_multiref import intermediate_proxy_points
        from src.utils import animal_mouth_geometry as g
        from PIL import Image
        c=np.asarray(Image.open(self.base.root/'middle.png').convert('RGB'))
        roi,names,p0,p1=g.validate_landmarks(self.base.spec)
        html=annotation_html([(.4,c,intermediate_proxy_points(.4,p0,p1))],names,roi,{'.4':cli.sha(self.base.root/'middle.png')})
        self.assertIn('data:image/png;base64,',html)
        self.assertIn('mouth_mid_landmarks_verified.json',html)
        self.assertIn('picked.some(fr=>fr.some(p=>!p.clicked))',html)
        self.assertNotIn('https://',html)
        self.assertNotIn('http://',html)

    def test_annotator_creates_browser_html(self):
        from scripts import annotate_animal_mouth_multiref as ann
        args=ann.parser().parse_args(self.base.argv+['--output-dir',str(self.base.root/'annotations')])
        a,m,h=ann.validate_args(args)
        result=ann.run(args,a,m,h)
        self.assertEqual(result['frame_count'],1)
        self.assertTrue((self.base.root/'annotations'/'annotate_mid_mouth.html').exists())
        data=json.loads((self.base.root/'annotations'/'mouth_mid_landmarks_template.json').read_text())
        self.assertFalse(data['verified'])
        with self.assertRaises(FileExistsError):
            ann.run(args,a,m,h)

    def test_annotator_manifest_accepted_and_hash_guard(self):
        from src.utils import animal_mouth_geometry as g
        roi,names,c,o=g.validate_landmarks(self.base.spec)
        template={
            'schema_version':1,'coordinate_system':'normalized_xy_on_512px_aligned_reference',
            'verified':True,'frames':{'0.4':self.base.spec['closed']},
            'source_sha256':{'0.4':cli.sha(self.base.root/'middle.png')},
        }
        p=self.base.root/'middle_verified.json'
        p.write_text(json.dumps(template))
        args=cli.parser().parse_args(self.base.argv+['--mid-points',str(p)])
        a,m,h=cli.validate_args(args)
        report=cli.run(args,a,m,h)
        self.assertEqual(report['middle_frames'][0]['landmark_source'],'manual_verified')
        template['source_sha256']['0.4']='bad'
        p.write_text(json.dumps(template))
        args.output_dir=self.base.root/'different_out'
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            cli.run(args,a,m,h)

    def test_render_both_modes_with_masks(self):
        args=cli.parser().parse_args(self.base.argv+['--method','both'])
        a,m,h=cli.validate_args(args)
        report=cli.run(args,a,m,h)
        self.assertEqual(report['methods'],['layered_nearest','layered_flow'])
        self.assertTrue((self.base.out/'mouth_outer_mask.png').exists())
        self.assertTrue((self.base.out/'mouth_interior_mask_500.png').exists())
        self.assertTrue((self.base.out/'mouth_layered_flow_500.png').exists())
        self.assertTrue(report['outside_roi_changed_all_zero'])

    def test_mid_endpoint_duplicate_rejected(self):
        args=cli.parser().parse_args(self.base.argv)
        args.mid=['0.4='+str(self.base.ref/'closed_reference.png')]
        with self.assertRaisesRegex(ValueError,'endpoint'):
            cli.validate_args(args)

    def test_holdout_endpoint_duplicate_rejected(self):
        args=cli.parser().parse_args(self.base.argv+['--holdout','0.7='+str(self.base.ref/'open_reference_aligned.png')])
        with self.assertRaisesRegex(ValueError,'endpoint'):
            cli.validate_args(args)
