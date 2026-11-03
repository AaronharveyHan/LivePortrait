# coding: utf-8
"""V0.6.5 CLI: real 512px integration, path safety, holdout and strict diagnostics."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts import run_animal_mouth_v065 as cli
from src.utils import animal_mouth_geometry as geo


class V065CLITests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory(prefix='v065_',dir=cli.ROOT)
        self.addCleanup(self.t.cleanup)
        self.root=Path(self.t.name)
        self.ref=self.root/'v062';self.ref.mkdir()
        self.out=self.root/'out'
        self.ann=self.root/'landmarks.json'
        self.midpoints=self.root/'mid_verified.json'
        self.tongue=self.root/'tongue_verified.json'
        self.spec=geo.create_template((.29,.49,.74,.83))
        self.spec['verified']=True
        self.spec['open']['lower_lip'][1]+=.041
        self.spec['open']['chin'][1]+=.022
        mid={name:xy.copy() for name,xy in self.spec['closed'].items()}
        mid['lower_lip'][1]+=.023
        mid['chin'][1]+=.012
        self.mid_pts=mid
        c=np.zeros((512,512,3),np.uint8);c[:]=[100,100,70]
        o=c.copy();o[330:393,239:281]=[20,50,70];o[350:395,246:268]=[190,125,150]
        m=c.copy();m[325:373,240:275]=[35,50,60];m[345:372,248:266]=[170,120,142]
        hold=c.copy();hold[327:379,241:273]=[35,60,65];hold[349:379,247:269]=[175,120,145]
        self.paths=[self.ref/'closed_reference.png',self.root/'middle.png',self.ref/'open_reference_aligned.png',self.root/'holdout.png']
        for p,im in zip(self.paths,[c,m,o,hold]):Image.fromarray(im,'RGB').save(p)
        (self.ref/'report.json').write_text('{"mode":"original"}')
        self.spec['reference_sha256']={'closed':cli.old.sha(self.paths[0]),'open':cli.old.sha(self.paths[2])}
        self.ann.write_text(json.dumps(self.spec),encoding='utf-8')
        self.midpoints.write_text(json.dumps({
            'schema_version':1,'verified':True,'coordinate_system':'normalized_xy_on_512px_aligned_reference',
            'frames':{'0.45':mid},'source_sha256':{'0.45':cli.old.sha(self.paths[1])},
        }),encoding='utf-8')
        self.tongue.write_text(json.dumps({
            'schema_version':1,'verified':True,'coordinate_system':'normalized_xy_on_512px_aligned_reference',
            'source_sha256':{'0.0':cli.old.sha(self.paths[0]),'0.45':cli.old.sha(self.paths[1]),'1.0':cli.old.sha(self.paths[2])},
            'polygons':{'0.0':[],'.45':[[.483,.67],[.52,.66],[.54,.68],[.54,.73],[.49,.72]],
                        '1.0':[[.482,.64],[.54,.64],[.553,.74],[.525,.786],[.482,.765]]}
        }),encoding='utf-8')
        self.base=['--reference-dir',str(self.ref),'--landmarks',str(self.ann),
                   '--mid','0.45='+str(self.paths[1]),'--mid-points',str(self.midpoints),
                   '--mid-aligned','--same-dog','--tongue-manifest',str(self.tongue),
                   '--output-dir',str(self.out)]

    def args(self,extra=()):
        a=cli.parser().parse_args(self.base+list(extra))
        vals=cli.validate_args(a)
        return a,vals

    def test_plan_doesnt_read_models(self):
        a=cli.parser().parse_args(['--mid','0.45='+str(self.paths[1]),'--plan'])
        self.assertEqual(cli.main(['--mid','0.45='+str(self.paths[1]),'--plan']),0)

    def test_requires_mid_points(self):
        a=cli.parser().parse_args([v for v in self.base if v not in ('--mid-points',str(self.midpoints))])
        with self.assertRaisesRegex(ValueError,'mid-points'):
            cli.validate_args(a)

    def test_bad_tongue_values(self):
        a=cli.parser().parse_args(self.base+['--tongue-values','0,1'])
        with self.assertRaisesRegex(ValueError,'length'):
            cli.validate_args(a)

    def test_same_dog_required(self):
        a=cli.parser().parse_args([v for v in self.base if v!='--same-dog'])
        with self.assertRaisesRegex(ValueError,'same-dog'):
            cli.validate_args(a)

    def test_no_output_equal_input_dir(self):
        a=cli.parser().parse_args(self.base+['--output-dir',str(self.ref)])
        with self.assertRaisesRegex(ValueError,'input'):
            cli.validate_args(a)

    def test_init_tongue_no_model_and_no_overwrite(self):
        a=cli.parser().parse_args(self.base+['--init-tongue'])
        a.tongue_manifest=None
        vals=cli.validate_args(a)
        result=cli.run(a,*vals)
        self.assertTrue((self.out/'annotate_tongue.html').exists())
        self.assertTrue((self.out/'aligned_ref_01.png').exists())
        self.assertFalse(json.loads((self.out/'mouth_tongue_template.json').read_text())['verified'])
        self.assertEqual(result['ref_count'],3)
        with self.assertRaises(FileExistsError):cli.run(a,*vals)

    def test_render_report_masks_and_continuity(self):
        a,vals=self.args()
        report=cli.run(a,*vals)
        self.assertEqual(report['opening_mode'],'geometry')
        self.assertEqual(report['stage'],'v0.6.5_independent_jaw_tongue')
        self.assertTrue(report['outside_roi_changed_all_zero'])
        self.assertEqual(len(report['frames']),5)
        self.assertEqual(len(report['adjacent_roi_mae']['v065_jaw_tongue']),4)
        self.assertTrue((self.out/'comparison_sheet.png').is_file())
        self.assertTrue((self.out/'tongue_alpha_500.png').is_file())
        self.assertTrue((self.out/'cavity_ref_01.png').is_file())
        self.assertGreater(report['reference_openings_normalized'][1],0)
        self.assertLess(report['reference_openings_normalized'][1],1)
        with self.assertRaises(FileExistsError):cli.run(a,*vals)

    def test_manifest_hash_guard(self):
        data=json.loads(self.tongue.read_text());data['source_sha256']['0.45']='bad'
        self.tongue.write_text(json.dumps(data))
        a,vals=self.args()
        with self.assertRaisesRegex(ValueError,'hashes mismatch'):
            cli.run(a,*vals)

    def test_holdout_independent_and_scores(self):
        a,vals=self.args(['--holdout','0.7='+str(self.paths[3]),'--holdout-tongue','.5'])
        report=cli.run(a,*vals)
        self.assertIn('v065_jaw_tongue',report['holdout_evaluation']['mouth_roi_mae'])
        self.assertTrue((self.out/'holdout_aligned.png').is_file())

    def test_duplicate_holdout_contents_refused(self):
        copy_path=self.root/'copy_of_middle.png'
        copy_path.write_bytes(self.paths[1].read_bytes())
        a,vals=self.args(['--holdout','0.7='+str(copy_path)])
        with self.assertRaisesRegex(ValueError,'duplicates'):
            cli.run(a,*vals)

    def test_strict_control_step_rejects(self):
        a=cli.parser().parse_args(self.base+['--strict-continuity','--tongue-values','0,0,0,0,1'])
        with self.assertRaisesRegex(ValueError,'Control-rate'):
            cli.validate_args(a)

    def test_strict_pixel_rejects_before_writes(self):
        a,vals=self.args(['--strict-continuity','--max-roi-mae','.01'])
        with self.assertRaisesRegex(ValueError,'Continuity threshold'):
            cli.run(a,*vals)
        self.assertFalse((self.out/'report.json').exists())

    def test_manual_independent_tongue_controls(self):
        a,vals=self.args(['--tongue-values','0,0,.1,.5,1'])
        report=cli.run(a,*vals)
        self.assertEqual(report['tongue_target_source'],'user_supplied_independent')
        self.assertEqual(report['frames'][1]['tongue_mask_nonzero_pixels'],0)
        self.assertGreater(report['frames'][-1]['tongue_mask_nonzero_pixels'],0)


if __name__=='__main__':unittest.main()
