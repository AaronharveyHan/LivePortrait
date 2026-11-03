# coding: utf-8
"""V0.6.6 executable end-to-end tests with disposable in-repository images."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from scripts import run_animal_mouth_v066 as cli
from src.utils import animal_mouth_geometry as geo

class V066CLITests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory(prefix='v066_',dir=cli.ROOT)
        self.addCleanup(t.cleanup)
        self.root=Path(t.name)
        self.ref=self.root/'v062';self.ref.mkdir()
        self.out=self.root/'out'
        self.ann=self.root/'landmarks.json'
        self.midpoints=self.root/'mid_verified.json'
        self.tongue=self.root/'tongue_verified.json'
        spec=geo.create_template((.29,.49,.74,.83))
        spec['verified']=True
        spec['open']['lower_lip'][1]+=.041
        spec['open']['chin'][1]+=.022
        mid={name:xy.copy() for name,xy in spec['closed'].items()}
        mid['lower_lip'][1]+=.023
        mid['chin'][1]+=.012
        c=np.full((512,512,3),[100,100,70],np.uint8)
        c[318:400,238:288]=[30,45,51]
        o=c.copy();o[350:395,246:270]=[205,133,160]
        m=c.copy();m[345:372,248:266]=[190,120,142]
        hold=c.copy();hold[340:378,242:270]=[150,120,145]
        self.paths=[self.ref/'closed_reference.png',self.root/'middle.png',
                    self.ref/'open_reference_aligned.png',self.root/'holdout.png']
        for p,im in zip(self.paths,[c,m,o,hold]):Image.fromarray(im,'RGB').save(p)
        (self.ref/'report.json').write_text('{"mode":"original"}')
        spec['reference_sha256']={'closed':cli.previous.old.sha(self.paths[0]),
                                  'open':cli.previous.old.sha(self.paths[2])}
        self.ann.write_text(json.dumps(spec),encoding='utf-8')
        self.midpoints.write_text(json.dumps({
            'schema_version':1,'verified':True,
            'coordinate_system':'normalized_xy_on_512px_aligned_reference',
            'frames':{'0.45':mid},
            'source_sha256':{'0.45':cli.previous.old.sha(self.paths[1])}}),encoding='utf-8')
        self.tongue.write_text(json.dumps({
            'schema_version':1,'verified':True,
            'coordinate_system':'normalized_xy_on_512px_aligned_reference',
            'source_sha256':{'0.0':cli.previous.old.sha(self.paths[0]),
                '0.45':cli.previous.old.sha(self.paths[1]),
                '1.0':cli.previous.old.sha(self.paths[2])},
            'polygons':{'0.0':[],
                '0.45':[[.483,.67],[.52,.66],[.54,.68],[.54,.73],[.49,.72]],
                '1.0':[[.482,.64],[.54,.64],[.553,.74],[.525,.786],[.482,.765]]}
        }),encoding='utf-8')
        self.base=['--reference-dir',str(self.ref),'--landmarks',str(self.ann),
                   '--mid','0.45='+str(self.paths[1]),'--mid-points',str(self.midpoints),
                   '--mid-aligned','--same-dog','--tongue-manifest',str(self.tongue),
                   '--output-dir',str(self.out)]

    def make(self,extra=()):
        args=cli.parser().parse_args(self.base+list(extra))
        return args,cli.validate_args(args)

    def test_plan(self):
        self.assertEqual(cli.main(['--plan']),0)

    def test_parser_controls(self):
        a,_=self.make(['--root-depth','.7','--donor-strength','0'])
        self.assertEqual(a.root_depth,.7)
        self.assertEqual(a.donor_strength,0)

    def test_full_run_report_and_images(self):
        a,v=self.make(['--tongue-values','0,0,.2,.6,1'])
        report=cli.run(a,*v)
        self.assertEqual(report['stage'],'v0.6.6_cavity_root')
        self.assertTrue(report['outside_roi_changed_all_zero'])
        self.assertEqual(len(report['frames']),5)
        self.assertEqual(len(report['cavity']['references']),3)
        self.assertTrue((self.out/'comparison_sheet.png').is_file())
        self.assertTrue((self.out/'donor_confidence_01.png').is_file())
        self.assertTrue((self.out/'tongue_root_alpha_750.png').is_file())
        self.assertTrue((self.out/'mouth_v065_baseline_250.png').is_file())
        self.assertIsNone(report['frames'][0]['tongue_root'])
        with self.assertRaises(FileExistsError):cli.run(a,*v)

    def test_geometry_peak_nonmonotone(self):
        spec=json.loads(self.midpoints.read_text(encoding='utf-8'))
        spec['frames']['0.45']['lower_lip'][1]+=.025
        self.midpoints.write_text(json.dumps(spec))
        a,v=self.make(['--opening-mode','geometry_peak'])
        report=cli.run(a,*v)
        self.assertEqual(report['peak_index'],1)
        self.assertLess(report['reference_openings_normalized'][-1],1)

    def test_hash_mismatch_fails_before_outputs(self):
        s=json.loads(self.tongue.read_text());s['source_sha256']['0.45']='BAD'
        self.tongue.write_text(json.dumps(s))
        a,v=self.make()
        with self.assertRaisesRegex(ValueError,'hashes mismatch'):cli.run(a,*v)
        self.assertFalse((self.out/'comparison_sheet.png').exists())

    def test_no_overwrite_ref_dir(self):
        a=cli.parser().parse_args(self.base+['--output-dir',str(self.ref)])
        with self.assertRaisesRegex(ValueError,'input directory'):cli.validate_args(a)

    def test_strict_continuity_fails_before_output(self):
        a,v=self.make(['--strict-continuity','--max-roi-mae','.001'])
        with self.assertRaisesRegex(ValueError,'Continuity threshold'):cli.run(a,*v)
        self.assertFalse((self.out/'report.json').exists())

    def test_holdout_is_independent(self):
        a,v=self.make(['--holdout','0.7='+str(self.paths[3]),'--holdout-tongue','.5'])
        report=cli.run(a,*v)
        self.assertIn('v066_rooted_tongue',report['holdout_evaluation']['roi_mae'])
        self.assertTrue((self.out/'holdout_aligned.png').exists())

    def test_holdout_duplicate_content_refused(self):
        duplicate=self.root/'duplicated.png';duplicate.write_bytes(self.paths[1].read_bytes())
        a,v=self.make(['--holdout','0.7='+str(duplicate)])
        with self.assertRaisesRegex(ValueError,'duplicates'):cli.run(a,*v)

    def test_invalid_parameters(self):
        for flag,value in (('--donor-strength','1.1'),('--root-depth','-0.1'),
                           ('--root-feather-px','33'),('--inpaint-radius','0'),
                           ('--dilate-px','4'),('--shadow-strength','nan')):
            with self.subTest(flag=flag),self.assertRaises(ValueError):
                cli.validate_args(cli.parser().parse_args(self.base+[flag,value]))

    def test_no_init_tongue_in_v066(self):
        with self.assertRaisesRegex(ValueError,'reuses'):
            cli.validate_args(cli.parser().parse_args(self.base+['--init-tongue']))

if __name__=='__main__':unittest.main()
