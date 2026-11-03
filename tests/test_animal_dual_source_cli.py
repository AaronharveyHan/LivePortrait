# coding: utf-8
"""V0.6.2 CLI validation & fast original-image smoke tests (no models)."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from scripts import run_animal_dual_source as cli


class DualSourceCliTests(unittest.TestCase):
    def test_plan_no_image_no_runtime(self):
        args=cli.parse_args(['--plan'])
        alphas, roi=cli.validate_args(args)
        meta=cli.plan(args,alphas,roi)
        self.assertEqual(meta['stage'],'v0.6.2_dual_source')
        self.assertEqual(len(meta['onnx_files']),4)

    def test_two_photos_required(self):
        args=cli.parse_args(['--same-dog','--source-closed','a.png'])
        with self.assertRaises((ValueError,FileNotFoundError)):
            cli.validate_args(args)

    def test_same_dog_explicit_required(self):
        args=cli.parse_args(['--source-closed','a.png','--source-open','b.png'])
        with self.assertRaisesRegex(ValueError,'--same-dog'):
            cli.validate_args(args)

    def test_bad_alphas(self):
        for value in ('0,.5,.5,1','0,0.5','0,2,1','0,nan,1','0,1,0.5','a,1','0,1,1','0,0.25001,0.25002,1'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                cli.parse_alphas(value)

    def test_valid_alphas(self):
        self.assertEqual(cli.parse_alphas('0,0.25,0.5,1'),(0,.25,.5,1))

    def test_bad_roi(self):
        for val in ('0,0,1','0,0,2,2','0,nan,1,1'):
            with self.subTest(val=val),self.assertRaises(ValueError):
                cli.parse_roi(val)

    def test_mutually_exclusive_identity_flags(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.parse_args(['--same-dog','--different-dog-demo'])

    def test_outputs_distinct(self):
        paths=cli.planned_files(Path('build/test_v062'),(0,.25,.5,.75,1))
        self.assertEqual(len(paths),10)
        self.assertEqual(len(set(paths)),len(paths))

    def test_original_mode_end_to_end(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=ROOT / 'build' if (ROOT/'build').is_dir() else ROOT) as temp:
            tmp=Path(temp)
            a=np.full((256,256,3),100,np.uint8)
            b=np.full((256,256,3),100,np.uint8)
            b[140:205,95:185,:]=(30,50,190)
            closed=tmp/'closed.png';opened=tmp/'open.png'
            Image.fromarray(a).save(closed);Image.fromarray(b).save(opened)
            out=tmp/'v062'
            argv=['--source-closed',str(closed),'--source-open',str(opened),
                  '--same-dog','--alignment','none','--input-mode','original',
                  '--output-dir',str(out)]
            args=cli.parse_args(argv)
            alphas,roi=cli.validate_args(args)
            with contextlib.redirect_stdout(io.StringIO()):
                report=cli.run(args,alphas,roi)
            self.assertTrue((out/'contact_sheet.png').exists())
            self.assertTrue((out/'report.json').exists())
            self.assertEqual(report['frames'][0]['metrics']['outside_region_changed_pixels'],0)
            self.assertEqual(report['frames'][-1]['metrics']['outside_region_changed_pixels'],0)
            from PIL import Image
            zero=np.array(Image.open(out/'mouth_000.png'))
            ref=np.array(Image.open(out/'closed_reference.png'))
            np.testing.assert_array_equal(zero,ref)
            with self.assertRaises(FileExistsError):
                cli.run(args,alphas,roi)

if __name__=='__main__':unittest.main()
