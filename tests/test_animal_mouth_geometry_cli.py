# coding: utf-8
"""V0.6.3 CLI integration tests, no PyTorch or actual ONNX models."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts import run_animal_mouth_geometry as cli
from src.utils import animal_mouth_geometry as g


class GeometryCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=cli.ROOT / 'build' if (cli.ROOT/'build').is_dir() else cli.ROOT)
        self.root=Path(self.tmp.name)
        self.refs=self.root/'references';self.refs.mkdir()
        c=np.full((512,512,3),80,np.uint8)
        o=c.copy();o[265:380,175:380]=[110,160,185]
        Image.fromarray(c).save(self.refs/'closed_reference.png')
        Image.fromarray(o).save(self.refs/'open_reference_aligned.png')
        self.out=self.root/'output'
        self.init_args=['--reference-dir',str(self.refs),'--output-dir',str(self.out),
                        '--same-dog','--init-landmarks']

    def tearDown(self):
        self.tmp.cleanup()

    def test_plan_no_images(self):
        args=cli.parser().parse_args(['--plan'])
        self.assertEqual(len(cli.validate_args(args)),5)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['--plan']),0)

    def test_parse_alphas_invalid(self):
        for raw in ('0,.5,.5,1','0,1,.5','nan,1','0,.5','0,1,1','0,1.2','0,0.001,0.0011,1'):
            with self.subTest(raw=raw),self.assertRaises(ValueError):cli.parse_alphas(raw)

    def test_same_dog_required(self):
        args=cli.parser().parse_args(['--reference-dir',str(self.refs),'--init-landmarks'])
        with self.assertRaisesRegex(ValueError,'same-dog'):
            cli.validate_args(args)

    def test_missing_landmarks_mode(self):
        args=cli.parser().parse_args(['--same-dog'])
        with self.assertRaisesRegex(ValueError,'landmarks'):
            cli.validate_args(args)

    def test_init_creates_files_and_no_overwrite_template(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(self.init_args),0)
        paths=['mouth_landmarks_template.json','closed_guide.png','open_guide.png','annotate_mouth.html']
        self.assertTrue(all((self.out/x).is_file() for x in paths))
        self.assertIn('reference_sha256',json.loads((self.out/paths[0]).read_text()))
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(FileExistsError):
            cli.main(self.init_args)
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(FileExistsError):
            cli.main(self.init_args+['--overwrite'])

    def _verified_json(self):
        spec=g.create_template((.29,.49,.74,.83))
        spec['verified']=True
        spec['open']['lower_lip'][1]+=.02
        spec['open']['chin'][1]+=.01
        spec['reference_sha256']={'closed':cli._hash(self.refs/'closed_reference.png'),
                                  'open':cli._hash(self.refs/'open_reference_aligned.png')}
        p=self.root/'landmarks.json';p.write_text(json.dumps(spec))
        return p

    def test_render_both_policies(self):
        p=self._verified_json()
        args=['--reference-dir',str(self.refs),'--output-dir',str(self.out),
              '--same-dog','--landmarks',str(p)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(args),0)
        report=json.loads((self.out/'report.json').read_text())
        self.assertEqual(len(report['frames']['soft']),5)
        self.assertEqual(len(report['frames']['dominant']),5)
        self.assertTrue((self.out/'comparison_sheet.png').exists())
        self.assertTrue(all(x['metrics']['outside_region_changed_pixels']==0
                            for group in report['frames'].values() for x in group))
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(FileExistsError):
            cli.main(args)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(args+['--overwrite']),0)

    def test_unconfirmed_rejected(self):
        p=self._verified_json();spec=json.loads(p.read_text());spec['verified']=False
        p.write_text(json.dumps(spec))
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'confirmed'):
            cli.main(['--reference-dir',str(self.refs),'--output-dir',str(self.out),
                      '--same-dog','--landmarks',str(p)])

    def test_wrong_reference_hash_rejected(self):
        p=self._verified_json();spec=json.loads(p.read_text())
        spec['reference_sha256']['open']='0'*64;p.write_text(json.dumps(spec))
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'hashes mismatch'):
            cli.main(['--reference-dir',str(self.refs),'--output-dir',str(self.out),
                      '--same-dog','--landmarks',str(p)])

    def test_output_dir_cannot_be_input_dir(self):
        args=cli.parser().parse_args(self.init_args[:])
        args.output_dir=self.refs
        with self.assertRaisesRegex(ValueError,'cannot be reference'):
            cli.validate_args(args)


if __name__=='__main__':unittest.main()
