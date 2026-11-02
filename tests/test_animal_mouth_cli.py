# coding: utf-8
"""V0.6.1 CLI parse/validation cases without NumPy, Torch or ORT."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts'/'run_animal_onnx_frame.py'


class MouthCliTests(unittest.TestCase):
    def run_cli(self,*args):
        return subprocess.run([sys.executable,'-S',str(SCRIPT),*args],capture_output=True,text=True,cwd=str(ROOT))

    def test_default_plan_without_site_packages(self):
        p=self.run_cli('--plan')
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(json.loads(p.stdout)['stage'],'v0.6_single_frame')

    def test_mouth_csv_error(self):
        p=self.run_cli('--synthetic','--mouth-kp-indices','1,,2')
        self.assertEqual(p.returncode,2)
        self.assertIn('comma-separated',p.stderr)

    def test_nondefault_scale_requires_indices(self):
        p=self.run_cli('--synthetic','--mouth-y-scale','1.2')
        self.assertEqual(p.returncode,2)
        self.assertIn('mouth-kp-indices',p.stderr)

    def test_invalid_index(self):
        p=self.run_cli('--synthetic','--mouth-kp-indices','21','--mouth-y-scale','1.2')
        self.assertEqual(p.returncode,2)
        self.assertIn('0..20',p.stderr)

    def test_lower_not_subset(self):
        p=self.run_cli('--synthetic','--mouth-kp-indices','1','--mouth-lower-kp-indices','2')
        self.assertEqual(p.returncode,2)
        self.assertIn('subset',p.stderr)

    def test_nonfinite_rejected(self):
        p=self.run_cli('--synthetic','--mouth-kp-indices','1','--mouth-y-scale','nan')
        self.assertEqual(p.returncode,2)
        self.assertIn('finite',p.stderr)

    def test_positive_and_negative_bias_accepted_before_onnx_preflight(self):
        with tempfile.TemporaryDirectory() as d:
            for bias in ('-0.005','0.005'):
                r=self.run_cli('--synthetic','--onnx-dir',d, '--mouth-kp-indices','0,1',
                               '--mouth-lower-kp-indices','1','--mouth-lower-y-bias',bias)
                self.assertEqual(r.returncode,2)
                self.assertIn('missing ONNX prerequisite',r.stderr)
                self.assertNotIn('lower_y_bias',r.stderr)

    def test_duplicate_output_paths_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'report.json'
            r=self.run_cli('--synthetic','--report-json',str(p),'--dump-kp-json',str(p))
            self.assertEqual(r.returncode,2)
            self.assertIn('different',r.stderr)

    def test_dump_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'kp.json'
            p.write_text('do not overwrite')
            r=self.run_cli('--synthetic','--dump-kp-json',str(p))
            self.assertEqual(r.returncode,2)
            self.assertIn('--overwrite',r.stderr)

    def test_dump_requires_json(self):
        p=self.run_cli('--synthetic','--dump-kp-json','foo.txt')
        self.assertEqual(p.returncode,2)
        self.assertIn('.json',p.stderr)


if __name__ == '__main__':
    unittest.main()
