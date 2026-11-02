# coding: utf-8
"""V0.6 CLI planning and fail-fast tests use no third-party packages."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'run_animal_onnx_frame.py'


class AnimalOnnxChainCliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, '-S', str(SCRIPT), *args],
                              capture_output=True, text=True, cwd=str(ROOT))

    def test_plan_without_third_party(self):
        r = self.run_cli('--plan')
        self.assertEqual(r.returncode, 0, r.stderr)
        report = json.loads(r.stdout)
        self.assertEqual([v['model'] for v in report['model_files']], ['F','M','W','G'])
        self.assertFalse(report['supported']['stitching'])

    def test_requires_all_three_photos(self):
        r = self.run_cli('--source','any.png')
        self.assertEqual(r.returncode, 2)
        self.assertIn('--driving-first', r.stderr)

    def test_invalid_threads(self):
        r = self.run_cli('--synthetic','--threads','0')
        self.assertEqual(r.returncode, 2)
        self.assertIn('--threads', r.stderr)

    def test_missing_graph_before_numpy_import(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.run_cli('--synthetic','--onnx-dir',d)
        self.assertEqual(r.returncode, 2)
        self.assertIn('missing ONNX prerequisite', r.stderr)

    def test_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'frame.png'
            p.write_bytes(b'test')
            r = self.run_cli('--synthetic', '--output',str(p))
            self.assertEqual(r.returncode, 2)
            self.assertIn('--overwrite',r.stderr)

    def test_reject_nan_multiplier(self):
        r = self.run_cli('--synthetic','--driving-multiplier','nan')
        self.assertEqual(r.returncode, 2)
        self.assertIn('--driving-multiplier',r.stderr)


if __name__ == '__main__':
    unittest.main()
