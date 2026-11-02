# coding: utf-8
"""Verify ONNX export planning/error gates without third-party packages."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / 'scripts/export_animal_onnx.py'


class AnimalOnnxExporterCliTests(unittest.TestCase):
    def run_cli(self, *args):
        # -S removes site-packages, proving plan/error checks need no torch.
        return subprocess.run([sys.executable, '-S', str(EXPORTER), *args],
                              capture_output=True, text=True, cwd=str(ROOT))

    def test_plan_all_without_third_party_packages(self):
        p = self.run_cli('--plan', '--model', 'all')
        self.assertEqual(p.returncode, 0, p.stderr)
        obj = json.loads(p.stdout)
        self.assertEqual(len(obj['models']), 4)
        self.assertEqual([m['model'] for m in obj['models']], ['F', 'M', 'W', 'G'])
        self.assertTrue(obj['plan_only'])

    def test_plan_custom_weight_root(self):
        with tempfile.TemporaryDirectory() as directory:
            p = self.run_cli('--plan', '--weights-root', directory)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertFalse(json.loads(p.stdout)['models'][0]['checkpoint_exists'])

    def test_w_export_requires_explicit_opt_in(self):
        p = self.run_cli('--model', 'w')
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('--experimental-w', p.stderr)

    def test_w_requires_opset20(self):
        p = self.run_cli('--model', 'w', '--experimental-w', '--opset', '17')
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('20 or later', p.stderr)

    def test_w_rejects_legacy_exporter(self):
        p = self.run_cli('--model', 'w', '--experimental-w', '--exporter', 'legacy')
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('dynamo', p.stderr)

    def test_missing_weights_before_loading_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            p = self.run_cli('--model', 'f', '--weights-root', directory)
        self.assertEqual(p.returncode, 2)
        self.assertIn('Missing weights', p.stderr)


if __name__ == '__main__':
    unittest.main()
