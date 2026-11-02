# coding: utf-8
"""V0.5 CLI preflight and input gates without PyTorch / ONNX packages."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / 'scripts/validate_animal_onnx.py'


class AnimalOnnxValidateCliTests(unittest.TestCase):
    def run_cli(self, *args):
        # -S deliberately disables site-packages; preflight stays pure stdlib.
        return subprocess.run([sys.executable, '-S', str(CLI), *args],
                              cwd=str(ROOT), capture_output=True, text=True)

    def test_preflight_all_without_site_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            x = self.run_cli('--mode', 'preflight', '--model', 'all',
                             '--weights-root', tmp, '--onnx-dir', tmp)
        self.assertEqual(x.returncode, 0, x.stderr)
        obj = json.loads(x.stdout)
        self.assertEqual([r['model'] for r in obj['models']], ['F', 'M', 'W', 'G'])
        self.assertTrue(all(not m['ready_compare'] for m in obj['models']))

    def test_preflight_strict_reports_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            x = self.run_cli('--mode', 'preflight', '--model', 'f',
                             '--weights-root', tmp, '--onnx-dir', tmp, '--strict')
        self.assertEqual(x.returncode, 2)
        self.assertTrue(json.loads(x.stdout)['models'])

    def test_preflight_w_does_not_require_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            x = self.run_cli('--mode', 'preflight', '--model', 'w',
                             '--weights-root', tmp, '--onnx-dir', tmp)
        self.assertEqual(x.returncode, 0, x.stderr)

    def test_compare_w_requires_explicit_opt_in(self):
        x = self.run_cli('--mode', 'compare', '--model', 'w')
        self.assertEqual(x.returncode, 2)
        self.assertIn('--experimental-w', x.stderr)

    def test_benchmark_w_requires_explicit_opt_in(self):
        x = self.run_cli('--mode', 'benchmark', '--model', 'all')
        self.assertEqual(x.returncode, 2)
        self.assertIn('--experimental-w', x.stderr)

    def test_benchmark_requires_graph_before_onnxruntime_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            x = self.run_cli('--mode', 'benchmark', '--onnx-dir', tmp)
        self.assertEqual(x.returncode, 2)
        self.assertIn('missing prerequisite', x.stderr)
        self.assertNotIn('ModuleNotFoundError', x.stderr)

    def test_compare_requires_checkpoint_before_torch_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            x = self.run_cli('--mode', 'compare', '--weights-root', tmp,
                             '--onnx-dir', tmp)
        self.assertEqual(x.returncode, 2)
        self.assertIn('missing prerequisite', x.stderr)

    def test_reject_bad_timings(self):
        for params in (('--repeat', '0'), ('--threads', '0'), ('--warmup', '-1')):
            with self.subTest(params=params):
                x = self.run_cli('--mode', 'preflight', *params)
                self.assertEqual(x.returncode, 2)

    def test_reject_nan_tolerance(self):
        x = self.run_cli('--atol', 'nan')
        self.assertEqual(x.returncode, 2)
        self.assertIn('finite', x.stderr)

    def test_json_report_saved_only_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'reports' / 'inventory.json'
            x = self.run_cli('--onnx-dir', tmp, '--weights-root', tmp,
                             '--output-json', str(target))
            self.assertEqual(x.returncode, 0, x.stderr)
            self.assertEqual(json.loads(target.read_text(encoding='utf-8')),
                             json.loads(x.stdout))


if __name__ == '__main__':
    unittest.main()
