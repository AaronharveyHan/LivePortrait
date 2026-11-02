# coding: utf-8
"""V0.5 unit tests; no ONNX Runtime, PyTorch, CUDA or real weights required."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from scripts.validate_animal_onnx import run_onnx, _inputs_for_model

from src.utils.animal_onnx_contract import get_contract
from src.utils.animal_onnx_validation import (
    CHECKPOINTS, compare_arrays, inspect_model_files, latency_summary,
    manifest_issues, validate_ort_io,
)


class FakeSession:
    def __init__(self, code):
        contract = get_contract(code)
        self.inputs = [SimpleNamespace(name=s.name, shape=list(s.shape), type='tensor(float)')
                       for s in contract.inputs]
        self.outputs = [SimpleNamespace(name=s.name, shape=list(s.shape), type='tensor(float)')
                        for s in contract.outputs]

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs


class AnimalOnnxValidationTests(unittest.TestCase):
    def test_latency_interpolation(self):
        data = latency_summary([30, 10, 20, 40])
        self.assertEqual(data['mean_ms'], 25)
        self.assertEqual(data['p50_ms'], 25)
        self.assertAlmostEqual(data['p95_ms'], 38.5)

    def test_latency_rejects_nonfinite_or_empty(self):
        for values in ([], [1, float('nan')], [-1], [float('inf')]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                latency_summary(values)

    def test_compare_equal(self):
        a = np.array([0.0, 1.0], dtype=np.float32)
        x = compare_arrays(a, a.copy())
        self.assertTrue(x['passed'])
        self.assertEqual(x['mismatches'], 0)

    def test_compare_within_tolerance(self):
        a = np.array([0, 2], dtype=np.float32)
        b = np.array([0.0005, 2.001], dtype=np.float32)
        self.assertTrue(compare_arrays(a, b, atol=1e-3, rtol=1e-3)['passed'])

    def test_compare_mismatch(self):
        a = np.array([0, 2], dtype=np.float32)
        b = np.array([0.1, 2], dtype=np.float32)
        result = compare_arrays(a, b, atol=1e-3, rtol=1e-3)
        self.assertFalse(result['passed'])
        self.assertEqual(result['mismatches'], 1)
        self.assertGreater(result['max_abs_error'], 0.09)

    def test_compare_nan_is_failure(self):
        a = np.array([1, np.nan], dtype=np.float32)
        result = compare_arrays(a, a.copy())
        self.assertFalse(result['passed'])
        self.assertEqual(result['nonfinite_elements'], 1)
        self.assertIsNone(result['max_abs_error'])
        json.dumps(result, allow_nan=False)

    def test_compare_wrong_shape_or_dtype(self):
        a = np.array([1, 2], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'shape'):
            compare_arrays(a, np.zeros((2, 1), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, 'float32'):
            compare_arrays(a, a.astype(np.float64))

    def test_compare_bad_tolerances(self):
        a = np.array([1.0], dtype=np.float32)
        for value in (-1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compare_arrays(a, a, atol=value)

    def test_ort_contract_accepts_all_models(self):
        for code in ('f', 'm', 'w', 'g'):
            with self.subTest(model=code):
                validate_ort_io(FakeSession(code), get_contract(code))

    def test_ort_contract_rejects_wrong_output(self):
        s = FakeSession('m')
        s.outputs[0].name = 'wrong'
        with self.assertRaisesRegex(ValueError, 'name mismatch'):
            validate_ort_io(s, get_contract('m'))

    def test_ort_contract_rejects_wrong_dtype(self):
        s = FakeSession('f')
        s.inputs[0].type = 'tensor(float16)'
        with self.assertRaisesRegex(ValueError, 'float32'):
            validate_ort_io(s, get_contract('f'))

    def test_ort_contract_rejects_dynamic_shape(self):
        s = FakeSession('f')
        s.outputs[0].shape[0] = 'batch'
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            validate_ort_io(s, get_contract('f'))

    def test_manifest_valid(self):
        c = get_contract('f')
        x = c.as_dict()
        x.update(animal_version='v1.1', onnx_file='animal_f_v1.1.onnx', opset_requested=20)
        self.assertEqual(manifest_issues(x, c, 'animal_f_v1.1.onnx'), [])

    def test_manifest_rejects_swapped_names(self):
        c = get_contract('w')
        x = c.as_dict()
        x['inputs'][1]['name'] = 'kp_source'
        x.update(animal_version='v1.1', onnx_file='animal_w_v1.1.onnx', opset_requested=19)
        issues = manifest_issues(x, c, 'animal_w_v1.1.onnx')
        self.assertTrue(any('inputs' in issue for issue in issues))
        self.assertTrue(any('opset' in issue for issue in issues))

    def test_inventory_missing_all(self):
        with tempfile.TemporaryDirectory() as directory:
            item = inspect_model_files('f', directory, directory)
        self.assertFalse(item['ready_compare'])
        self.assertFalse(item['ready_benchmark'])
        self.assertEqual(len(item['issues']), 3)

    def test_inventory_graph_ready_without_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx = root / 'animal_f_v1.1.onnx'
            onnx.write_bytes(b'nonempty placeholder, not validated by preflight')
            manifest = get_contract('f').as_dict()
            manifest.update(animal_version='v1.1', onnx_file=onnx.name, opset_requested=20)
            onnx.with_suffix('.json').write_text(json.dumps(manifest), encoding='utf-8')
            item = inspect_model_files('f', root, root)
        self.assertTrue(item['ready_benchmark'])
        self.assertFalse(item['ready_compare'])
        self.assertEqual(item['issues'], ['missing or empty animal v1.1 checkpoint'])

    def test_inventory_bad_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'animal_f_v1.1.onnx').write_bytes(b'test')
            (root / 'animal_f_v1.1.json').write_text('{oops', encoding='utf-8')
            item = inspect_model_files('f', root, root)
        self.assertFalse(item['ready_benchmark'])
        self.assertIn('invalid JSON', ' '.join(item['issues']))

    def test_mock_ort_session_executes_and_times(self):
        contract = get_contract('f')
        calls = []

        class S(FakeSession):
            def run(self, names, feed):
                calls.append((names, tuple(feed)))
                return [np.zeros(contract.outputs[0].shape, dtype=np.float32)]

        feed = _inputs_for_model('f', contract, np, seed=17)
        sess = S('f')
        outputs, timing = run_onnx(sess, contract, feed, warmup=2, repeat=3, benchmark=True)
        self.assertEqual(len(calls), 5)
        self.assertEqual(timing['count'], 3)
        self.assertEqual(outputs[0].shape, contract.outputs[0].shape)

    def test_mock_ort_session_rejects_wrong_runtime_shape(self):
        contract = get_contract('f')

        class S(FakeSession):
            def run(self, names, feed):
                return [np.zeros((1, 3), dtype=np.float32)]

        feed = _inputs_for_model('f', contract, np, seed=17)
        with self.assertRaisesRegex(ValueError, 'output feature_3d'):
            run_onnx(S('f'), contract, feed, warmup=0, repeat=1, benchmark=False)

    def test_synthetic_inputs_are_deterministic(self):
        for code in ('f', 'm', 'w', 'g'):
            contract = get_contract(code)
            a = _inputs_for_model(code, contract, np, seed=17)
            b = _inputs_for_model(code, contract, np, seed=17)
            self.assertEqual(list(a), [s.name for s in contract.inputs])
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
                self.assertTrue(a[key].flags.c_contiguous)
                self.assertEqual(a[key].dtype, np.float32)

    def test_checkpoint_path_is_version_specific(self):
        for name, path in CHECKPOINTS.items():
            self.assertIn('base_models_v1.1', path)
            self.assertTrue(path.endswith('.pth'))


if __name__ == '__main__':
    unittest.main()
