# coding: utf-8
"""CPU-only, no-torch/no-onnx tests of the animal ONNX I/O contracts."""

import unittest
from types import SimpleNamespace

from src.utils.animal_onnx_contract import (
    MODEL_NAMES, MOTION_OUTPUT_NAMES, get_contract, selected_models,
    validate_model_config, check_shapes, REQUIRED_MODEL_PARAMS,
)


class AnimalOnnxContractTests(unittest.TestCase):
    def test_model_selection(self):
        self.assertEqual(selected_models('ALL'), MODEL_NAMES)
        self.assertEqual(selected_models('W'), ('w',))
        with self.assertRaises(ValueError):
            selected_models('s')

    def test_appearance_contract(self):
        contract = get_contract('F')
        self.assertEqual(contract.inputs[0].shape, (1, 3, 256, 256))
        self.assertEqual(contract.outputs[0].shape, (1, 32, 16, 64, 64))
        self.assertEqual(contract.checkpoint_field, 'checkpoint_F_animal')

    def test_motion_dict_order_and_shapes(self):
        c = get_contract('m')
        self.assertEqual(tuple(item.name for item in c.outputs), MOTION_OUTPUT_NAMES)
        self.assertEqual(c.outputs[4].shape, (1, 63))
        self.assertEqual(c.outputs[5].shape, (1, 1))

    def test_warping_adapter_contract(self):
        c = get_contract('w')
        self.assertTrue(c.experimental)
        self.assertEqual(tuple(i.name for i in c.inputs),
                         ('feature_3d', 'kp_driving', 'kp_source'))
        self.assertEqual(c.outputs[0].shape, (1, 256, 64, 64))

    def test_generator_contract(self):
        c = get_contract('g')
        self.assertFalse(c.experimental)
        self.assertEqual(c.inputs[0].shape, get_contract('w').outputs[0].shape)
        self.assertEqual(c.outputs[0].shape, (1, 3, 512, 512))

    def test_json_manifest_primitive(self):
        import json
        m = get_contract('m').as_dict()
        self.assertEqual(json.loads(json.dumps(m))['outputs'][0]['name'], 'pitch')
        self.assertEqual(m['inputs'][0]['dtype'], 'float32')

    def test_config_matching(self):
        config = {'model_params': {key: dict(value) for key, value
                                   in REQUIRED_MODEL_PARAMS.items()}}
        validate_model_config(config)

    def test_config_mismatch_is_rejected(self):
        config = {'model_params': {key: dict(value) for key, value
                                   in REQUIRED_MODEL_PARAMS.items()}}
        config['model_params']['motion_extractor_params']['num_kp'] = 22
        with self.assertRaisesRegex(ValueError, 'num_kp'):
            validate_model_config(config)

    def test_shape_check(self):
        c = get_contract('f')
        check_shapes((SimpleNamespace(shape=(1, 32, 16, 64, 64)),), c)
        with self.assertRaisesRegex(ValueError, 'feature_3d'):
            check_shapes((SimpleNamespace(shape=(1, 16, 16, 64, 64)),), c)
        with self.assertRaises(ValueError):
            check_shapes((), c)


if __name__ == '__main__':
    unittest.main()
