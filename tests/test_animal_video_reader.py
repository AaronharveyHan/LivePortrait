# coding: utf-8
"""CPU-only tests; do not require torch, CUDA, weights or imageio-ffmpeg."""

import ast
from pathlib import Path
import unittest

import numpy as np

from src.utils.animal_video_reader import iter_video_frames_rgb


class FakeReader:
    def __init__(self, frames, error_at=None):
        self.frames = frames
        self.error_at = error_at
        self.index = 0
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.error_at is not None and self.index == self.error_at:
            raise ValueError('simulated decoding error')
        if self.index >= len(self.frames):
            raise StopIteration
        item = self.frames[self.index]
        self.index += 1
        return item

    def close(self):
        self.closed = True


class AnimalVideoReaderTests(unittest.TestCase):
    def setUp(self):
        # Different channels make RGB/BGR errors immediately detectable.
        self.frames = [np.full((12, 18, 3), (10+i, 20+i, 30+i),
                               dtype=np.uint8) for i in range(5)]
        self.reader = FakeReader(self.frames)
        self.factory = lambda _path: self.reader

    def test_rgb_order_and_resize(self):
        result = list(iter_video_frames_rgb('fake.mp4', (8, 6),
                                            reader_factory=self.factory))
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0].shape, (6, 8, 3))
        self.assertEqual(result[0][0, 0].tolist(), [10, 20, 30])
        self.assertTrue(self.reader.closed)

    def test_no_resize_preserves_values(self):
        result = list(iter_video_frames_rgb('fake.mp4', resize=None,
                                            reader_factory=self.factory))
        self.assertTrue(np.array_equal(result[2], self.frames[2]))
        self.assertTrue(self.reader.closed)

    def test_frame_limit_does_not_decode_extra_frame(self):
        result = list(iter_video_frames_rgb('fake.mp4', max_frames=2,
                                            reader_factory=self.factory))
        self.assertEqual(len(result), 2)
        self.assertEqual(self.reader.index, 2)
        self.assertTrue(self.reader.closed)

    def test_zero_limit_reads_nothing(self):
        result = list(iter_video_frames_rgb('fake.mp4', max_frames=0,
                                            reader_factory=self.factory))
        self.assertEqual(result, [])
        self.assertEqual(self.reader.index, 0)
        self.assertTrue(self.reader.closed)

    def test_explicit_close_after_early_exit(self):
        gen = iter_video_frames_rgb('fake.mp4', reader_factory=self.factory)
        next(gen)
        gen.close()
        self.assertTrue(self.reader.closed)
        self.assertEqual(self.reader.index, 1)

    def test_decoder_exception_still_closes(self):
        self.reader.error_at = 2
        with self.assertRaisesRegex(ValueError, 'simulated decoding error'):
            list(iter_video_frames_rgb('fake.mp4', reader_factory=self.factory))
        self.assertTrue(self.reader.closed)

    def test_invalid_limit(self):
        with self.assertRaises(ValueError):
            list(iter_video_frames_rgb('fake.mp4', max_frames=-1,
                                       reader_factory=self.factory))
        self.assertEqual(self.reader.index, 0)
        self.assertFalse(self.reader.closed)

    def test_motion_template_accepts_frame_generator(self):
        # Parse just this function from the production pipeline to avoid
        # importing torch or loading any model weights into the test process.
        pipeline = (Path(__file__).resolve().parents[1] /
                    'src/live_portrait_pipeline_animal.py')
        tree = ast.parse(pipeline.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body
                   if isinstance(n, ast.ClassDef) and n.name == 'LivePortraitPipelineAnimal')
        func = next(n for n in cls.body
                    if isinstance(n, ast.FunctionDef) and n.name == 'make_motion_template')

        class FakeTensor:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=np.float32)

            def cpu(self):
                return self

            def numpy(self):
                return self.values

        class FakeWrapper:
            def __init__(self):
                self.prepared = []

            def prepare_source(self, frame):
                self.prepared.append(frame[0, 0].tolist())
                return frame

            def get_kp_info(self, frame):
                return {key: FakeTensor([1.0])
                        for key in ('pitch', 'yaw', 'roll', 'scale', 'exp', 't')}

        namespace = {
            'np': np,
            'track': lambda iterable, **_kwargs: iterable,
            'get_rotation_matrix': lambda *args: FakeTensor(np.eye(3)),
        }
        exec(compile(ast.Module(body=[func], type_ignores=[]), str(pipeline), 'exec'),
             namespace)
        wrapper = FakeWrapper()
        owner = type('FakePipeline', (), {'live_portrait_wrapper_animal': wrapper})()
        previews = []
        # A generator has no len() and can be consumed only once.
        frames = (np.full((4, 4, 3), i, np.uint8) for i in range(3))
        template = namespace['make_motion_template'](
            owner, frames, output_fps=20, preview_frames=previews)
        self.assertEqual(template['n_frames'], 3)
        self.assertEqual(template['output_fps'], 20)
        self.assertEqual(len(template['motion']), 3)
        self.assertEqual(len(wrapper.prepared), 3)
        self.assertEqual([p[0, 0, 0] for p in previews], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
