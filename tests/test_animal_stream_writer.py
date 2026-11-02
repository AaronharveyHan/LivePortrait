# coding: utf-8
"""CPU-only, no-CUDA tests for the incremental animal-video writer."""

import unittest
import numpy as np

from src.utils.animal_stream_writer import AnimalStreamWriter


class FakeWriter:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.frames = []
        self.closed = False
        self.instances.append(self)

    def write(self, image):
        self.frames.append(image.copy())

    def close(self):
        self.closed = True


class StreamWriterTests(unittest.TestCase):
    def setUp(self):
        FakeWriter.instances.clear()
        self.source = np.full((16, 16, 3), 10, dtype=np.uint8)
        self.generated = np.full((16, 16, 3), 20, dtype=np.uint8)
        self.driving = np.full((16, 16, 3), 30, dtype=np.uint8)

    def test_no_concat_writes_one_frame_at_a_time(self):
        writer = AnimalStreamWriter('out.mp4', None, 25, self.source,
                                    writer_factory=FakeWriter)
        for _ in range(3):
            writer.write(self.generated)
        writer.close()
        self.assertEqual(len(FakeWriter.instances), 1)
        self.assertEqual(len(FakeWriter.instances[0].frames), 3)
        self.assertTrue(FakeWriter.instances[0].closed)
        self.assertEqual(FakeWriter.instances[0].kwargs['fps'], 25)

    def test_concat_preserves_panel_order_and_pasteback(self):
        pasteback = np.full((32, 32, 3), 40, dtype=np.uint8)
        writer = AnimalStreamWriter('out.mp4', 'concat.mp4', 25, self.source,
                                    writer_factory=FakeWriter)
        writer.write(self.generated, pasteback, self.driving)
        writer.close()
        final = FakeWriter.instances[0].frames[0]
        collage = FakeWriter.instances[1].frames[0]
        self.assertEqual(final.shape, (32, 32, 3))
        self.assertEqual(collage.shape, (16, 48, 3))
        self.assertEqual(tuple(collage[0, 0]), (30, 30, 30))
        self.assertEqual(tuple(collage[0, 16]), (10, 10, 10))
        self.assertEqual(tuple(collage[0, 32]), (20, 20, 20))
        self.assertTrue(all(w.closed for w in FakeWriter.instances))

    def test_template_concat_has_two_panels(self):
        writer = AnimalStreamWriter('out.mp4', 'concat.mp4', 25, self.source,
                                    writer_factory=FakeWriter)
        writer.write(self.generated)
        writer.close()
        self.assertEqual(FakeWriter.instances[1].frames[0].shape, (16, 32, 3))

    def test_writer_closed_cannot_append(self):
        writer = AnimalStreamWriter('out.mp4', None, 25, self.source,
                                    writer_factory=FakeWriter)
        writer.close()
        with self.assertRaises(RuntimeError):
            writer.write(self.generated)


if __name__ == '__main__':
    unittest.main()
