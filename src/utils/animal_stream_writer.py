# coding: utf-8
"""Incremental video sinks for animal animation; model/CUDA independent.

Frames written here are uint8 RGB, matching the existing images2video path.
"""

import cv2
import numpy as np


class AnimalStreamWriter:
    """Write each animation frame immediately, without retaining a frame list.

    `writer_factory` permits CPU-only tests without initializing imageio/FFmpeg.
    If a second writer fails to open, the first writer is closed before raising.
    """

    def __init__(self, output_path, concat_path, fps, source_image, writer_factory=None):
        if writer_factory is None:
            from .video import VideoWriter
            writer_factory = VideoWriter
        self.source_image = source_image
        self.output_writer = None
        self.concat_writer = None
        self.source_resized = None
        self.frame_shape = None
        kwargs = {'fps': fps, 'ffmpeg_params': ['-crf', '18']}
        try:
            self.output_writer = writer_factory(wfp=output_path, **kwargs)
            if concat_path is not None:
                self.concat_writer = writer_factory(wfp=concat_path, **kwargs)
        except BaseException:
            self.close()
            raise

    def write(self, generated_frame, output_frame=None, driving_frame=None):
        if self.output_writer is None:
            raise RuntimeError('AnimalStreamWriter is already closed')
        self.output_writer.write(
            generated_frame if output_frame is None else output_frame
        )
        if self.concat_writer is None:
            return

        h, w = generated_frame.shape[:2]
        if self.frame_shape != (h, w):
            self.source_resized = cv2.resize(self.source_image, (w, h))
            self.frame_shape = (h, w)
        panels = [self.source_resized, generated_frame]
        if driving_frame is not None:
            panels.insert(0, cv2.resize(driving_frame, (w, h)))
        self.concat_writer.write(np.hstack(panels))

    def close(self):
        # Always try to release both FFmpeg processes, even if one close fails.
        errors = []
        for name in ('output_writer', 'concat_writer'):
            writer = getattr(self, name, None)
            setattr(self, name, None)
            if writer is not None:
                try:
                    writer.close()
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise errors[0]
