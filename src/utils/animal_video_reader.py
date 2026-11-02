# coding: utf-8
"""Read an animal driving video one RGB frame at a time.

Using imageio's FFmpeg backend preserves the existing load_video() RGB
convention. No video list or GPU tensor batch is allocated by this module.
"""

import cv2


def iter_video_frames_rgb(path, resize=(256, 256), max_frames=None, reader_factory=None):
    """Yield RGB uint8 frames, releasing the reader even on decoding errors.

    Args:
        path: Driving video filename.
        resize: ``(width, height)`` or None to preserve native resolution.
        max_frames: Optional frame limit; zero yields no frames.
        reader_factory: Test injection point; must return an iterable with close().

    The generator holds a decoder resource while it is being iterated. If a
    caller stops before exhaustion, call ``generator.close()`` in a finally.
    """
    if max_frames is not None and max_frames < 0:
        raise ValueError('max_frames must be non-negative or None')
    if reader_factory is None:
        import imageio
        reader_factory = lambda filename: imageio.get_reader(filename, 'ffmpeg')

    reader = reader_factory(path)
    try:
        frames = iter(reader)
        count = 0
        while max_frames is None or count < max_frames:
            try:
                frame_rgb = next(frames)
            except StopIteration:
                break
            # imageio's FFmpeg reader already yields RGB (not OpenCV BGR).
            if resize is not None:
                frame_rgb = cv2.resize(frame_rgb, resize)
            yield frame_rgb
            count += 1
    finally:
        reader.close()
