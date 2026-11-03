# coding: utf-8
"""V0.7.0 CLI fail-fast and fake-IO integration, avoiding model weights."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts import run_animal_onnx_video as app
from tests.test_animal_onnx_video import make_sessions
from src.utils import animal_onnx_chain as ch

ROOT = Path(__file__).resolve().parents[1]


class FakeReader:
    def __init__(self, count=3, fail=False):
        self.frames = [np.full((80,120,3),10+i*20,np.uint8) for i in range(count)]
        self.closed=False
        self.fail=fail

    def get_meta_data(self):
        return {'fps':15.0}

    def __iter__(self):
        for i, frame in enumerate(self.frames):
            if self.fail and i==1:
                raise RuntimeError('video read failure')
            yield frame

    def close(self):
        self.closed=True


class FakeWriter:
    instances=[]
    def __init__(self, *, wfp, fps, ffmpeg_params=None):
        self.path=Path(wfp)
        self.fps=fps
        self.frames=0
        self.closed=False
        self.path.write_bytes(b'FAKE_MP4')
        type(self).instances.append(self)

    def write(self, rgb):
        self.frames+=1
        assert rgb.dtype==np.uint8

    def close(self):
        self.closed=True


class V070CliTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)
        self.source=self.base/'dog.png'
        Image.fromarray(np.zeros((256,256,3),np.uint8),'RGB').save(self.source)
        self.driver=self.base/'drive.mp4'
        self.driver.write_bytes(b'placeholder')
        self.out=self.base/'out.mp4'

    def args(self,*extra):
        return app.make_parser().parse_args(['--source',str(self.source),'--driving',str(self.driver),
                                             '--output',str(self.out),*extra])

    def run_cli(self,*args):
        return subprocess.run([sys.executable,'-S',str(ROOT/'scripts/run_animal_onnx_video.py'),*args],
                              cwd=str(ROOT),capture_output=True,text=True)

    def test_plan_without_numpy(self):
        r=self.run_cli('--plan')
        self.assertEqual(r.returncode,0,r.stderr)
        j=json.loads(r.stdout)
        self.assertEqual(j['stage'],'v0.7.0_animal_expression_video')
        self.assertEqual(len(j['models']),4)

    def test_missing_input_is_error(self):
        r=self.run_cli('--source','no.jpg','--driving','not.mp4')
        self.assertEqual(r.returncode,2)
        self.assertIn('Missing',r.stderr)

    def test_bad_params(self):
        for flags in (('--threads','0'),('--max-frames','-1'),('--fps','nan'),
                      ('--driving-multiplier','-1'),('--motion-mode','none')):
            with self.subTest(flags=flags), self.assertRaises((ValueError,SystemExit)):
                args=self.args(*flags)
                app.validate_args(args)

    def test_no_output_overwrite_without_flag(self):
        self.out.write_bytes(b'original')
        with self.assertRaisesRegex(FileExistsError, 'overwrite'):
            app.validate_args(self.args())
        self.assertEqual(self.out.read_bytes(),b'original')

    def test_source_cannot_be_output(self):
        args=self.args('--comparison',str(self.driver))
        with self.assertRaisesRegex(ValueError,'overwrite source'):
            app.validate_args(args)

    def test_no_decodable_frames(self):
        args=self.args()
        app.validate_args(args)
        reader=FakeReader(count=0)
        with self.assertRaisesRegex(ValueError,'no decodable'):
            app.run(args,reader_factory=lambda _:reader,writer_factory=FakeWriter,session_factory=lambda _:make_sessions())
        self.assertTrue(reader.closed)
        self.assertFalse(self.out.exists())

    def test_stream_pipeline_3_frames_no_history(self):
        args=self.args('--comparison',str(self.base/'compare.mp4'),'--report',str(self.base/'result.json'),
                       '--max-frames','3','--motion-mode','expression-only')
        app.validate_args(args)
        reader=FakeReader()
        FakeWriter.instances.clear()
        ss=make_sessions()
        with patch.object(ch,'motion_multiplier',return_value=1.0):
            report=app.run(args,reader_factory=lambda _:reader,writer_factory=FakeWriter,
                           session_factory=lambda _:ss)
        self.assertTrue(reader.closed)
        self.assertEqual(report['frame_count'],3)
        self.assertEqual(report['fps'],15)
        self.assertEqual(ss['f'].calls,1)
        self.assertEqual(ss['m'].calls,4)
        self.assertEqual(ss['w'].calls,3)
        self.assertEqual(ss['g'].calls,3)
        self.assertEqual([w.frames for w in FakeWriter.instances],[3,3])
        self.assertTrue(all(w.closed for w in FakeWriter.instances))
        self.assertEqual(self.out.read_bytes(),b'FAKE_MP4')
        self.assertEqual((self.base/'result.json').exists(),True)

    def test_decoder_failure_cleans_partials(self):
        args=self.args()
        app.validate_args(args)
        reader=FakeReader(fail=True)
        ss=make_sessions()
        FakeWriter.instances.clear()
        with patch.object(ch,'motion_multiplier',return_value=1):
            with self.assertRaisesRegex(RuntimeError,'video read failure'):
                app.run(args,reader_factory=lambda _:reader,writer_factory=FakeWriter,
                        session_factory=lambda _:ss)
        self.assertTrue(reader.closed)
        self.assertTrue(all(w.closed for w in FakeWriter.instances))
        self.assertFalse(self.out.exists())
        self.assertFalse(list(self.base.glob('.v070_*')))

    def test_fps_fallback(self):
        self.assertEqual(app.checked_fps({},None),(25.0,'fallback_25fps_invalid_video_metadata'))
        self.assertEqual(app.checked_fps({'fps':24},None),(24,'video_metadata'))
        self.assertEqual(app.checked_fps({'fps':24},30),(30,'user_override'))


if __name__=='__main__': unittest.main()
