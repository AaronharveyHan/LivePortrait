# coding: utf-8

"""
Pipeline of LivePortrait (Animal)
"""

import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.")
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None")

import torch
torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import numpy as np
import os
import time
import os.path as osp
from rich.progress import track

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.video import images2video, concat_frames, get_fps, add_audio_to_video, has_audio_stream, video2gif
from .utils.animal_stream_writer import AnimalStreamWriter
from .utils.animal_video_reader import iter_video_frames_rgb
from .utils.crop import _transform_img, prepare_paste_back, paste_back
from .utils.io import load_image_rgb, load_video, resize_to_limit, dump, load
from .utils.helper import mkdir, basename, dct2device, is_video, is_template, remove_suffix, is_image, calc_motion_multiplier
from .utils.rprint import rlog as log
# from .utils.viz import viz_lmk
from .live_portrait_wrapper import LivePortraitWrapperAnimal


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)

class LivePortraitPipelineAnimal(object):

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper_animal: LivePortraitWrapperAnimal = LivePortraitWrapperAnimal(inference_cfg=inference_cfg)
        self.cropper: Cropper = Cropper(
            crop_cfg=crop_cfg,
            image_type='animal_face',
            flag_use_half_precision=inference_cfg.flag_use_half_precision,
            flag_force_cpu=inference_cfg.flag_force_cpu,
            flag_detectors_cpu=inference_cfg.flag_detectors_cpu,
            device_id=inference_cfg.device_id,
        )

    def make_motion_template(self, I_lst, **kwargs):
        # Accept both preloaded image lists (legacy) and lazy video-frame
        # generators. Keep only the small pose/landmark arrays in the template.
        preview_frames = kwargs.get('preview_frames')
        template_dct = {
            'n_frames': 0,
            'output_fps': kwargs.get('output_fps', 25),
            'motion': [],
        }

        for frame in track(I_lst, description='Making driving motion templates...'):
            # The generator gives us RGB uint8; upload only this frame.
            if preview_frames is not None:
                preview_frames.append(frame)
            I_i = (self.live_portrait_wrapper_animal.prepare_source(frame)
                   if isinstance(frame, np.ndarray) else frame)
            x_i_info = self.live_portrait_wrapper_animal.get_kp_info(I_i)
            R_i = get_rotation_matrix(x_i_info['pitch'], x_i_info['yaw'], x_i_info['roll'])

            item_dct = {
                'scale': x_i_info['scale'].cpu().numpy().astype(np.float32),
                'R': R_i.cpu().numpy().astype(np.float32),
                'exp': x_i_info['exp'].cpu().numpy().astype(np.float32),
                't': x_i_info['t'].cpu().numpy().astype(np.float32),
            }

            template_dct['motion'].append(item_dct)

        template_dct['n_frames'] = len(template_dct['motion'])
        return template_dct

    def execute(self, args: ArgumentConfig):
        # for convenience
        inf_cfg = self.live_portrait_wrapper_animal.inference_cfg
        device = self.live_portrait_wrapper_animal.device
        crop_cfg = self.cropper.crop_cfg
        stream_video = getattr(args, 'flag_stream_video', False)
        stream_driving_video = getattr(args, 'flag_stream_driving_video', False)
        save_concat_video = getattr(args, 'flag_save_concat_video', True)
        save_gif = getattr(args, 'flag_save_gif', True)
        profile_enabled = os.getenv("LIVEPORTRAIT_PROFILE", "0").lower() in ("1", "true", "yes")
        profile_decode_ms = []
        rss_process = None
        if profile_enabled:
            try:
                import psutil  # optional; not needed for CPU unit tests
                rss_process = psutil.Process()
            except ImportError:
                log('[PROFILE] Install psutil to sample process RSS (optional).')

        def sample_rss(stage):
            if rss_process is not None:
                rss_mib = rss_process.memory_info().rss / (1024 ** 2)
                log(f'[PROFILE] RSS sampled ({stage}): {rss_mib:.1f} MiB')

        sample_rss('start')
        cuda_device = str(device).startswith("cuda")
        if profile_enabled and cuda_device:
            torch.cuda.reset_peak_memory_stats(device)

        ######## load source input ########
        if is_image(args.source):
            img_rgb = load_image_rgb(args.source)
            img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)
            log(f"Load source image from {args.source}")
        else:  # source input is an unknown format
            raise Exception(f"Unknown source format: {args.source}")

        ######## process driving info ########
        flag_load_from_template = is_template(args.driving)
        driving_rgb_crop_256x256_lst = None
        driving_preview_from_file = False
        wfp_template = None

        if flag_load_from_template:
            # NOTE: load from template, it is fast, but the cropping video is None
            log(f"Load from template: {args.driving}, NOT the video, so the cropping video and audio are both NULL.", style='bold green')
            driving_template_dct = load(args.driving)
            n_frames = driving_template_dct['n_frames']

            # set output_fps
            output_fps = driving_template_dct.get('output_fps', inf_cfg.output_fps)
            log(f'The FPS of template: {output_fps}')

            if args.flag_crop_driving_video:
                log("Warning: flag_crop_driving_video is True, but the driving info is a template, so it is ignored.")

        elif osp.exists(args.driving) and is_video(args.driving):
            # load from video file, AND make motion template
            output_fps = int(get_fps(args.driving))
            log(f"Load driving video from: {args.driving}, FPS is {output_fps}")

            log("Start making driving motion template...")
            if stream_driving_video and not inf_cfg.flag_crop_driving_video:
                # V0.3: decode + resize one RGB frame at a time. If the
                # generated output is also streamed, a second decoder pass
                # supplies concat previews without caching a full frame list.
                preview_frames = ([] if save_concat_video and not stream_video
                                  else None)
                driving_frames = iter_video_frames_rgb(args.driving, resize=(256, 256))
                try:
                    driving_template_dct = self.make_motion_template(
                        driving_frames, output_fps=output_fps,
                        preview_frames=preview_frames,
                    )
                finally:
                    driving_frames.close()
                n_frames = driving_template_dct['n_frames']
                if n_frames == 0:
                    raise ValueError(f'Driving video contains no decodable frames: {args.driving}')
                driving_rgb_crop_256x256_lst = preview_frames
                driving_preview_from_file = bool(save_concat_video and stream_video)
                log(f'[V0.3] Streaming driving decode: {n_frames} frames; '
                    f'preview caching: {preview_frames is not None}')
            else:
                if stream_driving_video and inf_cfg.flag_crop_driving_video:
                    log('[V0.3] Driving crop/tracking requires the legacy '
                        'full-video path; streaming decode disabled for this input.')
                driving_rgb_lst = load_video(args.driving)
                n_frames = len(driving_rgb_lst)
                if inf_cfg.flag_crop_driving_video:
                    ret_d = self.cropper.crop_driving_video(driving_rgb_lst)
                    log(f'Driving video is cropped, {len(ret_d["frame_crop_lst"])} frames are processed.')
                    if len(ret_d["frame_crop_lst"]) != n_frames:
                        n_frames = min(n_frames, len(ret_d["frame_crop_lst"]))
                    driving_rgb_crop_lst = ret_d['frame_crop_lst']
                    driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_crop_lst]
                else:
                    driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]
                # V0.2: even the fallback path uploads just one frame at a time.
                driving_template_dct = self.make_motion_template(
                    driving_rgb_crop_256x256_lst, output_fps=output_fps,
                )

            # save the motion template in the original pickle format

            wfp_template = remove_suffix(args.driving) + '.pkl'
            dump(wfp_template, driving_template_dct)
            log(f"Dump motion template to {wfp_template}")

        else:
            raise Exception(f"{args.driving} not exists or unsupported driving info types!")

        sample_rss('driving ready')
        ######## prepare for pasteback ########
        I_p_pstbk_lst = None
        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            I_p_pstbk_lst = []
            log("Prepared pasteback mask done.")

        ######## process source info ########
        if inf_cfg.flag_do_crop:
            crop_info = self.cropper.crop_source_image(img_rgb, crop_cfg)
            if crop_info is None:
                raise Exception("No animal face detected in the source image!")
            img_crop_256x256 = crop_info['img_crop_256x256']
        else:
            img_crop_256x256 = cv2.resize(img_rgb, (256, 256))  # force to resize to 256x256
        I_s = self.live_portrait_wrapper_animal.prepare_source(img_crop_256x256)
        x_s_info = self.live_portrait_wrapper_animal.get_kp_info(I_s)
        x_c_s = x_s_info['kp']
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.live_portrait_wrapper_animal.extract_feature_3d(I_s)
        x_s = self.live_portrait_wrapper_animal.transform_keypoint(x_s_info)

        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, crop_info['M_c2o'], dsize=(img_rgb.shape[1], img_rgb.shape[0]))

        ######## animate ########
        I_p_lst = [] if not stream_video else None
        pasteback_enabled = (inf_cfg.flag_pasteback and inf_cfg.flag_do_crop
                             and inf_cfg.flag_stitching)
        stream_writer = None
        if stream_video:
            mkdir(args.output_dir)
            output_basename = f'{basename(args.source)}--{basename(args.driving)}'
            stream_output = osp.join(args.output_dir, f'{output_basename}.mp4')
            stream_concat = (osp.join(args.output_dir, f'{output_basename}_concat.mp4')
                             if save_concat_video else None)
            stream_writer = AnimalStreamWriter(
                output_path=stream_output, concat_path=stream_concat,
                fps=output_fps, source_image=img_crop_256x256
            )
        driving_preview_iter = (iter_video_frames_rgb(args.driving, resize=(256, 256))
                                if driving_preview_from_file else None)
        animation_start = time.perf_counter() if profile_enabled else None
        try:
            for i in track(range(n_frames), description='🚀Animating...', total=n_frames):

                x_d_i_info = driving_template_dct['motion'][i]
                x_d_i_info = dct2device(x_d_i_info, device)

                R_d_i = x_d_i_info['R'] if 'R' in x_d_i_info.keys() else x_d_i_info['R_d']  # compatible with previous keys
                delta_new = x_d_i_info['exp']
                t_new = x_d_i_info['t']
                t_new[..., 2].fill_(0)  # zero tz
                scale_new = x_s_info['scale']

                x_d_i = scale_new * (x_c_s @ R_d_i + delta_new) + t_new

                if i == 0:
                    x_d_0 = x_d_i
                    motion_multiplier = calc_motion_multiplier(x_s, x_d_0)

                x_d_diff = (x_d_i - x_d_0) * motion_multiplier
                x_d_i = x_d_diff + x_s

                if not inf_cfg.flag_stitching:
                    pass
                else:
                    x_d_i = self.live_portrait_wrapper_animal.stitching(x_s, x_d_i)

                x_d_i = x_s + (x_d_i - x_s) * inf_cfg.driving_multiplier
                if profile_enabled:
                    if cuda_device:
                        torch.cuda.synchronize(device)
                    decode_start = time.perf_counter()
                out = self.live_portrait_wrapper_animal.warp_decode(f_s, x_s, x_d_i)
                if profile_enabled:
                    if cuda_device:
                        torch.cuda.synchronize(device)
                    profile_decode_ms.append((time.perf_counter() - decode_start) * 1000.0)
                I_p_i = self.live_portrait_wrapper_animal.parse_output(out['out'])[0]
                if not stream_video:
                    I_p_lst.append(I_p_i)

                if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
                    I_p_pstbk = paste_back(I_p_i, crop_info['M_c2o'], img_rgb, mask_ori_float)
                    if not stream_video:
                        I_p_pstbk_lst.append(I_p_pstbk)

                if stream_writer is not None:
                    if driving_preview_iter is not None:
                        try:
                            driving_frame = next(driving_preview_iter)
                        except StopIteration as exc:
                            raise RuntimeError('Driving video changed between '
                                               'motion and preview decode passes') from exc
                    else:
                        driving_frame = (driving_rgb_crop_256x256_lst[i]
                                         if driving_rgb_crop_256x256_lst is not None
                                         else None)
                    stream_writer.write(
                        I_p_i,
                        output_frame=I_p_pstbk if pasteback_enabled else I_p_i,
                        driving_frame=driving_frame,
                    )
        finally:
            try:
                if stream_writer is not None:
                    stream_writer.close()
            finally:
                if driving_preview_iter is not None:
                    driving_preview_iter.close()

        sample_rss('animation done')
        if profile_enabled:
            animation_elapsed = time.perf_counter() - animation_start
            if profile_decode_ms:
                avg = sum(profile_decode_ms) / len(profile_decode_ms)
                log(f"[PROFILE] W+G warp_decode: {avg:.2f} ms/frame "
                    f"({len(profile_decode_ms)} frames, max {max(profile_decode_ms):.2f} ms)")
            log(f"[PROFILE] animation loop: {animation_elapsed:.2f} s")
            if cuda_device:
                allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 2
                reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 2
                log(f"[PROFILE] CUDA peak allocated/reserved: "
                    f"{allocated:.1f}/{reserved:.1f} MiB")

        mkdir(args.output_dir)
        flag_driving_has_audio = (not flag_load_from_template) and has_audio_stream(args.driving)
        video_name = f'{basename(args.source)}--{basename(args.driving)}'
        wfp = osp.join(args.output_dir, f'{video_name}.mp4')
        wfp_concat = (osp.join(args.output_dir, f'{video_name}_concat.mp4')
                      if save_concat_video else None)

        # The non-streaming path preserves the original batch encoding behavior.
        if not stream_video:
            if save_concat_video:
                frames_concatenated = concat_frames(
                    driving_rgb_crop_256x256_lst, [img_crop_256x256], I_p_lst
                )
                images2video(frames_concatenated, wfp=wfp_concat, fps=output_fps)
            if I_p_pstbk_lst is not None and len(I_p_pstbk_lst) > 0:
                images2video(I_p_pstbk_lst, wfp=wfp, fps=output_fps)
            else:
                images2video(I_p_lst, wfp=wfp, fps=output_fps)

        if flag_driving_has_audio:
            if save_concat_video:
                wfp_concat_with_audio = osp.join(args.output_dir, f'{video_name}_concat_with_audio.mp4')
                add_audio_to_video(wfp_concat, args.driving, wfp_concat_with_audio)
                os.replace(wfp_concat_with_audio, wfp_concat)
            wfp_with_audio = osp.join(args.output_dir, f'{video_name}_with_audio.mp4')
            add_audio_to_video(wfp, args.driving, wfp_with_audio)
            os.replace(wfp_with_audio, wfp)

        # final log
        if wfp_template not in (None, ''):
            log(f'Animated template: {wfp_template}, you can specify `-d` argument with this template path next time to avoid cropping video, motion making and protecting privacy.', style='bold green')
        log(f'Animated video: {wfp}')
        log(f'Animated video with concat: {wfp_concat}')

        # build the gif
        wfp_gif = video2gif(wfp) if save_gif else None
        if wfp_gif is not None:
            log(f'Animated gif: {wfp_gif}')


        return wfp, wfp_concat, wfp_gif
