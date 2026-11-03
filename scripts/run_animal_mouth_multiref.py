#!/usr/bin/env python3
# coding: utf-8
"""V0.6.4: multi-reference + layered mouth texture, CPU-only experiment.

Use V0.6.2 references, V0.6.3 verified landmarks and optional REAL
intermediate frames from the same dog. Model F/M/W/G code is unchanged.
No audio, video, 3D anatomy or automatic mouth landmarks are inferred.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--reference-dir',type=Path,default=ROOT/'build/onnx_animal_v062/yawn_preview',help='V0.6.2 CLOSED and aligned OPEN reference PNGs')
    p.add_argument('--landmarks',type=Path,help='V0.6.3 mouth_landmarks_verified.json')
    p.add_argument('--mid',action='append',default=[],metavar='ALPHA=PNG',help='Add REAL intermediate state; repeat, e.g. 0.333=frame_01.png')
    p.add_argument('--mid-aligned',action='store_true',help='Mid PNGs are already registered RGB 512x512; do not resize or shift')
    p.add_argument('--mid-points',type=Path,help='Optional VERIFIED manual landmarks for all middle frames (see docs)')
    p.add_argument('--holdout',metavar='ALPHA=PNG',help='Unseen real frame, not also a --mid; report ROI pixel MAE')
    p.add_argument('--alignment',choices=('translation','none'),default='translation',help='Alignment of raw middle/holdout pictures to V0.6.2 closed reference')
    p.add_argument('--max-shift',type=float,default=20.)
    p.add_argument('--same-dog',action='store_true',help='User asserts all images show SAME animal')
    p.add_argument('--method',choices=('both','layered_nearest','layered_flow'),default='both')
    p.add_argument('--alphas',default='0,0.25,0.5,0.75,1')
    p.add_argument('--feather-px',type=float,default=24.)
    p.add_argument('--output-dir',type=Path,default=ROOT/'build/onnx_animal_v064/yawn_multiref')
    p.add_argument('--overwrite',action='store_true',help='Overwrite only V0.6.4 outputs; never inputs')
    p.add_argument('--plan',action='store_true',help='Read flags and file existence without image/model loading')
    return p


def parse_alphas(raw):
    from src.utils.animal_mouth_multiref import validate_target_alphas
    try:
        values=tuple(float(v.strip()) for v in raw.split(','))
    except ValueError as exc:
        raise ValueError('Expected comma-separated --alphas') from exc
    return validate_target_alphas(values)


def parse_mids(raws):
    from src.utils.animal_mouth_multiref import parse_frame_spec,validate_reference_alphas
    mids=[parse_frame_spec(s) for s in raws]
    if not mids:
        raise ValueError('At least one --mid frame required for V0.6.4 multi-reference evaluation')
    alphas=[0.]+[a for a,_ in mids]+[1.]
    validate_reference_alphas(alphas)
    return mids


def validate_args(args):
    from src.utils.animal_mouth_multiref import parse_frame_spec
    from math import isfinite
    alphas=parse_alphas(args.alphas)
    mids=parse_mids(args.mid)
    holdout=parse_frame_spec(args.holdout) if args.holdout else None
    if holdout and any(abs(a-holdout[0])<1e-7 for a,_ in mids):
        raise ValueError('Holdout must NOT be one of the reference alphas')
    if not isfinite(args.max_shift) or not 0<=args.max_shift<=64:
        raise ValueError('--max-shift must be 0..64 finite')
    if not isfinite(args.feather_px) or not 1<=args.feather_px<=128:
        raise ValueError('--feather-px must be 1..128 finite')
    if args.plan:
        return alphas,mids,holdout
    if not args.same_dog:
        raise ValueError('Explicit --same-dog required for multi-reference animation')
    if args.landmarks is None or not args.landmarks.is_file():
        raise FileNotFoundError('Verified V0.6.3 --landmarks JSON required')
    if args.mid_points is not None and not args.mid_points.is_file():
        raise FileNotFoundError('Missing --mid-points JSON')
    required=[args.reference_dir/'closed_reference.png',args.reference_dir/'open_reference_aligned.png']
    required += [p for _,p in mids]
    if holdout:
        required.append(holdout[1])
    for path in required:
        if not path.is_file():
            raise FileNotFoundError('Missing reference/holdout image: '+str(path))
    inputs={p.resolve() for p in (required+[args.landmarks]+([args.mid_points] if args.mid_points else []))}
    if len({p.resolve() for _,p in mids}) != len(mids):
        raise ValueError('Each --mid must have a distinct source image')
    anchors={p.resolve() for p in required[:2]}
    if any(p.resolve() in anchors for _,p in mids):
        raise ValueError('Middle frame cannot reuse a closed/open endpoint image')
    if holdout and holdout[1].resolve() in ({p.resolve() for _,p in mids} | anchors):
        raise ValueError('Holdout image cannot be a middle or endpoint reference')
    if args.output_dir.resolve() == args.reference_dir.resolve():
        raise ValueError('Output directory cannot be V0.6.2 reference directory')
    if not args.output_dir.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Output directory must be inside LivePortrait repository')
    if any(args.output_dir.resolve() == p.parent for p in inputs):
        raise ValueError('Output directory cannot be an input directory')
    return alphas,mids,holdout


def sha(path):
    d=hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda:f.read(1024*1024),b''):
            d.update(buf)
    return d.hexdigest()


def _read_reference(path):
    import numpy as np
    from PIL import Image
    from src.utils.animal_mouth_geometry import check_image
    with Image.open(path) as im:
        rgb=np.asarray(im.convert('RGB'),dtype=np.uint8)
    return check_image(rgb,str(path))


def load_mid_raw(path,closed_ref,*,alignment='translation',max_shift=20):
    """Match V0.6.2 original-image path: 256cv2 bilinear -> 512cv2 bilinear.

    Assumes source photos were pre-cropped similarly. ECC uses UPPER face only;
    high similarity score does not prove mouth correspondence.
    """
    import cv2
    from src.utils.animal_dual_source import estimate_translation,align_reconstruction
    # Match scripts/run_animal_onnx_frame.py read_rgb() OpenCV branch exactly,
    # but do not import any ONNX modules into this CPU-only experiment.
    raw=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError('Cannot decode middle/holdout image: '+str(path))
    inp=cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)
    if inp.shape[:2]!=(256,256):
        inp=cv2.resize(inp,(256,256),interpolation=cv2.INTER_LINEAR)
    resize='opencv-bilinear'
    ref256=cv2.resize(closed_ref,(256,256),interpolation=cv2.INTER_LINEAR)
    if alignment=='translation':
        reg=estimate_translation(ref256,inp,max_shift=max_shift)
    else:
        reg={'mode':'none','score':None,'dx_256':0.,'dy_256':0.}
    image=cv2.resize(inp,(512,512),interpolation=cv2.INTER_LINEAR)
    return align_reconstruction(image,reg),{'mode':'raw','resize':resize,'registration':reg}


def _manual_middle_points(path, mids, names, roi):
    """Optional manual annotations keyed by state alpha; all must be verified."""
    from src.utils.animal_mouth_multiref import validate_manual_points
    data=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data,dict) or data.get('schema_version')!=1 or data.get('verified') is not True or data.get('coordinate_system')!='normalized_xy_on_512px_aligned_reference':
        raise ValueError('--mid-points must be a verified v1 normalized coordinate manifest')
    frames=data.get('frames')
    checksums=data.get('source_sha256')
    if checksums is not None:
        if not isinstance(checksums,dict) or len(checksums)!=len(mids):
            raise ValueError('--mid-points source_sha256 must cover every middle image')
        for alpha,path in mids:
            candidates=[v for key,v in checksums.items() if abs(float(key)-alpha)<1e-6]
            if len(candidates)!=1 or candidates[0]!=sha(path):
                raise ValueError('Middle point annotation image hash mismatch at alpha {}'.format(alpha))
    if not isinstance(frames,dict) or len(frames)!=len(mids):
        raise ValueError('--mid-points must cover every --mid frame')
    mapped={}
    for key,pts in frames.items():
        try:
            val=float(key)
        except ValueError as exc:
            raise ValueError('Bad alpha key in --mid-points') from exc
        matching=[a for a,_ in mids if abs(a-val)<1e-6]
        if len(matching)!=1 or matching[0] in mapped:
            raise ValueError('Duplicate/unrecognized --mid-points alpha: '+str(key))
        mapped[matching[0]]=validate_manual_points(pts,names,roi)
    return mapped


def _sheet(imgs,alphas):
    from PIL import Image,ImageDraw
    w,h=256,283
    names=tuple(imgs)
    out=Image.new('RGB',(w*len(alphas),h*len(names)),'white')
    d=ImageDraw.Draw(out)
    for row,name in enumerate(names):
        for i,(alpha,img) in enumerate(zip(alphas,imgs[name])):
            x,y=w*i,h*row
            out.paste(Image.fromarray(img,'RGB').resize((w,w)),(x,y))
            d.text((x+7,y+260),f'{name}: alpha={alpha:.3f}',fill=(0,0,0))
    return out


def planned_outputs(args,alphas,mids,holdout):
    methods=('layered_nearest','layered_flow') if args.method=='both' else (args.method,)
    o=args.output_dir
    files=[o/'comparison_sheet.png',o/'report.json',o/'mouth_outer_mask.png']
    files.extend(o/f'mouth_interior_mask_{round(a*1000):03d}.png' for a in alphas)
    files.extend(o/f'mid_{i:02d}_aligned.png' for i in range(1,len(mids)+1))
    if holdout:
        files.append(o/'holdout_aligned.png')
    for m in methods:
        files.extend(o/f'mouth_{m}_{round(a*1000):03d}.png' for a in alphas)
    return files,methods


def run(args,alphas,mids,holdout):
    import numpy as np
    from PIL import Image
    from src.utils import animal_mouth_geometry as geo
    from src.utils.animal_mouth_multiref import (
        intermediate_proxy_points,prepare_meshes,render_layered,region_mae,outside_changed,
        target_control_points,_mouth_interior_mask
    )
    from src.utils.animal_dual_source import blend_mouth,mouth_mask
    import cv2
    output_paths,methods=planned_outputs(args,alphas,mids,holdout)
    inputpaths=[args.reference_dir/'closed_reference.png',args.reference_dir/'open_reference_aligned.png',args.landmarks]
    inputpaths.extend(p for _,p in mids)
    if holdout:inputpaths.append(holdout[1])
    if args.mid_points:inputpaths.append(args.mid_points)
    inputs={p.resolve() for p in inputpaths}
    for dest in output_paths:
        if dest.resolve() in inputs:
            raise ValueError('Refusing to overwrite input: '+str(dest))
        if dest.exists() and not args.overwrite:
            raise FileExistsError('Output exists: {} (use --overwrite)'.format(dest))
    cpath,opath=inputpaths[:2]
    c=_read_reference(cpath)
    o=_read_reference(opath)
    spec=json.loads(args.landmarks.read_text(encoding='utf-8'))
    roi,names,cp,op=geo.validate_landmarks(spec)
    hashes=spec.get('reference_sha256')
    if hashes and (hashes.get('closed')!=sha(cpath) or hashes.get('open')!=sha(opath)):
        raise ValueError('V0.6.3 landmark references do not match actual images')
    # Mixing raw GIF frames with reconstructed ONNX endpoints is not an honest
    # like-for-like experiment; require explicitly prepared/aligned mid outputs.
    mode='unknown'
    report_path=args.reference_dir/'report.json'
    if report_path.is_file():
        info=json.loads(report_path.read_text(encoding='utf-8'))
        mode=info.get('mode','unknown')
    if not args.mid_aligned and mode=='onnx':
        raise ValueError('Raw middle photos and ONNX-reconstructed endpoints use different appearances; use V0.6.2 original reference dir or --mid-aligned ONNX middle frames')
    manual=_manual_middle_points(args.mid_points,mids,names,roi) if args.mid_points else {}
    frames=[{'alpha':0.,'rgb':c,'points':cp}]
    alignments=[]
    prepared=[]
    for i,(a,p) in enumerate(mids,1):
        if args.mid_aligned:
            rgb=_read_reference(p)
            info={'mode':'already_aligned_512','registration':None,'resize':'none'}
        else:
            rgb,info=load_mid_raw(p,c,alignment=args.alignment,max_shift=args.max_shift)
        points=manual.get(a,intermediate_proxy_points(a,cp,op))
        frames.append({'alpha':a,'rgb':rgb,'points':points})
        prepared.append((f'mid_{i:02d}_aligned.png',rgb))
        alignments.append({'alpha':a,'source':str(p),'sha256':sha(p),
                           'landmark_source':'manual_verified' if a in manual else 'endpoint_linear_proxy_UNVERIFIED',**info})
    frames.append({'alpha':1.,'rgb':o,'points':op})
    meshes=prepare_meshes(frames,roi) # preflight fold protection before writing PNGs
    hold_rgb=None
    hold_info=None
    if holdout:
        ha,hp=holdout
        if args.mid_aligned:
            hold_rgb=_read_reference(hp)
            hold_info={'mode':'already_aligned_512','registration':None}
        else:
            hold_rgb,hold_info=load_mid_raw(hp,c,alignment=args.alignment,max_shift=args.max_shift)
    images={name:[] for name in ('v063_soft','v063_dominant',*methods)}
    refs_mask=mouth_mask(512,512,roi,args.feather_px)
    audit={method:[] for method in methods}
    for a in alphas:
        images['v063_soft'].append(geo.blend_geometry(c,o,a,spec,policy='soft',feather_px=args.feather_px))
        images['v063_dominant'].append(geo.blend_geometry(c,o,a,spec,policy='dominant',feather_px=args.feather_px))
        for method in methods:
            rgb,details=render_layered(c,frames,a,roi,names,meshes,method=method,feather_px=args.feather_px)
            images[method].append(rgb)
            audit[method].append({'alpha':a,**details,'outside_roi_changed_pixels':outside_changed(c,rgb,roi)})
    # Optional HOLDOUT renders are scored only if not included in the training/reference frames.
    evaluation=None
    if holdout:
        a,_=holdout
        results={'v062_rgb_blend':blend_mouth(c,o,a,refs_mask),
                 'v063_soft':geo.blend_geometry(c,o,a,spec,policy='soft',feather_px=args.feather_px),
                 'v063_dominant':geo.blend_geometry(c,o,a,spec,policy='dominant',feather_px=args.feather_px)}
        for m in methods:
            results[m]=render_layered(c,frames,a,roi,names,meshes,method=m,feather_px=args.feather_px)[0]
        evaluation={'alpha':a,'file':str(holdout[1]),'sha256':sha(holdout[1]),'alignment':hold_info,
                    'metric':'mouth_ROI_RGB_mean_abs_0_to_255_not_perceptual',
                    'roi_mae':{name:region_mae(arr,hold_rgb,roi) for name,arr in results.items()},
                    'note':'HOLDOUT frame was not among reference images; image-alignment differences can dominate MAE'}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.rint(refs_mask*255).astype(np.uint8),'L').save(args.output_dir/'mouth_outer_mask.png')
    for a in alphas:
        inner=_mouth_interior_mask(roi,target_control_points(frames,a),names)
        Image.fromarray(np.rint(inner*255).astype(np.uint8),'L').save(args.output_dir/f'mouth_interior_mask_{round(a*1000):03d}.png')
    for name,arr in prepared:
        Image.fromarray(arr,'RGB').save(args.output_dir/name)
    if hold_rgb is not None:
        Image.fromarray(hold_rgb,'RGB').save(args.output_dir/'holdout_aligned.png')
    for method in methods:
        for a,arr in zip(alphas,images[method]):
            Image.fromarray(arr,'RGB').save(args.output_dir/f'mouth_{method}_{round(a*1000):03d}.png')
    _sheet(images,alphas).save(args.output_dir/'comparison_sheet.png')
    report={'schema_version':1,'stage':'v0.6.4_multiref_layered_mouth','reference_dir':str(args.reference_dir),
            'reference_mode':mode,'same_dog_asserted':True,'roi':roi,'alphas':list(alphas),
            'reference_alphas':[f['alpha'] for f in frames],'raw_input_registration':args.alignment,
            'v063_landmarks':str(args.landmarks),'v063_landmark_sha256':sha(args.landmarks),
            'middle_frames':alignments,'methods':list(methods),'frames':audit,
            'outside_roi_changed_all_zero':all(row['outside_roi_changed_pixels']==0 for vals in audit.values() for row in vals),
            'adjacent_roi_mae':{name:[region_mae(a,b,roi) for a,b in zip(vals,vals[1:])] for name,vals in images.items()},
            'holdout_evaluation':evaluation,
            'limitations':['CPU 2D appearance experiment only; NO 3D jaw, phoneme sync or video synthesis',
                           'Unverified middle landmarks are linear proxies, NOT measured anatomy',
                           'Nearest texture can still POP when switching references',
                           'Optical flow may fail at teeth/tongue occlusion and produce double edges',
                           'A real reference used as an anchor is NOT an independent test example',
                           'Manual crop/alignment affects pixel MAE; inspect RGB, masks and holdout visually',
                           'No extra ONNX inference. Existing F/M/W/G untouched']}
    (args.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    result={'stage':report['stage'],'output_dir':str(args.output_dir),'methods':list(methods),
            'reference_alphas':report['reference_alphas'],'mid_geometry':'manual_verified' if args.mid_points else 'linear_proxy_unverified',
            'comparison_sheet':str(args.output_dir/'comparison_sheet.png'),
            'outside_roi_changed_all_zero':report['outside_roi_changed_all_zero'],
            'holdout_roi_mae':evaluation['roi_mae'] if evaluation else None}
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    return report


def main(argv=None):
    args=parser().parse_args(argv)
    alphas,mids,holdout=validate_args(args)
    if args.plan:
        data={'stage':'v0.6.4_multiref_layered_mouth','reference_dir':str(args.reference_dir),
              'reference_ready':{p:(args.reference_dir/p).is_file() for p in ('closed_reference.png','open_reference_aligned.png')},
              'landmarks_ready':bool(args.landmarks and args.landmarks.is_file()),
              'middle_frames':[{'alpha':a,'path':str(p),'exists':p.is_file()} for a,p in mids],
              'holdout':{'alpha':holdout[0],'exists':holdout[1].is_file()} if holdout else None,
              'mid_geometry':'manual' if args.mid_points else 'UNVERIFIED endpoint linear proxy',
              'method':args.method,'no_model_loading':True}
        print(json.dumps(data,indent=2,ensure_ascii=False))
        return 0
    run(args,alphas,mids,holdout)
    return 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except (ValueError,OSError,ImportError,RuntimeError,KeyError) as exc:
        print('[ERROR]',exc,file=sys.stderr)
        sys.exit(2)
