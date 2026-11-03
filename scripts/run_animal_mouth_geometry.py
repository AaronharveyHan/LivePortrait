#!/usr/bin/env python3
# coding: utf-8
"""V0.6.3: geometrically warp paired V0.6.2 mouth references (CPU only).

First run --init-landmarks; edit closed/open normalized landmarks using the
annotated guides and set verified=true. Then run with --landmarks JSON.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_alphas(raw):
    try:
        vals = [float(s.strip()) for s in raw.split(',')]
    except ValueError as exc:
        raise ValueError('alphas must be comma separated finite numbers') from exc
    if len(vals) < 2 or len(vals) > 21 or any(not math.isfinite(x) or not 0<=x<=1 for x in vals):
        raise ValueError('alphas need 2..21 finite values in [0,1]')
    if vals[0] != 0 or vals[-1] != 1 or any(vals[i]>=vals[i+1] for i in range(len(vals)-1)):
        raise ValueError('alphas must strictly increase and include 0 and 1')
    if len({round(v*1000) for v in vals}) != len(vals):
        raise ValueError('alpha names would collide at .001 resolution')
    return tuple(vals)


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--reference-dir',type=Path,default=ROOT/'build/onnx_animal_v062/yawn_preview',help='V0.6.2 output directory: closed_reference.png + open_reference_aligned.png')
    p.add_argument('--output-dir',type=Path,default=ROOT/'build/onnx_animal_v063/yawn_geometry')
    p.add_argument('--landmarks',type=Path,help='Annotated and confirmed landmarks JSON')
    p.add_argument('--init-landmarks',action='store_true',help='Write an unverified JSON TEMPLATE and two overlay guides; does not render')
    p.add_argument('--roi',default='0.29,0.49,0.74,0.83',help='Used when creating landmark template only')
    p.add_argument('--alphas',default='0,0.25,0.5,0.75,1')
    p.add_argument('--feather-px',type=float,default=24.)
    p.add_argument('--policies',choices=('both','soft','dominant'),default='both')
    p.add_argument('--overwrite',action='store_true',help='Overwrite V0.6.3 output files, never inputs or landmark JSON')
    p.add_argument('--same-dog',action='store_true',help='User asserts the two V0.6.2 references belong to the same dog')
    p.add_argument('--plan',action='store_true',help='List prerequisites; no third-party model or image imports')
    return p


def load_refs(refdir):
    from PIL import Image
    import numpy as np
    from src.utils.animal_mouth_geometry import check_image
    paths=[refdir/'closed_reference.png',refdir/'open_reference_aligned.png']
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError('Missing V0.6.2 reference: '+str(p))
    out=[]
    for p in paths:
        with Image.open(p) as im:
            rgb=np.asarray(im.convert('RGB'),dtype=np.uint8)
        out.append(check_image(rgb,p.name))
    return out[0],out[1],paths


def validate_args(args):
    from src.utils.animal_mouth_geometry import validate_roi
    alphas=parse_alphas(args.alphas)
    if not math.isfinite(args.feather_px) or not 1 <= args.feather_px <= 128:
        raise ValueError('--feather-px must be finite 1..128')
    if args.plan:
        return alphas
    if not args.same_dog:
        raise ValueError('Explicit --same-dog assertion is required')
    if args.init_landmarks and args.landmarks is not None:
        raise ValueError('--init-landmarks and --landmarks cannot both be used')
    if not args.init_landmarks and args.landmarks is None:
        raise ValueError('Provide --landmarks JSON or --init-landmarks')
    if args.output_dir.resolve() == args.reference_dir.resolve():
        raise ValueError('Output directory cannot be reference input directory')
    if not args.output_dir.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Output directory must be within LivePortrait repository')
    if args.landmarks and args.output_dir.resolve() == args.landmarks.resolve():
        raise ValueError('Output directory conflicts with landmarks')
    validate_roi(tuple(float(x) for x in args.roi.split(',')))
    return alphas


def _planned_paths(args, alphas):
    o=args.output_dir
    if args.init_landmarks:
        return [o/'mouth_landmarks_template.json',o/'closed_guide.png',o/'open_guide.png',o/'annotate_mouth.html']
    policies=('soft','dominant') if args.policies=='both' else (args.policies,)
    paths=[o/'report.json',o/'comparison_sheet.png']
    for policy in policies:
        paths.extend(o/f'mouth_{policy}_{round(a*1000):03d}.png' for a in alphas)
    return paths


def _assert_writable(paths, args):
    # Protect all inputs, even if --overwrite is supplied.
    inputs={args.reference_dir.resolve()/'closed_reference.png',args.reference_dir.resolve()/'open_reference_aligned.png'}
    if args.landmarks:
        inputs.add(args.landmarks.resolve())
    for p in paths:
        if p.resolve() in inputs:
            raise ValueError('Refusing to overwrite source/reference: '+str(p))
        if p.exists() and (not args.overwrite or (args.init_landmarks and p.name=='mouth_landmarks_template.json')):
            raise FileExistsError(f'Output exists: {p} (do not overwrite landmark JSON; use new directory)')


def _png(path,arr):
    from PIL import Image
    Image.fromarray(arr,'RGB').save(path)


def _sheet(images, alphas, policies):
    from PIL import Image,ImageDraw
    w,h=256,284
    sheet=Image.new('RGB',(len(alphas)*w,len(policies)*h),'white')
    draw=ImageDraw.Draw(sheet)
    for j, policy in enumerate(policies):
        for i, alpha in enumerate(alphas):
            rgb=images[policy][i]
            im=Image.fromarray(rgb,'RGB').resize((w,w))
            x,y=i*w,j*h
            sheet.paste(im,(x,y))
            draw.text((x+8,y+260),f'{policy}: alpha={alpha:.3f}',fill=(0,0,0))
    return sheet


def _hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def run(args, alphas):
    from src.utils.animal_mouth_geometry import (
        validate_roi, create_template, validate_landmarks, guides,
        blend_geometry, mouth_geometry_metrics,
    )
    from src.utils.animal_dual_source import mouth_mask, blend_mouth
    import numpy as np
    from PIL import Image
    c,o,paths=load_refs(args.reference_dir)
    outputs=_planned_paths(args,alphas)
    _assert_writable(outputs,args)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.init_landmarks:
        try:
            roi=validate_roi(tuple(float(x) for x in args.roi.split(',')))
        except ValueError as exc:
            raise ValueError('--roi must contain four valid comma-separated normalized coordinates') from exc
        spec=create_template(roi)
        spec['reference_sha256']={'closed':_hash(paths[0]),'open':_hash(paths[1])}
        (args.output_dir/'mouth_landmarks_template.json').write_text(
            json.dumps(spec,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        a,b=guides(c,o,spec)
        a.save(args.output_dir/'closed_guide.png')
        b.save(args.output_dir/'open_guide.png')
        from src.utils.animal_mouth_annotator import annotation_html
        (args.output_dir/'annotate_mouth.html').write_text(
            annotation_html(c,o,spec),encoding='utf-8')
        result={'mode':'init_landmarks','output_dir':str(args.output_dir),
                'verified':False,'annotation_html':str(args.output_dir/'annotate_mouth.html'),
                'message':'Open annotate_mouth.html in Chrome, click the 5 corresponding landmarks in BOTH images, download verified JSON.'}
        print(json.dumps(result,indent=2,ensure_ascii=False))
        return result
    spec=json.loads(args.landmarks.read_text(encoding='utf-8'))
    roi,names,cp,op=validate_landmarks(spec)
    # Prevent applying landmarks derived from different images accidentally.
    fingerprints=spec.get('reference_sha256',{})
    if fingerprints:
        if fingerprints.get('closed') != _hash(paths[0]) or fingerprints.get('open') != _hash(paths[1]):
            raise ValueError('Landmark reference image hashes mismatch; re-init / re-annotate')
    # Check all geometry before emitting first PNG, avoid partial outputs on fold.
    policies=('soft','dominant') if args.policies=='both' else (args.policies,)
    images={policy:[] for policy in policies}
    metrics={policy:[] for policy in policies}
    mask=mouth_mask(512,512,roi,args.feather_px)
    for a in alphas:
        for policy in policies:
            result=blend_geometry(c,o,a,spec,policy=policy,feather_px=args.feather_px)
            images[policy].append(result)
            metrics[policy].append(mouth_geometry_metrics(c,result,roi))
    # Standard V0.6.2 pixel blending comparator (not visual proof of quality).
    reference=[blend_mouth(c,o,a,mask) for a in alphas]
    comparator=[mouth_geometry_metrics(c,r,roi) for r in reference]
    # Include baseline images in comparison sheet for a complete side-by-side view.
    sheet_rows={'v062_rgb_blend':reference,**images}
    _sheet(sheet_rows,alphas,tuple(sheet_rows)).save(args.output_dir/'comparison_sheet.png')
    for policy in policies:
        for alpha,arr in zip(alphas,images[policy]):
            _png(args.output_dir/f'mouth_{policy}_{round(alpha*1000):03d}.png',arr)
    report={
        'schema_version':1,'stage':'v0.6.3_geometric_mouth','mode':'two_state_2d_manual_landmarks',
        'same_dog_asserted':True,'reference_dir':str(args.reference_dir),
        'reference_sha256':{'closed':_hash(paths[0]),'open':_hash(paths[1])},
        'landmarks_file':str(args.landmarks),'landmarks_sha256':_hash(args.landmarks),
        'landmark_names':names,'roi':roi,'feather_px':args.feather_px,'alphas':alphas,
        'policies':policies,'frames':{
            policy:[{'alpha':alpha,'path':str(args.output_dir/f'mouth_{policy}_{round(alpha*1000):03d}.png'),
                     'metrics':m} for alpha,m in zip(alphas,metrics[policy])]
            for policy in policies
        },
        'v062_reference_metrics':comparator,
        'limitations':[
            'Landmarks are user marked 2D photo correspondences, NOT a fitted 3D jaw',
            'Soft mode can ghost teeth/tongue despite geometry warp',
            'Dominant mode switches texture at alpha=0.5 and may visibly pop in a video',
            'No occlusion modeling, audio phonemes, automatic landmarks or temporal smoothing',
            'Exact outside-ROI closed pixels preserved; visual realism must be inspected',
        ],
    }
    (args.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'stage':report['stage'],'output_dir':str(args.output_dir),
                      'num_frames':sum(len(v) for v in images.values()),'policies':policies,
                      'reference_comparator':'v062_rgb_blend','comparison_sheet':str(args.output_dir/'comparison_sheet.png'),
                      'outside_changed_per_policy':{p:[m['outside_region_changed_pixels'] for m in ms] for p,ms in metrics.items()}},
                     indent=2,ensure_ascii=False))
    return report


def main(argv=None):
    args=parser().parse_args(argv)
    alphas=validate_args(args)
    if args.plan:
        detail={'stage':'v0.6.3_geometric_mouth','reference_dir':str(args.reference_dir),
                'required_references':{n:(args.reference_dir/n).is_file() for n in ('closed_reference.png','open_reference_aligned.png')},
                'landmarks':str(args.landmarks),'init_landmarks':args.init_landmarks,
                'alphas':alphas,'policies':args.policies,
                'note':'Stage after V0.6.2. Requires manual aligned image correspondences; no ONNX called by this command.'}
        print(json.dumps(detail,ensure_ascii=False,indent=2))
        return 0
    run(args,alphas)
    return 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except (OSError,ValueError,RuntimeError,KeyError,ImportError) as exc:
        print('[ERROR]',exc,file=sys.stderr)
        sys.exit(2)
