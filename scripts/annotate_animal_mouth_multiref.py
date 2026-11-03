#!/usr/bin/env python3
# coding: utf-8
"""V0.6.4: prepare OFFLINE browser manual labels for every intermediate frame."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from scripts.run_animal_mouth_multiref import parser as render_parser,validate_args,sha,_read_reference,load_mid_raw
from src.utils.animal_mouth_geometry import validate_landmarks
from src.utils.animal_mouth_multiref import intermediate_proxy_points
from src.utils.animal_mouth_multi_annotator import annotation_html


def parser():
    p=render_parser()
    p.description=__doc__
    p.set_defaults(output_dir=ROOT/'build/onnx_animal_v064/yawn_annotation')
    return p


def run(args,alphas,mids,holdout):
    import numpy as np
    from PIL import Image
    if args.mid_points:
        raise ValueError('--mid-points is output of annotation; do not supply it to annotation stage')
    if args.holdout:
        raise ValueError('--holdout is for evaluation, not point annotation')
    o=args.output_dir
    paths=[o/'annotate_mid_mouth.html',o/'mouth_mid_landmarks_template.json']
    paths.extend(o/f'mid_{i:02d}_aligned.png' for i in range(1,len(mids)+1))
    for path in paths:
        if path.exists():
            if not args.overwrite or path.name=='mouth_mid_landmarks_template.json':
                raise FileExistsError('Output already exists: {} (protecting template JSON)'.format(path))
    cpath=args.reference_dir/'closed_reference.png'
    opath=args.reference_dir/'open_reference_aligned.png'
    c=_read_reference(cpath)
    spec=json.loads(args.landmarks.read_text(encoding='utf-8'))
    roi,names,cp,op=validate_landmarks(spec)
    hashes=spec.get('reference_sha256',{})
    if hashes and (hashes.get('closed')!=sha(cpath) or hashes.get('open')!=sha(opath)):
        raise ValueError('Landmark reference hashes do not match V0.6.2 reference images')
    ref_report=args.reference_dir/'report.json'
    if ref_report.is_file() and not args.mid_aligned and json.loads(ref_report.read_text(encoding='utf-8')).get('mode')=='onnx':
        raise ValueError('Cannot use raw middle images with ONNX reconstructed endpoint references')
    frames=[]
    checksums={}
    for a,path in mids:
        if args.mid_aligned:
            img=_read_reference(path)
        else:
            img,_=load_mid_raw(path,c,alignment=args.alignment,max_shift=args.max_shift)
        key=format(a,'.9g')
        checksums[key]=sha(path)
        frames.append((a,img,intermediate_proxy_points(a,cp,op)))
    o.mkdir(parents=True,exist_ok=True)
    for i,(_,im,_) in enumerate(frames,1):
        Image.fromarray(im,'RGB').save(o/f'mid_{i:02d}_aligned.png')
    (o/'annotate_mid_mouth.html').write_text(annotation_html(frames,names,roi,checksums),encoding='utf-8')
    template={'schema_version':1,'coordinate_system':'normalized_xy_on_512px_aligned_reference','verified':False,
              'frames':{format(a,'.9g'):{n:[round(float(v)/512,6) for v in xy] for n,xy in zip(names,points)} for a,_,points in frames},
              'source_sha256':checksums,
              'note':'EXAMPLE ONLY. Do not edit verified=true unless EVERY landmark has been manually checked. Prefer HTML click-to-export.'}
    (o/'mouth_mid_landmarks_template.json').write_text(json.dumps(template,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    result={'stage':'v0.6.4_multiref_annotation','status':'UNVERIFIED','output_dir':str(o),
            'html':str(o/'annotate_mid_mouth.html'),'frame_count':len(mids),
            'next':'Open HTML, click ALL points on ALL frames, download mouth_mid_landmarks_verified.json'}
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return result


def main(argv=None):
    args=parser().parse_args(argv)
    alphas,mids,holdout=validate_args(args)
    if args.plan:
        print(json.dumps({'stage':'v0.6.4_annotation_plan','frames':[[a,str(p)] for a,p in mids],
                          'landmarks_exists':bool(args.landmarks and args.landmarks.is_file())},indent=2))
        return 0
    run(args,alphas,mids,holdout)
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except (OSError,ValueError,RuntimeError,ImportError,KeyError) as exc:
        print('[ERROR]',exc,file=sys.stderr)
        sys.exit(2)
