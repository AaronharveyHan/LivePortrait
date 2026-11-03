#!/usr/bin/env python3
# coding: utf-8
"""V0.6.5: measured jaw-opening + separately controlled, manually masked tongue.

Uses aligned original V0.6.2 photos, VERIFIED V0.6.3 + V0.6.4
landmarks, and human-drawn tongue polygons. CPU/OpenCV only; does NOT alter
ONNX F/M/W/G. Intended for jaw and tongue prototype evaluation, not lip sync.
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts import run_animal_mouth_multiref as old


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--reference-dir',type=Path,default=ROOT/'build/onnx_animal_v062/yawn_preview')
    p.add_argument('--landmarks',type=Path,help='Verified V0.6.3 endpoints JSON')
    p.add_argument('--mid',action='append',default=[],metavar='TIME=PNG',help='Original time position (repeat), e.g. 0.333333=frame_01.png')
    p.add_argument('--mid-points',type=Path,help='Verified V0.6.4 intermediate landmarks JSON')
    p.add_argument('--mid-aligned',action='store_true',help='Mid images are already registered 512px RGB')
    p.add_argument('--alignment',choices=('translation','none'),default='translation')
    p.add_argument('--max-shift',type=float,default=20)
    p.add_argument('--same-dog',action='store_true')
    p.add_argument('--opening-mode',choices=('geometry','time'),default='geometry',help='geometry=measured aperture, time=explicit diagnostic fallback')
    p.add_argument('--init-tongue',action='store_true',help='Generate offline HTML to hand-draw tongue polygons on each aligned reference')
    p.add_argument('--tongue-manifest',type=Path,help='Human-verified exported tongue polygons JSON')
    p.add_argument('--openings',default='0,0.25,0.5,0.75,1',help='Target measured normalized jaw opening 0..1 (not frame time)')
    p.add_argument('--tongue-values',default=None,help='Independent tongue extension 0..1 per target opening, comma separated')
    p.add_argument('--cavity-method',choices=('layered_nearest','layered_flow'),default='layered_flow')
    p.add_argument('--holdout',metavar='OPENING=PNG',help='Unseen photo, NOT included in any reference or tongue mask')
    p.add_argument('--holdout-tongue',type=float,default=None,help='Externally set tongue amount for holdout; otherwise heuristic curve')
    p.add_argument('--feather-px',type=float,default=24)
    p.add_argument('--max-opening-step',type=float,default=.30)
    p.add_argument('--max-tongue-step',type=float,default=.50)
    p.add_argument('--max-roi-mae',type=float,default=32)
    p.add_argument('--strict-continuity',action='store_true',help='FAIL before writing outputs if step/ROI change exceeds thresholds')
    p.add_argument('--output-dir',type=Path,default=ROOT/'build/onnx_animal_v065/yawn_independent')
    p.add_argument('--overwrite',action='store_true',help='Replace only named output PNG and report files, never source/annotation JSON')
    p.add_argument('--plan',action='store_true',help='Show prerequisites, no images or models opened')
    return p


def parse_values(raw,openings):
    from src.utils.animal_mouth_v065 import default_tongue_amount
    if raw is None:
        return tuple(default_tongue_amount(x) for x in openings), 'demonstration_delayed_smoothstep_UNVERIFIED'
    try:
        vals=tuple(float(v.strip()) for v in raw.split(','))
    except ValueError as exc:
        raise ValueError('--tongue-values must be comma separated 0..1 numbers') from exc
    if len(vals)!=len(openings) or any(not math.isfinite(x) or x<0 or x>1 for x in vals):
        raise ValueError('--tongue-values length must equal --openings and values finite 0..1')
    return vals,'user_supplied_independent'


def validate_args(args):
    from src.utils.animal_mouth_multiref import validate_target_alphas,parse_frame_spec,validate_reference_alphas
    try:
        openings=validate_target_alphas(tuple(float(x.strip()) for x in args.openings.split(',')))
    except ValueError as exc:
        raise ValueError('Invalid --openings; use ascending 0..1 values including endpoints') from exc
    tongue_values,source=parse_values(args.tongue_values,openings)
    mids=[parse_frame_spec(x) for x in args.mid]
    validate_reference_alphas([0]+[a for a,_ in mids]+[1])
    holdout=parse_frame_spec(args.holdout) if args.holdout else None
    for name in ('max_shift','feather_px','max_opening_step','max_tongue_step','max_roi_mae'):
        x=getattr(args,name)
        if not isinstance(x,(float,int)) or not math.isfinite(x) or x<=0:
            raise ValueError('--{} must be positive finite'.format(name.replace('_','-')))
    if args.max_shift>64 or not 1<=args.feather_px<=128:
        raise ValueError('Invalid alignment max-shift or feather-px')
    if args.holdout_tongue is not None and (not math.isfinite(args.holdout_tongue) or not 0<=args.holdout_tongue<=1):
        raise ValueError('Invalid --holdout-tongue')
    if args.plan:
        return openings,tongue_values,source,mids,holdout
    if not args.same_dog:
        raise ValueError('User must assert --same-dog')
    if args.landmarks is None or not args.landmarks.is_file():
        raise FileNotFoundError('--landmarks verified endpoint JSON is required')
    if args.mid_points is None and args.opening_mode=='geometry':
        raise ValueError('Measured --opening-mode geometry needs human-verified --mid-points')
    if args.mid_points is not None and not args.mid_points.is_file():
        raise FileNotFoundError('--mid-points JSON missing')
    if args.init_tongue and args.tongue_manifest:
        raise ValueError('Choose either --init-tongue or --tongue-manifest')
    if not args.init_tongue and (args.tongue_manifest is None or not args.tongue_manifest.is_file()):
        raise FileNotFoundError('Run --init-tongue then supply --tongue-manifest verified JSON')
    refs=[args.reference_dir/'closed_reference.png',args.reference_dir/'open_reference_aligned.png']
    paths=refs+[p for _,p in mids]
    if holdout:paths.append(holdout[1])
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError('Missing reference photo: '+str(p))
    if len({p.resolve() for p in paths})!=len(paths):
        raise ValueError('Repeated frame file path; holdout must be independent')
    if holdout and any(abs(a-holdout[0])<1e-7 for a,_ in mids):
        raise ValueError('Holdout has same labeled frame time as a reference; use an unseen frame and independently measured OPENING')
    inputs=paths+[args.landmarks]
    if args.mid_points:inputs.append(args.mid_points)
    if args.tongue_manifest:inputs.append(args.tongue_manifest)
    if not args.output_dir.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Output directory must be inside repository')
    if args.output_dir.resolve()==args.reference_dir.resolve() or any(args.output_dir.resolve()==p.parent.resolve() for p in inputs):
        raise ValueError('Output directory must not be an input directory')
    if args.strict_continuity:
        from src.utils.animal_mouth_v065 import continuity_report
        # Pre-check commanded rate limits before any image/model loading.
        zero=[__import__('numpy').zeros((512,512,3),dtype='uint8') for _ in openings]
        from src.utils import animal_mouth_geometry as geo
        metrics=continuity_report(zero,openings,tongue_values,(.25,.45,.8,.92),
                                  max_opening_step=args.max_opening_step,max_tongue_step=args.max_tongue_step,
                                  max_roi_mae=args.max_roi_mae)
        if not metrics['all_pass']:
            raise ValueError('Control-rate limit violated; add intermediate --openings or change --tongue-values')
    return openings,tongue_values,source,mids,holdout


def _read_sources(args,mids):
    """Align using exact V0.6.4 preparation, require same photo processing mode."""
    import numpy as np
    from src.utils import animal_mouth_geometry as geo
    cpath=args.reference_dir/'closed_reference.png'
    opath=args.reference_dir/'open_reference_aligned.png'
    c=old._read_reference(cpath)
    o=old._read_reference(opath)
    spec=json.loads(args.landmarks.read_text(encoding='utf-8'))
    roi,names,cp,op=geo.validate_landmarks(spec)
    checks=spec.get('reference_sha256')
    if checks and (checks.get('closed')!=old.sha(cpath) or checks.get('open')!=old.sha(opath)):
        raise ValueError('Endpoint landmark reference SHA256 mismatch')
    report=args.reference_dir/'report.json'
    mode=json.loads(report.read_text(encoding='utf-8')).get('mode','unknown') if report.is_file() else 'unknown'
    if mode=='onnx' and not args.mid_aligned:
        raise ValueError('ONNX endpoints cannot mix with raw middle photos; use original references or aligned ONNX intermediate frames')
    manual=old._manual_middle_points(args.mid_points,mids,names,roi) if args.mid_points else {}
    frames=[{'alpha':0.,'rgb':c,'points':cp}]
    notes=[]
    for alpha,path in mids:
        if args.mid_aligned:
            mid=old._read_reference(path)
            info={'mode':'already_aligned_512'}
        else:
            mid,info=old.load_mid_raw(path,c,alignment=args.alignment,max_shift=args.max_shift)
        pts=manual.get(alpha)
        if pts is None:
            from src.utils.animal_mouth_multiref import intermediate_proxy_points
            pts=intermediate_proxy_points(alpha,cp,op)
        frames.append({'alpha':alpha,'rgb':mid,'points':pts})
        notes.append({'alpha':alpha,'source_path':str(path),'geometry':'manual_verified' if alpha in manual else 'endpoint_proxy_UNVERIFIED',**info})
    frames.append({'alpha':1.,'rgb':o,'points':op})
    from src.utils import animal_mouth_multiref as multi
    mesh=multi.prepare_meshes(frames,roi)
    image_paths=[cpath]+[p for _,p in mids]+[opath]
    keys=[str(f['alpha']) for f in frames]
    hashes={k:old.sha(p) for k,p in zip(keys,image_paths)}
    if len(set(hashes.values()))!=len(hashes):
        raise ValueError('Two reference images have identical binary content')
    return c,frames,mesh,roi,names,hashes,notes,image_paths


def _output_paths(args,openings,nrefs):
    out=args.output_dir
    if args.init_tongue:
        return [out/'annotate_tongue.html',out/'mouth_tongue_template.json'] + [out/f'aligned_ref_{i:02d}.png' for i in range(nrefs)]
    paths=[out/'comparison_sheet.png',out/'report.json']
    paths += [out/f'cavity_ref_{i:02d}.png' for i in range(nrefs)]
    paths += [out/f'tongue_reference_mask_{i:02d}.png' for i in range(nrefs)]
    for a in openings:
        s=round(a*1000)
        paths.extend([out/f'mouth_v064_flow_{s:03d}.png',out/f'mouth_v065_cavity_{s:03d}.png',
                      out/f'mouth_v065_tongue_{s:03d}.png',out/f'tongue_alpha_{s:03d}.png'])
    if args.holdout:paths.append(out/'holdout_aligned.png')
    return paths


def _sheet(columns,openings):
    from PIL import Image,ImageDraw
    cell=256;row=284
    out=Image.new('RGB',(len(openings)*cell,len(columns)*row),'white')
    d=ImageDraw.Draw(out)
    for j,(name,imgs) in enumerate(columns.items()):
        for i,(opening,img) in enumerate(zip(openings,imgs)):
            x,y=i*cell,j*row
            out.paste(Image.fromarray(img,'RGB').resize((cell,cell)),(x,y))
            d.text((x+7,y+260),'{} opening={:.3f}'.format(name,opening),fill='black')
    return out


def _save_rgb(path,arr):
    from PIL import Image
    Image.fromarray(arr,'RGB').save(path)


def run(args,openings,tongue_values,tongue_source,mids,holdout):
    import numpy as np
    from PIL import Image
    from src.utils import animal_mouth_multiref as multi
    from src.utils.animal_mouth_v065 import (calibrate_openings,opening_to_time,
        validate_tongue_manifest,polygon_mask,prepare_cavity_frames,render_independent,continuity_report,
        default_tongue_amount)
    c,frames,oldmesh,roi,names,hashes,notes,inputs=_read_sources(args,mids)
    output_files=_output_paths(args,openings,len(frames))
    protected={p.resolve() for p in inputs+[args.landmarks]+([args.mid_points] if args.mid_points else [])+
               ([args.tongue_manifest] if args.tongue_manifest else [])+([holdout[1]] if holdout else [])}
    for path in output_files:
        if path.resolve() in protected:
            raise ValueError('Refusing to overwrite source/annotation: '+str(path))
        if path.exists() and (not args.overwrite or path.name=='mouth_tongue_template.json'):
            raise FileExistsError('Output exists: {} (use --overwrite or a new directory)'.format(path))
    if args.init_tongue:
        from src.utils.animal_tongue_annotator import tongue_annotation_html
        html=tongue_annotation_html(frames,hashes,roi)
        template={'schema_version':1,'coordinate_system':'normalized_xy_on_512px_aligned_reference',
                  'verified':False,'source_sha256':hashes,'polygons':{str(f['alpha']):[] for f in frames},
                  'note':'UNVERIFIED. In the HTML, mark each tongue polygon or explicitly confirm absence.'}
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/'annotate_tongue.html').write_text(html,encoding='utf-8')
        (args.output_dir/'mouth_tongue_template.json').write_text(json.dumps(template,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        for i,f in enumerate(frames):_save_rgb(args.output_dir/f'aligned_ref_{i:02d}.png',f['rgb'])
        result={'stage':'v0.6.5_tongue_annotation','html':str(args.output_dir/'annotate_tongue.html'),
                'ref_count':len(frames),'source_hashes':hashes,
                'note':'Open HTML offline; draw tongue polygons (or confirm absent) in every reference.'}
        print(json.dumps(result,indent=2,ensure_ascii=False))
        return result
    positions,raw_ratios=calibrate_openings(frames,names,mode=args.opening_mode)
    manifest=json.loads(args.tongue_manifest.read_text(encoding='utf-8'))
    polygons=validate_tongue_manifest(manifest,frames,hashes,roi)
    cavity,masks=prepare_cavity_frames(frames,polygons)
    cavmesh=multi.prepare_meshes(cavity,roi)
    rows={'v064_flow':[],'v065_cavity_only':[],'v065_jaw_tongue':[]}
    audits=[];tongue_masks=[]
    for opening,q in zip(openings,tongue_values):
        t=opening_to_time(opening,positions,frames)
        oldrgb,_=multi.render_layered(c,frames,t,roi,names,oldmesh,method='layered_flow',feather_px=args.feather_px)
        zero,_,_=render_independent(c,frames,cavity,cavmesh,polygons,masks,opening,0,positions,roi,names,
                                     feather_px=args.feather_px,cavity_method=args.cavity_method)
        rgb,tm,audit=render_independent(c,frames,cavity,cavmesh,polygons,masks,opening,q,positions,roi,names,
                                       feather_px=args.feather_px,cavity_method=args.cavity_method)
        rows['v064_flow'].append(oldrgb)
        rows['v065_cavity_only'].append(zero)
        rows['v065_jaw_tongue'].append(rgb)
        tongue_masks.append(tm)
        audits.append(audit)
    continuity=continuity_report(rows['v065_jaw_tongue'],openings,tongue_values,roi,
        max_opening_step=args.max_opening_step,max_tongue_step=args.max_tongue_step,max_roi_mae=args.max_roi_mae)
    if args.strict_continuity and not continuity['all_pass']:
        raise ValueError('Continuity threshold failed; no V0.6.5 files written. See --max-roi-mae or reduce per-frame control changes')
    evaluation=None;hold_rgb=None
    if holdout:
        ho, hp=holdout
        # Reject identical image content even if the file was copied to a new path.
        if old.sha(hp) in hashes.values():
            raise ValueError('Holdout content duplicates a reference image')
        if args.mid_aligned:hold_rgb=old._read_reference(hp);info={'mode':'already_aligned_512'}
        else:hold_rgb,info=old.load_mid_raw(hp,c,alignment=args.alignment,max_shift=args.max_shift)
        hq=default_tongue_amount(ho) if args.holdout_tongue is None else args.holdout_tongue
        tt=opening_to_time(ho,positions,frames)
        baseline=multi.render_layered(c,frames,tt,roi,names,oldmesh,method='layered_flow',feather_px=args.feather_px)[0]
        predicted=render_independent(c,frames,cavity,cavmesh,polygons,masks,ho,hq,positions,roi,names,
                                     feather_px=args.feather_px,cavity_method=args.cavity_method)[0]
        evaluation={'opening_external_target':ho,'tongue_external_or_heuristic_target':hq,
                    'tongue_target_source':'user' if args.holdout_tongue is not None else 'heuristic_unverified',
                    'path':str(hp),'sha256':old.sha(hp),'alignment':info,
                    'mouth_roi_mae':{'v064_flow':multi.region_mae(baseline,hold_rgb,roi),
                                     'v065_jaw_tongue':multi.region_mae(predicted,hold_rgb,roi)},
                    'note':'HOLDOUT not used in references or tongue polygons; exact image alignment and target controls affect MAE'}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for i,(f,cf,m) in enumerate(zip(frames,cavity,masks)):
        _save_rgb(args.output_dir/f'cavity_ref_{i:02d}.png',cf['rgb'])
        Image.fromarray(m,'L').save(args.output_dir/f'tongue_reference_mask_{i:02d}.png')
    for i,opening in enumerate(openings):
        suffix=f'{round(opening*1000):03d}'
        for name,prefix in (('v064_flow','mouth_v064_flow'),('v065_cavity_only','mouth_v065_cavity'),('v065_jaw_tongue','mouth_v065_tongue')):
            _save_rgb(args.output_dir/f'{prefix}_{suffix}.png',rows[name][i])
        Image.fromarray(np.rint(tongue_masks[i]*255).astype(np.uint8),'L').save(args.output_dir/f'tongue_alpha_{suffix}.png')
    _sheet(rows,openings).save(args.output_dir/'comparison_sheet.png')
    if hold_rgb is not None:_save_rgb(args.output_dir/'holdout_aligned.png',hold_rgb)
    report={'schema_version':1,'stage':'v0.6.5_independent_jaw_tongue',
            'reference_dir':str(args.reference_dir),'reference_alphas_time':[f['alpha'] for f in frames],
            'reference_openings_normalized':positions,'reference_aperture_width_ratios':raw_ratios,
            'opening_mode':args.opening_mode,'opening_target_values':list(openings),
            'tongue_target_values':list(tongue_values),'tongue_target_source':tongue_source,
            'reference_alignment':notes,'tongue_manifest':str(args.tongue_manifest),
            'tongue_manifest_sha256':old.sha(args.tongue_manifest),'tongue_reference_source':audits[0]['source_tongue_alpha'],
            'inpainted_cavity_is_proxy':True,'same_dog_asserted':True,'roi':roi,
            'frames':audits,'continuity':continuity,'outside_roi_changed_all_zero':all(x['outside_roi_changed_pixels']==0 for x in audits),
            'adjacent_roi_mae':{k:[multi.region_mae(a,b,roi) for a,b in zip(v,v[1:])] for k,v in rows.items()},
            'holdout_evaluation':evaluation,
            'limitations':['Tongue mask is human-annotated; empty marks are manually confirmed',
                'Jaw opening is a 2D relative metric, NOT an actual animal jaw angle',
                'Cavity under tongue reconstructed by inpainting, may be blurry/wrong',
                'Tongue is affinely scaled; no true tongue-root articulations or 3D occlusion',
                'Aperture and tongue controls do not constitute phonemes/audio lipsync',
                'No automatic segmentation, video generation or temporal learned consistency',
                'Independent holdout cannot be one of the reference photos or polygon masks',
                'Existing ONNX F/M/W/G and V0.6.4 algorithm untouched']}
    (args.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    result={'stage':report['stage'],'output_dir':str(args.output_dir),
            'comparison_sheet':str(args.output_dir/'comparison_sheet.png'),
            'measured_opening_positions':positions,'tongue_source':tongue_source,
            'continuity_all_pass':continuity['all_pass'],
            'outside_roi_changed_all_zero':report['outside_roi_changed_all_zero'],
            'holdout_roi_mae':evaluation['mouth_roi_mae'] if evaluation else None}
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    return report


def main(argv=None):
    args=parser().parse_args(argv)
    openings,values,source,mids,holdout=validate_args(args)
    if args.plan:
        detail={'stage':'v0.6.5_independent_jaw_tongue',
                'required_files':{'landmarks':bool(args.landmarks and args.landmarks.is_file()),
                                  'mid_points':bool(args.mid_points and args.mid_points.is_file()),
                                  'tongue_manifest':bool(args.tongue_manifest and args.tongue_manifest.is_file())},
                'init_tongue':args.init_tongue,'opening_mode':args.opening_mode,
                'openings':openings,'tongue_values':values,'tongue_values_source':source,
                'mids':[{'alpha':a,'path':str(p),'exists':p.is_file()} for a,p in mids],
                'holdout':str(holdout) if holdout else None,'no_model_loading':True}
        print(json.dumps(detail,indent=2,ensure_ascii=False))
        return 0
    run(args,openings,values,source,mids,holdout)
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except (OSError,ValueError,RuntimeError,ImportError,KeyError) as exc:
        print('[ERROR]',exc,file=sys.stderr)
        sys.exit(2)
