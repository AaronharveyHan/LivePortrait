#!/usr/bin/env python3
# coding: utf-8
"""V0.6.6 CPU experiment: multi-reference tongue-hole repair + anchored tongue.

Uses EXACT V0.6.5 manually verified references/landmarks/tongue manifest.
Does not invoke ONNX, infer animal anatomy, or overwrite source annotations.
"""
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import run_animal_mouth_v065 as previous


def parser():
    p=previous.parser()
    p.description=__doc__
    p.set_defaults(output_dir=ROOT/'build/onnx_animal_v066/yawn_cavity_root')
    p.add_argument('--donor-strength',type=float,default=.9,
                   help='0=Telea-only baseline; 1=fully accept valid multi-reference pixels')
    p.add_argument('--dilate-px',type=int,default=5,
                   help='Odd tongue-inpainting mask dilation, 1..31')
    p.add_argument('--inpaint-radius',type=int,default=7,
                   help='Telea fallback radius 1..15 pixels')
    p.add_argument('--root-depth',type=float,default=.58,
                   help='Tongue root position between upper lip (0) and lower lip (1)')
    p.add_argument('--root-feather-px',type=float,default=9,
                   help='Fade tongue emergence at anchored root (1..32 pixels)')
    p.add_argument('--shadow-strength',type=float,default=.12,
                   help='Subtle mouth/tongue contact shadow intensity in [0,1]')
    return p


def validate_args(args):
    from src.utils.animal_mouth_v066 import _unit_float
    if args.init_tongue:
        raise ValueError('V0.6.6 reuses the V0.6.5 verified polygons; run old --init-tongue instead')
    values=previous.validate_args(args)
    for key in ('donor_strength','root_depth','shadow_strength'):
        _unit_float(getattr(args,key),key)
    if not isinstance(args.dilate_px,int) or args.dilate_px<1 or args.dilate_px>31 or args.dilate_px%2!=1:
        raise ValueError('--dilate-px must be an odd integer 1..31')
    if not isinstance(args.inpaint_radius,int) or not 1<=args.inpaint_radius<=15:
        raise ValueError('--inpaint-radius must be 1..15')
    if not isinstance(args.root_feather_px,(float,int)) or not math.isfinite(args.root_feather_px) or not 1<=args.root_feather_px<=32:
        raise ValueError('--root-feather-px must be 1..32')
    return values


def planned_paths(args,opens,nframes):
    out=args.output_dir
    paths=[out/'report.json',out/'comparison_sheet.png']
    paths.extend(out/f'cavity_ref_{i:02d}.png' for i in range(nframes))
    paths.extend(out/f'donor_confidence_{i:02d}.png' for i in range(nframes))
    for a in opens:
        n=round(a*1000)
        paths.extend([out/f'mouth_v065_baseline_{n:03d}.png',
                      out/f'mouth_v066_cavity_{n:03d}.png',
                      out/f'mouth_v066_rooted_{n:03d}.png',
                      out/f'tongue_root_alpha_{n:03d}.png'])
    if args.holdout:paths.append(out/'holdout_aligned.png')
    return paths


def run(args, opens, tongue_values, tongue_source, mids, holdout):
    import numpy as np
    from PIL import Image
    from src.utils import animal_mouth_v065 as old
    from src.utils import animal_mouth_v066 as new
    from src.utils import animal_mouth_multiref as multi
    c,frames,oldmesh,roi,names,hashes,notes,inputs=previous._read_sources(args,mids)
    output_files=planned_paths(args,opens,len(frames))
    protected={p.resolve() for p in inputs+[args.landmarks,args.tongue_manifest]+
               ([args.mid_points] if args.mid_points else [])+
               ([holdout[1]] if holdout else [])}
    for path in output_files:
        if path.resolve() in protected:
            raise ValueError('Refusing to overwrite source or annotation: '+str(path))
        if path.exists() and not args.overwrite:
            raise FileExistsError('Output exists: {} (use --overwrite or new directory)'.format(path))
    positions,ratios=old.calibrate_openings(frames,names,mode=args.opening_mode)
    manifest=json.loads(args.tongue_manifest.read_text(encoding='utf-8'))
    polygons=old.validate_tongue_manifest(manifest,frames,hashes,roi)
    old_cavity,masks=old.prepare_cavity_frames(frames,polygons,
                               dilate_px=args.dilate_px,inpaint_radius=args.inpaint_radius)
    new_cavity,new_masks,confidence,cavity_report=new.prepare_multiref_cavity(
        frames,polygons,roi,names,dilate_px=args.dilate_px,
        inpaint_radius=args.inpaint_radius,donor_strength=args.donor_strength)
    old_mesh=multi.prepare_meshes(old_cavity,roi)
    new_mesh=multi.prepare_meshes(new_cavity,roi)
    rows={'v065_baseline':[],'v066_cavity_only':[],'v066_rooted_tongue':[]}
    frame_audits=[]
    tongue_alphas=[]
    for opening,q in zip(opens,tongue_values):
        baseline,_,baseline_audit=old.render_independent(
            c,frames,old_cavity,old_mesh,polygons,masks,opening,q,positions,roi,names,
            feather_px=args.feather_px,cavity_method=args.cavity_method)
        jaw,result,alpha,audit=new.render_rooted(
            c,frames,new_cavity,new_mesh,new_masks,opening,q,positions,roi,names,
            feather_px=args.feather_px,cavity_method=args.cavity_method,
            root_depth=args.root_depth,root_feather_px=args.root_feather_px,
            shadow_strength=args.shadow_strength)
        rows['v065_baseline'].append(baseline)
        rows['v066_cavity_only'].append(jaw)
        rows['v066_rooted_tongue'].append(result)
        tongue_alphas.append(alpha)
        audit['v065_baseline_outside_changed']=baseline_audit['outside_roi_changed_pixels']
        frame_audits.append(audit)
    continuity=old.continuity_report(rows['v066_rooted_tongue'],opens,tongue_values,roi,
        max_opening_step=args.max_opening_step,max_tongue_step=args.max_tongue_step,
        max_roi_mae=args.max_roi_mae)
    if args.strict_continuity and not continuity['all_pass']:
        raise ValueError('Continuity threshold failed; no V0.6.6 output files written')
    hold_eval=None;hold_rgb=None
    if holdout:
        ho,hp=holdout
        if previous.old.sha(hp) in hashes.values():
            raise ValueError('Holdout content duplicates a reference image')
        if args.mid_aligned:
            hold_rgb=previous.old._read_reference(hp);info={'mode':'already_aligned_512'}
        else:
            hold_rgb,info=previous.old.load_mid_raw(hp,c,alignment=args.alignment,
                                                     max_shift=args.max_shift)
        hq=old.default_tongue_amount(ho) if args.holdout_tongue is None else args.holdout_tongue
        baseline=old.render_independent(c,frames,old_cavity,old_mesh,polygons,masks,
                       ho,hq,positions,roi,names,feather_px=args.feather_px,
                       cavity_method=args.cavity_method)[0]
        prediction=new.render_rooted(c,frames,new_cavity,new_mesh,new_masks,
                    ho,hq,positions,roi,names,feather_px=args.feather_px,
                    cavity_method=args.cavity_method,root_depth=args.root_depth,
                    root_feather_px=args.root_feather_px,
                    shadow_strength=args.shadow_strength)[1]
        hold_eval={'external_opening':ho,'tongue_amount':hq,'image_path':str(hp),
                   'sha256':previous.old.sha(hp),'alignment':info,
                   'roi_mae':{'v065_baseline':multi.region_mae(baseline,hold_rgb,roi),
                              'v066_rooted_tongue':multi.region_mae(prediction,hold_rgb,roi)},
                   'note':'Unseen, independent holdout only; MAE is not a realism score'}
    # Complete all analysis before creating files. A failure in geometry or
    # continuity should not leave a partially successful output directory.
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for i,(f,co) in enumerate(zip(new_cavity,confidence)):
        previous._save_rgb(args.output_dir/f'cavity_ref_{i:02d}.png',f['rgb'])
        Image.fromarray(np.rint(co*255).astype(np.uint8),'L').save(
            args.output_dir/f'donor_confidence_{i:02d}.png')
    for i,a in enumerate(opens):
        n=round(a*1000)
        for row,stem in (('v065_baseline','mouth_v065_baseline'),
                         ('v066_cavity_only','mouth_v066_cavity'),
                         ('v066_rooted_tongue','mouth_v066_rooted')):
            previous._save_rgb(args.output_dir/f'{stem}_{n:03d}.png',rows[row][i])
        Image.fromarray(np.rint(tongue_alphas[i]*255).astype(np.uint8),'L').save(
            args.output_dir/f'tongue_root_alpha_{n:03d}.png')
    previous._sheet(rows,opens).save(args.output_dir/'comparison_sheet.png')
    if hold_rgb is not None:previous._save_rgb(args.output_dir/'holdout_aligned.png',hold_rgb)
    peak=int(np.argmax(ratios))
    report={'schema_version':1,'stage':'v0.6.6_cavity_root',
        'same_dog_asserted':True,'opening_mode':args.opening_mode,
        'reference_alphas_time':[f['alpha'] for f in frames],
        'reference_openings_normalized':positions,
        'reference_aperture_width_ratios':ratios,
        'peak_index':peak,'opening_branch_ends_at':frames[peak]['alpha'],
        'tongue_manifest_sha256':previous.old.sha(args.tongue_manifest),
        'landmark_sha256':previous.old.sha(args.landmarks),
        'midpoint_sha256':previous.old.sha(args.mid_points) if args.mid_points else None,
        'alignment':notes,'mouth_roi':roi,'mouth_feather_px':args.feather_px,
        'targets':{'opening':list(opens),'tongue':list(tongue_values),
                   'tongue_source':tongue_source},
        'cavity':{'method':'multi_reference_donor_plus_telea_fallback',
                  'donor_strength':args.donor_strength,'inpaint_radius':args.inpaint_radius,
                  'dilate_px':args.dilate_px,'references':cavity_report,
                  'not_actual_occluded_ground_truth':True},
        'tongue':{'root_depth':args.root_depth,'root_feather_px':args.root_feather_px,
                  'shadow_strength':args.shadow_strength},
        'frames':frame_audits,'continuity':continuity,
        'outside_roi_changed_all_zero':all(v['outside_roi_changed_pixels']==0 for v in frame_audits),
        'adjacent_roi_mae':{key:[multi.region_mae(x,y,roi) for x,y in zip(seq,seq[1:])]
                            for key,seq in rows.items()},
        'holdout_evaluation':hold_eval,
        'limitations':['Multi-frame donor pixels are 2-D proxies, may carry wrong teeth/skin or incorrect occlusion',
                       'Telea fallback remains where no unoccluded mouth interior is available',
                       'Tongue root is estimated from polygon top, not a verified anatomical anchor',
                       'Tongue overlay remains a 2-D warp with approximate occlusion/shadow',
                       'Low discontinuity / ROI MAE does not establish natural animal motion',
                       'No audio, learned motion, re-training or ONNX graph changes']}
    (args.output_dir/'report.json').write_text(
        json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'stage':report['stage'],'output_dir':str(args.output_dir),
        'comparison_sheet':str(args.output_dir/'comparison_sheet.png'),
        'reference_openings_normalized':positions,
        'cavity_donor_coverage':[r['donor_coverage_fraction'] for r in cavity_report],
        'continuity_all_pass':continuity['all_pass'],
        'outside_roi_changed_all_zero':report['outside_roi_changed_all_zero'],
        'holdout_roi_mae':hold_eval['roi_mae'] if hold_eval else None},
        indent=2,ensure_ascii=False,allow_nan=False))
    return report


def main(argv=None):
    args=parser().parse_args(argv)
    vals=validate_args(args)
    if args.plan:
        print(json.dumps({'stage':'v0.6.6_cavity_root','no_models_loaded':True,
         'reference_dir':str(args.reference_dir),
         'mouth_landmarks_exists':bool(args.landmarks and args.landmarks.is_file()),
         'mid_landmarks_exists':bool(args.mid_points and args.mid_points.is_file()),
         'tongue_manifest_exists':bool(args.tongue_manifest and args.tongue_manifest.is_file()),
         'new_controls':{'donor_strength':args.donor_strength,
                         'root_depth':args.root_depth,
                         'root_feather_px':args.root_feather_px,
                         'shadow_strength':args.shadow_strength}},indent=2,ensure_ascii=False))
        return 0
    run(args,*vals)
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except (OSError,ValueError,RuntimeError,KeyError,ImportError) as exc:
        print('[ERROR]',exc,file=sys.stderr);sys.exit(2)
