# coding: utf-8
"""V0.6.5 CPU-only geometric jaw opening and independently masked tongue texture.

No semantic segmentation or 3-D articulation is inferred. All tongue polygons are
human-confirmed on the *aligned* 512px photos. The cavity below each polygon is
approximated by classical inpainting; occluded real mouth detail is unknowable.
"""

import math
import numpy as np
from . import animal_mouth_geometry as geo
from . import animal_mouth_multiref as multi
from .animal_dual_source import mouth_mask

SIZE = geo.SIZE


def jaw_aperture(points, names):
    """Perpendicular upper-to-lower lip separation / corner-to-corner mouth width.

    Dimensionless 2D proxy, NOT a calibrated physical jaw angle.
    """
    if not all(n in names for n in ('left_corner', 'right_corner', 'upper_lip', 'lower_lip')):
        raise ValueError('Missing jaw landmarks')
    pts = np.asarray(points, np.float64)
    if pts.shape != (len(names), 2) or not np.all(np.isfinite(pts)):
        raise ValueError('Invalid jaw point shape or values')
    d = dict(zip(names, pts))
    corners = d['right_corner'] - d['left_corner']
    width = float(np.linalg.norm(corners))
    if width < 6:
        raise ValueError('Mouth corners too close for opening measurement')
    # Orient normal downward in image coordinates, robust to mild head roll.
    normal = np.asarray([-corners[1], corners[0]], np.float64) / width
    if normal[1] < 0:
        normal *= -1
    opening = float(np.dot(d['lower_lip'] - d['upper_lip'], normal) / width)
    if opening <= 0 or not math.isfinite(opening):
        raise ValueError('Lower lip must lie below upper lip along jaw axis')
    return opening


def calibrate_openings(frames, names, *, mode='geometry'):
    """Calibrate jaw opening from manual landmarks.

    geometry: strict time-ordered monotone opening, unchanged from V0.6.5.
    geometry_peak: use only the *increasing* branch through the first maximum
    as 0..1 opening. Later closing frames remain in the source data but are
    never used to infer increased opening. This is not a closing animation.
    time: diagnostic chronological control, not measured jaw opening.
    """
    if mode not in ('geometry', 'geometry_peak', 'time'):
        raise ValueError('opening mode must be geometry, geometry_peak or time')
    ratios = [jaw_aperture(f['points'], names) for f in frames]
    if mode == 'time':
        return [float(f['alpha']) for f in frames], ratios
    if mode == 'geometry':
        span = ratios[-1] - ratios[0]
        if span <= 0.02:
            raise ValueError('Insufficient measured opening change; check mouth points')
        positions = [(x-ratios[0]) / span for x in ratios]
        if any(y-x < 0.005 for x,y in zip(positions,positions[1:])):
            raise ValueError('Opening measures non-monotone or too close: check verified points, or use --opening-mode time for DIAGNOSTIC only')
        return [float(np.clip(x,0,1)) for x in positions], ratios

    # Explicit opt-in. This may shorten the opening trajectory: the reference
    # image at chronological alpha=1 is NOT necessarily the maximum aperture.
    peak = int(np.argmax(ratios))
    span = ratios[peak] - ratios[0]
    if peak == 0 or span <= 0.02:
        raise ValueError('geometry_peak needs a measured opening peak later than the first frame')
    positions = [(x-ratios[0]) / span for x in ratios]
    if any(b-a < 0.005 for a,b in zip(positions[:peak], positions[1:peak+1])):
        raise ValueError('geometry_peak opening branch is non-monotone or too close; check landmarks')
    if any(b-a > 0.005 for a,b in zip(positions[peak:-1], positions[peak+1:])):
        raise ValueError('geometry_peak has a second opening rise after the peak; split the motion into phases')
    if any(x < -0.005 for x in positions):
        raise ValueError('geometry_peak contains a closing state below the first reference; use a separate phase')
    return [float(x) for x in positions], ratios


def opening_to_time(opening, positions, frames):
    """Invert an opening-only reference curve; excludes post-peak closure.

    Strict geometry/time have monotone positions and behave as before.
    geometry_peak positions end below their maximum; map only through the
    maximum and never silently sort/reference a closing photo as open=1.
    """
    if not isinstance(opening, (int,float)) or isinstance(opening,bool) or not math.isfinite(opening) or not 0<=opening<=1:
        raise ValueError('Opening must be finite in [0,1]')
    if len(positions)!=len(frames) or not positions:
        raise ValueError('Position / reference mismatch')
    pos=np.asarray(positions, dtype=np.float64)
    if not np.all(np.isfinite(pos)):
        raise ValueError('Opening positions must be finite')
    peak=int(np.argmax(pos))
    if abs(pos[0]) > 1e-6 or abs(pos[peak]-1) > 1e-6:
        raise ValueError('Opening positions must start at 0 and reach 1 at the peak')
    if np.any(np.diff(pos[:peak+1]) <= 0):
        raise ValueError('Opening branch must be strictly increasing')
    if np.any(np.diff(pos[peak:]) > 0):
        raise ValueError('Post-peak reference positions cannot rise; use separate phases')
    return float(np.interp(float(opening), pos[:peak+1],
                           [f['alpha'] for f in frames[:peak+1]]))


def default_tongue_amount(opening):
    """Explicitly heuristic demonstration curve with delayed tongue emergence."""
    q = float(np.clip((opening - 0.25) / 0.75, 0, 1))
    return q*q*(3-2*q)


def validate_tongue_manifest(spec, frames, source_hashes, roi):
    """Validate all human-verified tongue polygons and raw-file fingerprints.

    Empty polygon [] is a human-confirmed absence, not an automatic detection.
    """
    if not isinstance(spec,dict) or spec.get('schema_version')!=1 or spec.get('verified') is not True:
        raise ValueError('Tongue mask manifest must be v1 verified by human clicks')
    if spec.get('coordinate_system')!='normalized_xy_on_512px_aligned_reference':
        raise ValueError('Wrong tongue polygon coordinate system')
    if spec.get('source_sha256')!=source_hashes:
        raise ValueError('Tongue annotation source hashes mismatch; regenerate on these exact inputs')
    raw = spec.get('polygons')
    if not isinstance(raw,dict) or len(raw)!=len(frames):
        raise ValueError('Tongue polygons must cover every reference frame')
    result={}
    x0,y0,x1,y1=geo.validate_roi(roi)
    for f in frames:
        alpha=f['alpha']
        keys=[k for k in raw if _close_key(k, alpha)]
        if len(keys)!=1:
            raise ValueError('Exactly one tongue polygon required for alpha {}'.format(alpha))
        points=raw[keys[0]]
        if not isinstance(points,list) or (points and not 3 <= len(points) <= 60):
            raise ValueError('Tongue polygon must be [] or 3..60 vertices')
        pixels=[]
        for xy in points:
            if not isinstance(xy,list) or len(xy)!=2 or any(
                not isinstance(v,(float,int)) or isinstance(v,bool) or not math.isfinite(v) for v in xy
            ):
                raise ValueError('Tongue polygon must contain finite normalized XY')
            x,y=map(float,xy)
            if not (x0 < x < x1 and y0 < y < y1):
                raise ValueError('Tongue polygon outside mouth ROI')
            pixels.append([x*SIZE,y*SIZE])
        if pixels:
            import cv2
            poly=np.asarray(pixels,dtype=np.float32)
            area=float(cv2.contourArea(poly))
            roi_area=(x1-x0)*(y1-y0)*SIZE*SIZE
            if not 20 < area < .8*roi_area:
                raise ValueError('Tongue polygon area too small or too large')
        result[alpha]=np.asarray(pixels,np.float32).reshape(-1,2)
    if not any(len(p)>0 for p in result.values()):
        raise ValueError('All tongue masks empty: cannot demonstrate independent tongue control')
    return result


def _close_key(raw,alpha):
    try:
        return abs(float(raw)-float(alpha))<1e-6
    except (TypeError,ValueError):
        return False


def polygon_mask(poly):
    import cv2
    mask=np.zeros((SIZE,SIZE),np.uint8)
    if len(poly):
        cv2.fillPoly(mask,[np.rint(poly).astype(np.int32)],255)
    return mask


def prepare_cavity_frames(frames,polygons,*,dilate_px=5,inpaint_radius=7):
    """Remove visible tongue using marked masks; inpainted cavity is a proxy."""
    import cv2
    if dilate_px<1 or dilate_px%2!=1 or not 1<=inpaint_radius<=15:
        raise ValueError('Invalid cavity inpaint settings')
    cavity=[]
    masks=[]
    kernel=np.ones((dilate_px,dilate_px),np.uint8)
    for f in frames:
        m=polygon_mask(polygons[f['alpha']])
        if np.any(m):
            rm=cv2.dilate(m,kernel,iterations=1)
            rgb=cv2.inpaint(f['rgb'],rm,inpaint_radius,cv2.INPAINT_TELEA)
        else:
            rgb=f['rgb'].copy()
        cavity.append({'alpha':f['alpha'],'rgb':np.ascontiguousarray(rgb),'points':f['points']})
        masks.append(m)
    return cavity,masks


def _tongue_render(source, source_mask, src_points, target_points, names, amount, roi):
    """Affine jaw-relative tongue projection with independent extension amount.

    Explicit mask controls visibility. Premultiplied RGB avoids black polygon
    edges; still only an image-space proxy, not tongue biomechanics.
    """
    import cv2
    if amount <= 0:
        return np.zeros((SIZE,SIZE,3),np.uint8), np.zeros((SIZE,SIZE),np.float32)
    yy,xx=np.where(source_mask>0)
    if len(xx)<20:
        raise ValueError('Selected source tongue polygon too small')
    top=int(yy.min())
    near=yy <= top+8
    sx=float(np.median(xx[near]))
    sy=float(top)
    dsrc=dict(zip(names,np.asarray(src_points,np.float64)))
    ddst=dict(zip(names,np.asarray(target_points,np.float64)))
    sw=float(np.linalg.norm(dsrc['right_corner']-dsrc['left_corner']))
    tw=float(np.linalg.norm(ddst['right_corner']-ddst['left_corner']))
    xratio=float(np.clip(tw/sw,0.65,1.5))
    sopen=jaw_aperture(src_points,names)
    topen=jaw_aperture(target_points,names)
    yratio=float(np.clip(topen/sopen,.5,1.5))
    xscale=xratio*(.5+.5*math.sqrt(float(amount)))
    yscale=yratio*float(amount)
    tx=float(ddst['upper_lip'][0]+(sx-dsrc['upper_lip'][0])*xratio)
    ty=float(ddst['upper_lip'][1]+(sy-dsrc['upper_lip'][1])*yratio)
    matrix=np.asarray([[xscale,0,tx-xscale*sx],[0,yscale,ty-yscale*sy]],np.float32)
    # Premultiply the source tongue RGB by a softly feathered segmentation mask.
    src_alpha=cv2.GaussianBlur(source_mask.astype(np.float32)/255,(0,0),1.35)
    premul=source.astype(np.float32)*src_alpha[:,:,None]
    warped=cv2.warpAffine(premul,matrix,(SIZE,SIZE),flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    wm=cv2.warpAffine(src_alpha,matrix,(SIZE,SIZE),flags=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    outer=mouth_mask(SIZE,SIZE,roi,8)
    wm=np.clip(wm*outer,0,1)
    tongue=np.clip(warped/np.maximum(wm[:,:,None]/np.maximum(outer[:,:,None],1e-6),1e-4),0,255)
    # Lower amounts are allowed but reduced opacity only near tongue emergence.
    if amount < .1:
        v=amount/.1
        wm*=v*v*(3-2*v)
    return np.ascontiguousarray(np.rint(tongue).astype(np.uint8)),wm.astype(np.float32)


def render_independent(closed,reference_frames,cavity_frames,cavity_meshes,polygons,masks,
                       opening,tongue_amount,positions,roi,names,*,feather_px=24,
                       cavity_method='layered_flow'):
    """Return (RGB512, tongue_alpha_mask, audit). Jaw/tongue controls are separate."""
    if not isinstance(tongue_amount,(int,float)) or isinstance(tongue_amount,bool) or not math.isfinite(tongue_amount) or not 0<=tongue_amount<=1:
        raise ValueError('tongue_amount must be finite in [0,1]')
    if cavity_method not in multi.METHODS:
        raise ValueError('Unknown cavity method')
    geo.check_image(closed,'closed')
    t=opening_to_time(opening,positions,reference_frames)
    base,base_audit=multi.render_layered(closed,cavity_frames,t,roi,names,cavity_meshes,
                                         method=cavity_method,feather_px=feather_px)
    srcidx=int(np.argmax([np.count_nonzero(m) for m in masks]))
    src=reference_frames[srcidx]
    target=multi.target_control_points(reference_frames,t)
    tongue,tm=_tongue_render(src['rgb'],masks[srcidx],src['points'],target,names,tongue_amount,roi)
    out=np.clip(np.rint(base.astype(np.float32)*(1-tm[:,:,None])+
                        tongue.astype(np.float32)*tm[:,:,None]),0,255).astype(np.uint8)
    outside=mouth_mask(SIZE,SIZE,roi,feather_px)==0
    out[outside]=closed[outside]
    audit={'opening':float(opening),'mapped_time_alpha':t,
           'tongue_amount':float(tongue_amount),'source_tongue_alpha':src['alpha'],
           'jaw_ratio':jaw_aperture(target,names), 'base':base_audit,
           'outside_roi_changed_pixels':multi.outside_changed(closed,out,roi),
           'tongue_mask_nonzero_pixels':int(np.count_nonzero(tm>.05))}
    return np.ascontiguousarray(out),tm,audit


def continuity_report(outputs, openings, tongue_values, roi, *, max_opening_step=.3,
                      max_tongue_step=.5, max_roi_mae=32):
    """Audit hard control-rate and visual pixel-change limits, not realism."""
    for x in (max_opening_step,max_tongue_step,max_roi_mae):
        if not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x) or x<=0:
            raise ValueError('Continuity limits must be positive finite')
    if len(outputs)!=len(openings) or len(openings)!=len(tongue_values):
        raise ValueError('Continuity sequence lengths mismatch')
    steps=[]
    for i in range(1,len(outputs)):
        do=float(abs(openings[i]-openings[i-1]))
        dt=float(abs(tongue_values[i]-tongue_values[i-1]))
        delta=float(multi.region_mae(outputs[i-1],outputs[i],roi))
        steps.append({'from':i-1,'to':i,'opening_step':do,'tongue_step':dt,
                      'mouth_roi_mae':delta,
                      'pass':bool(do<=max_opening_step+1e-8 and dt<=max_tongue_step+1e-8 and delta<=max_roi_mae)})
    return {'limits':{'max_opening_step':max_opening_step,'max_tongue_step':max_tongue_step,
                      'max_roi_mae':max_roi_mae},
            'all_pass':all(s['pass'] for s in steps),'steps':steps,
            'note':'Pixel-change limit detects jumps, but low change does not prove natural motion'}
