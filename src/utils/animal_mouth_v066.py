# coding: utf-8
"""V0.6.6: donor-aware cavity approximation and tongue-root-anchored 2-D overlay.

Experimental CPU image processing, not anatomical reconstruction. The verified
V0.6.5 tongue masks and V0.6.3/4 landmarks are the only semantic inputs.
Existing F/M/W/G and all V0.6.5 functions remain untouched.
"""
import math
import numpy as np
from . import animal_mouth_geometry as geo
from . import animal_mouth_multiref as multi
from . import animal_mouth_v065 as old
from .animal_dual_source import mouth_mask

SIZE = geo.SIZE


def _unit_float(value, name, *, inclusive_zero=True):
    if (not isinstance(value, (float, int)) or isinstance(value, bool)
            or not math.isfinite(value) or value > 1 or
            (value < 0 if inclusive_zero else value <= 0)):
        raise ValueError(f'{name} must be finite in '+('[0,1]' if inclusive_zero else '(0,1]'))
    return float(value)


def prepare_multiref_cavity(frames, polygons, roi, names, *, dilate_px=5,
                            inpaint_radius=7, donor_strength=0.9):
    """Build tongue-removed cavity proxies from verified multi-frame visibility.

    For each missing tongue pixel, search warped other frames for pixels that
    are BOTH tongue-unmasked and inside the donor/target landmark cavity hull.
    Missing evidence falls back to Telea. No region outside the target's
    original tongue mask is edited. Report donor coverage explicitly.
    """
    import cv2
    multi.validate_reference_frames(frames, roi)
    _unit_float(donor_strength, 'donor_strength')
    if not isinstance(dilate_px,int) or dilate_px < 1 or dilate_px > 31 or dilate_px%2 != 1:
        raise ValueError('dilate_px must be odd 1..31')
    if not isinstance(inpaint_radius,int) or not 1 <= inpaint_radius <= 15:
        raise ValueError('inpaint_radius must be integer 1..15')
    base, masks = old.prepare_cavity_frames(frames, polygons, dilate_px=dilate_px,
                                             inpaint_radius=inpaint_radius)
    inner = [multi._mouth_interior_mask(roi, f['points'], names) for f in frames]
    enhanced=[]; confidence_images=[]; records=[]
    for i, (target, fallback, mask) in enumerate(zip(frames, base, masks)):
        target_rgb=geo.check_image(target['rgb'],'reference')
        missing = cv2.dilate(mask, np.ones((dilate_px,dilate_px),np.uint8)) > 0
        restored=fallback['rgb'].astype(np.float32).copy()
        conf=np.zeros((SIZE,SIZE),np.float32)
        attempted=[]
        if np.any(missing) and donor_strength > 0:
            accumulated=np.zeros((SIZE,SIZE,3),np.float32)
            weight_sum=np.zeros((SIZE,SIZE),np.float32)
            for j, donor in enumerate(frames):
                if i==j:continue
                try:
                    # Pair-wise fold guard, not the chronological mesh; the
                    # donor and target need not be adjacent reference frames.
                    src,dst,tri=multi.pair_mesh(donor['points'],target['points'],roi)
                except ValueError as exc:
                    attempted.append({'index':j,'status':'geometry_rejected','reason':str(exc)})
                    continue
                donor_rgb=geo._warp_with_mesh(donor['rgb'],src,dst,tri)
                donor_visibility=((masks[j]==0).astype(np.uint8)*255)
                donor_interior=np.rint(np.clip(inner[j],0,1)*255).astype(np.uint8)
                visible=geo._warp_with_mesh(donor_visibility,src,dst,tri).astype(np.float32)/255
                cavity=geo._warp_with_mesh(donor_interior,src,dst,tri).astype(np.float32)/255
                valid=(np.clip(visible,0,1)*np.clip(cavity,0,1)*inner[i])
                valid=np.where(missing,valid,0).astype(np.float32)
                jaw_distance=abs(old.jaw_aperture(target['points'],names)-
                                 old.jaw_aperture(donor['points'],names))
                factor=float(np.exp(-1.5*jaw_distance))
                weight=valid*factor
                accumulated+=donor_rgb.astype(np.float32)*weight[:,:,None]
                weight_sum+=weight
                attempted.append({'index':j,'status':'considered',
                                  'valid_target_pixels':int(np.count_nonzero(weight>0.1)),
                                  'jaw_aperture_distance':float(jaw_distance)})
            valid=weight_sum>1e-4
            if np.any(valid):
                color=np.divide(accumulated,np.maximum(weight_sum[:,:,None],1e-5))
                # Feather donor availability but never paint outside missing.
                blurred=cv2.GaussianBlur(np.clip(weight_sum,0,1),(0,0),1.8)
                conf=np.where(missing,np.clip(blurred*donor_strength,0,1),0).astype(np.float32)
                restored=restored*(1-conf[:,:,None])+color*conf[:,:,None]
        out=np.clip(np.rint(restored),0,255).astype(np.uint8)
        # Conservation invariant: only the originally hidden tongue region
        # and its dilation may change, never arbitrary mouth/face pixels.
        out[~missing]=target_rgb[~missing]
        enhanced.append({'alpha':target['alpha'],'rgb':np.ascontiguousarray(out),
                         'points':target['points']})
        confidence_images.append(np.ascontiguousarray(conf))
        nmissing=int(np.count_nonzero(missing))
        records.append({'alpha':float(target['alpha']),
                        'tongue_mask_pixels':int(np.count_nonzero(mask)),
                        'repair_pixels':nmissing,
                        'donor_confidence_mean_in_repair':float(conf[missing].mean()) if nmissing else 0.,
                        'donor_coverage_fraction':float(np.count_nonzero(conf[missing]>0.05)/nmissing) if nmissing else 0.,
                        'telea_fallback_fraction_estimate':float(1-conf[missing].mean()) if nmissing else 0.,
                        'candidate_donors':attempted})
    return enhanced,masks,confidence_images,records


def _root_and_overlay(source, source_mask, src_points, target_points, names,
                      amount, roi, *, root_depth=0.58, root_feather_px=9):
    """Keep source tongue root fixed in jaw-relative coordinates as tip extends.

    Root is estimated from TOP of human polygon, not an anatomical landmark.
    User may tune root_depth. A vertical alpha fade makes tongue emerge behind
    upper mouth boundary instead of looking like an opaque rectangular sticker.
    """
    import cv2
    _unit_float(amount,'tongue_amount')
    _unit_float(root_depth,'root_depth')
    if not isinstance(root_feather_px,(float,int)) or isinstance(root_feather_px,bool) or not math.isfinite(root_feather_px) or not 1<=root_feather_px<=32:
        raise ValueError('root_feather_px must be 1..32')
    if amount == 0:
        return np.zeros((SIZE,SIZE,3),np.uint8),np.zeros((SIZE,SIZE),np.float32),None
    src=geo.check_image(source,'tongue source')
    srcmask=np.asarray(source_mask)
    if srcmask.shape!=(SIZE,SIZE) or srcmask.dtype!=np.uint8:
        raise ValueError('source tongue mask must be uint8 512x512')
    y,x=np.where(srcmask>127)
    if len(x)<20:raise ValueError('Source tongue polygon has too few visible pixels')
    sy=float(np.quantile(y,.01))
    sx=float(np.median(x[y<=sy+8]))
    if not np.isfinite(sx):raise ValueError('Cannot derive source tongue root')
    sdict=dict(zip(names,np.asarray(src_points,np.float64)))
    tdict=dict(zip(names,np.asarray(target_points,np.float64)))
    sc=np.linalg.norm(sdict['right_corner']-sdict['left_corner'])
    tc=np.linalg.norm(tdict['right_corner']-tdict['left_corner'])
    if sc<6 or tc<6:raise ValueError('Mouth reference width too narrow')
    scale_x=float(np.clip(tc/sc,.65,1.5))
    scale_y=float(np.clip(old.jaw_aperture(target_points,names)/old.jaw_aperture(src_points,names),.5,1.5))
    sa=(1-root_depth)*sdict['upper_lip']+root_depth*sdict['lower_lip']
    ta=(1-root_depth)*tdict['upper_lip']+root_depth*tdict['lower_lip']
    # Retain source polygon-root offset relative to mouth geometry; so at
    # amount=1 and same target pose its root stays exactly at the source root.
    root=np.asarray([ta[0]+(sx-sa[0])*scale_x,ta[1]+(sy-sa[1])*scale_y])
    stretch_x=scale_x*(.5+.5*math.sqrt(amount))
    stretch_y=scale_y*amount
    matrix=np.asarray([[stretch_x,0,root[0]-sx*stretch_x],
                       [0,stretch_y,root[1]-sy*stretch_y]],np.float32)
    a=cv2.GaussianBlur(srcmask.astype(np.float32)/255,(0,0),1.2)
    premul=src.astype(np.float32)*a[:,:,None]
    warped=cv2.warpAffine(premul,matrix,(SIZE,SIZE),cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    alpha=cv2.warpAffine(a,matrix,(SIZE,SIZE),cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    result=np.clip(warped/np.maximum(alpha[:,:,None],1e-5),0,255).astype(np.uint8)
    # Fade just at anchored root. Do not fade the whole tongue with amount.
    yy=np.arange(SIZE,dtype=np.float32)[:,None]
    fade=np.clip((yy-float(root[1])+2)/float(root_feather_px),0,1)
    fade=fade*fade*(3-2*fade)
    outer=mouth_mask(SIZE,SIZE,roi,8)
    alpha=np.clip(alpha*fade*outer,0,1).astype(np.float32)
    if amount<.10:
        p=amount/.10
        alpha*=p*p*(3-2*p)
    return np.ascontiguousarray(result),alpha, {
            'root_xy':[float(root[0]),float(root[1])],
            'estimated_source_root_xy':[sx,sy],
            'root_depth':float(root_depth),
            'root_feather_px':float(root_feather_px),
            'tongue_visible_pixels':int(np.count_nonzero(alpha>.05))}


def render_rooted(closed, frames, cavity_frames, cavity_meshes, masks,
                  opening, tongue_amount, positions, roi, names, *,
                  feather_px=24, cavity_method='layered_flow',
                  root_depth=.58, root_feather_px=9, shadow_strength=.12):
    """Return jaw-only, jaw+tongue, tongue alpha, and diagnostics."""
    import cv2
    _unit_float(tongue_amount,'tongue_amount')
    _unit_float(shadow_strength,'shadow_strength')
    t=old.opening_to_time(opening,positions,frames)
    jaw,audit=multi.render_layered(closed,cavity_frames,t,roi,names,cavity_meshes,
                                   method=cavity_method,feather_px=feather_px)
    source_idx=int(np.argmax([np.count_nonzero(m) for m in masks]))
    src=frames[source_idx]
    target=multi.target_control_points(frames,t)
    rgb,ta,rootinfo=_root_and_overlay(src['rgb'],masks[source_idx],src['points'],
                                      target,names,tongue_amount,roi,
                                      root_depth=root_depth,root_feather_px=root_feather_px)
    dst=jaw.astype(np.float32)
    if np.any(ta):
        # Subtle contact shadow at tongue boundary. Strictly clipped to support
        # and image-space only; not a physical shadow or occlusion solution.
        blur=cv2.GaussianBlur(ta,(0,0),3.0)
        contact=np.clip(blur-ta,0,1)*float(shadow_strength)
        dst*=1-contact[:,:,None]
        dst=dst*(1-ta[:,:,None])+rgb.astype(np.float32)*ta[:,:,None]
    out=np.clip(np.rint(dst),0,255).astype(np.uint8)
    mask=mouth_mask(SIZE,SIZE,roi,feather_px)
    out[mask==0]=closed[mask==0]
    if multi.outside_changed(closed,out,roi):
        raise AssertionError('V0.6.6 output changed pixels outside mouth ROI')
    return jaw,np.ascontiguousarray(out),ta,{'opening':float(opening),
        'mapped_time_alpha':float(t),'tongue_amount':float(tongue_amount),
        'tongue_source_alpha':float(src['alpha']),
        'tongue_root':rootinfo,'base':audit,
        'outside_roi_changed_pixels':multi.outside_changed(closed,out,roi)}
