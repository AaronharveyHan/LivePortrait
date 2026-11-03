# coding: utf-8
"""V0.6.4 CPU-only multi-reference, layered mouth appearance experiment.

Works on already aligned V0.6.2 reference images plus registered same-dog
intermediate frames. The 2D mesh is NOT anatomical or a 3D jaw model.

The outer mouth/muzzle uses geometric warping and smooth low-frequency color
mixing. The inner mouth selects sharp reference texture (nearest) or tries
local bidirectional optical flow (flow). Both can still produce artifacts.
"""

import math

import numpy as np

from . import animal_mouth_geometry as geo
from .animal_dual_source import mouth_mask

SIZE = geo.SIZE
METHODS = ('layered_nearest', 'layered_flow')


def parse_frame_spec(raw):
    """Parse `ALPHA=PATH` without splitting Windows drive letters."""
    from pathlib import Path
    if not isinstance(raw, str) or '=' not in raw:
        raise ValueError('Frame must be ALPHA=PATH (for example 0.333=frame_01.png)')
    a, path = raw.split('=', 1)
    try:
        alpha = float(a)
    except ValueError as exc:
        raise ValueError('Frame alpha must be a number') from exc
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('Intermediate frame alpha must be strictly inside (0,1)')
    if not path.strip():
        raise ValueError('Frame image path must not be empty')
    return alpha, Path(path.strip())


def validate_reference_alphas(alphas, *, max_refs=10):
    alphas = tuple(alphas)
    if not 2 <= len(alphas) <= max_refs:
        raise ValueError('Need 2..{} reference images'.format(max_refs))
    if any(not isinstance(a, (int, float)) or isinstance(a, bool) or not math.isfinite(a)
           or not 0 <= a <= 1 for a in alphas):
        raise ValueError('Reference alphas must be finite numbers in [0,1]')
    if alphas[0] != 0 or alphas[-1] != 1:
        raise ValueError('Reference alphas must include 0 and 1')
    if any(y-x < .025 for x,y in zip(alphas, alphas[1:])):
        raise ValueError('Reference alphas must be increasing and >=.025 apart')
    return tuple(float(a) for a in alphas)


def validate_target_alphas(alphas):
    a = tuple(alphas)
    if len(a) < 2 or len(a) > 21 or a[0] != 0 or a[-1] != 1:
        raise ValueError('Targets require 2..21 alphas from 0 to 1')
    if any(not isinstance(v, (float,int)) or not math.isfinite(v) or not 0 <= v <= 1 for v in a):
        raise ValueError('Bad target alpha')
    if any(y <= x for x,y in zip(a,a[1:])) or len({round(v*1000) for v in a}) != len(a):
        raise ValueError('Targets must strictly increase without 0.001 filename collisions')
    return a


def intermediate_proxy_points(alpha, closed, opened):
    """Illustrative *not actual* landmark positions at time alpha."""
    return closed*(1-alpha)+opened*alpha


def validate_manual_points(points, names, roi):
    """Optional human-confirmed middle state landmarks (normalized coordinates)."""
    if not isinstance(points, dict) or set(points) != set(names):
        raise ValueError('Middle frame points must have exact V0.6.3 landmark names')
    x0,y0,x1,y1 = geo.validate_roi(roi)
    result = []
    for name in names:
        xy = points[name]
        if not isinstance(xy, (tuple, list)) or len(xy) != 2 or any(
            not isinstance(v, (int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in xy
        ):
            raise ValueError('Invalid normalized coordinates for '+name)
        x,y = xy
        if not x0+.008 < x < x1-.008 or not y0+.008 < y < y1-.008:
            raise ValueError('Middle frame point outside ROI: '+name)
        result.append((x*SIZE, y*SIZE))
    arr = np.asarray(result,np.float64)
    # Avoid zero-length edges, which can lead to folded triangles.
    if len(arr)>1:
        d = np.linalg.norm(arr[:,None,:]-arr[None,:,:], axis=2)
        d[np.diag_indices(len(arr))] = 1e9
        if np.min(d) < 2:
            raise ValueError('Duplicated middle frame points')
    return arr


def validate_reference_frames(frames, roi):
    """Each ref: {'alpha': float, 'rgb': RGB512, 'points': Nx2 px}."""
    if not isinstance(frames, (list,tuple)):
        raise ValueError('Reference list missing')
    validate_reference_alphas([f['alpha'] for f in frames])
    count=None
    for f in frames:
        geo.check_image(f['rgb'],'reference frame')
        p=np.asarray(f['points'],dtype=np.float64)
        if p.ndim!=2 or p.shape[1]!=2 or (count is not None and p.shape[0]!=count):
            raise ValueError('Reference point count mismatch')
        if p.shape[0]<5 or p.shape[0]>12 or not np.all(np.isfinite(p)):
            raise ValueError('Bad reference control points')
        x0,y0,x1,y1 = roi
        if np.any(p[:,0] <= (x0+.008)*SIZE) or np.any(p[:,0] >= (x1-.008)*SIZE) or np.any(p[:,1] <= (y0+.008)*SIZE) or np.any(p[:,1] >= (y1-.008)*SIZE):
            raise ValueError('Reference point outside ROI')
        count=p.shape[0]
    return True


def pair_mesh(points_a, points_b, roi):
    """Build/fold-check one adjacent interval. Reject invalid geometry."""
    from scipy.spatial import Delaunay, QhullError
    anchors=geo.boundary_anchors(roi)
    a=np.vstack([anchors,np.asarray(points_a,dtype=np.float64)])
    b=np.vstack([anchors,np.asarray(points_b,dtype=np.float64)])
    if np.max(np.linalg.norm(a-b,axis=1))>SIZE*.30:
        raise ValueError('Adjacent reference displacement exceeds 30% of frame')
    try:
        tris=Delaunay((a+b)*.5).simplices
    except (ValueError,QhullError) as exc:
        raise ValueError('Cannot triangulate adjacent reference positions') from exc
    for tri in tris:
        signs=[]
        for p in (a,b):
            v=p[tri]
            ab=v[1]-v[0]
            ac=v[2]-v[0]
            area=float(ab[0]*ac[1]-ab[1]*ac[0])
            if abs(area)<.2:
                raise ValueError('Degenerate mouth triangle in multi-reference pair')
            signs.append(math.copysign(1,area))
        if signs[0] != signs[1]:
            raise ValueError('Adjacent mouth triangles fold: adjust middle-frame landmarks')
    return a,b,tris


def prepare_meshes(frames,roi):
    validate_reference_frames(frames,roi)
    return [pair_mesh(f0['points'],f1['points'],roi) for f0,f1 in zip(frames,frames[1:])]


def _mouth_interior_mask(roi, target_points, names, *, feather=7):
    """Approximate 2D visible cavity; NOT a semantic teeth/tongue segmenter."""
    import cv2
    required=('left_corner','upper_lip','right_corner','lower_lip')
    if any(n not in names for n in required):
        raise ValueError('Named mouth landmarks missing')
    d=dict(zip(names,target_points))
    polygon=np.asarray([d[n] for n in required],np.float32)
    hull=cv2.convexHull(np.rint(polygon).astype(np.int32))
    mask=np.zeros((SIZE,SIZE),np.uint8)
    cv2.fillConvexPoly(mask,hull,255)
    # Slightly extend region to cover teeth/lip boundary; still restricted to mouth ROI.
    mask=cv2.dilate(mask,np.ones((7,7),np.uint8),iterations=1)
    mask=cv2.GaussianBlur(mask,(0,0),feather)
    x0,y0,x1,y1=roi
    boundary=mouth_mask(SIZE,SIZE,roi,12)
    return np.clip(mask.astype(np.float32)/255,0,1)*boundary


def _local_flow_texture(warped_a, warped_b, t, roi):
    """Bidirectional Farneback alignment INSIDE mouth crop; exploratory only.

    Optical flow is not an occlusion model. Wrong flow can double teeth/tongues.
    """
    import cv2
    x0,y0,x1,y1=roi
    margin=16
    xa=max(0,int(x0*SIZE)-margin);xb=min(SIZE,int(math.ceil(x1*SIZE))+margin)
    ya=max(0,int(y0*SIZE)-margin);yb=min(SIZE,int(math.ceil(y1*SIZE))+margin)
    a=warped_a[ya:yb,xa:xb]
    b=warped_b[ya:yb,xa:xb]
    ga=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
    gb=cv2.cvtColor(b,cv2.COLOR_RGB2GRAY)
    kw=dict(pyr_scale=.5,levels=3,winsize=19,iterations=3,poly_n=5,poly_sigma=1.2,flags=0)
    fwd=cv2.calcOpticalFlowFarneback(ga,gb,None,**kw)
    rev=cv2.calcOpticalFlowFarneback(gb,ga,None,**kw)
    # Bound erratic flow in low-texture areas, especially black fur.
    fwd=np.clip(fwd,-24,24)
    rev=np.clip(rev,-24,24)
    yy,xx=np.indices(ga.shape,dtype=np.float32)
    fa=cv2.remap(a,xx-t*fwd[:,:,0],yy-t*fwd[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    fb=cv2.remap(b,xx-(1-t)*rev[:,:,0],yy-(1-t)*rev[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    inside=np.clip(np.rint(fa.astype(np.float32)*(1-t)+fb.astype(np.float32)*t),0,255).astype(np.uint8)
    result=warped_a.copy()
    result[ya:yb,xa:xb]=inside
    return result


def render_layered(closed, frames, alpha, roi, names, meshes, *, method='layered_nearest', feather_px=24):
    """Use nearest two adjacent photos; separate outer geometric and inner textures.

    Outside the mouth mask output is bitwise-identical to CLOSED. For exact
    reference alpha the mouth is from that reference (no inter-frame RGB ghost).
    """
    if method not in METHODS:
        raise ValueError('Unknown V0.6.4 texture method')
    if not isinstance(alpha,(int,float)) or isinstance(alpha,bool) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0,1]')
    geo.check_image(closed,'closed')
    if alpha == 0:
        return closed.copy(), {'left_alpha':0.,'right_alpha':0.,'interior_source':'closed','exact_anchor':True}
    mask=mouth_mask(SIZE,SIZE,roi,feather_px)
    # Exact reference alpha: preserve its entire mouth appearance.
    for f in frames:
        if abs(float(alpha)-f['alpha']) < 1e-8:
            texture=f['rgb']
            dst=np.clip(np.rint(closed.astype(np.float32)*(1-mask[:,:,None]) +
                               texture.astype(np.float32)*mask[:,:,None]),0,255).astype(np.uint8)
            dst[mask==0]=closed[mask==0]
            return np.ascontiguousarray(dst),{'left_alpha':f['alpha'],'right_alpha':f['alpha'],
                       'interior_source':'exact_reference','exact_anchor':True}
    ix=next((i for i in range(len(frames)-1) if frames[i]['alpha']<alpha<frames[i+1]['alpha']),None)
    if ix is None:
        raise ValueError('Alpha does not fall in reference range')
    f0,f1=frames[ix],frames[ix+1]
    t=(alpha-f0['alpha'])/(f1['alpha']-f0['alpha'])
    src0,src1,triangles=meshes[ix]
    dst=src0*(1-t)+src1*t
    wa=geo._warp_with_mesh(f0['rgb'],src0,dst,triangles)
    wb=geo._warp_with_mesh(f1['rgb'],src1,dst,triangles)
    outer=wa.astype(np.float32)*(1-t)+wb.astype(np.float32)*t
    local_points=f0['points']*(1-t)+f1['points']*t
    interior_mask=_mouth_interior_mask(roi,local_points,names)
    if method == 'layered_nearest':
        interior=wa if t<.5 else wb
        selected=f0['alpha'] if t<.5 else f1['alpha']
    else:
        interior=_local_flow_texture(wa,wb,t,roi)
        selected='bidirectional_flow'
    im=interior_mask[:,:,None]
    layered=outer*(1-im)+interior.astype(np.float32)*im
    out=np.clip(np.rint(closed.astype(np.float32)*(1-mask[:,:,None])+layered*mask[:,:,None]),0,255).astype(np.uint8)
    out[mask==0]=closed[mask==0]
    return np.ascontiguousarray(out),{'left_alpha':f0['alpha'],'right_alpha':f1['alpha'],
             'local_t':float(t),'interior_source':selected,'exact_anchor':False}



def target_control_points(frames, alpha):
    """Interpolated mesh controls at alpha; proxy if middle points are unverified."""
    if not 0 <= alpha <= 1:
        raise ValueError('alpha out of range')
    for f in frames:
        if abs(alpha-f['alpha'])<1e-8:
            return np.asarray(f['points'],np.float64).copy()
    for a,b in zip(frames,frames[1:]):
        if a['alpha']<alpha<b['alpha']:
            t=(alpha-a['alpha'])/(b['alpha']-a['alpha'])
            return a['points']*(1-t)+b['points']*t
    raise ValueError('No corresponding reference interval')

def region_mae(a,b,roi):
    """Pixel-space MAE diagnostic, not a realism score."""
    a=geo.check_image(a,'prediction');b=geo.check_image(b,'holdout')
    x0,y0,x1,y1=roi
    p=a[int(y0*SIZE):int(math.ceil(y1*SIZE)),int(x0*SIZE):int(math.ceil(x1*SIZE))]
    q=b[int(y0*SIZE):int(math.ceil(y1*SIZE)),int(x0*SIZE):int(math.ceil(x1*SIZE))]
    return float(np.mean(np.abs(p.astype(np.float32)-q.astype(np.float32))))


def outside_changed(closed, render, roi):
    return geo.mouth_geometry_metrics(closed,render,roi)['outside_region_changed_pixels']
