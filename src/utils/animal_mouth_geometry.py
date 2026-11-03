# coding: utf-8
"""V0.6.3: experimental paired-photo mouth geometry morphing.

The points are user annotated correspondences in ALIGNED V0.6.2 references;
they are not learned 3-D jaw joints. This is a 2-D appearance experiment.
"""

import hashlib
import math

import numpy as np


SIZE = 512
POINT_NAMES = ('left_corner', 'upper_lip', 'right_corner', 'lower_lip', 'chin')


def check_image(rgb, name):
    arr = np.asarray(rgb)
    if arr.shape != (SIZE, SIZE, 3) or arr.dtype != np.uint8:
        raise ValueError(f'{name} must be RGB uint8 {SIZE}x{SIZE}x3')
    return np.ascontiguousarray(arr)


def validate_roi(roi):
    if not isinstance(roi, (tuple, list)) or len(roi) != 4:
        raise ValueError('roi must be four normalized floats')
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in roi):
        raise ValueError('roi must contain finite numbers')
    x0, y0, x1, y1 = map(float, roi)
    if not (0.05 <= x0 < x1 <= 0.95 and 0.05 <= y0 < y1 <= 0.95):
        raise ValueError('roi out of bounds')
    if x1 - x0 < 0.16 or y1 - y0 < 0.16:
        raise ValueError('roi too small for piecewise affine morph')
    return x0, y0, x1, y1


def create_template(roi):
    """UNVERIFIED illustrative positions, NOT an anatomical detector."""
    x0, y0, x1, y1 = validate_roi(roi)
    cx = (x0 + x1) / 2
    w = x1 - x0
    h = y1 - y0
    pts = {
        'left_corner': [round(cx - .26*w, 4), round(y0 + .39*h, 4)],
        'upper_lip': [round(cx, 4), round(y0 + .36*h, 4)],
        'right_corner': [round(cx + .26*w, 4), round(y0 + .39*h, 4)],
        'lower_lip': [round(cx, 4), round(y0 + .62*h, 4)],
        'chin': [round(cx, 4), round(y0 + .81*h, 4)],
    }
    return {
        'schema_version': 1,
        'coordinate_system': 'normalized_xy_on_512px_aligned_reference',
        'verified': False,
        'roi': list(roi),
        'closed': pts,
        'open': {name: xy.copy() for name, xy in pts.items()},
        'note': 'EXAMPLE ONLY. Edit BOTH closed/open points using the guides; set verified=true after inspection. Closed and open must be the SAME dog.',
    }


def validate_landmarks(spec, *, require_verified=True):
    if not isinstance(spec, dict) or spec.get('schema_version') != 1:
        raise ValueError('landmarks JSON schema_version must be 1')
    if spec.get('coordinate_system') != 'normalized_xy_on_512px_aligned_reference':
        raise ValueError('coordinate_system must match aligned 512px references')
    if require_verified and spec.get('verified') is not True:
        raise ValueError('Landmarks not confirmed: edit both point sets and set verified=true')
    roi = validate_roi(spec.get('roi'))
    c = spec.get('closed')
    o = spec.get('open')
    if not isinstance(c, dict) or not isinstance(o, dict):
        raise ValueError('closed and open landmarks must be named dictionaries')
    if set(c) != set(o) or not set(POINT_NAMES).issubset(c) or len(c) > 12:
        raise ValueError('Closed/open landmark names must match and contain left_corner, upper_lip, right_corner, lower_lip, chin (max 12)')
    names = tuple(sorted(c))
    points = []
    for group in (c, o):
        rows = []
        for name in names:
            xy = group[name]
            if not isinstance(xy, (list, tuple)) or len(xy) != 2 or any(
                not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in xy
            ):
                raise ValueError(f'bad finite XY for {name}')
            x, y = map(float, xy)
            x0, y0, x1, y1 = roi
            if not (x0+.008 < x < x1-.008 and y0+.008 < y < y1-.008):
                raise ValueError(f'{name} must be inside mouth ROI (not on boundary)')
            rows.append([x * SIZE, y * SIZE])
        p = np.asarray(rows, dtype=np.float64)
        if np.min(np.linalg.norm(p[:, None, :] - p[None, :, :] + np.eye(len(p))[:,:,None]*1e5, axis=2)) < 2:
            raise ValueError('Duplicate/near-duplicate landmarks within a state')
        points.append(p)
    closed, opened = points
    if require_verified and np.max(np.abs(closed - opened)) < .25:
        raise ValueError('Closed and open points are nearly identical; template must be manually corrected')
    if np.max(np.linalg.norm(closed-opened, axis=1)) > SIZE * .30:
        raise ValueError('Landmark displacement > 30% of image; verify alignment and annotation')
    return roi, names, closed, opened


def boundary_anchors(roi):
    x0, y0, x1, y1 = roi
    a = []
    for t in (0,.25,.5,.75,1):
        a.append((x0 + t*(x1-x0), y0))
        a.append((x0 + t*(x1-x0), y1))
    for t in (.25,.5,.75):
        a.append((x0, y0 + t*(y1-y0)))
        a.append((x1, y0 + t*(y1-y0)))
    return np.asarray(a, np.float64) * SIZE


def _point_mapping(alpha, spec):
    roi, names, closed, opened = validate_landmarks(spec)
    anchors = boundary_anchors(roi)
    src_c = np.vstack([anchors, closed])
    src_o = np.vstack([anchors, opened])
    target = src_c * (1-alpha) + src_o * alpha
    try:
        from scipy.spatial import Delaunay
        from scipy.spatial import QhullError
        canonical = (src_c + src_o) * .5
        triangulation = Delaunay(canonical).simplices
    except (ImportError, ValueError, QhullError) as exc:
        raise ValueError('Cannot triangulate mouth landmarks; check point spacing / scipy') from exc
    for tri in triangulation:
        # Guard against folds or near-degenerate triangle for either state.
        signs = []
        for p in (src_c, target, src_o):
            v = p[tri]
            ab = v[1]-v[0]
            ac = v[2]-v[0]
            area = float(ab[0]*ac[1]-ab[1]*ac[0])
            if abs(area) < .2:
                raise ValueError('Degenerate mouth triangle; adjust landmarks')
            signs.append(math.copysign(1, area))
        if len(set(signs)) > 1:
            raise ValueError('Mouth landmark triangle folds between states; move control points closer')
    return roi, names, src_c, src_o, target, triangulation


def _warp_with_mesh(img, src, dst, triangles):
    """Inverse triangle affine remap, identity outside the ROI triangulated hull."""
    import cv2
    yy, xx = np.indices((SIZE, SIZE), dtype=np.float32)
    mx, my = xx.copy(), yy.copy()
    for tri in triangles:
        d = dst[tri].astype(np.float32)
        s = src[tri].astype(np.float32)
        x0 = max(0, int(math.floor(np.min(d[:,0])))-1)
        x1 = min(SIZE, int(math.ceil(np.max(d[:,0])))+2)
        y0 = max(0, int(math.floor(np.min(d[:,1])))-1)
        y1 = min(SIZE, int(math.ceil(np.max(d[:,1])))+2)
        if x0>=x1 or y0>=y1:
            continue
        patch = np.zeros((y1-y0, x1-x0), np.uint8)
        poly = np.rint(d - [x0, y0]).astype(np.int32)
        cv2.fillConvexPoly(patch, poly, 255)
        inv = cv2.getAffineTransform(d, s)
        subx = mx[y0:y1,x0:x1]
        suby = my[y0:y1,x0:x1]
        sel = patch.astype(bool)
        xs = xx[y0:y1,x0:x1]
        ys = yy[y0:y1,x0:x1]
        subx[sel] = (inv[0,0]*xs + inv[0,1]*ys + inv[0,2])[sel]
        suby[sel] = (inv[1,0]*xs + inv[1,1]*ys + inv[1,2])[sel]
    return cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


def blend_geometry(closed, opened, alpha, spec, *, policy='soft', feather_px=24):
    """Warp two mouth textures to intermediate geometry; composite only within ROI.

    'soft': geometry + RGB crossfade (possible tooth/tongue ghosts)
    'dominant': geometry + select one source for the mouth interior at alpha>=.5;
                avoids double texture but produces a temporal texture discontinuity.
    """
    from .animal_dual_source import mouth_mask
    c = check_image(closed, 'closed')
    o = check_image(opened, 'aligned open')
    if not isinstance(alpha, (float, int)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must be finite in [0,1]')
    if policy not in ('soft', 'dominant'):
        raise ValueError('policy must be soft or dominant')
    if alpha == 0:
        return c.copy()
    roi, names, src_c, src_o, dst, triangles = _point_mapping(float(alpha), spec)
    mask = mouth_mask(SIZE, SIZE, roi, feather_px)
    if policy == 'dominant':
        texture = _warp_with_mesh(c if alpha < .5 else o,
                                  src_c if alpha < .5 else src_o, dst, triangles)
    else:
        a = _warp_with_mesh(c, src_c, dst, triangles)
        b = _warp_with_mesh(o, src_o, dst, triangles)
        texture = np.clip(np.rint(a.astype(np.float32)*(1-alpha) +
                                   b.astype(np.float32)*alpha),0,255).astype(np.uint8)
    wt = mask[:,:,None]
    out = np.clip(np.rint(c.astype(np.float32)*(1-wt) +
                            texture.astype(np.float32)*wt), 0, 255).astype(np.uint8)
    # Exactly preserve the closed reference outside support.
    out[mask == 0] = c[mask == 0]
    return np.ascontiguousarray(out)


def mouth_geometry_metrics(closed, frame, roi):
    """Descriptive metrics; not a realism/phoneme score."""
    c = check_image(closed,'closed')
    f = check_image(frame,'frame')
    x0,y0,x1,y1 = roi
    inside = np.zeros((SIZE,SIZE),bool)
    inside[int(y0*SIZE):int(math.ceil(y1*SIZE)),int(x0*SIZE):int(math.ceil(x1*SIZE))] = True
    diff = np.abs(c.astype(np.float32)-f.astype(np.float32)).mean(axis=2)
    return {'mouth_region_mean_abs': float(diff[inside].mean()),
            'outside_region_changed_pixels': int(np.count_nonzero(diff[~inside]>0)),
            'mean_abs_all': float(diff.mean())}


def guides(closed, opened, spec, *, step=0.05):
    """Create annotated coordinate guides; template positions NOT verified."""
    from PIL import Image, ImageDraw, ImageFont
    roi, names, cp, op = validate_landmarks(spec, require_verified=False)
    results = []
    for img, pts, title in ((closed,cp,'CLOSED'),(opened,op,'OPEN ALIGNED')):
        out = Image.fromarray(check_image(img,title)).convert('RGB')
        draw = ImageDraw.Draw(out)
        for v in np.arange(0,1.001,step):
            n = round(float(v)*SIZE)
            draw.line((n,0,n,SIZE), fill=(220,220,80), width=1)
            draw.line((0,n,SIZE,n), fill=(220,220,80), width=1)
            draw.text((n+2,2),f'{v:.2f}',fill=(255,255,0))
            draw.text((2,n+2),f'{v:.2f}',fill=(255,255,0))
        x0,y0,x1,y1=roi
        draw.rectangle((x0*SIZE,y0*SIZE,x1*SIZE,y1*SIZE),outline=(255,0,0),width=3)
        for i,name in enumerate(names):
            x,y = pts[i]
            draw.ellipse((x-5,y-5,x+5,y+5),fill=(0,235,235),outline=(0,0,0),width=2)
            draw.text((x+7,y-8),name,fill=(255,255,255),stroke_width=2,stroke_fill=(0,0,0))
        results.append(out)
    return results
