"""Layout backend: FrameSet -> Plan (rooms, walls, openings, ceiling heights).

Pipeline:
  fuse -> (drift correction) -> Manhattan alignment -> floor / ceiling ->
  2D evidence maps -> wall lines + gaps -> room segmentation by closing
  doors -> per-room rectilinear polygon -> wall planes refined on raw points
  -> openings (doors / passages / windows) -> measurements with intervals.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .frames import FrameSet
from .fusion import Cloud, fuse, select_keyframes, voxel_downsample
from .manhattan import dominant_yaw, floor_and_ceiling, horizontal_plane_peaks, refine_plane_height, yaw_rotation
from .maps import Maps, build_maps, consensus_labels, refine_partition, segment_rooms_by_walls
from .measure import Measurement, area_measurement, combine, length_measurement
from .plan_types import Opening, Plan, Room, Wall
from .walls import Gap, extract_wall_lines, find_gaps

DOOR_MAX = 1.3
# A wall plane whose supporting points never reach this height is not evidence of a
# wall: kitchen units, wardrobes, beds and sofas present vertical faces up to ~1.5 m.
# Such an edge is reported as "occluded" with OCCLUDED_SIGMA (true wall at or behind it).
FURNITURE_H = 1.6
OCCLUDED_SIGMA = 0.25
UNSEEN_SIGMA = 0.30


# --------------------------------------------------------------------------- utils

def _cloud_transform(cloud: Cloud, T: np.ndarray) -> Cloud:
    R, t = T[:3, :3], T[:3, 3]
    return Cloud(cloud.points @ R.T + t, cloud.normals @ R.T, cloud.frame,
                 cloud.cam_pos @ R.T + t, cloud.keyframes, cloud.ranges)


class WallPoints:
    """Vertical-surface points (voxel-averaged) indexed for fast per-edge queries."""

    def __init__(self, cloud: Cloud, floor_y: float, top: float, voxel: float = 0.01):
        h = cloud.points[:, 1] - floor_y
        sel = (np.abs(cloud.normals[:, 1]) < 0.3) & (h > 0.08) & (h < top - 0.08)
        p, n, r, cnt = voxel_downsample(cloud.points[sel], voxel, cloud.normals[sel],
                                        cloud.ranges[sel][:, None])
        nn = np.linalg.norm(n, axis=1, keepdims=True)
        self.p = p
        self.n = n / np.maximum(nn, 1e-6)
        self.r = r[:, 0]
        self.w = cnt.astype(np.float64)
        self.h = p[:, 1] - floor_y
        order = np.argsort(p[:, 0])
        self._xs = p[order, 0]
        self._order = order

    def box(self, xmin, xmax, zmin, zmax) -> np.ndarray:
        lo, hi = np.searchsorted(self._xs, [xmin, xmax])
        idx = self._order[lo:hi]
        z = self.p[idx, 2]
        return idx[(z >= zmin) & (z <= zmax)]


# --------------------------------------------------------------------------- polygons

def mask_to_rectilinear(mask: np.ndarray, grid, min_edge: float = 0.12):
    """Room mask -> list of axis lines [(axis, c)] describing a rectilinear polygon.

    axis 0: vertical edge on line x = c; axis 1: horizontal edge on line z = c.
    Consecutive lines alternate axes; vertices are their intersections.
    """
    res = grid.res
    m = ndi.binary_fill_holes(mask)
    m = ndi.binary_closing(m, structure=np.ones((3, 3)), iterations=3)
    m = ndi.binary_opening(m, structure=np.ones((3, 3)), iterations=2)
    lab, nl = ndi.label(m)
    if nl > 1:
        sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, nl + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    img = np.ascontiguousarray(m.astype(np.uint8))
    cs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    ap = cv2.approxPolyDP(c, max(1.0, 0.05 / res), True)[:, 0, :].astype(float)
    # contour coords are (col=j -> z, row=i -> x) at cell centres
    xz = np.c_[grid.x0 + (ap[:, 1] + 0.5) * res, grid.z0 + (ap[:, 0] + 0.5) * res]
    n = len(xz)
    if n < 3:
        return None
    lines = []
    for k in range(n):
        a, b = xz[k], xz[(k + 1) % n]
        d = b - a
        L = float(np.hypot(*d))
        if L < 1e-6:
            continue
        axis = 0 if abs(d[0]) < abs(d[1]) else 1          # mostly along z -> line x=c
        cval = 0.5 * (a[0] + b[0]) if axis == 0 else 0.5 * (a[1] + b[1])
        lines.append([axis, cval, L])
    lines = _merge_lines(lines)
    # drop short edges: merge their two (same-axis) neighbours
    changed = True
    while changed and len(lines) > 4:
        changed = False
        verts = _vertices(lines)
        lens = [np.linalg.norm(verts[(k + 1) % len(verts)] - verts[k]) for k in range(len(verts))]
        # edge k runs from vertex k to k+1 and lies on line k
        k = int(np.argmin(lens))
        if lens[k] < min_edge:
            lines.pop(k)
            lines = _merge_lines(lines)
            changed = True
    if len(lines) < 4:
        return None
    return lines


def mask_to_rectilinear_snapped(mask: np.ndarray, grid, wall_lines, min_edge: float = 0.12,
                                merge_tol: float = 0.06, inside_frac: float = 0.5):
    """Room mask -> rectilinear polygon whose edges lie on detected wall lines.

    Builds a cell complex from the wall-line coordinates near the room and
    keeps the cells mostly covered by the room mask. Edges can then only
    appear where a wall was actually observed (or at the mask extent), which
    removes the notches a free-space mask produces under furniture.
    Returns axis lines [(axis, c, len)] or None.
    """
    from shapely.geometry import box as sbox
    res = grid.res
    m = ndi.binary_fill_holes(mask)
    m = ndi.binary_closing(m, structure=np.ones((3, 3)), iterations=3)
    ii, jj = np.nonzero(m)
    if len(ii) < 20:
        return None
    xmin, xmax = grid.x0 + ii.min() * res, grid.x0 + (ii.max() + 1) * res
    zmin, zmax = grid.z0 + jj.min() * res, grid.z0 + (jj.max() + 1) * res
    pad = 0.3
    cand = {0: [xmin, xmax], 1: [zmin, zmax]}
    strength = {0: [0.0, 0.0], 1: [0.0, 0.0]}
    for ln in wall_lines:
        lo_a, hi_a = (zmin, zmax) if ln.axis == 0 else (xmin, xmax)
        lo_c, hi_c = (xmin, xmax) if ln.axis == 0 else (zmin, zmax)
        if not (lo_c - pad <= ln.c <= hi_c + pad):
            continue
        overlap = sum(max(0.0, min(b, hi_a) - max(a, lo_a)) for a, b in ln.segments)
        if overlap < 0.25:
            continue
        cand[ln.axis].append(ln.c)
        strength[ln.axis].append(overlap)
    coords = {}
    for ax in (0, 1):
        order = np.argsort(cand[ax])
        cs = np.asarray(cand[ax])[order]
        st = np.asarray(strength[ax])[order]
        merged, mst = [], []
        for c, s_ in zip(cs, st):
            if merged and c - merged[-1] < merge_tol:
                if s_ > mst[-1]:
                    merged[-1], mst[-1] = c, s_
            else:
                merged.append(c)
                mst.append(s_)
        coords[ax] = np.asarray(merged)
    xs, zs = coords[0], coords[1]
    # fraction of each complex cell covered by the mask
    cells = []
    for a in range(len(xs) - 1):
        i0 = int(round((xs[a] - grid.x0) / res)); i1 = int(round((xs[a + 1] - grid.x0) / res))
        if i1 <= i0:
            continue
        for b in range(len(zs) - 1):
            j0 = int(round((zs[b] - grid.z0) / res)); j1 = int(round((zs[b + 1] - grid.z0) / res))
            if j1 <= j0:
                continue
            sub = m[max(i0, 0):max(i1, 0), max(j0, 0):max(j1, 0)]
            if sub.size and sub.mean() >= inside_frac:
                cells.append(sbox(xs[a], zs[b], xs[a + 1], zs[b + 1]))
    if not cells:
        return None
    u = unary_union(cells)
    if u.geom_type == "MultiPolygon":
        u = max(u.geoms, key=lambda g_: g_.area)
    u = Polygon(u.exterior).simplify(0.005)
    if u.area < 0.5:
        return None
    xz = np.asarray(u.exterior.coords)[:-1]
    lines = []
    n = len(xz)
    for k in range(n):
        a, b = xz[k], xz[(k + 1) % n]
        d = b - a
        L = float(np.hypot(*d))
        if L < 1e-6:
            continue
        axis = 0 if abs(d[0]) < abs(d[1]) else 1
        cval = 0.5 * (a[0] + b[0]) if axis == 0 else 0.5 * (a[1] + b[1])
        lines.append([axis, cval, L])
    lines = _merge_lines(lines)
    changed = True
    while changed and len(lines) > 4:
        changed = False
        verts = _vertices(lines)
        lens = [np.linalg.norm(verts[(k + 1) % len(verts)] - verts[k]) for k in range(len(verts))]
        k = int(np.argmin(lens))
        if lens[k] < min_edge:
            lines.pop(k)
            lines = _merge_lines(lines)
            changed = True
    return lines if len(lines) >= 4 else None


def _merge_lines(lines):
    """Merge consecutive same-axis lines (length-weighted) until axes alternate."""
    out = [list(l) for l in lines]
    changed = True
    while changed and len(out) > 1:
        changed = False
        for k in range(len(out)):
            a, b = out[k], out[(k + 1) % len(out)]
            if a[0] == b[0]:
                w = a[2] + b[2]
                merged = [a[0], (a[1] * a[2] + b[1] * b[2]) / max(w, 1e-9), w]
                if (k + 1) % len(out) == 0:
                    out[0] = merged
                    out.pop(k)
                else:
                    out[k] = merged
                    out.pop(k + 1)
                changed = True
                break
    return out


def _vertices(lines) -> np.ndarray:
    """Vertex k = intersection of line k-1 and line k."""
    V = []
    for k in range(len(lines)):
        p, q = lines[k - 1], lines[k]
        if p[0] == 0:          # p: x = c, q: z = c
            V.append([p[1], q[1]])
        else:
            V.append([q[1], p[1]])
    return np.array(V, dtype=float)


def _ccw(V: np.ndarray, lines):
    area = 0.5 * np.sum(V[:, 0] * np.roll(V[:, 1], -1) - np.roll(V[:, 0], -1) * V[:, 1])
    if area < 0:
        lines = lines[::-1]
        # with reversed lines vertex k = intersect(line k-1, line k) still holds
        V = _vertices(lines)
    return V, lines


# --------------------------------------------------------------------------- wall refinement

def refine_edge(wp: WallPoints, axis: int, c: float, span: tuple[float, float], out_sign: float,
                search_in: float = 0.20, search_out: float = 0.30):
    """Locate the wall plane for one polygon edge from raw points.

    axis 0: plane x = c; axis 1: plane z = c.  out_sign: +1 if outward (away
    from the room) is the +axis direction. Picks the outermost well-supported
    plane facing into the room (so cabinet fronts do not win over the wall
    behind them when the wall is visible above them).
    Returns (c_refined, sigma_fit, n_points, coverage, max_support_height).
    """
    a, b = min(span), max(span)
    trim = min(0.08, 0.2 * (b - a))
    lo, hi = a + trim, b - trim
    if axis == 0:
        idx = wp.box(c - search_in - search_out, c + search_in + search_out, lo, hi)
        coord, along = wp.p[idx, 0], wp.p[idx, 2]
        ncomp = wp.n[idx, 0]
    else:
        lo_x, hi_x = lo, hi
        idx = wp.box(lo_x, hi_x, c - search_in - search_out, c + search_in + search_out)
        coord, along = wp.p[idx, 2], wp.p[idx, 0]
        ncomp = wp.n[idx, 2]
    if len(idx) == 0:
        return c, None, 0, 0.0, 0.0
    s = (coord - c) * out_sign                 # outward offset
    facing = (-ncomp * out_sign) > 0.8         # normal points into the room
    sel = facing & (s > -search_in) & (s < search_out)
    if sel.sum() < 20:
        return c, None, int(sel.sum()), 0.0, 0.0
    s, along, w, hh = s[sel], along[sel], wp.w[idx][sel], wp.h[idx][sel]
    bins = np.arange(-search_in, search_out + 0.005, 0.005)
    hist, e = np.histogram(s, bins=bins, weights=w)
    hs = np.convolve(hist, [1, 2, 3, 2, 1], mode="same") / 9.0
    peak_max = hs.max()
    peaks = [i for i in range(1, len(hs) - 1) if hs[i] >= hs[i - 1] and hs[i] > hs[i + 1]
             and hs[i] > 0.25 * peak_max]
    best = None
    for i in sorted(peaks, key=lambda i: -e[i]):          # outermost first
        s0 = 0.5 * (e[i] + e[i + 1])
        near = np.abs(s - s0) < 0.02
        cov = _coverage(along[near], lo, hi)
        if cov >= 0.25 or best is None:
            best = (s0, cov)
            if cov >= 0.25:
                break
    if best is None:
        best = (0.5 * (e[int(np.argmax(hs))] + e[int(np.argmax(hs)) + 1]), 0.0)
    s0, _ = best
    m, sd, n = refine_plane_height(s, s0, window=0.02)
    inl = np.abs(s - m) < 0.025
    cov = _coverage(along[inl], lo, hi)
    hmax = float(np.percentile(hh[inl], 98)) if inl.sum() > 10 else 0.0
    return c + m * out_sign, sd, n, cov, hmax


def _coverage(along: np.ndarray, lo: float, hi: float, bin_size: float = 0.05) -> float:
    if hi <= lo or len(along) == 0:
        return 0.0
    nb = max(1, int(math.ceil((hi - lo) / bin_size)))
    k = np.clip(((along - lo) / bin_size).astype(int), 0, nb - 1)
    return float(len(np.unique(k)) / nb)


# --------------------------------------------------------------------------- main

def build_plan(fs: FrameSet, drift_correction: bool = True, res: float = 0.02,
               progress=None, cloud: Cloud | None = None, single_room: bool = False,
               room_name: str | None = None, low_walls: bool = False, neck_split: bool = False,
               bagging: int = 1) -> Plan:
    """single_room: the capture is one room (photo tier folder); skip door segmentation."""
    t0 = time.time()
    log = progress or (lambda *_: None)
    errs = fs.errors
    if cloud is None:
        kf = select_keyframes(fs)
        log(f"fusing {len(kf)} keyframes")
        cloud = fuse(fs, kf, progress=log)

    drift_info: dict = {"enabled": bool(drift_correction)}
    yaw, mscore = dominant_yaw(cloud.normals)
    T_align = yaw_rotation(yaw)
    cloud = _cloud_transform(cloud, T_align)
    if drift_correction:
        from .drift import correct_drift
        cloud, T_corr, drift_info = correct_drift(cloud, log=log)
        drift_info["enabled"] = True
    else:
        drift_info.update({"method": "none (poses used as captured)"})

    (floor_y, floor_sd, _), ceil = floor_and_ceiling(cloud.points, cloud.normals)
    ceil_y = ceil[0] if ceil is not None else None
    top = (ceil_y - floor_y) if ceil_y is not None else 2.6
    log(f"floor y={floor_y:.3f}  ceiling={'%.3f' % ceil_y if ceil_y is not None else 'not observed'}")

    maps = build_maps(cloud, floor_y, ceil_y, res=res)
    g = maps.grid
    h = cloud.points[:, 1] - floor_y
    if low_walls:
        _add_low_wall_lines(maps, cloud, h)
    # wall lines from points on wall-like (barrier) cells
    ij = g.ij(cloud.points[:, [0, 2]])
    ok = g.inside(ij)
    on_bar = np.zeros(len(h), bool)
    on_bar[ok] = maps.barrier[ij[ok, 0], ij[ok, 1]]
    sel = (np.abs(cloud.normals[:, 1]) < 0.3) & (h > 0.25) & (h < min(1.95, top - 0.1)) & on_bar
    wxz, wn = cloud.points[sel][:, [0, 2]], cloud.normals[sel][:, [0, 2]]
    lines = extract_wall_lines(wxz, wn)
    gaps = find_gaps(lines)
    labels, n_rooms, closures = segment_rooms_by_walls(maps, gaps, door_max=DOOR_MAX)
    if bagging > 1:
        # bagging: rooms whose partition flips when 15% of the points are removed are coin flips;
        # the majority partition over bootstrap rebuilds of the evidence maps is the stable one
        rng = np.random.default_rng(0)
        runs = [labels]
        for b in range(bagging - 1):
            keep = rng.random(len(cloud.points)) < 0.85
            cb = cloud.subset(keep)
            mb = build_maps(cb, floor_y, ceil_y, res=res, grid=g, seed=b + 1)
            hb = h[keep]
            ijb = g.ij(cb.points[:, [0, 2]])
            okb = g.inside(ijb)
            barb = np.zeros(len(hb), bool)
            barb[okb] = mb.barrier[ijb[okb, 0], ijb[okb, 1]]
            sb = (np.abs(cb.normals[:, 1]) < 0.3) & (hb > 0.25) & (hb < min(1.95, top - 0.1)) & barb
            g_b = find_gaps(extract_wall_lines(cb.points[sb][:, [0, 2]], cb.normals[sb][:, [0, 2]]))
            runs.append(segment_rooms_by_walls(mb, g_b, door_max=DOOR_MAX)[0])
        labels, n_rooms = consensus_labels(maps, runs)
    if neck_split:
        labels, n_rooms = refine_partition(labels, maps, door_max=DOOR_MAX)
    if single_room:
        lab_all, nl_all = ndi.label(maps.interior)
        if nl_all:
            score = ndi.sum(maps.traj, lab_all, index=np.arange(1, nl_all + 1)) * 1e6 + \
                ndi.sum(np.ones_like(lab_all), lab_all, index=np.arange(1, nl_all + 1))
            labels = (lab_all == 1 + int(np.argmax(score))).astype(np.int32)
            n_rooms = 1
    log(f"{len(lines)} wall lines, {len(gaps)} gaps, {n_rooms} rooms")

    wp = WallPoints(cloud, floor_y, top)
    up = cloud.normals[:, 1] > 0.85
    down = cloud.normals[:, 1] < -0.85
    floor_pts = cloud.points[up & (np.abs(h) < 0.06)]
    ceil_pts = cloud.points[down & (h > 1.8)]

    rooms: list[Room] = []
    warnings: list[str] = []
    if mscore < 0.5:
        warnings.append(f"weak Manhattan structure (score {mscore:.2f}); non-orthogonal walls are snapped to the dominant axes")
    for r in range(1, n_rooms + 1):
        room = _build_room(f"R{len(rooms) + 1}", labels == r, g, wp, floor_pts, ceil_pts,
                           floor_y, ceil_y, errs, maps, lines)
        if room is not None:
            room._label = r
            rooms.append(room)
        else:
            warnings.append(f"region {r} could not be converted to a polygon")
    tester = ThroughTester(cloud, floor_y, rooms)
    _assign_openings(rooms, gaps, closures, labels, g, wp, floor_y, errs, maps, tester)
    adjacency = _adjacency(rooms)
    _name_rooms(rooms)
    if single_room and room_name and rooms:
        rooms[0].name = room_name

    from shapely.validation import make_valid
    polys = [make_valid(Polygon(r.polygon)) for r in rooms]
    try:
        fp = unary_union(polys) if polys else None
    except Exception:                   # GEOS topology error on a degenerate polygon
        fp = unary_union([p.buffer(0) for p in polys]) if polys else None
    footprint = None
    if fp is not None:
        per = sum(r.perimeter.value for r in rooms)
        edge_sig = float(np.mean([w.offset_sigma for r in rooms for w in r.walls])) if rooms else 0.0
        footprint = area_measurement(float(fp.area), per, edge_sig, errs.scale_sigma_rel,
                                     method="union of room floor polygons (net internal area)")
    drift_info["runtime_s"] = round(time.time() - t0, 2)
    plan = Plan(fs.tier, rooms, adjacency, T_align, floor_y, footprint, drift_info, warnings)
    plan.debug = {"maps": maps, "labels": labels, "lines": lines, "gaps": gaps, "closures": closures,
                  "cloud": cloud, "manhattan_score": mscore, "floor_sigma": floor_sd}
    return plan


def _add_low_wall_lines(maps: Maps, cloud: Cloud, h: np.ndarray, min_len: float = 1.2):
    """Fix loop (attempt 2): long straight walls seen only low down are still walls.

    A capture that rarely looks above ~1 m (phone pointed at the floor) never
    sees walls in the >1.25 m band, so they fail the tall-structure test and
    rooms merge. Straight vertical planes >= min_len long observed in the
    0.1-1.0 m band are rasterised into the barrier map. Furniture faces this
    long (counters, sofa backs) mostly bound areas that are already not
    interior (no floor seen under / behind them), so they rarely change the
    partition.
    """
    from .maps import _interior, rasterize_segment
    g = maps.grid
    sel = (np.abs(cloud.normals[:, 1]) < 0.3) & (h > 0.1) & (h < 1.0)
    low = extract_wall_lines(cloud.points[sel][:, [0, 2]], cloud.normals[sel][:, [0, 2]],
                             min_seg=min_len, close_gap=0.08, min_occ=3)
    extra = np.zeros(maps.barrier.shape, bool)
    for ln in low:
        for a, b in ln.segments:
            if b - a < min_len:
                continue
            p0 = np.array([ln.c, a]) if ln.axis == 0 else np.array([a, ln.c])
            p1 = np.array([ln.c, b]) if ln.axis == 0 else np.array([b, ln.c])
            rasterize_segment(extra, g, p0, p1, half_width=0)
    maps.barrier = maps.barrier | extra
    maps.interior = _interior(g, maps.barrier, maps.floor_obs, maps.free)


def _build_room(rid, mask, g, wp: WallPoints, floor_pts, ceil_pts, floor_y_glob, ceil_y_glob, errs, maps,
                wall_lines=()) -> Room | None:
    lines = mask_to_rectilinear_snapped(mask, g, wall_lines) or mask_to_rectilinear(mask, g)
    if lines is None:
        return None
    V = _vertices(lines)
    V, lines = _ccw(V, lines)
    # local floor height
    poly0 = Polygon(V)
    inner = poly0.buffer(-0.15)
    fl = _points_in(floor_pts, inner if not inner.is_empty else poly0)
    if len(fl) > 50:
        floor_y, floor_sd, floor_n = refine_plane_height(fl[:, 1], np.median(fl[:, 1]), 0.04)
    else:
        floor_y, floor_sd, floor_n = floor_y_glob, 0.01, 0
    # refine each line's plane on raw points; flatten unobserved notches
    for _ in range(12):
        V = _vertices(lines)
        sig, cov, _ = _refine_lines(lines, V, wp, errs)
        nl = len(lines)
        if nl <= 4:
            break
        lens = [np.linalg.norm(V[(k + 1) % nl] - V[k]) for k in range(nl)]
        bad = [k for k in range(nl) if cov[k] < 0.15 and lens[k] < 0.8]
        if not bad:
            break
        k = min(bad, key=lambda q: lens[q])
        prv, nxt = (k - 1) % nl, (k + 1) % nl
        keep_c = lines[prv][1] if cov[prv] * lens[prv] >= cov[nxt] * lens[nxt] else lines[nxt][1]
        lines[prv][1] = keep_c
        lines[nxt][1] = keep_c
        lines.pop(k)
        lines = _merge_lines(lines)
        V = _vertices(lines)
        V, lines = _ccw(V, lines)
    V = _vertices(lines)
    sig, cov, state = _refine_lines(lines, V, wp, errs)
    # lines are now refined; vertex k = line k-1 ∩ line k  => edge k is on line k
    V = _vertices_edges(lines)
    poly = Polygon(V)
    if not poly.is_valid or poly.area < 0.5:
        poly = poly.buffer(0)
        if poly.is_empty or poly.area < 0.5:
            return None

    # ceiling
    inner = poly.buffer(-0.2)
    cp = _points_in(ceil_pts, inner if not inner.is_empty else poly)
    notes = []
    ceil_y = None
    if len(cp) > 200:
        lvl = ceiling_level(cp)
        if lvl is not None:
            ceil_y, c_sd, c_n, levels = lvl
            if len(levels) > 1:
                notes.append("multiple ceiling levels observed (m above floor, plan area m2): "
                             + ", ".join(f"{h - floor_y:.2f} ({a:.1f})" for h, a in levels)
                             + "; reported height is the level covering the largest area")
    if ceil_y is not None:
        H = ceil_y - floor_y
        s_fit = combine(c_sd / math.sqrt(max(1.0, min(c_n / 25.0, 400.0))),
                        floor_sd / math.sqrt(max(1.0, min(floor_n / 25.0, 400.0))))
        ceiling_height = Measurement(H, combine(s_fit, errs.plane_bias * math.sqrt(2), H * errs.scale_sigma_rel),
                                     "m", "ceiling plane - floor plane (robust plane fits)",
                                     {"fit": s_fit, "plane_bias": errs.plane_bias * math.sqrt(2),
                                      "scale": H * errs.scale_sigma_rel})
    else:
        # not observed: lower bound from the highest wall points, widened prior
        minx, minz, maxx, maxz = poly.bounds
        idx = wp.box(minx - 0.3, maxx + 0.3, minz - 0.3, maxz + 0.3)
        lb = float(np.percentile(wp.h[idx], 99.5)) if len(idx) > 50 else 2.2
        lb = max(lb, 2.0)
        val = max(lb, 2.5)
        ceiling_height = Measurement(val, max(0.15, (3.2 - lb) / 3.92), "m",
                                     f"NOT OBSERVED: ceiling not in capture; lower bound {lb:.2f} m from wall extent, prior 2.5-3.2 m",
                                     {"prior": max(0.15, (3.2 - lb) / 3.92)})
        notes.append("ceiling not observed: ceiling height is a bounded prior, not a measurement")

    walls = []
    nl = len(lines)
    for k in range(nl):
        a, b = V[k], V[(k + 1) % nl]
        L = float(np.linalg.norm(b - a))
        d = (b - a) / max(L, 1e-9)
        n_in = np.array([-d[1], d[0]])               # left of CCW edge
        s_prev, s_next = sig[(k - 1) % nl], sig[(k + 1) % nl]
        lm = length_measurement(L, (s_prev, s_next), errs.scale_sigma_rel,
                                method="distance between adjacent wall planes")
        walls.append(Wall(f"{rid}.W{k + 1}", a, b, n_in, lm, ceiling_height, sig[k], cov[k],
                          observed=state[k] == "wall"))
    n_occ = sum(1 for st in state if st != "wall")
    if n_occ:
        notes.append(f"{n_occ} wall plane(s) not observed above {FURNITURE_H} m (furniture face or unseen); "
                     f"their position carries a {OCCLUDED_SIGMA:.2f} m sigma")
    perim = sum(w.length.value for w in walls)
    per_m = Measurement(perim, combine(*[w.length.sigma for w in walls]) / math.sqrt(2), "m",
                        "sum of wall lengths")
    area = area_measurement(float(poly.area), perim, float(np.mean(sig)), errs.scale_sigma_rel,
                            method="polygon area of refined wall planes")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mrr = poly.minimum_rotated_rectangle
    ex = mrr.exterior.coords
    e1 = np.hypot(ex[1][0] - ex[0][0], ex[1][1] - ex[0][1])
    e2 = np.hypot(ex[2][0] - ex[1][0], ex[2][1] - ex[1][1])
    kind = "connector" if (max(e1, e2) / max(min(e1, e2), 1e-6) > 2.2 and min(e1, e2) < 1.6) else "room"
    return Room(rid, rid, kind, np.asarray(V), float(floor_y), ceil_y, area, per_m, ceiling_height,
                walls, [], notes)


def ceiling_level(cp: np.ndarray, cell: float = 0.1):
    """Main ceiling plane of a room: the level covering the largest plan AREA.

    Point counts depend on where the camera looked (a bulkhead seen up close
    outnumbers a ceiling glimpsed once); covered area does not. Returns
    (y, sd, n, [(level_y, area_m2), ...]) or None.
    """
    peaks = horizontal_plane_peaks(cp[:, 1], min_frac=0.02)
    if not peaks:
        return None
    big = max(c for _, c in peaks)
    levels = []
    for y0, cnt in peaks:
        if cnt < 0.1 * big:
            continue
        sel = np.abs(cp[:, 1] - y0) < 0.03
        if sel.sum() < 50:
            continue
        cells = np.unique(np.floor(cp[sel][:, [0, 2]] / cell).astype(np.int64), axis=0)
        levels.append((float(y0), len(cells) * cell * cell))
    if not levels:
        return None
    y_best = max(levels, key=lambda t: t[1])[0]
    y, sd, n = refine_plane_height(cp[:, 1], y_best, 0.03)
    return y, sd, n, sorted(levels)


def _refine_lines(lines, V, wp, errs):
    """Refine every line on raw points in place.

    Returns (sigma per line, coverage per line, state per line) with state
    "wall" (plane supported above FURNITURE_H), "occluded" (supported only
    lower: may be furniture in front of the wall) or "unseen" (no points).
    """
    sig, cov, state = [], [], []
    nl = len(lines)
    for k in range(nl):
        # line k carries the edge from vertex k to vertex k+1
        a, b = V[k], V[(k + 1) % nl]
        axis, c, _ = lines[k]
        span = (a[1], b[1]) if axis == 0 else (a[0], b[0])
        d = b - a
        # CCW polygon: interior on the left of the edge direction; outward = right
        right = np.array([d[1], -d[0]])
        out_sign = float(np.sign(right[0] if axis == 0 else right[1])) or 1.0
        c_new, sd, n, cv, hmax = refine_edge(wp, axis, c, span, out_sign)
        if sd is None:
            s_k, st = combine(errs.plane_bias, UNSEEN_SIGMA), "unseen"
        else:
            if abs(c_new - c) < 0.35:
                lines[k][1] = c_new
            n_eff = max(1.0, min(n / 25.0, 400.0))
            s_k = combine(sd / math.sqrt(n_eff), errs.plane_bias)
            st = "wall"
            if hmax < FURNITURE_H:
                s_k, st = combine(s_k, OCCLUDED_SIGMA), "occluded"
        sig.append(combine(s_k, errs.coverage_penalty * (1.0 - cv)))
        cov.append(cv)
        state.append(st)
    return sig, cov, state


def _vertices_edges(lines) -> np.ndarray:
    """Vertex k = start of edge k = intersection of line k-1 and line k."""
    return _vertices(lines)


def _points_in(pts: np.ndarray, poly) -> np.ndarray:
    if len(pts) == 0 or poly.is_empty:
        return pts[:0]
    minx, minz, maxx, maxz = poly.bounds
    sel = (pts[:, 0] >= minx) & (pts[:, 0] <= maxx) & (pts[:, 2] >= minz) & (pts[:, 2] <= maxz)
    cand = pts[sel]
    if len(cand) == 0:
        return cand
    from matplotlib.path import Path as MPath
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda p: p.area)
    path = MPath(np.asarray(poly.exterior.coords))
    return cand[path.contains_points(cand[:, [0, 2]])]


# --------------------------------------------------------------------------- openings

class ThroughTester:
    """Is a wall gap an opening? Count camera rays that actually passed through it.

    An opening is somewhere the sensor saw *through* the wall plane: points
    beyond the plane whose camera was on the room side and whose ray crossed
    the plane inside the gap rectangle. An unobserved stretch of wall (hidden
    behind furniture, or simply not swept) has none, even when another room
    lies behind it. Density is normalised by the capture's own density of
    points on observed walls, so thresholds do not depend on scan length.

    Mirrors also pass rays "through" the plane, into a virtual room. Reflecting
    those points back across the plane lands them on real points of the room;
    for a doorway the reflected points mostly land in empty space.
    """

    def __init__(self, cloud: Cloud, floor_y: float, rooms, max_points: int = 2_500_000):
        from scipy.spatial import cKDTree
        rng = np.random.default_rng(0)
        sub = rng.random(len(cloud.points)) < min(1.0, max_points / max(len(cloud.points), 1))
        self.P = cloud.points[sub].astype(np.float64)
        self.C = cloud.cam_pos[cloud.frame[sub]].astype(np.float64)
        self.N = cloud.normals[sub]
        self.floor_y = floor_y
        self.tree = cKDTree(self.P[::4])
        self.ref = self._reference_density(rooms)

    def _plane_terms(self, center, n_in, along):
        n3 = np.array([n_in[0], 0.0, n_in[1]])
        u3 = np.array([along[0], 0.0, along[1]])
        c0 = np.array([center[0], self.floor_y, center[1]])
        return n3, u3, c0

    def _reference_density(self, rooms):
        dens = []
        for r in rooms:
            for w in r.walls:
                if not w.observed or w.length.value < 0.8:
                    continue
                mid = 0.5 * (w.start + w.end)
                along = (w.end - w.start) / w.length.value
                n3, u3, c0 = self._plane_terms(mid, w.normal_in, along)
                s_p = (self.P - c0) @ n3
                al = (self.P - c0) @ u3
                hh = self.P[:, 1] - self.floor_y
                band = (np.abs(s_p) < 0.03) & (np.abs(al) < 0.5 * w.length.value - 0.1) & (hh > 0.1) & (hh < 2.0)
                band &= (self.N @ n3) > 0.7
                area = max(0.5, (w.length.value - 0.2) * 1.9 * max(w.coverage, 0.2))
                dens.append(band.sum() / area)
        return float(np.median(dens)) if dens else 1.0

    def test(self, center, n_in, along, width):
        n3, u3, c0 = self._plane_terms(center, n_in, along)
        s_p = (self.P - c0) @ (-n3)
        s_c = (self.C - c0) @ (-n3)
        cand = (s_p > 0.1) & (s_p < 4.0) & (s_c < -0.05)
        if not cand.any():
            return 0.0, None, (None, None)
        Pc, Cc = self.P[cand], self.C[cand]
        t = s_c[cand] / (s_c[cand] - s_p[cand])
        q = Cc + t[:, None] * (Pc - Cc)
        al = (q - c0) @ u3
        hh = q[:, 1] - self.floor_y
        hit = (np.abs(al) < 0.5 * width - 0.03) & (hh > 0.1) & (hh < 2.0)
        n_hit = int(hit.sum())
        ratio = n_hit / max(width * 1.9, 0.1) / max(self.ref, 1e-9)
        if n_hit < 50:
            return ratio, None, (None, None)
        # only surfaces parallel to the plane can tell a mirror from a doorway: floor, ceiling and
        # perpendicular walls map onto themselves under any reflection across a vertical plane
        par = np.abs(self.N[cand][hit] @ n3) > 0.8
        lo, hi = np.percentile(hh[hit], [3, 97])
        if par.sum() < 50:
            return ratio, 0.0, (float(lo), float(hi))
        X = Pc[hit][par][:: max(1, int(par.sum()) // 3000)]
        sx = (X - c0) @ (-n3)
        R = X + 2 * sx[:, None] * n3            # reflect back to the room side
        d, _ = self.tree.query(R)
        mirror = float(np.mean(d < 0.04))
        return ratio, mirror, (float(lo), float(hi))


# through-ratio thresholds, calibrated on the three sample captures against openings labelled by eye
# in the camera frames (bench/label_openings.py, docs/technical_report.md section 2): real walk-through
# doors measured 2.8-6.0, phantoms on unobserved wall 0-0.28, glass doors / stair voids 0.4-1.7
OPEN_MIN = 0.3
DOOR_MIN = 2.0
MIRROR_MIN = 0.7


def _assign_openings(rooms, gaps, closures, labels, g, wp: WallPoints, floor_y, errs, maps: Maps,
                     tester: "ThroughTester | None" = None):
    """Attach gaps to the room walls they lie on; classify and measure them."""
    closure_kind = {id(gp): kind for gp, kind in closures}
    count = {r.id: 0 for r in rooms}
    for room in rooms:
        for w in room.walls:
            d = w.end - w.start
            L = np.linalg.norm(d)
            if L < 0.4:
                continue
            u = d / L
            axis = 0 if abs(u[0]) < abs(u[1]) else 1      # wall along z -> plane x=c
            c = w.start[0] if axis == 0 else w.start[1]
            t0, t1 = sorted([w.start[1 - axis], w.end[1 - axis]])
            cands = []
            for gp in gaps:
                if gp.line.axis != axis or abs(gp.line.c - c) > 0.3:
                    continue
                a, b = max(gp.a, t0), min(gp.b, t1)
                if b - a < 0.4 or (b - a) < 0.6 * gp.width:
                    continue
                cands.append(gp)
            # keep non-overlapping gaps, narrowest first
            chosen = []
            for gp in sorted(cands, key=lambda x: x.width):
                if all(gp.b <= q.a + 0.05 or gp.a >= q.b - 0.05 for q in chosen):
                    chosen.append(gp)
            for gp in sorted(chosen, key=lambda x: x.a):
                op = _make_opening(room, w, gp, axis, c, u, labels, g, wp, floor_y, errs, maps,
                                   closure_kind.get(id(gp)), tester)
                if op is None:
                    continue
                # the two faces of one wall can both carry the same opening: keep the better-evidenced one
                dup = next((q for q in room.openings if abs(float(np.dot(q.along, op.along))) > 0.9
                            and np.linalg.norm(q.center - op.center) < 0.45), None)
                if dup is not None:
                    if _through_of(op) > _through_of(dup):
                        op.id = dup.id
                        room.openings[room.openings.index(dup)] = op
                    continue
                count[room.id] += 1
                op.id = f"{room.id}.O{count[room.id]}"
                room.openings.append(op)


def _make_opening(room, wall, gp: Gap, axis, c, u, labels, g, wp, floor_y, errs, maps, closure, tester=None):
    a, b = gp.a, gp.b
    mid = 0.5 * (a + b)
    center = np.array([c, mid]) if axis == 0 else np.array([mid, c])
    n_in = wall.normal_in
    # which rooms are on either side?
    def label_at(pt):
        q = g.ij(pt[None])[0]
        if 0 <= q[0] < labels.shape[0] and 0 <= q[1] < labels.shape[1]:
            win = labels[max(0, q[0] - 3):q[0] + 4, max(0, q[1] - 3):q[1] + 4]
            v = win[win > 0]
            if len(v):
                return int(np.bincount(v).argmax())
        return 0
    inside = label_at(center + n_in * 0.35)
    outside = label_at(center - n_in * 0.35)
    if outside == 0:
        outside = label_at(center - n_in * 0.55)
    # evidence that space continues through the gap (rays went through it)
    q = g.ij((center - n_in * 0.3)[None])[0]
    seen_through = False
    if 0 <= q[0] < maps.free.shape[0] and 0 <= q[1] < maps.free.shape[1]:
        win = maps.free[max(0, q[0] - 2):q[0] + 3, max(0, q[1] - 2):q[1] + 3]
        seen_through = bool((win >= 2).mean() > 0.4)
    # vertical profile of points in the gap on the wall plane
    if axis == 0:
        idx = wp.box(c - 0.06, c + 0.06, a + 0.05, b - 0.05)
    else:
        idx = wp.box(a + 0.05, b - 0.05, c - 0.06, c + 0.06)
    hs = wp.h[idx]
    low = hs[hs < 0.9]
    high = hs[hs > 1.75]
    has_sill = len(low) > 30 and (np.percentile(low, 90) > 0.3)
    head_h = float(np.percentile(high, 5)) if len(high) > 20 else None

    # jamb refinement: perpendicular faces at the gap ends
    left, s_l = _jamb(wp, axis, c, a, +1, n_in)
    right, s_r = _jamb(wp, axis, c, b, -1, n_in)
    width = right - left
    if not (0.3 < width < gp.width + 0.15):
        left, right, s_l, s_r = a, b, 0.02, 0.02
        width = gp.width
    mid = 0.5 * (left + right)
    center = np.array([c, mid]) if axis == 0 else np.array([mid, c])
    through = mirror = None
    if tester is not None:
        through, mirror, (t_lo, t_hi) = tester.test(center, n_in, u, width)
        if through < OPEN_MIN:
            return None                            # nothing seen through it: unobserved wall, not an opening
        if mirror is not None and mirror > MIRROR_MIN:
            room.notes.append(f"mirror on {wall.id} at {np.round(center, 2).tolist()} "
                              f"({width:.2f} m wide): not an opening")
            return None
        if through >= DOOR_MIN and (t_lo is None or t_lo < 0.7):
            kind = "door" if width <= DOOR_MAX else "passage"
        else:
            kind = "window"                        # glass / window / partial-height opening
    else:
        if not seen_through and closure is None and outside == 0:
            return None
        if has_sill and outside == 0:
            kind = "window"
        elif gp.width <= DOOR_MAX:
            kind = "door"
        else:
            kind = "passage"
    wm = length_measurement(width, (s_l, s_r), errs.scale_sigma_rel,
                            method="jamb-to-jamb" if s_l < 0.02 else "gap between wall segments")
    connects = []
    ev = "closure:" + closure if closure else "gap"
    if through is not None:
        ev += f", through-ratio {through:.2f}" + (f", mirror-score {mirror:.2f}" if mirror is not None else "")
    op = Opening("", kind, wall.id, center, u, wm, connects=connects, evidence=ev)
    op._labels = (inside, outside)
    if head_h is not None:
        op.height = Measurement(head_h, combine(0.01, errs.plane_bias), "m", "lowest header point above gap")
    if kind == "window" and len(low):
        sill = float(np.percentile(low, 95))
        op.sill = Measurement(sill, combine(0.015, errs.plane_bias), "m", "top of wall points below gap")
    return op


def _through_of(op) -> float:
    import re
    m = re.search(r"through-ratio ([0-9.]+)", op.evidence or "")
    return float(m.group(1)) if m else 0.0


def _jamb(wp: WallPoints, axis, c, t_end, direction, n_in):
    """Jamb face near along-coordinate t_end. direction +1: left end (jamb face looks +along)."""
    # jamb faces have normals along the wall direction, located within the wall depth
    if axis == 0:
        idx = wp.box(c - 0.30, c + 0.30, t_end - 0.12, t_end + 0.12)
        along, ncomp, depth = wp.p[idx, 2], wp.n[idx, 2], (wp.p[idx, 0] - c) * -n_in[0]
    else:
        idx = wp.box(t_end - 0.12, t_end + 0.12, c - 0.30, c + 0.30)
        along, ncomp, depth = wp.p[idx, 0], wp.n[idx, 0], (wp.p[idx, 2] - c) * -n_in[1]
    hs = wp.h[idx] if len(idx) else np.array([])
    sel = (ncomp * direction > 0.8) & (depth > -0.03) & (depth < 0.30) & (hs > 0.3) & (hs < 1.9)
    if sel.sum() < 15:
        return t_end, 0.02
    m, sd, n = refine_plane_height(along[sel], float(np.median(along[sel])), 0.03)
    return m, combine(sd / math.sqrt(max(1.0, min(n / 10.0, 200.0))), 0.003)


def _adjacency(rooms):
    lab2room = {r._label: r.id for r in rooms}
    adj = {}
    for r in rooms:
        for op in r.openings:
            ins, outs = getattr(op, "_labels", (0, 0))
            other = lab2room.get(outs)
            if other and other != r.id:
                op.connects = [r.id, other]
                key = tuple(sorted([r.id, other]))
                adj.setdefault(key, {"rooms": list(key), "via": [], "kind": op.kind})
                adj[key]["via"].append(op.id)
                if op.kind == "door":
                    adj[key]["kind"] = "door"
            else:
                op.connects = [r.id]
    return list(adj.values())


def _name_rooms(rooms):
    # largest first gets the lowest number; connectors are named as such
    order = sorted(rooms, key=lambda r: -r.area.value)
    n_room = n_conn = 0
    for r in order:
        if r.kind == "connector":
            n_conn += 1
            r.name = f"Hallway {n_conn}"
        else:
            n_room += 1
            r.name = f"Room {n_room}"
