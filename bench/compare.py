"""Plan-to-plan comparison: registration, room / wall / opening matching.

Two captures of the same property live in different ARKit (or VO) world
frames. Both plans are Manhattan-aligned, so registration is a search over the
four 90-degree yaws plus a 2D translation, found by FFT cross-correlation of
rasterised room masks and refined on room-polygon IoU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

RES = 0.05


def load(path):
    return json.loads(Path(path).read_text())


def room_polys(doc):
    return {r["id"]: Polygon(r["polygon"]).buffer(0) for r in doc["rooms"]}


def _raster(polys, lo, shape):
    from matplotlib.path import Path as MPath
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    pts = np.c_[(xs.ravel() + 0.5) * RES + lo[0], (ys.ravel() + 0.5) * RES + lo[1]]
    m = np.zeros(len(pts), bool)
    for p in polys:
        g = [p] if p.geom_type == "Polygon" else list(p.geoms)
        for q in g:
            m |= MPath(np.asarray(q.exterior.coords)).contains_points(pts)
    return m.reshape(shape).astype(float)


def _xf(p, k, t):
    return translate(rotate(p, 90 * k, origin=(0, 0)), t[0], t[1])


def register(doc_a, doc_b):
    """Find (k quarter turns, translation) mapping plan A into plan B's frame."""
    A = list(room_polys(doc_a).values())
    B = list(room_polys(doc_b).values())
    if not A or not B:
        return (0, np.zeros(2), 0.0)
    ub = unary_union(B)
    best = None
    for k in range(4):
        Ak = [rotate(p, 90 * k, origin=(0, 0)) for p in A]
        ua = unary_union(Ak)
        lo = np.minimum(ua.bounds[:2], ub.bounds[:2]) - 2.0
        hi = np.maximum(ua.bounds[2:], ub.bounds[2:]) + 2.0
        size = np.ceil((hi - lo) / RES).astype(int)
        shape = (int(size[1]) * 2, int(size[0]) * 2)
        ma = _raster(Ak, lo, shape)
        mb = _raster(B, lo, shape)
        F = np.fft.ifft2(np.fft.fft2(mb) * np.conj(np.fft.fft2(ma))).real
        iy, ix = np.unravel_index(np.argmax(F), F.shape)
        if iy > shape[0] // 2:
            iy -= shape[0]
        if ix > shape[1] // 2:
            ix -= shape[1]
        t = np.array([ix * RES, iy * RES])
        t = _refine(Ak, ub, t)
        iou = _overlap(unary_union([translate(p, *t) for p in Ak]), ub)
        if best is None or iou > best[2]:
            best = (k, t, iou)
    return best


def _overlap(ua, ub):
    """Overlap coefficient: intersection / smaller footprint (handles partial captures)."""
    return ua.intersection(ub).area / max(min(ua.area, ub.area), 1e-9)


def _refine(Ak, ub, t, steps=(0.04, 0.02, 0.01, 0.005)):
    def score(tt):
        return _overlap(unary_union([translate(p, *tt) for p in Ak]), ub)
    cur, s = np.array(t, float), score(t)
    for st in steps:
        improved = True
        while improved:
            improved = False
            for d in ((st, 0), (-st, 0), (0, st), (0, -st)):
                c = cur + d
                sc = score(c)
                if sc > s + 1e-6:
                    cur, s, improved = c, sc, True
    return cur


def match_rooms(doc_a, doc_b, reg, min_iou=0.4):
    k, t, _ = reg
    A = {rid: _xf(p, k, t) for rid, p in room_polys(doc_a).items()}
    B = room_polys(doc_b)
    pairs = []
    for ia, pa in A.items():
        bestb, bi = None, 0.0
        for ib, pb in B.items():
            u = pa.union(pb).area
            iou = pa.intersection(pb).area / u if u > 0 else 0
            if iou > bi:
                bestb, bi = ib, iou
        if bestb is not None and bi >= min_iou:
            pairs.append((ia, bestb, bi))
    # one-to-one: keep the best IoU per B room
    out = {}
    for ia, ib, iou in sorted(pairs, key=lambda x: -x[2]):
        if ib not in {v[0] for v in out.values()} and ia not in out:
            out[ia] = (ib, iou)
    return [(a, b, i) for a, (b, i) in out.items()]


def _rot(v, k):
    v = np.asarray(v, float)
    for _ in range(k % 4):
        v = np.array([-v[1], v[0]])
    return v


def match_walls(room_a, room_b, k, t, max_dist=0.35):
    """Pair walls by orientation + midpoint distance after registration."""
    pairs = []
    used = set()
    for wa in room_a["walls"]:
        sa = _rot(wa["start"], k) + t
        ea = _rot(wa["end"], k) + t
        na = _rot(wa["normal_in"], k)
        ma = 0.5 * (sa + ea)
        best, bd = None, max_dist
        for wb in room_b["walls"]:
            if wb["id"] in used:
                continue
            nb = np.asarray(wb["normal_in"])
            if float(np.dot(na, nb)) < 0.9:
                continue
            mb = 0.5 * (np.asarray(wb["start"]) + np.asarray(wb["end"]))
            # perpendicular offset must be small; along-wall midpoint can shift
            d_perp = abs(float(np.dot(ma - mb, nb)))
            d_along = abs(float(np.dot(ma - mb, np.array([-nb[1], nb[0]]))))
            la, lb = wa["length"]["value"], wb["length"]["value"]
            d = d_perp + 0.5 * d_along / max(1.0, 0.5 * (la + lb))
            if d_perp < 0.15 and d_along < 0.5 * max(la, lb) and d < bd:
                best, bd = wb, d
        if best is not None:
            used.add(best["id"])
            pairs.append((wa, best))
    return pairs


def match_openings(room_a, room_b, k, t, max_dist=0.35):
    pairs, used = [], set()
    for oa in room_a["openings"]:
        ca = _rot(oa["center"], k) + t
        best, bd = None, max_dist
        for ob in room_b["openings"]:
            if ob["id"] in used:
                continue
            d = float(np.linalg.norm(ca - np.asarray(ob["center"])))
            if d < bd:
                best, bd = ob, d
        if best is not None:
            used.add(best["id"])
            pairs.append((oa, best))
    return pairs
