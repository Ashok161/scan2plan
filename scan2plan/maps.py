"""2D evidence maps and room segmentation.

All maps live in the Manhattan-aligned world frame, plan coordinates (x, z),
on a square grid of `res` metres. Cell (i, j) covers x in [x0+i*res, ...),
z in [z0+j*res, ...).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .fusion import Cloud


@dataclass
class Grid:
    x0: float
    z0: float
    res: float
    shape: tuple[int, int]

    def ij(self, xz: np.ndarray) -> np.ndarray:
        return np.floor((xz - [self.x0, self.z0]) / self.res).astype(np.int64)

    def inside(self, ij: np.ndarray) -> np.ndarray:
        return (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < self.shape[0]) & (ij[:, 1] < self.shape[1])

    def xz(self, ij: np.ndarray) -> np.ndarray:
        return (np.asarray(ij, dtype=float) + 0.5) * self.res + [self.x0, self.z0]

    def count(self, xz: np.ndarray, weights=None) -> np.ndarray:
        ij = self.ij(xz)
        ok = self.inside(ij)
        lin = ij[ok, 0] * self.shape[1] + ij[ok, 1]
        w = None if weights is None else weights[ok]
        c = np.bincount(lin, weights=w, minlength=self.shape[0] * self.shape[1])
        return c.reshape(self.shape)


@dataclass
class Maps:
    grid: Grid
    floor_y: float
    ceil_y: float | None
    wall_hi: np.ndarray      # wall evidence above furniture height
    wall_any: np.ndarray     # wall evidence at any height
    floor_obs: np.ndarray    # floor plane observed
    free: np.ndarray         # ray-carved free space (counts)
    traj: np.ndarray         # camera trajectory cells (bool)
    interior: np.ndarray     # final interior mask (bool)
    barrier: np.ndarray      # wall-like cells: tall vertical structure (bool)


def build_maps(cloud: Cloud, floor_y: float, ceil_y: float | None, res: float = 0.02,
               max_rays_per_frame: int = 400, seed: int = 0) -> Maps:
    p, n = cloud.points, cloud.normals
    h = p[:, 1] - floor_y
    top = (ceil_y - floor_y) if ceil_y is not None else 2.6
    keep = (h > -0.1) & (h < top + 0.1)
    xz_all = p[keep][:, [0, 2]]
    lo = np.percentile(xz_all, 0.05, axis=0) - 0.5
    hi = np.percentile(xz_all, 99.95, axis=0) + 0.5
    cam = cloud.cam_pos[:, [0, 2]]
    lo = np.minimum(lo, cam.min(0) - 0.5)
    hi = np.maximum(hi, cam.max(0) + 0.5)
    shape = tuple(np.ceil((hi - lo) / res).astype(int))
    g = Grid(float(lo[0]), float(lo[1]), res, shape)

    vert = np.abs(n[:, 1]) < 0.3
    band_hi = vert & (h > 1.25) & (h < min(1.95, top - 0.15))
    band_any = vert & (h > 0.1) & (h < top - 0.1)
    flo = (n[:, 1] > 0.85) & (np.abs(h) < 0.04)
    wall_hi = g.count(p[band_hi][:, [0, 2]])
    wall_any = g.count(p[band_any][:, [0, 2]])
    floor_obs = g.count(p[flo][:, [0, 2]])
    # tall vertical structure: points in >= 3 of 5 height slabs (spans >~1.2 m);
    # catches walls in captures that never looked above 1.25 m, ignores
    # counters, sofas and tables
    slabs = np.zeros(shape, dtype=np.int32)
    for lo_h, hi_h in ((0.1, 0.5), (0.5, 0.9), (0.9, 1.3), (1.3, 1.7), (1.7, 2.1)):
        if lo_h > top - 0.1:
            break
        s = vert & (h > lo_h) & (h < min(hi_h, top - 0.1))
        slabs += (g.count(p[s][:, [0, 2]]) >= 2).astype(np.int32)
    barrier = (wall_hi >= 3) | (slabs >= 3)

    # 2D ray carving: camera -> hit for points in a mid-height band
    rng = np.random.default_rng(seed)
    band_mid = (h > 0.3) & (h < min(1.9, top - 0.1))
    idx = np.flatnonzero(band_mid)
    fr = cloud.frame[idx]
    order = rng.permutation(len(idx))
    # cap rays per frame
    _, first = np.unique(fr[order], return_index=True)
    counts = np.bincount(fr, minlength=len(cloud.keyframes))
    cap = np.minimum(counts, max_rays_per_frame)
    sel = []
    sorted_by_frame = order[np.argsort(fr[order], kind="stable")]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    for k in np.flatnonzero(cap):
        sel.append(sorted_by_frame[starts[k]:starts[k] + cap[k]])
    sel = idx[np.concatenate(sel)] if sel else np.array([], int)
    free = np.zeros(shape, dtype=np.float64)
    if len(sel):
        a = cloud.cam_pos[cloud.frame[sel]][:, [0, 2]]
        b = p[sel][:, [0, 2]]
        d = b - a
        L = np.linalg.norm(d, axis=1)
        ok = L > 0.1
        a, d, L = a[ok], d[ok], L[ok]
        # stop 6 cm short of the hit so the hit surface itself is not carved
        t_end = np.maximum(L - 0.06, 0) / L
        nstep = np.ceil(L / (res * 0.8)).astype(int)
        for chunk in np.array_split(np.arange(len(a)), max(1, len(a) // 20000)):
            ns = nstep[chunk]
            rep = np.repeat(chunk, ns)
            t = (np.arange(ns.sum()) - np.repeat(np.cumsum(ns) - ns, ns)) / np.repeat(ns, ns)
            t = t * t_end[rep]
            pts = a[rep] + d[rep] * t[:, None]
            free += g.count(pts)

    traj = g.count(cam) > 0
    interior = _interior(g, barrier, floor_obs, free)
    return Maps(g, floor_y, ceil_y, wall_hi, wall_any, floor_obs, free, traj, interior, barrier)


def _interior(g: Grid, barrier, floor_obs, free) -> np.ndarray:
    res = g.res
    wall = ndi.binary_dilation(barrier, iterations=1)
    obs = (floor_obs >= 1) | (free >= 2)
    obs = ndi.binary_closing(obs, structure=np.ones((3, 3)), iterations=2)
    interior = obs & ~wall
    # fill furniture-sized holes (sofas, beds, tables hide the floor)
    holes = ndi.binary_fill_holes(interior) & ~interior
    lab, nl = ndi.label(holes)
    if nl:
        areas = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, nl + 1)) * res * res
        wall_frac = ndi.mean(barrier, lab, index=np.arange(1, nl + 1))
        fill = np.flatnonzero((areas < 6.0) & (wall_frac < 0.25)) + 1
        interior |= np.isin(lab, fill)
    interior = ndi.binary_opening(interior, structure=np.ones((3, 3)), iterations=1)
    return interior


def geodesic_grow(markers: np.ndarray, domain: np.ndarray, max_iter: int = 2000) -> np.ndarray:
    """Grow labelled seeds through domain one cell per step (4-connected).

    Equivalent to assigning each domain cell to its geodesically nearest seed;
    unlike watershed_ift it has no plateau-flooding artefacts.
    """
    labels = np.where(domain, markers, 0).astype(np.int32)
    cross = ndi.generate_binary_structure(2, 1)
    for _ in range(max_iter):
        grown = ndi.grey_dilation(labels, footprint=cross)
        new = (labels == 0) & domain & (grown > 0)
        if not new.any():
            break
        labels[new] = grown[new]
    return labels


def boundary_lengths(labels: np.ndarray) -> dict[tuple[int, int], int]:
    """Number of 4-neighbour cell pairs between each pair of distinct labels."""
    out: dict[tuple[int, int], int] = {}
    for a, b in ((labels[1:, :], labels[:-1, :]), (labels[:, 1:], labels[:, :-1])):
        m = (a != b) & (a > 0) & (b > 0)
        pa, pb = a[m], b[m]
        lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
        if len(lo) == 0:
            continue
        key = lo.astype(np.int64) * 100000 + hi
        u, c = np.unique(key, return_counts=True)
        for k, cnt in zip(u, c):
            pair = (int(k // 100000), int(k % 100000))
            out[pair] = out.get(pair, 0) + int(cnt)
    return out


def _merge_regions(labels: np.ndarray, min_cells: float) -> np.ndarray:
    while True:
        bl = boundary_lengths(labels)
        cand = [(c, p) for p, c in bl.items() if c >= min_cells]
        if not cand:
            return labels
        _, (a, b) = max(cand)
        labels[labels == b] = a


def _drop_unvisited(labels: np.ndarray, traj: np.ndarray, min_cells: float) -> np.ndarray:
    near = ndi.binary_dilation(traj, iterations=5)
    out = labels.copy()
    for r in range(1, labels.max() + 1):
        m = labels == r
        if not m.any():
            continue
        if m.sum() < min_cells or not (m & near).any():
            out[m] = 0
    return out


def _relabel(labels: np.ndarray) -> np.ndarray:
    u = np.unique(labels)
    u = u[u > 0]
    lut = np.zeros(labels.max() + 1, dtype=np.int32)
    lut[u] = np.arange(1, len(u) + 1)
    return lut[labels]


def rasterize_segment(mask: np.ndarray, g: Grid, p0: np.ndarray, p1: np.ndarray, half_width: int = 1):
    L = np.linalg.norm(p1 - p0)
    n = max(2, int(np.ceil(L / (g.res * 0.5))))
    pts = p0 + (p1 - p0) * np.linspace(0, 1, n)[:, None]
    ij = g.ij(pts)
    for di in range(-half_width, half_width + 1):
        for dj in range(-half_width, half_width + 1):
            q = ij + [di, dj]
            ok = g.inside(q)
            mask[q[ok, 0], q[ok, 1]] = True


def segment_rooms_by_walls(maps: Maps, gaps, door_max: float = 1.3, min_area: float = 1.0,
                           min_split_area: float = 0.5, sliver_area: float = 1.5):
    """Rooms = visited connected components of the interior after closing openings.

    1. Door-sized gaps (<= door_max) are closed: a door separates rooms.
       A closure is kept only if it actually separates two interior regions.
    2. Wider gaps are closed only if that cuts off a region the camera never
       entered (glass walls, windows, views into unscanned space).
    3. Visited slivers below sliver_area are re-opened into their neighbour
       (a door closure that clipped the mouth of a hallway).
    Returns (labels, n_rooms, closures) with closures = [(gap, kind)].
    """
    g = maps.grid
    res = g.res
    cross = ndi.generate_binary_structure(2, 1)
    near = ndi.binary_dilation(maps.traj, iterations=int(round(0.3 / res)))

    def raster(gp):
        m = np.zeros(maps.interior.shape, bool)
        e = gp.endpoints()
        rasterize_segment(m, g, e[0], e[1])
        return m

    def label(cl):
        return ndi.label(maps.interior & ~cl, structure=cross)[0]

    def sides(lab, r):
        ring = ndi.binary_dilation(r, iterations=2)
        t = np.unique(lab[ring & (lab > 0)])
        return [int(x) for x in t]

    rasters = {}
    closures: list[tuple] = []
    for gp in sorted(gaps, key=lambda x: x.width):
        if gp.width <= door_max:
            rasters[id(gp)] = raster(gp)
            closures.append((gp, "door"))
    union = np.zeros(maps.interior.shape, bool)
    for gp, _ in closures:
        union |= rasters[id(gp)]
    lab = label(union)
    kept = []
    for gp, kind in closures:
        if len(sides(lab, rasters[id(gp)])) >= 2:
            kept.append((gp, kind))
    closures = kept
    union = np.zeros(maps.interior.shape, bool)
    for gp, _ in closures:
        union |= rasters[id(gp)]

    for gp in sorted((x for x in gaps if x.width > door_max), key=lambda x: x.width):
        r = raster(gp)
        lab = label(union | r)
        t = sides(lab, r)
        if len(t) < 2:
            continue
        vis = [bool(((lab == k) & near).any()) for k in t]
        areas = [(lab == k).sum() * res * res for k in t]
        if any(vis) and any((not v) and a > min_split_area for v, a in zip(vis, areas)):
            rasters[id(gp)] = r
            union |= r
            closures.append((gp, "window_or_glass"))

    # re-open door closures that only clip off a small visited sliver
    changed = True
    while changed:
        changed = False
        lab = label(union)
        for k, (gp, kind) in enumerate(closures):
            if kind != "door":
                continue
            t = sides(lab, rasters[id(gp)])
            small = [x for x in t if ((lab == x).sum() * res * res < sliver_area) and ((lab == x) & near).any()]
            if small and len(t) >= 2:
                closures.pop(k)
                union = np.zeros(maps.interior.shape, bool)
                for gq, _ in closures:
                    union |= rasters[id(gq)]
                changed = True
                break

    lab = label(union)
    out = np.zeros_like(lab)
    n = 0
    for r in range(1, lab.max() + 1):
        m = lab == r
        if (m & near).any() and m.sum() * res * res >= min_area:
            n += 1
            out[m] = n
    return out, n, closures
