"""Manhattan wall-line extraction and gap (opening) detection.

Wall lines are found per orientation and per facing side: an X-wall is a plane
z = c whose points have normals along +-Z; the two faces of one physical wall
are separate lines with opposite normal sign. Along each line the 1D point
occupancy is split into solid segments; the gaps between collinear segments are
opening candidates (doors, passages, windows).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .manhattan import refine_plane_height


@dataclass
class WallLine:
    axis: int            # 0: plane x = c (normal along x); 1: plane z = c (normal along z)
    sign: int            # +1 / -1: direction of the outward-facing normal (into the room it bounds)
    c: float             # plane coordinate
    c_sigma: float       # robust std of point offsets
    n_points: int
    segments: list[tuple[float, float]] = field(default_factory=list)   # along-axis [a, b]

    @property
    def along_axis(self) -> int:   # 0 -> x, 1 -> z (coordinate index in plan xz)
        return 1 - self.axis


@dataclass
class Gap:
    line: WallLine
    a: float          # along-axis start (end of left segment)
    b: float          # along-axis end (start of right segment)

    @property
    def width(self) -> float:
        return self.b - self.a

    def endpoints(self) -> np.ndarray:
        """2x2 plan points (x, z)."""
        if self.line.axis == 0:
            return np.array([[self.line.c, self.a], [self.line.c, self.b]])
        return np.array([[self.a, self.line.c], [self.b, self.line.c]])


def extract_wall_lines(xz: np.ndarray, nxz: np.ndarray, bin_size: float = 0.01,
                       min_seg: float = 0.2, close_gap: float = 0.12,
                       occ_res: float = 0.02, min_occ: int = 2) -> list[WallLine]:
    """xz: Nx2 plan coords of wall points; nxz: Nx2 horizontal normal components."""
    lines: list[WallLine] = []
    for axis in (0, 1):
        for sign in (1, -1):
            sel = nxz[:, axis] * sign > 0.85
            if sel.sum() < 50:
                continue
            c_all = xz[sel, axis]
            t_all = xz[sel, 1 - axis]
            lo, hi = c_all.min() - 0.05, c_all.max() + 0.05
            edges = np.arange(lo, hi + bin_size, bin_size)
            h, e = np.histogram(c_all, bins=edges)
            hs = np.convolve(h, [1, 2, 3, 2, 1], mode="same") / 9.0
            # local maxima, then greedy non-maximum suppression within 6 cm
            cand = [i for i in range(1, len(hs) - 1) if hs[i] >= hs[i - 1] and hs[i] > hs[i + 1] and hs[i] >= 15]
            cand.sort(key=lambda i: -hs[i])
            taken: list[float] = []
            for i in cand:
                c0 = 0.5 * (e[i] + e[i + 1])
                if any(abs(c0 - t) < 0.06 for t in taken):
                    continue
                taken.append(c0)
                near = np.abs(c_all - c0) < 0.035
                if near.sum() < 30:
                    continue
                c, s, n = refine_plane_height(c_all[near], c0, window=0.035)
                segs = _segments(t_all[near], occ_res, min_occ, close_gap, min_seg)
                if not segs:
                    continue
                lines.append(WallLine(axis, sign, c, s, n, segs))
    return lines


def _segments(t: np.ndarray, res: float, min_occ: int, close_gap: float, min_seg: float):
    lo = t.min()
    idx = np.floor((t - lo) / res).astype(int)
    occ = np.bincount(idx) >= min_occ
    segs = []
    i, n = 0, len(occ)
    while i < n:
        if not occ[i]:
            i += 1
            continue
        j = i
        while j < n and occ[j]:
            j += 1
        segs.append([lo + i * res, lo + j * res])
        i = j
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= close_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(a, b) for a, b in merged if b - a >= min_seg]


def find_gaps(lines: list[WallLine], min_w: float = 0.45, max_w: float = 4.5,
              coline_tol: float = 0.12, perp_reach: float = 0.25) -> list[Gap]:
    """Openings along (near-)collinear same-facing wall lines.

    A gap runs from the end of a solid segment to the next "stop": the start
    of the next collinear segment, or a perpendicular wall reaching the line
    (an opening that ends in a corner, e.g. a hallway mouth).
    """
    gaps = []
    groups: dict[tuple[int, int], list[WallLine]] = {}
    for ln in lines:
        groups.setdefault((ln.axis, ln.sign), []).append(ln)
    for (axis, sign), lns in groups.items():
        perp = [l for l in lines if l.axis != axis]
        lns = sorted(lns, key=lambda l: l.c)
        clusters: list[list[WallLine]] = []
        for ln in lns:
            if clusters and ln.c - clusters[-1][-1].c < coline_tol:
                clusters[-1].append(ln)
            else:
                clusters.append([ln])
        for cl in clusters:
            c = float(np.median([l.c for l in cl]))
            segs = sorted((a, b, ln) for ln in cl for a, b in ln.segments)
            host = max(cl, key=lambda l: l.n_points)
            crossings = sorted(p.c for p in perp
                               if any(a - perp_reach <= c <= b + perp_reach for a, b in p.segments))
            starts = [a for a, _, _ in segs]
            ends = [b for _, b, _ in segs]
            seen = set()

            def add(a, b):
                key = (round(a, 2), round(b, 2))
                if min_w <= b - a <= max_w and key not in seen:
                    # the gap must not be covered by any solid segment
                    if not any(s < b - 0.05 and e > a + 0.05 for s, e in zip(starts, ends)):
                        seen.add(key)
                        gaps.append(Gap(host, a, b))

            for b in ends:
                nxt = [s for s in starts if s > b + 0.02] + [x for x in crossings if x > b + 0.02]
                if nxt:
                    add(b, min(nxt))
            for a in starts:
                prv = [e for e in ends if e < a - 0.02] + [x for x in crossings if x < a - 0.02]
                if prv:
                    add(max(prv), a)
    return gaps
