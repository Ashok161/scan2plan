"""Ceiling-height split-half repeatability (disclosed proxy).

Only one capture (c7d28f72c6) observes the ceiling, so a true two-capture
ceiling spread cannot be measured on this sample data. As a proxy we split
that capture's keyframes into the first and second half of the walk and
measure every room's floor-to-ceiling height independently from each half.
The two halves share the ARKit session (not independent captures), so this
bounds measurement noise + intra-session drift, not capture-to-capture bias.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scan2plan.io.stray import load_stray
from scan2plan.layout import _points_in, build_plan
from scan2plan.manhattan import horizontal_plane_peaks, refine_plane_height
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parent.parent
CAPTURE = "c7d28f72c6"


def _height(cloud, mask, poly):
    p, n = cloud.points[mask], cloud.normals[mask]
    fl = _points_in(p[n[:, 1] > 0.85], poly.buffer(-0.15))
    ce = _points_in(p[n[:, 1] < -0.85], poly.buffer(-0.2))
    if len(fl) < 100 or len(ce) < 200:
        return None
    f, _, _ = refine_plane_height(fl[:, 1], float(np.median(fl[:, 1])), 0.04)
    ce = ce[ce[:, 1] > f + 1.8]
    if len(ce) < 200:
        return None
    pk = horizontal_plane_peaks(ce[:, 1], min_frac=0.02)
    if not pk:
        return None
    c, _, _ = refine_plane_height(ce[:, 1], max(pk, key=lambda t: t[1])[0], 0.03)
    return c - f


def run(capture: str = CAPTURE):
    fs = load_stray(ROOT / "data" / capture)
    plan = build_plan(fs)
    cloud = plan.debug["cloud"]
    K = len(cloud.cam_pos)
    halves = [cloud.frame < K // 2, cloud.frame >= K // 2]
    rows = []
    for r in plan.rooms:
        poly = Polygon(r.polygon)
        h = [_height(cloud, m, poly) for m in halves]
        rows.append({"room": r.id, "name": r.name, "full": round(r.ceiling_height.value, 4),
                     "half1": None if h[0] is None else round(h[0], 4),
                     "half2": None if h[1] is None else round(h[1], 4),
                     "spread_mm": None if None in h else round(abs(h[0] - h[1]) * 1000, 1)})
    out = {"capture": capture, "method": __doc__.strip().splitlines()[0], "rooms": rows}
    p = ROOT / "reports" / "ceiling_split_half.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
