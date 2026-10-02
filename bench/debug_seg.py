"""Debug render of the room segmentation: barrier, interior labels, wall lines, gaps, closures.

    python -m bench.debug_seg data/1a8384c3f6 /tmp/seg.png
Drawn top-down in the same orientation as render.py: (x, -z).
"""
from __future__ import annotations

import sys

import cv2
import numpy as np


def debug_image(plan, path: str, scale: int = 3):
    d = plan.debug
    maps, labels = d["maps"], d["labels"]
    g = maps.grid
    nl = int(labels.max())
    rng = np.random.default_rng(7)
    cols = rng.integers(70, 230, (nl + 1, 3)).astype(np.uint8)
    cols[0] = 0
    img = cols[labels]
    img[maps.interior & (labels == 0)] = (60, 60, 60)
    img[maps.barrier] = (255, 255, 255)
    img[maps.traj] = (0, 0, 255)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    # array is [i=x, j=z]; cv2 point is (col, row) = (j, i) before the final transpose
    def P(x, z):
        return (int((z - g.z0) / g.res * scale), int((x - g.x0) / g.res * scale))
    for ln in d["lines"]:
        for a, b in ln.segments:
            p0 = (ln.c, a) if ln.axis == 0 else (a, ln.c)
            p1 = (ln.c, b) if ln.axis == 0 else (b, ln.c)
            cv2.line(img, P(*p0), P(*p1), (255, 200, 0), 1)
    closed = {id(gp) for gp, _ in d["closures"]}
    for gp in d["gaps"]:
        e = gp.endpoints()
        col = (0, 255, 255) if id(gp) in closed else (255, 0, 255)
        cv2.line(img, P(*e[0]), P(*e[1]), col, 2 if id(gp) in closed else 1)
    # transpose to rows=z, cols=x, then flip rows so +z points down (top-down view, matches render)
    img = np.transpose(img, (1, 0, 2)).copy()
    for xm in np.arange(np.ceil(g.x0), g.x0 + g.shape[0] * g.res, 1.0):
        c = int((xm - g.x0) / g.res * scale)
        cv2.line(img, (c, 0), (c, img.shape[0] - 1), (90, 90, 90), 1)
        cv2.putText(img, f"x{xm:.0f}", (c + 2, 14), 0, 0.45, (0, 255, 0), 1)
    for zm in np.arange(np.ceil(g.z0), g.z0 + g.shape[1] * g.res, 1.0):
        r = int((zm - g.z0) / g.res * scale)
        cv2.line(img, (0, r), (img.shape[1] - 1, r), (90, 90, 90), 1)
        cv2.putText(img, f"z{zm:.0f}", (2, r - 3), 0, 0.45, (0, 255, 0), 1)
    for r in plan.rooms:
        c = r.polygon.mean(axis=0)
        cv2.putText(img, f"{r.id} {r.area.value:.1f}", (int((c[0] - g.x0) / g.res * scale),
                    int((c[1] - g.z0) / g.res * scale)), 0, 0.6, (255, 255, 255), 2)
    cv2.imwrite(path, img[:, :, ::-1])


if __name__ == "__main__":
    from scan2plan.io.stray import load_stray
    from scan2plan.layout import build_plan
    plan = build_plan(load_stray(sys.argv[1]))
    debug_image(plan, sys.argv[2])
    print("rooms", [(r.id, round(r.area.value, 2)) for r in plan.rooms])
