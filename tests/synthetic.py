"""Synthetic LiDAR captures with exact ground truth (ray-cast box geometry).

Two rooms joined by a door, solid walls as axis-aligned boxes, floor y=0,
ceiling y=H. A camera walks a loop through both rooms; each keyframe gets a
ray-cast depth image (256x192, ARKit-like intrinsics) plus Gaussian noise.
Ground truth is exact, so this is the one place the pipeline is scored
against true dimensions rather than proxies.
"""
from __future__ import annotations

import numpy as np

from scan2plan.frames import Frame, FrameSet, LIDAR_ERRORS

W, H_IMG = 256, 192
K = np.array([[213.0, 0, 127.5], [0, 213.0, 95.5], [0, 0, 1]])


def apartment(ceiling=2.5, t=0.1, door=(1.0, 1.9)):
    """Room A: x[0,4] z[0,3]; room B: x[4+t, 7+t] z[0,3]; door in the partition at z in door."""
    X1, X2 = 4.0, 7.0 + t
    boxes = [
        (-t, 0, -t, X2 + t, ceiling, 0),          # wall z<0
        (-t, 0, 3.0, X2 + t, ceiling, 3.0 + t),    # wall z>3
        (-t, 0, -t, 0, ceiling, 3.0 + t),          # wall x<0
        (X2, 0, -t, X2 + t, ceiling, 3.0 + t),     # wall x>7.1
        (X1, 0, 0, X1 + t, ceiling, door[0]),      # partition below door
        (X1, 0, door[1], X1 + t, ceiling, 3.0),    # partition above door
        (X1, 2.05, door[0], X1 + t, ceiling, door[1]),   # door header
    ]
    gt = {"A": (4.0, 3.0), "B": (3.0, 3.0), "door": door[1] - door[0], "ceiling": ceiling}
    return np.array(boxes, float), ceiling, gt


def _raycast(o, d, boxes, ceiling):
    """o: (3,), d: (N,3) unit. Returns hit distance along ray (N,)."""
    tmin_all = np.full(len(d), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        # floor / ceiling
        for yp in (0.0, ceiling):
            tt = (yp - o[1]) / d[:, 1]
            tt[(tt <= 1e-6) | ~np.isfinite(tt)] = np.inf
            tmin_all = np.minimum(tmin_all, tt)
        inv = 1.0 / d
        for b in boxes:
            lo, hi = b[:3], b[3:]
            t1 = (lo - o) * inv
            t2 = (hi - o) * inv
            tn = np.nanmax(np.minimum(t1, t2), axis=1)
            tf = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (tf >= tn) & (tf > 1e-6)
            tt = np.where(tn > 1e-6, tn, np.inf)
            tmin_all = np.where(hit, np.minimum(tmin_all, tt), tmin_all)
    return tmin_all


def _look(pos, yaw, pitch):
    """Camera-to-world (OpenCV camera: x right, y down, z forward), world +Y up."""
    fwd = np.array([np.cos(pitch) * np.cos(yaw), np.sin(pitch), np.cos(pitch) * np.sin(yaw)])
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, down, fwd, pos
    return T


def trajectory(n=260):
    """Loop: room A perimeter-ish, through the door, room B, and back."""
    pts = np.array([[1.0, 1.0], [3.0, 1.0], [3.0, 2.2], [3.6, 1.45], [4.6, 1.45], [6.0, 0.9],
                    [6.2, 2.2], [5.0, 2.3], [4.6, 1.45], [3.6, 1.45], [1.2, 2.2], [1.0, 1.0]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.linspace(0, seg.sum(), n)
    cum = np.r_[0, np.cumsum(seg)]
    xz = np.array([pts[np.searchsorted(cum, si, side="right") - 1] +
                   (pts[min(np.searchsorted(cum, si, side="right"), len(pts) - 1)] -
                    pts[np.searchsorted(cum, si, side="right") - 1]) *
                   ((si - cum[np.searchsorted(cum, si, side="right") - 1]) /
                    max(seg[min(np.searchsorted(cum, si, side="right") - 1, len(seg) - 1)], 1e-9))
                   for si in s])
    return xz


def make_capture(noise=0.005, n=260, seed=0, yaw_drift_deg_per_m=0.0) -> tuple[FrameSet, dict]:
    boxes, ceiling, gt = apartment()
    rng = np.random.default_rng(seed)
    xz = trajectory(n)
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H_IMG) + 0.0)
    rays_c = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    rays_c /= np.linalg.norm(rays_c, axis=1, keepdims=True)
    frames = []
    walked = 0.0
    for i in range(n):
        pos = np.array([xz[i, 0], 1.45, xz[i, 1]])
        if i:
            walked += float(np.linalg.norm(xz[i] - xz[i - 1]))
        yaw = 0.35 * i                       # keep turning: every wall gets seen
        pitch = 0.55 * np.sin(0.21 * i)      # sweep floor <-> ceiling
        T = _look(pos, yaw, pitch)
        d = rays_c @ T[:3, :3].T
        r = _raycast(pos, d, boxes, ceiling)
        z = (r * rays_c[:, 2]).reshape(H_IMG, W)          # range -> depth along optical axis
        valid = np.isfinite(z) & (z < 5.0)
        z = np.where(valid, z + rng.normal(0, noise, z.shape), 0).astype(np.float32)
        T_rep = T.copy()
        if yaw_drift_deg_per_m:
            a = np.deg2rad(yaw_drift_deg_per_m * walked)
            Ry = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
            T_rep[:3, :3] = Ry @ T[:3, :3]
        frames.append(Frame(i, i / 30.0, T_rep, K.copy(), (lambda z=z, m=valid: (z, m))))
    return FrameSet("lidar", frames, LIDAR_ERRORS, source="synthetic"), gt
