"""Gravity / Manhattan frame and floor / ceiling estimation."""
from __future__ import annotations

import numpy as np

UP = np.array([0.0, 1.0, 0.0])


def dominant_yaw(normals: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    """Yaw (radians, in [-pi/4, pi/4)) of the dominant wall directions modulo 90 deg.

    Uses the 4-fold symmetric circular mean of horizontal wall normals.
    Returns (yaw, manhattan_score), score in [0,1] = resultant length.
    """
    horiz = np.abs(normals[:, 1]) < 0.2
    n = normals[horiz]
    w = np.ones(len(n)) if weights is None else weights[horiz]
    ang = np.arctan2(n[:, 2], n[:, 0])
    z = np.sum(w * np.exp(4j * ang)) / max(np.sum(w), 1e-9)
    yaw = np.angle(z) / 4.0
    return float(yaw), float(np.abs(z))


def yaw_rotation(yaw: float) -> np.ndarray:
    """4x4 rotation about +Y that maps the dominant wall direction onto +X."""
    c, s = np.cos(-yaw), np.sin(-yaw)
    # rotate (x,z) by -yaw:  x' = c x - s z ; z' = s x + c z
    T = np.eye(4)
    T[0, 0], T[0, 2], T[2, 0], T[2, 2] = c, -s, s, c
    return T


def gravity_from_normals(normals: np.ndarray) -> np.ndarray:
    """Estimate the up vector from floor/ceiling normals (video/photo tiers, no IMU).

    Finds the axis most aligned with clustered, near-parallel normals: the
    dominant eigenvector of the scatter of normals that are roughly parallel
    to the current best guess. Starts from camera 'up' (-Y in OpenCV).
    """
    g = np.array([0.0, 1.0, 0.0])
    for _ in range(10):
        c = normals @ g
        sel = np.abs(c) > 0.85
        if sel.sum() < 50:
            break
        n = normals[sel] * np.sign(c[sel])[:, None]
        g_new = n.mean(axis=0)
        g_new /= np.linalg.norm(g_new)
        if np.dot(g_new, g) > 0.99999:
            g = g_new
            break
        g = g_new
    return g


def rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 rotation taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        axis = np.cross(a, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 0, 1])
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * K @ K
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def horizontal_plane_peaks(y: np.ndarray, bin_size: float = 0.01, min_frac: float = 0.02):
    """Histogram peaks of heights. Returns list of (height, count) sorted by height."""
    if len(y) == 0:
        return []
    lo, hi = np.percentile(y, 0.1) - 0.05, np.percentile(y, 99.9) + 0.05
    edges = np.arange(lo, hi + bin_size, bin_size)
    h, e = np.histogram(y, bins=edges)
    hs = np.convolve(h, [1, 2, 3, 2, 1], mode="same") / 9.0
    peaks = []
    thr = max(min_frac * len(y) / 5.0, 20)
    for i in range(1, len(hs) - 1):
        if hs[i] >= hs[i - 1] and hs[i] > hs[i + 1] and hs[i] > thr:
            peaks.append((0.5 * (e[i] + e[i + 1]), float(hs[i])))
    return peaks


def refine_plane_height(y: np.ndarray, guess: float, window: float = 0.04):
    """Robust plane height near guess. Returns (height, std_of_residuals, n)."""
    sel = np.abs(y - guess) < window
    if sel.sum() < 10:
        return guess, window, int(sel.sum())
    v = y[sel]
    m = np.median(v)
    for _ in range(3):
        r = v - m
        s = 1.4826 * np.median(np.abs(r)) + 1e-4
        keep = np.abs(r) < 2.5 * s
        m = float(np.mean(v[keep]))
    s = float(np.std(v[keep]))
    return m, s, int(keep.sum())


def floor_and_ceiling(points: np.ndarray, normals: np.ndarray):
    """Global floor and ceiling heights (Y up). Ceiling may be None (not observed)."""
    up = normals[:, 1] > 0.9
    down = normals[:, 1] < -0.9
    fp = horizontal_plane_peaks(points[up, 1])
    if not fp:
        raise RuntimeError("no floor observed")
    # floor: the lowest peak holding a substantial share of upward points
    big = max(c for _, c in fp)
    floor_guess = min(h for h, c in fp if c > 0.25 * big)
    floor = refine_plane_height(points[up, 1], floor_guess)
    cp = horizontal_plane_peaks(points[down, 1], min_frac=0.01)
    ceiling = None
    if cp and down.sum() > 2000:
        bigc = max(c for _, c in cp)
        cands = [h for h, c in cp if c > 0.25 * bigc and h - floor[0] > 1.9]
        if cands:
            ceiling = refine_plane_height(points[down, 1], max(cands))
    return floor, ceiling
