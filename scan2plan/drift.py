"""Drift accountability: plane-anchored pose correction.

ARKit / VO poses accumulate drift on long multi-room walks: the same wall seen
at the start and at the end of the capture lands a few cm apart, and the
floor tilts or steps. We correct this by anchoring every chunk of the
trajectory to the globally consistent structure it observes:

  * yaw:  each chunk's wall normals must agree with the global Manhattan frame
  * y:    each chunk's floor points must lie on the global floor plane
  * x, z: each chunk's wall points must lie on the consensus wall planes
          (wall planes extracted from the whole capture)

Per-chunk measured offsets are noisy and only partially observed, so the
corrections are solved jointly as a 1D pose graph per degree of freedom:
measurement edges (chunk vs anchor, weighted by support) plus smoothness edges
between consecutive chunks (drift is a slowly varying random walk). Revisits
of the same wall from distant parts of the trajectory act as loop closures
through the shared anchor. The solve is iterated so anchors sharpen as chunks
align. `build_plan(..., drift_correction=False)` is the ablation.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from .fusion import Cloud
from .manhattan import floor_and_ceiling


def _chunks(cloud: Cloud, path_len: float = 2.0, min_kf: int = 8) -> np.ndarray:
    """Chunk id per keyframe: split the trajectory every ~path_len metres."""
    cp = cloud.cam_pos
    step = np.r_[0.0, np.linalg.norm(np.diff(cp, axis=0), axis=1)]
    cum = np.cumsum(step)
    cid = np.floor(cum / path_len).astype(int)
    # merge tiny chunks into predecessor
    out = cid.copy()
    for c in np.unique(cid):
        if (cid == c).sum() < min_kf and c > 0:
            out[cid == c] = out[np.flatnonzero(cid == c)[0] - 1]
    _, out = np.unique(out, return_inverse=True)
    return out


def _wall_anchor_lines(p, n, axis: int, bin_size=0.01, min_count=200):
    """Consensus wall-plane coordinates for one axis and both facing signs."""
    out = {}
    for sign in (1, -1):
        sel = n[:, axis] * sign > 0.9
        if sel.sum() < min_count:
            out[sign] = np.array([])
            continue
        c = p[sel, axis]
        edges = np.arange(c.min() - 0.05, c.max() + 0.05, bin_size)
        h, e = np.histogram(c, bins=edges)
        hs = np.convolve(h, np.ones(5) / 5.0, mode="same")
        pk = [0.5 * (e[i] + e[i + 1]) for i in range(1, len(hs) - 1)
              if hs[i] >= hs[i - 1] and hs[i] > hs[i + 1] and hs[i] > min_count / 10]
        out[sign] = np.array(sorted(pk))
    return out


def _offset_to_anchors(coord, sign_arr, anchors, max_d=0.08):
    """Median signed residual of coords to the nearest same-sign anchor plane."""
    res = []
    for sign in (1, -1):
        a = anchors.get(sign, np.array([]))
        cc = coord[sign_arr == sign]
        if len(a) == 0 or len(cc) == 0:
            continue
        j = np.clip(np.searchsorted(a, cc), 1, len(a) - 1) if len(a) > 1 else np.zeros(len(cc), int)
        if len(a) > 1:
            d0 = cc - a[j - 1]
            d1 = cc - a[j]
            d = np.where(np.abs(d0) < np.abs(d1), d0, d1)
        else:
            d = cc - a[0]
        res.append(d[np.abs(d) < max_d])
    if not res:
        return 0.0, 0
    r = np.concatenate(res)
    if len(r) < 50:
        return 0.0, len(r)
    return float(np.median(r)), len(r)


def _solve_smooth(meas, w, lam):
    """argmin sum w_k (x_k - m_k)^2 + lam sum (x_{k+1} - x_k)^2."""
    K = len(meas)
    if K == 1:
        return np.array([meas[0] if w[0] > 0 else 0.0])
    main = w + lam * np.r_[1.0, 2.0 * np.ones(K - 2), 1.0]
    off = -lam * np.ones(K - 1)
    A = diags([off, main + 1e-9, off], [-1, 0, 1], format="csc")
    return spsolve(A, w * meas)


def _wall_sharpness(p, n, anchors_x, anchors_z):
    """Robust spread (m) of wall points around their consensus planes: lower is better."""
    out = []
    for axis, anchors in ((0, anchors_x), (2, anchors_z)):
        sel = np.abs(n[:, axis]) > 0.9
        sign = np.sign(n[sel, axis]).astype(int)
        coord = p[sel, axis]
        for s in (1, -1):
            a = anchors.get(s, np.array([]))
            cc = coord[sign == s]
            if len(a) == 0 or len(cc) == 0:
                continue
            d = np.min(np.abs(cc[:, None] - a[None, :]), axis=1) if len(cc) < 200000 else \
                np.min(np.abs(cc[::10, None] - a[None, :]), axis=1)
            out.append(d[d < 0.08])
    if not out:
        return float("nan")
    d = np.concatenate(out)
    return float(1.4826 * np.median(d))


def correct_drift(cloud: Cloud, iterations: int = 3, lam_t: float = 4.0, lam_r: float = 8.0,
                  log=None):
    """Returns (corrected cloud, per-keyframe 4x4 corrections, info dict)."""
    log = log or (lambda *_: None)
    K = len(cloud.cam_pos)
    chunk = _chunks(cloud)
    nC = int(chunk.max()) + 1
    T_total = np.tile(np.eye(4), (K, 1, 1))
    cur = cloud
    rng = np.random.default_rng(0)
    sub = rng.random(len(cloud.points)) < min(1.0, 3e6 / max(len(cloud.points), 1))
    (floor0, _, _), _ = floor_and_ceiling(cur.points[sub], cur.normals[sub])

    ax0 = _wall_anchor_lines(cur.points[sub], cur.normals[sub], 0)
    az0 = _wall_anchor_lines(cur.points[sub], cur.normals[sub], 2)
    sharp_before = _wall_sharpness(cur.points[sub], cur.normals[sub], ax0, az0)
    hist = []
    for it in range(iterations):
        p, n = cur.points[sub], cur.normals[sub]
        f_pt = cur.frame[sub]
        c_pt = chunk[f_pt]
        anchors_x = _wall_anchor_lines(p, n, 0)
        anchors_z = _wall_anchor_lines(p, n, 2)
        (floor_y, _, _), _ = floor_and_ceiling(p, n)
        m = np.zeros((4, nC))
        w = np.zeros((4, nC))
        for c in range(nC):
            sel = c_pt == c
            if sel.sum() < 200:
                continue
            pc, nc = p[sel], n[sel]
            # yaw from 4-fold circular mean of horizontal normals
            hz = np.abs(nc[:, 1]) < 0.2
            if hz.sum() > 300:
                ang = np.arctan2(nc[hz, 2], nc[hz, 0])
                zc = np.mean(np.exp(4j * ang))
                dyaw = np.angle(zc) / 4.0
                if abs(np.rad2deg(dyaw)) < 8 and abs(zc) > 0.4:
                    m[0, c], w[0, c] = dyaw, min(hz.sum() / 2000.0, 5.0) * abs(zc)
            fl = (nc[:, 1] > 0.9) & (np.abs(pc[:, 1] - floor_y) < 0.1)
            if fl.sum() > 200:
                m[1, c], w[1, c] = float(np.median(pc[fl, 1]) - floor_y), min(fl.sum() / 2000.0, 5.0)
            for dof, axis, anchors in ((2, 0, anchors_x), (3, 2, anchors_z)):
                ws = np.abs(nc[:, axis]) > 0.9
                off, cnt = _offset_to_anchors(pc[ws, axis], np.sign(nc[ws, axis]).astype(int), anchors)
                if cnt >= 200:
                    m[dof, c], w[dof, c] = off, min(cnt / 2000.0, 5.0)
        # corrections are minus the measured offsets
        corr = np.zeros((4, nC))
        for dof, lam in ((0, lam_r), (1, lam_t), (2, lam_t), (3, lam_t)):
            corr[dof] = -_solve_smooth(m[dof], w[dof], lam)
        # gauge: corrections are relative; remove the weighted mean so the map does not slide
        for dof in range(4):
            if w[dof].sum() > 0:
                corr[dof] -= np.sum(corr[dof] * w[dof]) / w[dof].sum()
        # per-keyframe transform: yaw about the chunk's camera centroid, then translate
        T = np.tile(np.eye(4), (K, 1, 1))
        for c in range(nC):
            ks = np.flatnonzero(chunk == c)
            pivot = cur.cam_pos[ks].mean(axis=0)
            yaw = corr[0, c]
            cs, sn = np.cos(yaw), np.sin(yaw)
            R = np.array([[cs, 0, -sn], [0, 1, 0], [sn, 0, cs]])
            Tc = np.eye(4)
            Tc[:3, :3] = R
            Tc[:3, 3] = pivot - R @ pivot + np.array([corr[2, c], corr[1, c], corr[3, c]])
            T[ks] = Tc
        cur = cur.apply_frame_transforms(T)
        T_total = np.einsum("kij,kjl->kil", T, T_total)
        hist.append({"iter": it, "max_abs_xz_cm": round(float(np.abs(corr[2:]).max()) * 100, 2),
                     "max_abs_y_cm": round(float(np.abs(corr[1]).max()) * 100, 2),
                     "max_abs_yaw_deg": round(float(np.rad2deg(np.abs(corr[0]).max())), 3)})
        log(f"drift iter {it}: {hist[-1]}")
    p, n = cur.points[sub], cur.normals[sub]
    sharp_after = _wall_sharpness(p, n, _wall_anchor_lines(p, n, 0), _wall_anchor_lines(p, n, 2))
    info = {
        "method": "plane-anchored correction: per-chunk yaw (Manhattan), height (floor plane) and x/z "
                  "(consensus wall planes) offsets solved as a smoothed 1D pose graph per DoF, 3 iterations",
        "n_chunks": nC,
        "iterations": hist,
        "wall_plane_spread_before_mm": round(sharp_before * 1000, 2),
        "wall_plane_spread_after_mm": round(sharp_after * 1000, 2),
    }
    return cur, T_total, info
