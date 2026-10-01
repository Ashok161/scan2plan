"""Depth-anchored visual odometry + pose-graph loop closure for the video tier.

No IMU, no ARKit pose, no SfM library: relative camera motion between
keyframes is recovered by matching ORB features and solving a *metric* PnP
(frame a's matched pixels are back-projected with frame a's own monocular
depth into 3D, then PnP finds the rigid motion that reprojects them onto frame
b's matching pixels). Because the depth is already metric, this sidesteps the
usual monocular SfM scale ambiguity: translation comes out in metres directly
from a single pair, no separate scale-recovery step needed per edge.

Drift is handled by chaining these edges sequentially and then correcting the
chain with a sparse pose-graph optimisation over any loop-closure edges found
by re-matching keyframes against spatially nearby earlier keyframes.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

cv2.setRNGSeed(0)


@dataclass
class Edge:
    a: int
    b: int
    R: np.ndarray      # 3x3, X_b = R @ X_a + t  (both in camera frames)
    t: np.ndarray      # 3
    n_inliers: int
    kind: str          # "sequential" | "loop"


def detect_and_describe(gray: np.ndarray, n_features: int = 1500):
    orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=7, edgeThreshold=15)
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des


def match_descriptors(des_a, des_b, max_ratio: float = 0.8):
    if des_a is None or des_b is None or len(des_a) < 8 or len(des_b) < 8:
        return np.empty(0, int), np.empty(0, int)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des_a, des_b, k=2)
    ia, ib = [], []
    for m in knn:
        if len(m) < 2:
            continue
        m0, m1 = m
        if m0.distance < max_ratio * m1.distance:
            ia.append(m0.queryIdx)
            ib.append(m0.trainIdx)
    return np.array(ia, int), np.array(ib, int)


def relative_pose_pnp(pts_a_px: np.ndarray, pts_b_px: np.ndarray, depth_a: np.ndarray,
                       K: np.ndarray, depth_range=(0.2, 8.0), min_inliers: int = 15,
                       depth_scale: tuple[float, float] = (1.0, 1.0)):
    """Metric relative pose a->b from matches, using frame a's depth as the object points.

    ``K`` and ``pts_*_px`` are in full-resolution pixel coordinates; depth_a
    may live at a different (smaller) resolution, related by depth_scale =
    (depth_w/full_w, depth_h/full_h) -- matched pixels are rescaled before
    indexing into it.

    Returns a dict with R, t, n_inliers, obj (3xN inlier points in cam-a frame),
    img_b (Nx2 inlier pixel coords in frame b) -- or None if too few inliers.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = depth_a.shape
    sx, sy = depth_scale
    ui = np.clip(np.round(pts_a_px[:, 0] * sx).astype(int), 0, w - 1)
    vi = np.clip(np.round(pts_a_px[:, 1] * sy).astype(int), 0, h - 1)
    z = depth_a[vi, ui]
    valid = np.isfinite(z) & (z > depth_range[0]) & (z < depth_range[1])
    if valid.sum() < min_inliers:
        return None
    X = (pts_a_px[:, 0] - cx) / fx * z
    Y = (pts_a_px[:, 1] - cy) / fy * z
    obj = np.stack([X, Y, z], axis=1)[valid].astype(np.float64)
    img = pts_b_px[valid].astype(np.float64)
    if len(obj) < min_inliers:
        return None
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj, img, K.astype(np.float64), None,
        iterationsCount=300, reprojectionError=4.0, confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < min_inliers:
        return None
    inliers = inliers.ravel()
    ok, rvec, tvec = cv2.solvePnP(obj[inliers], img[inliers], K.astype(np.float64), None,
                                   rvec, tvec, useExtrinsicGuess=True,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvec)
    return {"R": R, "t": tvec.ravel(), "n_inliers": int(len(inliers)),
            "obj": obj[inliers], "img_b": img[inliers]}


def chain_sequential(frames_gray, depths, K, min_inliers: int = 15,
                      depth_scale: tuple[float, float] = (1.0, 1.0)):
    """Sequential VO. Returns (poses_cam_to_world Nx4x4, edges, scale_ratios, failed_idx)."""
    n = len(frames_gray)
    sx, sy = depth_scale
    kps, dess = [], []
    for g in frames_gray:
        kp, des = detect_and_describe(g)
        kps.append(kp)
        dess.append(des)

    poses = np.tile(np.eye(4), (n, 1, 1))
    edges: list[Edge] = []
    scale_ratios = []
    failed = []
    last_good = 0
    for i in range(1, n):
        res = None
        anchor = None
        for cand in (i - 1, last_good):
            if cand < 0 or cand == anchor:
                continue
            ia, ib = match_descriptors(dess[cand], dess[i])
            if len(ia) < min_inliers:
                continue
            pts_a = np.array([kps[cand][k].pt for k in ia])
            pts_b = np.array([kps[i][k].pt for k in ib])
            r = relative_pose_pnp(pts_a, pts_b, depths[cand], K, min_inliers=min_inliers,
                                   depth_scale=depth_scale)
            anchor = cand
            if r is not None:
                res = (cand, r)
                break
        if res is None:
            failed.append(i)
            poses[i] = poses[last_good]
            continue
        cand, r = res
        T_cand_to_i = np.eye(4)
        T_cand_to_i[:3, :3] = r["R"]
        T_cand_to_i[:3, 3] = r["t"]
        poses[i] = poses[cand] @ np.linalg.inv(T_cand_to_i)
        edges.append(Edge(cand, i, r["R"], r["t"], r["n_inliers"], "sequential"))
        last_good = i

        # scale-consistency: reproject frame cand's inlier points into frame i and
        # compare predicted depth against frame i's own independently-predicted depth.
        pred_z = (r["R"] @ r["obj"].T + r["t"][:, None])[2, :]
        ub = np.clip(np.round(r["img_b"][:, 0] * sx).astype(int), 0, depths[i].shape[1] - 1)
        vb = np.clip(np.round(r["img_b"][:, 1] * sy).astype(int), 0, depths[i].shape[0] - 1)
        own_z = depths[i][vb, ub]
        ok = np.isfinite(own_z) & (own_z > 0.2) & (own_z < 8.0) & (pred_z > 0.05)
        if ok.sum() >= 5:
            scale_ratios.extend((pred_z[ok] / own_z[ok]).tolist())
    return poses, edges, np.array(scale_ratios), failed, kps, dess


def find_loop_closures(poses, kps, dess, depths, K, min_gap: int = 15,
                        radius: float = 1.2, min_inliers: int = 30, max_per_frame: int = 2,
                        depth_scale: tuple[float, float] = (1.0, 1.0)):
    """Match each keyframe against spatially-nearby earlier keyframes (outside a
    temporal window) using the current (drift-affected) chain as a proximity
    prior. Returns a list of loop Edges."""
    n = len(poses)
    pos = poses[:, :3, 3]
    loops = []
    for i in range(min_gap, n):
        cand = np.arange(0, i - min_gap)
        if len(cand) == 0:
            continue
        d = np.linalg.norm(pos[cand] - pos[i], axis=1)
        order = np.argsort(d)
        picked = 0
        for c in order:
            j = int(cand[c])
            if d[c] > radius:
                break
            ia, ib = match_descriptors(dess[j], dess[i])
            if len(ia) < min_inliers:
                continue
            pts_a = np.array([kps[j][k].pt for k in ia])
            pts_b = np.array([kps[i][k].pt for k in ib])
            r = relative_pose_pnp(pts_a, pts_b, depths[j], K, min_inliers=min_inliers,
                                   depth_scale=depth_scale)
            if r is None:
                continue
            loops.append(Edge(j, i, r["R"], r["t"], r["n_inliers"], "loop"))
            picked += 1
            if picked >= max_per_frame:
                break
    return loops


def optimize_pose_graph(poses_init: np.ndarray, edges: list[Edge], anchor_weight: float = 1e3,
                         max_nfev: int = 60) -> np.ndarray:
    """Sparse least-squares pose-graph refinement. No-op (returns poses_init) if
    there are no loop edges, since a pure sequential chain is already exactly
    consistent with its own measurements."""
    if not any(e.kind == "loop" for e in edges):
        return poses_init
    n = len(poses_init)
    rv0 = Rotation.from_matrix(poses_init[:, :3, :3]).as_rotvec()
    t0 = poses_init[:, :3, 3].copy()
    x0 = np.concatenate([rv0.ravel(), t0.ravel()])

    a_idx = np.array([e.a for e in edges])
    b_idx = np.array([e.b for e in edges])
    R_meas = np.stack([e.R for e in edges])
    t_meas = np.stack([e.t for e in edges])
    w = np.array([e.n_inliers for e in edges], dtype=np.float64)
    w = np.sqrt(w / max(w.max(), 1.0))
    E = len(edges)

    def unpack(x):
        rv = x[: 3 * n].reshape(n, 3)
        tt = x[3 * n :].reshape(n, 3)
        R = Rotation.from_rotvec(rv).as_matrix()
        return R, tt

    def residuals(x):
        R, t = unpack(x)
        Ra, Rb = R[a_idx], R[b_idx]
        ta, tb = t[a_idx], t[b_idx]
        Rbt = np.transpose(Rb, (0, 2, 1))
        R_ab_pred = Rbt @ Ra
        t_ab_pred = np.einsum("eij,ej->ei", Rbt, ta - tb)
        R_err = np.transpose(R_meas, (0, 2, 1)) @ R_ab_pred
        rot_res = Rotation.from_matrix(R_err).as_rotvec() * w[:, None]
        trans_res = (t_ab_pred - t_meas) * w[:, None]
        anchor_res = np.concatenate([
            (x[:3] - x0[:3]) * anchor_weight,
            (x[3 * n : 3 * n + 3] - x0[3 * n : 3 * n + 3]) * anchor_weight,
        ])
        return np.concatenate([rot_res.ravel(), trans_res.ravel(), anchor_res])

    n_res = E * 6 + 6
    sp = lil_matrix((n_res, 6 * n), dtype=np.int8)
    for ei, e in enumerate(edges):
        for blk, idx in ((e.a, 0), (e.b, 1)):
            for row_off in range(6):
                sp[ei * 6 + row_off, 3 * n * 0 + 3 * blk : 3 * n * 0 + 3 * blk + 3] = 1
                sp[ei * 6 + row_off, 3 * n + 3 * blk : 3 * n + 3 * blk + 3] = 1
    sp[E * 6 : E * 6 + 3, 0:3] = 1
    sp[E * 6 + 3 : E * 6 + 6, 3 * n : 3 * n + 3] = 1

    result = least_squares(residuals, x0, jac_sparsity=sp, method="trf",
                            loss="soft_l1", max_nfev=max_nfev, verbose=0)
    R_opt, t_opt = unpack(result.x)
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :3] = R_opt
    out[:, :3, 3] = t_opt
    return out
