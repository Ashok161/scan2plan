"""Video tier with phone motion: RGB frames + ARKit metric poses, NO depth sensor.

Every iPhone (not only Pro models) runs ARKit visual-inertial tracking, whose
poses are metric because the accelerometer fixes scale. This tier uses those
poses plus a monocular depth model and fixes the depth model's scale per frame
from multi-view geometry:

  for keyframe i, match SIFT features to neighbouring keyframes j with a useful
  baseline, triangulate with the known metric projection matrices, and take
  scale_i = median(z_triangulated / z_mono) over well-conditioned points.

Scales are smoothed along the walk and missing ones are filled from the median.
LiDAR depth, if present in the input folder, is never read.

Accepted inputs
  * StrayScanner folder: rgb.mp4 + odometry.csv + camera_matrix.csv (depth/ ignored)
  * NeRFCapture offline export (any iPhone): transforms.json + images/
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation

from ..frames import Frame, FrameSet, TierErrorModel
from ..video_io import VideoFrames

# Metric scale comes from ARKit VIO (sub-percent), but wall positions inherit the monocular depth
# model's local shape error, which multi-view averaging only partly removes. Measured on the sample
# data: median wall-length error 11-14% vs the LiDAR plan, i.e. sigma ~0.18 of length. That length-
# proportional term is carried in scale_sigma_rel so intervals stay calibrated.
VIDEO_MOTION_ERRORS = TierErrorModel("video", 0.03, 0.03, 0.02, 0.18, 0.0005, 0.05)

DEPTH_W, DEPTH_H = 256, 192
DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


def is_nerfcapture(path: Path) -> bool:
    if path.is_dir() and (path / "transforms.json").exists():
        return True
    if path.is_dir():
        return any(path.rglob("transforms.json"))
    return path.is_file() and path.suffix.lower() == ".zip"


def _resolve_input(path: Path) -> Path:
    """Accept a NeRFCapture export as its unzipped folder OR as the raw .zip AirDrop
    hand-off, and as either `transforms.json` directly at the top level or one level
    down (a real export unzips into a dated subfolder, e.g. '251002143000/')."""
    if path.is_file() and path.suffix.lower() == ".zip":
        import tempfile
        import zipfile
        tmp = Path(tempfile.mkdtemp(prefix="nerfcapture_"))
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp)
        path = tmp
    if path.is_dir() and not (path / "transforms.json").exists():
        hits = sorted(path.rglob("transforms.json"))
        if hits:
            path = hits[0].parent
    return path


def _rot_k_from_poses(Ts: list[np.ndarray]) -> int:
    """Pick the quarter-turn image rotation that makes gravity point image-down,
    from the camera-to-world poses alone (no constant per capture-format).

    Camera convention here is OpenCV (x right, y down, z forward); in a +Y-up
    world, gravity is the world vector (0,-1,0), which in camera frame is
    -R_wc[1, :] (second row of the camera-to-world rotation, negated).
    Dropping the forward/z component gives (gx, gy): the direction gravity
    points within the RAW (unrotated) image plane. Rotating the image by k
    quarter turns clockwise (np.rot90(img, -k), the convention used below)
    carries an image-plane vector (x, y) to:
        k=0: (x, y)       k=1: (-y, x)      k=2: (-x, -y)      k=3: (y, -x)
    (derived empirically from np.rot90's index mapping). The correct upright
    k is whichever makes the rotated vector's y-component largest (gravity
    pointing straight down the rotated image); the per-frame mode is used so
    a handful of noisy poses can't swing the whole capture's orientation.
    """
    ks = []
    for T in Ts:
        gx, gy = -T[1, 0], -T[1, 1]
        scores = [gy, gx, -gy, -gx]
        ks.append(int(np.argmax(scores)))
    counts = np.bincount(np.asarray(ks), minlength=4)
    return int(np.argmax(counts))


# ----------------------------------------------------------------------------- inputs

def _stray_frames(path: Path):
    from ..io.stray import read_odometry
    od = read_odometry(path)
    vf = VideoFrames(path / "rgb.mp4")
    out = []
    for row in od:
        idx = int(row[1])
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(row[5:9]).as_matrix()
        T[:3, 3] = row[2:5]
        K = np.array([[row[9], 0, row[11]], [0, row[10], row[12]], [0, 0, 1.0]])
        out.append((idx, float(row[0]), T, K, (lambda i=idx: vf.get(i))))
    return out, vf.size


def _nerfcapture_frames(path: Path):
    meta = json.loads((path / "transforms.json").read_text())
    flip = np.diag([1.0, -1.0, -1.0, 1.0])          # NeRF/OpenGL camera -> OpenCV camera
    out = []
    size = None
    for k, fr in enumerate(meta["frames"]):
        T = np.asarray(fr["transform_matrix"], float) @ flip
        fx = fr.get("fl_x", meta.get("fl_x"))
        fy = fr.get("fl_y", meta.get("fl_y", fx))
        cx = fr.get("cx", meta.get("cx"))
        cy = fr.get("cy", meta.get("cy"))
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        f = path / fr["file_path"]
        if not f.suffix:
            f = f.with_suffix(".png")
        out.append((k, float(fr.get("timestamp", k)), T, K,
                    (lambda f=f: np.asarray(Image.open(f).convert("RGB")))))
        size = size or (int(meta.get("w", 0)), int(meta.get("h", 0)))
    # transform_matrix is raw ARKit frame.camera.transform: world +Y up (gravity-aligned)
    # is an ARKit guarantee, not something to re-derive. An earlier version of this loader
    # re-estimated "up" as the mean camera-up direction across all frames and rotated the
    # whole trajectory to match -- averaging camera-up over an arbitrary walk (which yaws
    # and tilts through many directions, and need not average anywhere near true up) is
    # unreliable and was measured to corrupt the per-frame gravity direction enough to flip
    # the pose-derived upright rotation (_rot_k_from_poses) to the wrong quarter-turn on
    # real captures. Trust the documented convention instead, exactly like
    # scan2plan.io.stray trusts StrayScanner's odometry.csv convention without
    # re-estimating it.
    return out, size


# ----------------------------------------------------------------------------- helpers

def _select(frames, min_trans=0.10, min_rot_deg=10.0, max_kf=260):
    keep = [0]
    last = frames[0][2]
    cos_thr = np.cos(np.deg2rad(min_rot_deg))
    for i in range(1, len(frames)):
        T = frames[i][2]
        if np.linalg.norm(T[:3, 3] - last[:3, 3]) > min_trans or float(T[:3, 2] @ last[:3, 2]) < cos_thr:
            keep.append(i)
            last = T
    if len(keep) > max_kf:
        keep = [keep[i] for i in np.linspace(0, len(keep) - 1, max_kf).astype(int)]
    return keep


def _projection(T_wc, K):
    T_cw = np.linalg.inv(T_wc)
    return K @ T_cw[:3, :]


def _triangulate_scale(fi, fj, mono_i, kp_i, des_i, kp_j, des_j, matcher):
    """Scale ratios z_tri / z_mono for keyframe i from its matches with keyframe j."""
    if des_i is None or des_j is None or len(kp_i) < 20 or len(kp_j) < 20:
        return np.array([])
    m = matcher.knnMatch(des_i, des_j, k=2)
    good = [a for a, b in (p for p in m if len(p) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < 15:
        return np.array([])
    pi = np.float32([kp_i[g.queryIdx].pt for g in good])
    pj = np.float32([kp_j[g.trainIdx].pt for g in good])
    Pi, Pj = _projection(fi[2], fi[3]), _projection(fj[2], fj[3])
    X = cv2.triangulatePoints(Pi, Pj, pi.T.astype(np.float64), pj.T.astype(np.float64))
    X = (X[:3] / X[3]).T
    T_cw_i = np.linalg.inv(fi[2])
    T_cw_j = np.linalg.inv(fj[2])
    Xi = X @ T_cw_i[:3, :3].T + T_cw_i[:3, 3]
    Xj = X @ T_cw_j[:3, :3].T + T_cw_j[:3, 3]
    ok = (Xi[:, 2] > 0.2) & (Xj[:, 2] > 0.2) & (Xi[:, 2] < 6.0)
    # reprojection check in both views
    for Xc, K, p in ((Xi, fi[3], pi), (Xj, fj[3], pj)):
        uv = (Xc @ K.T)
        uv = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6)
        ok &= np.linalg.norm(uv - p, axis=1) < 2.0
    # triangulation angle > 2 deg
    ci, cj = fi[2][:3, 3], fj[2][:3, 3]
    a = X - ci
    b = X - cj
    cosang = np.einsum("ij,ij->i", a, b) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-9)
    ok &= cosang < np.cos(np.deg2rad(2.0))
    if ok.sum() < 8:
        return np.array([])
    h, w = mono_i.shape
    K = fi[3]
    sx, sy = w / (2 * K[0, 2] + 1), h / (2 * K[1, 2] + 1)
    u = np.clip((pi[ok, 0] * sx).astype(int), 0, w - 1)
    v = np.clip((pi[ok, 1] * sy).astype(int), 0, h - 1)
    zm = mono_i[v, u]
    good_m = zm > 0.1
    return Xi[ok, 2][good_m] / zm[good_m]


def _pair_ratio(za, Ka, Ta, zb, Kb, Tb, step=4):
    """Median of (a's depth reprojected into b) / (b's depth) over overlapping pixels."""
    H, W = za.shape
    u, v = np.meshgrid(np.arange(0, W, step, dtype=np.float32), np.arange(0, H, step, dtype=np.float32))
    z = za[::step, ::step]
    P = np.stack([(u - Ka[0, 2]) / Ka[0, 0] * z, (v - Ka[1, 2]) / Ka[1, 1] * z, z], -1).reshape(-1, 3)
    Pw = P @ Ta[:3, :3].T + Ta[:3, 3]
    T_cw = np.linalg.inv(Tb)
    Pb = Pw @ T_cw[:3, :3].T + T_cw[:3, 3]
    zp = Pb[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        ub = Kb[0, 0] * Pb[:, 0] / zp + Kb[0, 2]
        vb = Kb[1, 1] * Pb[:, 1] / zp + Kb[1, 2]
    inb = (zp > 0.2) & (ub >= 0) & (ub < W - 1) & (vb >= 0) & (vb < H - 1) & (z.ravel() > 0.2)
    if inb.sum() < 200:
        return None, 0
    q = zp[inb] / np.maximum(zb[vb[inb].astype(int), ub[inb].astype(int)], 1e-3)
    q = q[(q > 0.5) & (q < 2.0)]
    if len(q) < 200:
        return None, 0
    return float(np.median(q)), len(q)


def _joint_scales(monos, Kds, Ts, pos, tri_scale, tri_n, init, log, iters=3, n_nbr=4):
    """Per-keyframe depth scales: triangulation anchors + pairwise depth-overlap constraints.

    Least squares in log scale. Anchor rows: log s_n = log t_n (weight ~ sqrt(#points)).
    Pair rows: log s_b - log s_a = log(s_b q_ab / s_a), with q_ab the median ratio of a's
    scaled depth reprojected into b over b's scaled depth (re-linearised each iteration).
    """
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import lsqr
    N = len(monos)
    s = np.asarray(init, float).copy()
    pairs = []
    for a in range(N):
        d = np.linalg.norm(pos - pos[a], axis=1)
        pairs += [(a, int(b)) for b in np.argsort(d) if b != a and d[b] < 1.2][:n_nbr]
    anchors = [n for n in range(N) if np.isfinite(tri_scale[n])]
    for it in range(iters):
        rows, rhs, wts = [], [], []
        for a, b in pairs:
            q, cnt = _pair_ratio(monos[a] * s[a], Kds[a], Ts[a], monos[b] * s[b], Kds[b], Ts[b])
            if q is None:
                continue
            rows.append((b, a))
            rhs.append(np.log(s[b] * q / s[a]))
            wts.append(1.0)
        M = lil_matrix((len(rows) + len(anchors), N))
        y = np.zeros(len(rows) + len(anchors))
        for r, ((b, a), v, w) in enumerate(zip(rows, rhs, wts)):
            M[r, b], M[r, a], y[r] = w, -w, w * v
        for k, n in enumerate(anchors):
            w = 0.5 * np.sqrt(tri_n[n] / 15.0)
            M[len(rows) + k, n] = w
            y[len(rows) + k] = w * np.log(tri_scale[n])
        sol = lsqr(M.tocsr(), y, atol=1e-8, btol=1e-8)[0]
        s = np.exp(sol)
        log(f"joint scale iter {it}: {len(rows)} pair constraints, {len(anchors)} anchors, "
            f"scale range {np.percentile(s, 5):.3f}-{np.percentile(s, 95):.3f}")
    return s


def _consistent(n, deps, Kds, Ts, pos, rel_tol=0.06, n_nbr=4):
    """Pixels of keyframe n whose depth agrees with >= 1 neighbouring keyframe after reprojection.

    Monocular depth is locally warped differently in every view; warped regions do
    not survive reprojection into a neighbour, while true surfaces do.
    """
    d = np.linalg.norm(pos - pos[n], axis=1)
    nbrs = [b for b in np.argsort(d) if b != n and d[b] < 1.5][:n_nbr]
    H, W = deps[n].shape
    if not nbrs:
        return np.ones((H, W), bool)
    K = Kds[n]
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    z = deps[n]
    P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1).reshape(-1, 3)
    Pw = P @ Ts[n][:3, :3].T + Ts[n][:3, 3]
    good = np.zeros(H * W, bool)
    seen = np.zeros(H * W, bool)
    for b in nbrs:
        T_cw = np.linalg.inv(Ts[b])
        Pb = Pw @ T_cw[:3, :3].T + T_cw[:3, 3]
        zb = Pb[:, 2]
        Kb = Kds[b]
        with np.errstate(divide="ignore", invalid="ignore"):
            ub = Kb[0, 0] * Pb[:, 0] / zb + Kb[0, 2]
            vb = Kb[1, 1] * Pb[:, 1] / zb + Kb[1, 2]
        inb = (zb > 0.2) & (ub >= 0) & (ub < W - 1) & (vb >= 0) & (vb < H - 1)
        ui, vi = ub[inb].astype(int), vb[inb].astype(int)
        zref = deps[b][vi, ui]
        agree = np.abs(zb[inb] - zref) < rel_tol * zref
        idx = np.flatnonzero(inb)
        seen[idx] = True
        good[idx[agree]] = True
    # pixels no neighbour could see are kept (nothing to contradict them)
    return (good | ~seen).reshape(H, W)


# ----------------------------------------------------------------------------- main

def load_posed_video(path: str | Path, cache_dir: str | Path = ".cache", device: str | None = None,
                     model_id: str = DEFAULT_MODEL, progress=None) -> FrameSet:
    from . import mono_depth
    t0 = time.time()
    log = progress or (lambda *_: None)
    path = Path(path)
    nerf = is_nerfcapture(path)
    path = _resolve_input(path) if nerf else path
    if nerf:
        frames, size = _nerfcapture_frames(path)
        fmt = "NeRFCapture"
    else:
        frames, size = _stray_frames(path)
        fmt = "StrayScanner (depth ignored)"
    kf = _select(frames)
    # Orientation is derived from the poses themselves (see _rot_k_from_poses), not a
    # constant per capture-format: this must work for a portrait hold (StrayScanner's
    # sample captures, and most NeRFCapture captures) as well as a landscape-left or
    # landscape-right hold, on either format.
    rot_k = _rot_k_from_poses([frames[i][2] for i in kf])
    log(f"posed video ({fmt}): {len(frames)} frames -> {len(kf)} keyframes, "
        f"pose-derived upright rotation +{90 * rot_k} deg")

    sift = cv2.SIFT_create(nfeatures=1500)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    monos, feats = [], []
    for n, i in enumerate(kf):
        rgb = frames[i][4]()
        up = np.ascontiguousarray(np.rot90(rgb, -rot_k)) if rot_k else rgb    # upright for the model
        key = f"posedvideo:{path.name}:{frames[i][0]}:{rot_k}"
        d, _ = mono_depth.predict_depth(up, model_id, key, cache_dir=cache_dir, device=device,
                                        resize_to_input=False)
        d = np.asarray(d, np.float32)
        if rot_k:
            d = np.ascontiguousarray(np.rot90(d, rot_k))                       # back to sensor layout
        d = cv2.resize(d, (DEPTH_W, DEPTH_H), interpolation=cv2.INTER_LINEAR)
        monos.append(d)
        # Resize preserving the raw frame's own aspect ratio (a hard-coded 960x720
        # landscape canvas silently distorted/mis-scaled keypoints for any raw frame
        # that isn't ~4:3 landscape, e.g. a portrait-shaped raw frame): scale the
        # longer side down to 960 and keep the shorter side proportional.
        h0, w0 = rgb.shape[:2]
        resize_scale = 960.0 / max(h0, w0)
        gray = cv2.cvtColor(cv2.resize(rgb, (max(1, round(w0 * resize_scale)),
                                             max(1, round(h0 * resize_scale)))),
                            cv2.COLOR_RGB2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        # keypoints at full resolution
        for p in kp:
            p.pt = (p.pt[0] / resize_scale, p.pt[1] / resize_scale)
        feats.append((kp, des))
        if n % 40 == 0:
            log(f"depth + features {n}/{len(kf)}")

    # per-keyframe scale from neighbours with a useful baseline
    pos = np.array([frames[i][2][:3, 3] for i in kf])
    scales = np.full(len(kf), np.nan)
    n_pts = np.zeros(len(kf), int)
    for a in range(len(kf)):
        d = np.linalg.norm(pos - pos[a], axis=1)
        cand = [b for b in np.argsort(d) if b != a and 0.15 < d[b] < 0.8][:4]
        ratios = []
        for b in cand:
            r = _triangulate_scale(frames[kf[a]], frames[kf[b]], monos[a], *feats[a], *feats[b], matcher)
            if len(r):
                ratios.append(r)
        if ratios:
            r = np.concatenate(ratios)
            r = r[(r > 0.2) & (r < 5.0)]
            if len(r) >= 15:
                scales[a] = float(np.median(r))
                n_pts[a] = len(r)
    ok = np.isfinite(scales)
    if ok.sum() == 0:
        raise RuntimeError("could not triangulate any scale points; capture too short or textureless")
    s_global = float(np.median(scales[ok]))
    filled = np.where(ok, scales, s_global)
    smooth = median_filter(filled, size=5, mode="nearest")
    Kds = []
    for i in kf:
        K = frames[i][3]
        w0, h0 = size if size and size[0] else (2 * K[0, 2] + 1, 2 * K[1, 2] + 1)
        Kd = K.copy()
        Kd[0] *= DEPTH_W / w0
        Kd[1] *= DEPTH_H / h0
        Kds.append(Kd)
    Ts = [frames[i][2] for i in kf]
    # _joint_scales (pairwise chaining) was tried and measured worse (wall error 31-59% vs 11-14%):
    # chained ratios accumulate error like odometry drift. Kept for reference, not used.
    log(f"mono-depth scale: global {s_global:.3f}, per-frame spread {np.nanstd(scales[ok] / s_global):.3f}, "
        f"{ok.sum()}/{len(kf)} keyframes triangulated")

    deps, Kds = [], []
    for n, i in enumerate(kf):
        K = frames[i][3]
        w0, h0 = size if size and size[0] else (2 * K[0, 2] + 1, 2 * K[1, 2] + 1)
        Kd = K.copy()
        Kd[0] *= DEPTH_W / w0
        Kd[1] *= DEPTH_H / h0
        deps.append(monos[n] * smooth[n])
        Kds.append(Kd)
    keep_frac = []
    out = []
    for n, i in enumerate(kf):
        idx, ts, T, K, ld = frames[i]
        dep = deps[n]
        valid = (dep > 0.2) & (dep < 5.0)
        valid &= _consistent(n, deps, Kds, [frames[j][2] for j in kf], pos)
        keep_frac.append(float(valid.mean()))
        out.append(Frame(idx, ts, T, Kds[n], (lambda d=dep, v=valid: (d, v)), ld, K))
    log(f"multi-view consistency keeps {np.mean(keep_frac) * 100:.0f}% of depth pixels")
    fs = FrameSet("video", out, VIDEO_MOTION_ERRORS, source=str(path), meta={
        "video_mode": "rgb + phone motion (ARKit metric poses), monocular depth rescaled by multi-view triangulation",
        "input_format": fmt, "depth_model": model_id, "depth_sensor_used": False,
        "upright_rotation_deg": 90 * rot_k, "upright_rotation_source": "pose_derived",
        "scale_global": round(s_global, 4),
        "scale_frame_spread_rel": round(float(np.nanstd(scales[ok] / s_global)), 4),
        "keyframes_triangulated": int(ok.sum()), "n_keyframes": len(kf),
        "consistent_pixel_fraction": round(float(np.mean(keep_frac)), 3),
        "runtime_s": round(time.time() - t0, 1)})
    return fs
