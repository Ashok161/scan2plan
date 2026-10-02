"""Video tier: RGB-only iPhone clip -> FrameSet.

Pipeline (see module docstrings in mono_depth.py / video_vo.py for the pieces):

  1. Decode forward-only (video_io.VideoFrames), handling iPhone rotation
     metadata so frames come out right-side-up regardless of portrait/landscape
     capture.
  2. Pick ~3 keyframes/second (capped at max_keyframes) -- dense enough for
     robust frame-to-frame feature matching, sparse enough to hit the runtime
     budget.
  3. Self-calibrate the focal length with DepthPro's field-of-view head on a
     sparse subset of frames (median over the subset); DepthPro is ~60x slower
     than the dense model so it is deliberately not run on every frame.
  4. Predict per-keyframe metric depth with Depth-Anything-V2-Metric-Indoor-
     Small (Apache-2.0, fast) at its native output resolution.
  5. Visual odometry: ORB matches between keyframes are lifted to 3D with the
     *source* frame's metric depth and solved with metric PnP -- this gives
     scale directly, no monocular SfM scale ambiguity. Chained sequentially,
     corrected with a sparse pose-graph optimisation over any loop-closure
     edges found by re-matching against spatially-nearby earlier keyframes.
  6. Gravity: back-project a depth sample from several keyframes into the
     (ungravity-aligned) VO world, cluster floor/ceiling normals
     (manhattan.gravity_from_normals) and rotate the whole trajectory so that
     axis is +Y.
  7. Scale-consistency diagnostic: for every sequential VO edge, compare the
     depth of inlier matches predicted by the estimated motion against that
     frame's own independently-predicted depth; a mild global correction is
     applied from the aggregate ratio.

Everything measurable along the way (model ids/licences, focal estimate, VO
inlier stats, loop-closure count, scale-consistency stats, runtime) is written
to FrameSet.meta.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from ..frames import Frame, FrameSet, VIDEO_ERRORS, backproject, to_world
from ..manhattan import floor_and_ceiling, gravity_from_normals, rotation_aligning
from ..video_io import VideoFrames
from . import mono_depth
from .video_vo import chain_sequential, find_loop_closures, optimize_pose_graph

DEPTH_MODEL_ID = mono_depth.DEPTH_ANYTHING_SMALL
CALIB_MODEL_ID = mono_depth.DEPTH_PRO
TARGET_KEYFRAME_FPS = 3.0
ASSUMED_CEILING_HEIGHT_M = 2.5  # residential prior, used only as a last-resort scale sanity-check
N_CALIB_FRAMES = 8
MIN_VO_INLIERS = 25
LOOP_MIN_GAP = 15
LOOP_RADIUS_M = 1.2
LOOP_MIN_INLIERS = 30


def prefetch_models() -> None:
    """Download/cache model weights ahead of time (used by a setup script)."""
    mono_depth.prefetch_models([DEPTH_MODEL_ID, CALIB_MODEL_ID])


def _probe_orientation(path: Path):
    """Handle iPhone rotation metadata.

    OpenCV's FFmpeg backend auto-applies the QuickTime rotation matrix by
    default (CAP_PROP_ORIENTATION_AUTO=1), so for native Camera-app clips
    .read() should already come out right-side-up. This defends against the
    case where the backend *reports* rotated dimensions but doesn't actually
    rotate on decode, by comparing reported vs decoded frame shape and
    rotating ourselves if they disagree.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except Exception:
        pass
    rep_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rep_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    meta = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot decode first frame of {path}")
    act_h, act_w = frame.shape[:2]
    if (act_w, act_h) == (rep_w, rep_h):
        return (lambda im: im), (act_w, act_h), meta
    if (act_h, act_w) == (rep_w, rep_h) and meta in (90, 270):
        k = cv2.ROTATE_90_CLOCKWISE if meta == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE
        return (lambda im, k=k: cv2.rotate(im, k)), (rep_w, rep_h), meta
    return (lambda im: im), (act_w, act_h), meta


_ROT_FNS = {
    0: lambda im: im,
    1: lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE),
    2: lambda im: cv2.rotate(im, cv2.ROTATE_180),
    3: lambda im: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE),
}


def _upright_score(rgb: np.ndarray, cache_dir, vhash: str, fi: int, k: int, device) -> float | None:
    """How well a candidate rotation looks gravity-upright, from a single frame.

    No pose, no real intrinsics available yet (this runs before VO) -- just
    monocular depth + a generic centred-principal-point K. Finds pixels with a
    near-horizontal surface normal (floor/ceiling candidates) and measures
    what fraction sit in the *bottom half* of the image. A phone filming a
    room is overwhelmingly more likely to have floor than ceiling in frame
    (same prior as manhattan.floor_and_ceiling's "lowest peak holding a
    substantial share" rule), and floor is physically below the camera -- so
    the correct upright rotation should push most near-horizontal-normal
    pixels toward the bottom half; a 90-degree-off rotation scatters that same
    physical surface across left/right instead, collapsing the fraction
    toward 0.5. Returns None if the frame has no usable horizontal-plane
    pixels at all (e.g. no floor/ceiling in view), so it can be skipped
    without biasing the vote.
    """
    d, _ = mono_depth.predict_depth(rgb, DEPTH_MODEL_ID, f"{vhash}_orient{fi:06d}_r{k}",
                                     cache_dir=cache_dir, device=device, resize_to_input=False)
    dh, dw = d.shape
    K = np.array([[1.2 * dw, 0.0, dw / 2.0], [0.0, 1.2 * dw, dh / 2.0], [0.0, 0.0, 1.0]])
    valid = np.isfinite(d) & (d > 0.2) & (d < 8.0)
    pts, nrm, ok = backproject(d, valid, K, stride=2)
    if len(nrm) < 300:
        return None
    horiz = np.abs(nrm[:, 1]) > 0.8
    if horiz.sum() < 100:
        return None
    rows, _cols = np.where(ok)
    return float((rows[horiz] > dh / 2).mean())


def _detect_upright_rotation(vf: VideoFrames, rotate_meta_fn, n_total: int, vhash: str,
                              cache_dir, device, n_sample: int = 10) -> tuple[int, dict]:
    """Pick among {0, 90, 180, 270} degree extra rotations on top of whatever
    container rotation metadata already applied, using _upright_score. Falls
    back to 0 (no change) if no frame yields a usable horizontal-plane cluster
    at any rotation (e.g. no floor/ceiling visible anywhere in the sample)."""
    sample = np.linspace(0, max(n_total - 10, 0), n_sample).astype(int).tolist()
    scores = {k: [] for k in range(4)}
    n_used = 0
    for fi in sample:
        try:
            base = rotate_meta_fn(vf.get(int(fi)))
        except IndexError:
            continue
        n_used += 1
        for k in range(4):
            s = _upright_score(_ROT_FNS[k](base), cache_dir, vhash, int(fi), k, device)
            if s is not None:
                scores[k].append(s)
    means = {k: (float(np.mean(v)) if v else -1.0) for k, v in scores.items()}
    best_k = max(means, key=means.get) if any(v for v in scores.values()) else 0
    info = {"mean_bottom_fraction_by_quarter_turn": means, "n_frames_used": n_used,
            "n_frames_with_floor_or_ceiling": {k: len(v) for k, v in scores.items()},
            "chosen_extra_rotation_deg": 90 * best_k}
    return best_k, info


def _select_indices(n_frames: int, fps: float, max_keyframes: int,
                     target_fps: float = TARGET_KEYFRAME_FPS) -> list[int]:
    step = max(1, int(round(fps / target_fps)))
    idx = list(range(0, n_frames, step))
    if len(idx) > max_keyframes:
        sel = np.linspace(0, len(idx) - 1, max_keyframes).astype(int)
        idx = [idx[i] for i in sel]
    return idx


def _estimate_gravity(poses: np.ndarray, depths: list[np.ndarray], K_depth: np.ndarray,
                       n_sample: int = 40, stride: int = 4) -> np.ndarray:
    """World 'up' direction in raw (ungravity-aligned) VO world coordinates.

    gravity_from_normals clusters floor/ceiling normals around a vertical
    axis but can't tell which end is "up" from clustering alone (floor-up and
    ceiling-down normals are forced to the same sign by construction). Sign is
    disambiguated with the same prior manhattan.floor_and_ceiling relies on --
    floor dominates ceiling in a typical scan -- so most scene mass should sit
    *below* the camera along the correct "up" axis.
    """
    n = len(poses)
    sel = np.linspace(0, n - 1, min(n_sample, n)).astype(int)
    normals, rel_pts = [], []
    for k in sel:
        d = depths[k]
        valid = np.isfinite(d) & (d > 0.2) & (d < 8.0)
        pts, nrm, ok = backproject(d, valid, K_depth, stride=stride)
        if len(nrm) == 0:
            continue
        pw, nw = to_world(poses[k], pts, nrm)
        normals.append(nw)
        rel_pts.append(pw - poses[k][:3, 3])
    if not normals:
        return np.array([0.0, 1.0, 0.0])
    g = gravity_from_normals(np.concatenate(normals))
    proj = np.concatenate(rel_pts) @ g
    if np.median(proj) > 0:
        g = -g
    return g


def _sample_world_cloud(poses: np.ndarray, depths: list[np.ndarray], K_depth: np.ndarray,
                         n_sample: int = 60, stride: int = 4):
    n = len(poses)
    sel = np.linspace(0, n - 1, min(n_sample, n)).astype(int)
    pts_all, nrm_all = [], []
    for k in sel:
        d = depths[k]
        valid = np.isfinite(d) & (d > 0.2) & (d < 8.0)
        pts, nrm, ok = backproject(d, valid, K_depth, stride=stride)
        if len(nrm) == 0:
            continue
        pw, nw = to_world(poses[k], pts, nrm)
        pts_all.append(pw)
        nrm_all.append(nw)
    if not pts_all:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.concatenate(pts_all), np.concatenate(nrm_all)


def _ceiling_prior_scale(poses: np.ndarray, depths: list[np.ndarray], K_depth: np.ndarray,
                          assumed_height: float = ASSUMED_CEILING_HEIGHT_M,
                          plausible_range: tuple[float, float] = (1.8, 4.2)) -> dict:
    """Last-resort absolute-scale sanity check.

    There's no IMU/ARKit and no absolute scale reference in a bare RGB clip,
    so a systematic metric-depth bias (see FrameSet.meta['scale_consistency'],
    which only checks *relative* cross-view consistency) can leave the whole
    reconstruction globally too big or too small -- we saw this directly on
    the dev captures (e.g. an estimated ceiling height of several metres).
    Residential ceiling height is a strong, nearly-universal prior (~2.2-3.0 m);
    if our own floor/ceiling detector confidently finds both planes and the
    gap is wildly outside plausible range, rescale depth+poses to match the
    prior rather than silently emitting an implausible reconstruction.
    """
    pts, nrm = _sample_world_cloud(poses, depths, K_depth)
    out = {"applied": False, "detected_height_m": None, "scale_factor": 1.0}
    if len(pts) < 500:
        return out
    try:
        floor, ceiling = floor_and_ceiling(pts, nrm)
    except RuntimeError:
        return out
    if ceiling is None:
        return out
    height = ceiling[0] - floor[0]
    out["detected_height_m"] = float(height)
    if plausible_range[0] <= height <= plausible_range[1]:
        return out
    factor = float(np.clip(assumed_height / max(height, 1e-3), 0.3, 3.0))
    out["applied"] = True
    out["scale_factor"] = factor
    return out


def load_video(path: str | Path, cache_dir: str | Path = ".cache", device: str | None = None,
               max_keyframes: int = 400, progress=None, rotation_k: int | None = None) -> FrameSet:
    """rotation_k: force k quarter-turns clockwise (skip content-based upright detection).

    Used for StrayScanner rgb.mp4, which stores raw ARKit frames in the sensor's native
    landscape orientation with no rotation tag: a portrait-held capture is always 90 deg CW
    off. That is a fixed property of the file format, not per-capture sensor data.
    """
    t_start = time.time()
    path = Path(path)

    def log(msg: str):
        if progress:
            progress(msg)

    vhash = mono_depth.video_content_hash(path)
    rotate_meta_fn, (W0, H0), orient_meta = _probe_orientation(path)

    vf = VideoFrames(path)
    n_total, fps = vf.n_frames, vf.fps
    idx = _select_indices(n_total, fps, max_keyframes)
    log(f"video {path.name}: {n_total} frames @ {fps:.1f}fps -> {len(idx)} keyframes")

    # Container rotation metadata (if any) is honoured above, but it's absent
    # on e.g. StrayScanner exports even though the phone was held in portrait
    # -- monocular depth/focal models are trained on upright photos, so detect
    # any further 90-degree misalignment content-wise before running them for
    # real.
    if rotation_k is not None:
        extra_k = int(rotation_k) % 4
        orient_info = {"chosen_extra_rotation_deg": 90 * extra_k, "n_frames_used": 0,
                       "mean_bottom_fraction_by_quarter_turn": None,
                       "source": "forced by caller (capture-format convention)"}
    else:
        extra_k, orient_info = _detect_upright_rotation(vf, rotate_meta_fn, n_total, vhash,
                                                          cache_dir, device)
        orient_info["source"] = "content-based detection"
    rotate_fn = (lambda im, k=extra_k: _ROT_FNS[k](rotate_meta_fn(im)))
    W, H = (H0, W0) if extra_k % 2 else (W0, H0)
    log(f"upright detection: +{orient_info['chosen_extra_rotation_deg']} deg on top of "
        f"container metadata ({orient_info['n_frames_used']} probe frames, "
        f"scores={orient_info['mean_bottom_fraction_by_quarter_turn']})")

    calib_order = np.linspace(0, len(idx) - 1, min(N_CALIB_FRAMES, len(idx))).astype(int)
    calib_set = {idx[c] for c in calib_order}

    grays: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    focals: list[float] = []
    t0 = time.time()
    for ci, fi in enumerate(idx):
        try:
            rgb = rotate_fn(vf.get(fi))
        except IndexError:
            # cv2's reported CAP_PROP_FRAME_COUNT can overshoot the actually
            # decodable frames on iPhone HEVC files; truncate rather than crash.
            log(f"decode stopped at keyframe {ci}/{len(idx)} (frame {fi}): "
                f"reported frame count overshot the decodable stream")
            break
        grays.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        d, _ = mono_depth.predict_depth(rgb, DEPTH_MODEL_ID, f"{vhash}_{fi:06d}",
                                         cache_dir=cache_dir, device=device, resize_to_input=False)
        depths.append(d)
        if fi in calib_set:
            _, f_px = mono_depth.predict_depth(rgb, CALIB_MODEL_ID, f"{vhash}_{fi:06d}",
                                                cache_dir=cache_dir, device=device,
                                                resize_to_input=True)
            if f_px is not None and np.isfinite(f_px) and 0.3 * W < f_px < 5.0 * W:
                focals.append(f_px)
        if progress and ci % 50 == 0:
            log(f"depth {ci}/{len(idx)}")
    idx = idx[: len(grays)]
    if not idx:
        raise RuntimeError(f"could not decode any frames from {path}")
    depth_runtime = time.time() - t0

    if focals:
        focal_px = float(np.median(focals))
        focal_source = CALIB_MODEL_ID
    else:
        focal_px = 1.2 * W  # phone-camera heuristic fallback (~55 deg hfov)
        focal_source = "fallback_heuristic"
    K_full = np.array([[focal_px, 0.0, W / 2.0], [0.0, focal_px, H / 2.0], [0.0, 0.0, 1.0]])
    log(f"focal_px={focal_px:.1f} from {len(focals)}/{len(calib_order)} calib frames ({focal_source})")

    dh, dw = depths[0].shape
    sx, sy = dw / W, dh / H
    K_depth = K_full.copy()
    K_depth[0, :] *= sx
    K_depth[1, :] *= sy

    # ---- visual odometry ----
    t0 = time.time()
    poses, edges, scale_ratios, failed, kps, dess = chain_sequential(
        grays, depths, K_full, min_inliers=MIN_VO_INLIERS, depth_scale=(sx, sy))
    vo_runtime = time.time() - t0

    t0 = time.time()
    loops = find_loop_closures(poses, kps, dess, depths, K_full, min_gap=LOOP_MIN_GAP,
                                radius=LOOP_RADIUS_M, min_inliers=LOOP_MIN_INLIERS,
                                depth_scale=(sx, sy))
    loop_runtime = time.time() - t0
    all_edges = edges + loops

    t0 = time.time()
    poses = optimize_pose_graph(poses, all_edges)
    pg_runtime = time.time() - t0

    # ---- gravity alignment (no IMU: floor/ceiling normal clustering) ----
    g_raw = _estimate_gravity(poses, depths, K_depth)
    R_align = rotation_aligning(g_raw, np.array([0.0, 1.0, 0.0]))
    T_align = np.eye(4)
    T_align[:3, :3] = R_align
    poses = np.einsum("ij,kjl->kil", T_align, poses)

    # ---- scale-consistency diagnostic + mild global correction ----
    scale_ratios = scale_ratios[np.isfinite(scale_ratios)]
    scale_ratios = scale_ratios[(scale_ratios > 0.1) & (scale_ratios < 10.0)]
    if len(scale_ratios) >= 20:
        ratio_med = float(np.median(scale_ratios))
        ratio_std = float(1.4826 * np.median(np.abs(scale_ratios - ratio_med)))  # robust (MAD) std
        s = float(np.clip(1.0 / max(ratio_med, 1e-3), 0.7, 1.4))
    else:
        ratio_med, ratio_std, s = float("nan"), float("nan"), 1.0
    if s != 1.0:
        depths = [d * s for d in depths]
        poses = poses.copy()
        poses[:, :3, 3] *= s

    # ---- absolute-scale sanity check (residential ceiling-height prior) ----
    ceiling_check = _ceiling_prior_scale(poses, depths, K_depth)
    if ceiling_check["applied"]:
        s2 = ceiling_check["scale_factor"]
        log(f"ceiling-height prior correction: detected {ceiling_check['detected_height_m']:.2f}m, "
            f"assuming {ASSUMED_CEILING_HEIGHT_M}m -> global scale x{s2:.3f}")
        depths = [d * s2 for d in depths]
        poses = poses.copy()
        poses[:, :3, 3] *= s2

    n_inlier_seq = [e.n_inliers for e in edges]
    frames = []
    for k, fi in enumerate(idx):
        d_k = depths[k]
        frames.append(Frame(
            index=k,
            timestamp=float(fi) / fps,
            T_wc=poses[k],
            K_depth=K_depth,
            K_rgb=K_full,
            load_depth=_depth_loader(d_k),
            load_rgb=(lambda i=fi: rotate_fn(vf.get(i))),
        ))

    meta = {
        "source_frames": n_total,
        "fps": fps,
        "n_keyframes": len(idx),
        "frame_size_wh": [W, H],
        "orientation_meta": orient_meta,
        "orientation_detection": orient_info,
        "depth_model": DEPTH_MODEL_ID,
        "depth_model_license": mono_depth.MODEL_LICENSES[DEPTH_MODEL_ID],
        "depth_resolution_wh": [dw, dh],
        "calib_model": CALIB_MODEL_ID,
        "calib_model_license": mono_depth.MODEL_LICENSES[CALIB_MODEL_ID],
        "focal_px_estimate": focal_px,
        "focal_source": focal_source,
        "focal_samples": focals,
        "vo": {
            "n_sequential_edges": len(edges),
            "n_failed_frames": len(failed),
            "failed_indices": failed,
            "mean_inliers": float(np.mean(n_inlier_seq)) if n_inlier_seq else 0.0,
            "min_inliers": int(np.min(n_inlier_seq)) if n_inlier_seq else 0,
        },
        "loop_closure": {
            "n_loop_edges": len(loops),
            "mean_inliers": float(np.mean([e.n_inliers for e in loops])) if loops else 0.0,
            "pose_graph_optimized": bool(loops),
        },
        "scale_consistency": {
            "n_samples": int(len(scale_ratios)),
            "ratio_median": ratio_med,
            "ratio_std": ratio_std,
            "global_correction_applied": s,
        },
        "ceiling_prior_check": ceiling_check,
        "gravity_up_raw": g_raw.tolist(),
        "runtime_sec": {
            "depth_and_calib": depth_runtime,
            "vo_sequential": vo_runtime,
            "loop_closure_search": loop_runtime,
            "pose_graph_opt": pg_runtime,
            "total": time.time() - t_start,
        },
        "device": device or mono_depth._device(None),
        "seed": 0,
    }
    log(f"load_video done in {meta['runtime_sec']['total']:.1f}s "
        f"({len(idx)} keyframes, {len(loops)} loop edges)")
    return FrameSet(tier="video", frames=frames, errors=VIDEO_ERRORS, source=str(path), meta=meta)


def _depth_loader(depth: np.ndarray):
    def load():
        valid = np.isfinite(depth) & (depth > 0.2) & (depth < 8.0)
        return depth, valid
    return load
