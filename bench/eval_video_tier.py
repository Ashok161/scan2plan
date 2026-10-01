"""Evaluate the video tier against StrayScanner LiDAR ground truth.

For each capture directory, calls scan2plan.tiers.video.load_video on ONLY
``rgb.mp4`` (exactly the production video-tier contract -- no depth, no
odometry). The capture's ``odometry.csv`` (ARKit poses) and ``depth/*.png``
(LiDAR depth) are then used *purely as evaluation ground truth*: Sim3-aligned
ATE, global scale error, raw-scale drift along the trajectory, and depth
AbsRel against the LiDAR frames at the same video-frame indices (StrayScanner
exports one video frame per odometry row, so indices line up directly, no
timestamp matching needed).

Usage:
    .venv/bin/python bench/eval_video_tier.py --captures data/c00a170fe1 data/1a8384c3f6 data/c7d28f72c6 \
        --max-keyframes 400 --cache-dir .cache
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from scan2plan.io.stray import DEPTH_H, DEPTH_W, read_odometry
from scan2plan.tiers.video import load_video


def gt_poses(path: Path) -> dict[int, np.ndarray]:
    od = read_odometry(path)
    out = {}
    for row in od:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(row[5:9]).as_matrix()
        T[:3, 3] = row[2:5]
        out[int(row[1])] = T
    return out


def gt_depth(path: Path, idx: int, min_conf: int = 2):
    d = np.asarray(Image.open(path / "depth" / f"{idx:06d}.png"), dtype=np.float32) / 1000.0
    c = np.asarray(Image.open(path / "confidence" / f"{idx:06d}.png"))
    valid = (c >= min_conf) & (d > 0.15) & (d < 5.0)
    return d, valid


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Least-squares similarity transform (s, R, t) with dst ~= s * R @ src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    sigma = Dc.T @ Sc / len(src)
    U, D, Vt = np.linalg.svd(sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_s = (Sc ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s)
    t = mu_d - s * R @ mu_s
    return s, R, t


def eval_capture(path: str | Path, max_keyframes: int = 400, cache_dir: str | Path = ".cache",
                  device: str | None = None) -> dict:
    path = Path(path)
    print(f"=== {path.name} ===")
    t0 = time.time()
    fs = load_video(path / "rgb.mp4", cache_dir=cache_dir, max_keyframes=max_keyframes,
                     device=device, progress=lambda m: print("  ", m))
    runtime = time.time() - t0

    gtp = gt_poses(path)
    fps = fs.meta["fps"]
    frame_idx = [int(round(f.timestamp * fps)) for f in fs.frames]

    est_pos, gt_pos, matched = [], [], []
    for f, fi in zip(fs.frames, frame_idx):
        if fi in gtp:
            est_pos.append(f.position)
            gt_pos.append(gtp[fi][:3, 3])
            matched.append(fi)
    est_pos, gt_pos = np.array(est_pos), np.array(gt_pos)

    s, R, t = umeyama(est_pos, gt_pos)
    aligned = s * (est_pos @ R.T) + t
    ate = float(np.sqrt(np.mean(np.sum((aligned - gt_pos) ** 2, axis=1))))
    scale_error_pct = abs(s - 1.0) * 100.0

    # Raw-scale drift: our own metric step distances vs ground truth, with no
    # Sim3 fit at all (R cancels in the norm) -- checks whether the tier's
    # intrinsic metric scale (from depth-anchored PnP) holds steady over the
    # trajectory, independent of any post-hoc alignment.
    d_est = np.linalg.norm(np.diff(est_pos, axis=0), axis=1)
    d_gt = np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)
    ok = d_gt > 0.01
    ratio = d_est[ok] / d_gt[ok]
    half = max(len(ratio) // 2, 1)
    s1, s2 = np.median(ratio[:half]), np.median(ratio[half:]) if len(ratio) > half else np.median(ratio[:half])
    scale_ratio_std = float(np.std(ratio)) if len(ratio) else float("nan")
    scale_drift_half_pct = float(abs(s2 - s1) / max(s1, 1e-6) * 100.0) if len(ratio) else float("nan")

    absrels = []
    for f, fi in zip(fs.frames, frame_idx):
        if fi not in gtp:
            continue
        d_est_map, _ = f.load_depth()
        d_gt_map, v_gt = gt_depth(path, fi)
        d_est_r = np.asarray(
            Image.fromarray(d_est_map.astype(np.float32)).resize((DEPTH_W, DEPTH_H), Image.BILINEAR))
        ok_px = v_gt & np.isfinite(d_est_r) & (d_est_r > 0.1)
        if ok_px.sum() < 50:
            continue
        absrels.append(float(np.mean(np.abs(d_est_r[ok_px] - d_gt_map[ok_px]) / d_gt_map[ok_px])))
    absrel = float(np.mean(absrels)) if absrels else float("nan")

    result = {
        "capture": path.name,
        "n_source_frames": fs.meta["source_frames"],
        "duration_sec": fs.meta["source_frames"] / fps,
        "n_keyframes": len(fs.frames),
        "n_matched_gt_poses": len(est_pos),
        "ate_m_sim3": ate,
        "scale_error_pct_sim3": scale_error_pct,
        "scale_ratio_std_raw": scale_ratio_std,
        "scale_drift_half_pct_raw": scale_drift_half_pct,
        "depth_absrel": absrel,
        "n_depth_frames_compared": len(absrels),
        "vo_sequential_edges": fs.meta["vo"]["n_sequential_edges"],
        "vo_failed_frames": fs.meta["vo"]["n_failed_frames"],
        "vo_mean_inliers": fs.meta["vo"]["mean_inliers"],
        "loop_edges": fs.meta["loop_closure"]["n_loop_edges"],
        "pose_graph_optimized": fs.meta["loop_closure"]["pose_graph_optimized"],
        "scale_consistency_ratio_median": fs.meta["scale_consistency"]["ratio_median"],
        "scale_consistency_ratio_std": fs.meta["scale_consistency"]["ratio_std"],
        "global_scale_correction_applied": fs.meta["scale_consistency"]["global_correction_applied"],
        "focal_px_estimate": fs.meta["focal_px_estimate"],
        "focal_source": fs.meta["focal_source"],
        "device": fs.meta["device"],
        "runtime_sec": runtime,
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", nargs="+", default=[
        "data/c00a170fe1", "data/1a8384c3f6", "data/c7d28f72c6"])
    ap.add_argument("--max-keyframes", type=int, default=400)
    ap.add_argument("--cache-dir", default=".cache")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="write JSON results to this path")
    args = ap.parse_args()

    results = []
    for cap in args.captures:
        results.append(eval_capture(cap, max_keyframes=args.max_keyframes,
                                     cache_dir=args.cache_dir, device=args.device))

    print("\n=== summary ===")
    for r in results:
        print(f"{r['capture']:>12s}  ATE={r['ate_m_sim3']*100:5.1f}cm  "
              f"scale_err={r['scale_error_pct_sim3']:5.1f}%  "
              f"depth_AbsRel={r['depth_absrel']*100:5.1f}%  "
              f"loop_edges={r['loop_edges']:3d}  "
              f"vo_failed={r['vo_failed_frames']:3d}/{r['n_keyframes']:3d}  "
              f"runtime={r['runtime_sec']:6.1f}s")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
