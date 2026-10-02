"""Step 1: pure per-image metric-depth scale error against LiDAR depth.

For each StrayScanner capture (`data/<id>/`), samples ~N upright RGB frames
spread evenly across the clip, runs each candidate monocular depth model on
every frame, and compares the raw (unscaled) metric depth output against the
StrayScanner LiDAR depth for the SAME frame (confidence == 2 pixels only).

LiDAR depth/confidence is used here PURELY as offline ground truth, exactly
like bench/eval_video_tier.py and bench/eval_photo_tier.py; nothing here
feeds back into the photo/video tiers.

Orientation: StrayScanner stores raw ARKit sensor-landscape frames with no
rotation tag even though the phone was held portrait (same fixed convention
documented in scan2plan.tiers.video.load_video and used by the CLI via
`rotation_k=1`). Both the RGB frame and the LiDAR depth/confidence maps are
rotated 90 degrees clockwise here, identically, before comparison, so pixel
(u, v) in the rotated RGB and the rotated depth/confidence line up.

Metric, per frame:
  ratio = median(pred_depth / lidar_depth) over valid pixels      -- scale bias
  absrel = mean(|pred_depth - lidar_depth| / lidar_depth)         -- accuracy

Aggregated per (capture, model) as the median-of-per-frame ratio (bias) and
a robust spread (MAD-based std), plus the median-of-per-frame AbsRel.

Models compared:
  - depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf  (current; Apache-2.0)
  - depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf   (CC-BY-NC-4.0, non-commercial)
  - apple/DepthPro-hf, own FOV-head focal estimate            (Apple ML Research License)
  - apple/DepthPro-hf, given the TRUE per-frame focal (from StrayScanner's own
    odometry.csv fx/fy, i.e. exactly what an EXIF focal length would give for a
    real photo) -- implemented as an exact post-hoc rescale of the model's own
    output, not an approximation: DepthPro's post_process_depth_estimation
    converts its canonical inverse-depth output to metric depth via
    `depth = focal_pred / (width * raw_inv_depth)` (see
    transformers/models/depth_pro/image_processing_depth_pro.py). That is
    LINEAR in the focal length used, so
    `depth(f_true) = depth(f_pred) * (f_true / f_pred)` exactly reproduces
    what the model would have output had it been given f_true directly (the
    HF API has no argument to inject a focal length into the forward pass;
    the FOV head only ever predicts one).

Usage:
  .venv/bin/python -m bench.eval_depth_scale --n-frames 24 \
      --captures data/c00a170fe1 data/1a8384c3f6 data/c7d28f72c6 --out reports/depth_scale.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scan2plan.io.stray import read_odometry
from scan2plan.tiers import mono_depth as md
from scan2plan.video_io import VideoFrames

MODELS = {
    "depth_anything_small": {"id": md.DEPTH_ANYTHING_SMALL, "focal_hint": False},
    "depth_anything_base": {"id": md.DEPTH_ANYTHING_BASE, "focal_hint": False},
    "depthpro_own_focal": {"id": md.DEPTH_PRO, "focal_hint": False},
    "depthpro_true_focal": {"id": md.DEPTH_PRO, "focal_hint": True},
}


def _frame_true_focal(od_row: np.ndarray) -> float:
    """fx (== fy here) in px, native sensor-landscape resolution; a rotation by
    a multiple of 90 degrees does not change this scalar (square pixels)."""
    return float(od_row[9])


def _load_lidar_depth_conf(path: Path, idx: int, min_conf: int = 2):
    d = np.asarray(Image.open(path / "depth" / f"{idx:06d}.png"), dtype=np.float32) / 1000.0
    c = np.asarray(Image.open(path / "confidence" / f"{idx:06d}.png"))
    d = cv2.rotate(d, cv2.ROTATE_90_CLOCKWISE)
    c = cv2.rotate(c, cv2.ROTATE_90_CLOCKWISE)
    valid = (c >= min_conf) & (d > 0.15) & (d < 5.0)
    return d, valid


def _select_frames(od: np.ndarray, path: Path, n: int) -> list[int]:
    idxs = sorted(int(r[1]) for r in od)
    have = []
    for i in idxs:
        if (path / "depth" / f"{i:06d}.png").exists() and (path / "confidence" / f"{i:06d}.png").exists():
            have.append(i)
    if len(have) <= n:
        return have
    sel = np.linspace(0, len(have) - 1, n).astype(int)
    return sorted({have[k] for k in sel})


def eval_capture(path: Path, n_frames: int, cache_dir: str, device: str | None, progress=print) -> dict:
    od = read_odometry(path)
    od_by_idx = {int(r[1]): r for r in od}
    frame_idxs = _select_frames(od, path, n_frames)
    vf = VideoFrames(path / "rgb.mp4")
    vhash = md.video_content_hash(path / "rgb.mp4")
    progress(f"[{path.name}] {len(frame_idxs)} frames selected")

    per_model_frames: dict[str, list[dict]] = {m: [] for m in MODELS}
    for fi in frame_idxs:
        try:
            rgb = cv2.rotate(vf.get(fi), cv2.ROTATE_90_CLOCKWISE)
        except Exception as exc:
            progress(f"  frame {fi}: decode failed ({exc}); skipped")
            continue
        lidar_d, lidar_v = _load_lidar_depth_conf(path, fi)
        if lidar_v.sum() < 200:
            continue
        f_true = _frame_true_focal(od_by_idx[fi])

        dp_depth, dp_focal = None, None
        for mname, spec in MODELS.items():
            model_id = spec["id"]
            if spec["focal_hint"]:
                # exact post-hoc rescale of the already-computed own-focal
                # DepthPro output (see module docstring); no extra model call.
                if dp_depth is None:
                    continue
                depth = dp_depth * (f_true / dp_focal)
            else:
                depth, focal_px = md.predict_depth(rgb, model_id, f"{vhash}_{fi:06d}",
                                                    cache_dir=cache_dir, device=device, resize_to_input=True)
                if model_id == md.DEPTH_PRO:
                    dp_depth, dp_focal = depth, focal_px
            d_cmp = cv2.resize(depth, (lidar_v.shape[1], lidar_v.shape[0]), interpolation=cv2.INTER_LINEAR)
            ok = lidar_v & np.isfinite(d_cmp) & (d_cmp > 0.05)
            if ok.sum() < 100:
                continue
            ratio = float(np.median(d_cmp[ok] / lidar_d[ok]))
            absrel = float(np.mean(np.abs(d_cmp[ok] - lidar_d[ok]) / lidar_d[ok]))
            per_model_frames[mname].append({"frame": fi, "ratio": ratio, "absrel": absrel, "n_px": int(ok.sum())})

    results = {}
    for mname, rows in per_model_frames.items():
        if not rows:
            results[mname] = {"n_frames": 0}
            continue
        ratios = np.array([r["ratio"] for r in rows])
        absrels = np.array([r["absrel"] for r in rows])
        med = float(np.median(ratios))
        mad_std = float(1.4826 * np.median(np.abs(ratios - med)))
        results[mname] = {
            "n_frames": len(rows),
            "scale_bias_median_pred_over_lidar": round(med, 4),
            "scale_bias_spread_mad_std": round(mad_std, 4),
            "scale_bias_pct": round((med - 1.0) * 100.0, 1),
            "absrel_median": round(float(np.median(absrels)), 4),
            "absrel_mean": round(float(np.mean(absrels)), 4),
        }
    return {"capture": path.name, "n_frames_selected": len(frame_idxs), "models": results}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures", nargs="+", default=[
        "data/c00a170fe1", "data/1a8384c3f6", "data/c7d28f72c6"])
    ap.add_argument("--n-frames", type=int, default=24)
    ap.add_argument("--cache-dir", default=".cache")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="reports/depth_scale.json")
    a = ap.parse_args(argv)

    licenses = {md.DEPTH_ANYTHING_SMALL: md.MODEL_LICENSES[md.DEPTH_ANYTHING_SMALL],
                md.DEPTH_ANYTHING_BASE: md.MODEL_LICENSES[md.DEPTH_ANYTHING_BASE],
                md.DEPTH_PRO: md.MODEL_LICENSES[md.DEPTH_PRO]}

    results = []
    t0 = time.time()
    for cap in a.captures:
        r = eval_capture(Path(cap), a.n_frames, a.cache_dir, a.device)
        print(json.dumps(r, indent=2))
        results.append(r)
    out = {"licenses": licenses, "captures": results, "runtime_sec": round(time.time() - t0, 1)}

    print("\n=== summary (median scale bias % / spread / AbsRel %, pooled across captures) ===")
    for mname in MODELS:
        all_ratios, all_absrel = [], []
        for r in results:
            m = r["models"].get(mname, {})
            if m.get("n_frames"):
                all_ratios.append(m["scale_bias_median_pred_over_lidar"])
                all_absrel.append(m["absrel_median"])
        if all_ratios:
            med = float(np.median(all_ratios))
            print(f"{mname:>22s}: bias={100*(med-1):+6.1f}%  "
                  f"per-capture biases={[round(x,3) for x in all_ratios]}  "
                  f"absrel_med={np.median(all_absrel)*100:5.1f}%")
        else:
            print(f"{mname:>22s}: no data")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
