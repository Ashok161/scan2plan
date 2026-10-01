"""Synthetic staged-damage benchmark.

We have no staged-damage capture: the three StrayScanner captures in
`data/` are all of the same clean apartment, and there is no iPhone to shoot
a furnished room with real staged damage. This script is the disclosed
substitute: it composites procedurally generated damage textures (water
stain, mould, crack, hole) onto a real wall plane in a real capture,
*using the capture's own camera poses and intrinsics* so the synthetic
damage looks correctly perspective-warped and consistent across many real
viewpoints, with a known physical size and location. It is not a
replacement for a real staged-damage room; it validates that the detector
pipeline (damage.py) finds the right class in the right place at the right
metric size, and reports recall/precision/extent error honestly as
"synthetic benchmark" numbers, never mixed into any real-capture metric.

Two things this script does:

  1. `build_fixture_plan(fs)` -- a hand-built, single-room Plan (real wall
     lines + floor fit from the capture's own LiDAR, via
     scan2plan.manhattan / scan2plan.walls / scan2plan.fusion) used because
     the coordinator's scan2plan/layout.py did not exist yet when this was
     written. This is a development fixture, not a layout implementation;
     once layout.build_plan lands it should produce a comparable Plan for
     the same captures (sanity-checked in tests/test_damage.py).

  2. The synthetic-damage harness: `make_instances`, `composite_frameset`,
     `run_benchmark` (recall/precision/extent error against known ground
     truth), and `measure_clean_fp_rate` (false-positive rate on the three
     *unmodified* clean captures -- the honesty check required alongside
     the synthetic recall numbers).

Run directly: `.venv/bin/python bench/synthetic_damage.py --data-dir data`
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scan2plan.frames import FrameSet, Frame
from scan2plan.fusion import fuse, select_keyframes, voxel_downsample
from scan2plan.io.stray import load_stray
from scan2plan.manhattan import dominant_yaw, floor_and_ceiling, yaw_rotation
from scan2plan.measure import Measurement, combine
from scan2plan.plan_types import Plan, Room, Wall
from scan2plan.walls import extract_wall_lines, find_gaps
from scan2plan.damage import detect_damage, _surface_basis

RNG_SEED = 20260101


# --------------------------------------------------------------------------
# 1. Fixture Plan (dev stand-in for the coordinator's layout.build_plan)
# --------------------------------------------------------------------------
def build_fixture_plan(fs: FrameSet, voxel: float = 0.03, max_frames: int = 400) -> Plan:
    """Single-room Plan fit directly from LiDAR points. See module docstring."""
    keyframes = select_keyframes(fs, max_frames=max_frames)
    cloud = fuse(fs, keyframes)
    pts_ds, nrm_ds, _ = voxel_downsample(cloud.points, voxel, cloud.normals)
    norm = np.linalg.norm(nrm_ds, axis=1, keepdims=True)
    keep = norm[:, 0] > 1e-6
    pts_ds, nrm_ds = pts_ds[keep], nrm_ds[keep] / norm[keep]

    yaw, yaw_score = dominant_yaw(nrm_ds)
    R = yaw_rotation(yaw)[:3, :3]
    pts_r = pts_ds @ R.T
    nrm_r = nrm_ds @ R.T

    floor, ceiling = floor_and_ceiling(pts_r, nrm_r)
    floor_y, floor_sigma, _ = floor
    ceiling_y = ceiling[0] if ceiling else None
    ceiling_sigma = ceiling[1] if ceiling else 0.05

    horiz = np.abs(nrm_r[:, 1]) < 0.2
    xz = pts_r[horiz][:, [0, 2]]
    nxz = nrm_r[horiz][:, [0, 2]]
    lines = extract_wall_lines(xz, nxz)
    gaps = find_gaps(lines)
    gap_set = {id(g.line): g for g in gaps}   # noqa: F841 (kept for future opening use)

    walls = []
    endpoints = []
    for k, ln in enumerate(lines):
        if not ln.segments:
            continue
        a = min(s[0] for s in ln.segments)
        b = max(s[1] for s in ln.segments)
        covered = sum(e - s for s, e in ln.segments)
        if ln.axis == 0:
            start, end = np.array([ln.c, a]), np.array([ln.c, b])
            normal_in = np.array([float(ln.sign), 0.0])
        else:
            start, end = np.array([a, ln.c]), np.array([b, ln.c])
            normal_in = np.array([0.0, float(ln.sign)])
        length_val = float(b - a)
        if length_val < 0.5:
            continue
        top = ceiling_y if ceiling_y is not None else floor_y + 2.5
        height_val = float(top - floor_y)
        wid = f"room0.W{k}"
        walls.append(Wall(
            id=wid, start=start, end=end, normal_in=normal_in,
            length=Measurement(length_val, max(ln.c_sigma, 0.01), "m", method="fixture: wall-line segment extent"),
            height=Measurement(height_val, combine(floor_sigma, ceiling_sigma), "m", method="fixture: floor/ceiling fit"),
            offset_sigma=float(ln.c_sigma), coverage=float(covered / max(length_val, 1e-6)),
        ))
        endpoints.extend([start, end])

    if not walls:
        raise RuntimeError("fixture: no wall lines extracted; capture too sparse")

    endpoints = np.array(endpoints)
    from shapely.geometry import MultiPoint
    hull = MultiPoint(endpoints).convex_hull
    poly_xy = np.array(hull.exterior.coords[:-1])
    if cv2.contourArea(poly_xy.astype(np.float32)) < 0:
        poly_xy = poly_xy[::-1]
    perim = float(hull.length)
    area_val = float(hull.area)
    sigma_c = float(np.mean([w.offset_sigma for w in walls]))
    room = Room(
        id="room0", name="room0", kind="room", polygon=poly_xy, floor_y=float(floor_y),
        ceiling_y=float(ceiling_y) if ceiling_y is not None else None,
        area=Measurement(area_val, perim * sigma_c / np.sqrt(2.0), "m2", method="fixture: wall-endpoint convex hull"),
        perimeter=Measurement(perim, perim * sigma_c, "m", method="fixture: wall-endpoint convex hull"),
        ceiling_height=Measurement(
            float((ceiling_y if ceiling_y is not None else floor_y + 2.5) - floor_y),
            combine(floor_sigma, ceiling_sigma), "m", method="fixture: floor/ceiling fit"),
        walls=walls,
    )
    T_align = yaw_rotation(yaw)
    return Plan(tier=fs.tier, rooms=[room], adjacency=[], T_align=T_align, floor_y=float(floor_y),
               footprint_area=room.area,
               warnings=["fixture plan: hand-built for damage-stage development, "
                        f"not the coordinator's layout.build_plan (yaw_score={yaw_score:.2f}, "
                        f"ceiling_observed={ceiling_y is not None})"])


# --------------------------------------------------------------------------
# 2. Synthetic damage textures
# --------------------------------------------------------------------------
PX_PER_M = 400   # texture raster density; purely a rendering resolution, not detector resolution


def _smooth_noise(h: int, w: int, rng: np.random.Generator, blur: int = 15) -> np.ndarray:
    n = rng.random((h, w)).astype(np.float32)
    k = blur | 1
    return cv2.GaussianBlur(n, (k, k), 0)


def render_water_stain(width_m: float, height_m: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w, h = int(width_m * PX_PER_M), int(height_m * PX_PER_M)
    noise = _smooth_noise(h, w, rng, blur=w // 3 | 1)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    rad = np.sqrt(((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2)
    alpha = np.clip(1.2 - rad - 0.4 * (1 - noise), 0, 1) ** 1.5
    alpha = cv2.GaussianBlur(alpha, (9, 9), 0)
    rgba = np.zeros((h, w, 4), np.uint8)
    brown = np.array([120, 95, 60])      # BGR-ish mid tan-brown
    rgba[..., 0] = brown[0] * (0.7 + 0.3 * noise)
    rgba[..., 1] = brown[1] * (0.7 + 0.3 * noise)
    rgba[..., 2] = brown[2] * (0.7 + 0.3 * noise)
    rgba[..., 3] = (alpha * 200).astype(np.uint8)
    return rgba


def render_mould(width_m: float, height_m: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w, h = int(width_m * PX_PER_M), int(height_m * PX_PER_M)
    base = _smooth_noise(h, w, rng, blur=w // 4 | 1)
    base = (base - base.min()) / (base.max() - base.min() + 1e-6)
    speck = rng.random((h, w)) > 0.90
    speck = cv2.GaussianBlur(speck.astype(np.float32), (3, 3), 0)
    yy, xx = np.mgrid[0:h, 0:w]
    rad = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    envelope = np.clip(1.1 - rad, 0, 1)
    alpha = np.clip(0.55 * base * envelope + 0.9 * speck * envelope, 0, 1)
    rgba = np.zeros((h, w, 4), np.uint8)
    dark_green = np.array([30, 45, 25])
    rgba[..., 0] = dark_green[0]
    rgba[..., 1] = dark_green[1]
    rgba[..., 2] = dark_green[2]
    rgba[..., 3] = (alpha * 190).astype(np.uint8)
    return rgba


def render_crack(length_m: float, thickness_m: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # render in a square canvas sized by the long dimension so a near-vertical
    # or near-horizontal random walk both fit; caller places/rotates in uv.
    n = max(int(length_m * PX_PER_M), 32)
    canvas = np.zeros((n, n, 4), np.uint8)
    x, y = 2.0, n / 2.0
    pts = [(x, y)]
    step = n / 40.0
    for _ in range(40):
        x += step
        y += rng.normal(0, step * 0.5)
        y = np.clip(y, 2, n - 3)
        pts.append((x, y))
        if x >= n - 2:
            break
    pts = np.array(pts, np.int32)
    thickness_px = max(int(thickness_m * PX_PER_M), 1)
    cv2.polylines(canvas, [pts], False, (15, 15, 15, 255), thickness=thickness_px, lineType=cv2.LINE_AA)
    canvas[..., 3] = cv2.GaussianBlur(canvas[..., 3], (3, 3), 0)
    return canvas


def render_hole(width_m: float, height_m: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w, h = int(width_m * PX_PER_M), int(height_m * PX_PER_M)
    canvas = np.zeros((h, w, 4), np.uint8)
    cx, cy = w / 2, h / 2
    pts = []
    n_pts = 10
    for i in range(n_pts):
        ang = 2 * np.pi * i / n_pts
        r = min(w, h) / 2 * (0.6 + 0.4 * rng.random())
        pts.append((cx + r * np.cos(ang), cy + r * np.sin(ang)))
    pts = np.array(pts, np.int32)
    cv2.fillPoly(canvas, [pts], (8, 8, 8, 255))
    canvas[..., 3] = cv2.GaussianBlur(canvas[..., 3], (5, 5), 0)
    ring = cv2.dilate(canvas[..., 3], np.ones((9, 9), np.uint8)) - canvas[..., 3]
    canvas[..., 0] = np.where(ring > 0, 60, canvas[..., 0])
    canvas[..., 1] = np.where(ring > 0, 55, canvas[..., 1])
    canvas[..., 2] = np.where(ring > 0, 50, canvas[..., 2])
    canvas[..., 3] = np.maximum(canvas[..., 3], (ring * 0.6).astype(np.uint8))
    return canvas


RENDERERS = {
    "water_stain": lambda w, h, s: render_water_stain(w, h, s),
    "mould": lambda w, h, s: render_mould(w, h, s),
    "crack": lambda w, h, s: render_crack(max(w, h), min(w, h), s),
    "hole_or_impact": lambda w, h, s: render_hole(w, h, s),
}


# --------------------------------------------------------------------------
# 3. Instance definitions + compositing into real frames
# --------------------------------------------------------------------------
def make_instances(wall_extent_uv: tuple[float, float, float, float]) -> list[dict]:
    """Ground-truth synthetic damage instances, placed inside the wall's observed uv box.

    Spans four classes (more than the two the spec requires) at distinct,
    non-overlapping locations with known physical width/height in metres.
    """
    u0, u1, v0, v1 = wall_extent_uv
    uc, vc = (u0 + u1) / 2, (v0 + v1) / 2
    span_u, span_v = (u1 - u0), (v1 - v0)
    specs = [
        {"id": "syn.water_stain.1", "class": "water_stain", "width": 0.45, "height": 0.35,
         "center_uv": (uc - 0.30 * span_u, vc + 0.15 * span_v)},
        {"id": "syn.mould.1", "class": "mould", "width": 0.28, "height": 0.22,
         "center_uv": (uc + 0.28 * span_u, vc - 0.05 * span_v)},
        {"id": "syn.crack.1", "class": "crack", "width": 0.6, "height": 0.03,
         "center_uv": (uc, vc - 0.30 * span_v)},
        {"id": "syn.hole_or_impact.1", "class": "hole_or_impact", "width": 0.18, "height": 0.18,
         "center_uv": (uc - 0.05 * span_u, vc + 0.33 * span_v)},
    ]
    out = []
    for s in specs:
        cu, cv = s["center_uv"]
        if not (u0 + 0.1 < cu < u1 - 0.1 and v0 + 0.1 < cv < v1 - 0.1):
            continue   # keep instances inside the actually-observed wall patch
        out.append(s)
    return out


def _project_plan_point(p_plan: np.ndarray, T_align_inv: np.ndarray, T_wc: np.ndarray, K: np.ndarray):
    p_world = T_align_inv[:3, :3] @ p_plan + T_align_inv[:3, 3]
    R_wc, t_wc = T_wc[:3, :3], T_wc[:3, 3]
    p_cam = R_wc.T @ (p_world - t_wc)
    if p_cam[2] <= 0.05:
        return None
    x = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
    y = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
    return np.array([x, y]), p_cam[2]


def composite_frameset(fs: FrameSet, plan: Plan, wall_id: str, instances: list[dict],
                       n_frames: int = 40, img_size: tuple[int, int] = (1920, 1440)) -> tuple[FrameSet, list[dict]]:
    """Return a FrameSet whose RGB on selected frames carries the composited damage."""
    surface = next(s for s in plan.surfaces() if s["id"] == wall_id)
    origin, u_hat, v_hat, uv_corners = _surface_basis(surface)
    T_align_inv = np.linalg.inv(plan.T_align)

    textures = {}
    for inst in instances:
        rgba = RENDERERS[inst["class"]](inst["width"], inst["height"], seed=hash(inst["id"]) % (2**31))
        textures[inst["id"]] = rgba

    # pick frames that view the wall roughly frontally within 0.6-4m
    n = surface["normal"] / np.linalg.norm(surface["normal"])
    candidates = []
    center_plan = origin + 0.5 * (uv_corners[:, 0].max() + uv_corners[:, 0].min()) * u_hat \
        + 0.5 * (uv_corners[:, 1].max() + uv_corners[:, 1].min()) * v_hat
    for fr in fs.frames:
        proj = _project_plan_point(center_plan, T_align_inv, fr.T_wc, fr.K_rgb)
        if proj is None:
            continue
        xy, z = proj
        if not (0.6 <= z <= 4.0 and 0 <= xy[0] < img_size[0] and 0 <= xy[1] < img_size[1]):
            continue
        view_dir = fr.T_wc[:3, 2]   # camera +Z (forward, OpenCV)
        facing = float(-view_dir @ (T_align_inv[:3, :3] @ n))
        if facing < 0.5:
            continue
        candidates.append(fr.index)
    if len(candidates) == 0:
        raise RuntimeError("synthetic bench: no frame views the chosen wall frontally")
    sel = sorted({candidates[i] for i in np.linspace(0, len(candidates) - 1, min(n_frames, len(candidates))).astype(int)})

    composite_cache: dict[int, np.ndarray] = {}

    def make_loader(frame_idx: int, orig_loader):
        def _load():
            if frame_idx in composite_cache:
                return composite_cache[frame_idx]
            rgb = orig_loader().copy()
            h, w = rgb.shape[:2]
            fr = next(f for f in fs.frames if f.index == frame_idx)
            canvas = rgb.copy()
            for inst in instances:
                cu, cv = inst["center_uv"]
                hw, hh = inst["width"] / 2, inst["height"] / 2
                corners_uv = [(cu - hw, cv - hh), (cu + hw, cv - hh), (cu + hw, cv + hh), (cu - hw, cv + hh)]
                img_pts = []
                ok = True
                for (uu, vv) in corners_uv:
                    p_plan = origin + uu * u_hat + vv * v_hat
                    proj = _project_plan_point(p_plan, T_align_inv, fr.T_wc, fr.K_rgb)
                    if proj is None:
                        ok = False
                        break
                    img_pts.append(proj[0])
                if not ok:
                    continue
                tex = textures[inst["id"]]
                th, tw = tex.shape[:2]
                src = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], np.float32)
                dst = np.array(img_pts, np.float32)
                H = cv2.getPerspectiveTransform(src, dst)
                warped = cv2.warpPerspective(tex, H, (w, h), flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
                alpha = (warped[..., 3:4].astype(np.float32)) / 255.0
                canvas = (canvas.astype(np.float32) * (1 - alpha) + warped[..., :3].astype(np.float32) * alpha).astype(np.uint8)
            composite_cache[frame_idx] = canvas
            return canvas
        return _load

    new_frames = []
    for fr in fs.frames:
        if fr.index in sel and fr.load_rgb is not None:
            new_fr = Frame(index=fr.index, timestamp=fr.timestamp, T_wc=fr.T_wc, K_depth=fr.K_depth,
                           load_depth=fr.load_depth, load_rgb=make_loader(fr.index, fr.load_rgb),
                           K_rgb=fr.K_rgb, group=fr.group)
            new_frames.append(new_fr)
        else:
            new_frames.append(fr)
    fs2 = FrameSet(tier=fs.tier, frames=new_frames, errors=fs.errors, source=fs.source + "+synthetic",
                  meta={**fs.meta, "synthetic_frames": sel})

    ground_truth = []
    for inst in instances:
        ground_truth.append({**inst, "surface_id": wall_id, "n_frames_composited": len(sel)})
    return fs2, ground_truth


# --------------------------------------------------------------------------
# 4. Evaluation
# --------------------------------------------------------------------------
def evaluate_detections(detections: list[dict], ground_truth: list[dict], wall_id: str,
                        dist_thresh: float = 0.35) -> dict:
    gt_on_wall = [g for g in ground_truth if g["surface_id"] == wall_id]
    det_on_wall = [d for d in detections if d["surface_id"] == wall_id]
    matched_gt, matched_det = set(), set()
    matches = []
    for gi, g in enumerate(gt_on_wall):
        best_j, best_d = None, 1e9
        for dj, d in enumerate(det_on_wall):
            if dj in matched_det or d["class"] != g["class"]:
                continue
            # centroid in surface uv: need basis-independent distance -> use detector's own uv centroid approx
            dc = np.array(d["polygon_surface"]).mean(axis=0) if d["polygon_surface"] else np.array([1e9, 1e9])
            gc = np.array(g["center_uv"])
            dist = float(np.linalg.norm(dc - gc))
            if dist < dist_thresh and dist < best_d:
                best_j, best_d = dj, dist
        if best_j is not None:
            matched_gt.add(gi)
            matched_det.add(best_j)
            d = det_on_wall[best_j]
            w_err = d["extent"]["width"]["value"] - g["width"]
            h_err = d["extent"]["height"]["value"] - g["height"]
            matches.append({
                "gt_id": g["id"], "det_id": d["id"], "class": g["class"],
                "gt_width": g["width"], "det_width": d["extent"]["width"]["value"],
                "gt_height": g["height"], "det_height": d["extent"]["height"]["value"],
                "width_err_m": w_err, "height_err_m": h_err, "centroid_dist_m": best_d,
                "n_views": d["n_views"], "confidence": d["confidence"],
            })
    recall = len(matched_gt) / max(len(gt_on_wall), 1)
    precision = len(matched_det) / max(len(det_on_wall), 1)
    extent_errs = [abs(m["width_err_m"]) for m in matches] + [abs(m["height_err_m"]) for m in matches]
    return {
        "n_gt": len(gt_on_wall), "n_det": len(det_on_wall), "n_matched": len(matches),
        "recall": recall, "precision": precision,
        "mean_abs_extent_err_m": float(np.mean(extent_errs)) if extent_errs else None,
        "max_abs_extent_err_m": float(np.max(extent_errs)) if extent_errs else None,
        "matches": matches,
        "unmatched_gt": [g["id"] for i, g in enumerate(gt_on_wall) if i not in matched_gt],
        "unmatched_det": [d["id"] for j, d in enumerate(det_on_wall) if j not in matched_det],
    }


def measure_clean_fp_rate(data_dir: Path, cache_dir: str, max_frames: int = 200, progress=None) -> dict:
    results = {}
    for cap in sorted(p.name for p in data_dir.iterdir() if p.is_dir()):
        fs = load_stray(data_dir / cap)
        plan = build_fixture_plan(fs)
        regions = detect_damage(fs, plan, cache_dir=cache_dir, max_frames=max_frames,
                                progress=(lambda m, c=cap: progress(f"[{c}] {m}")) if progress else None)
        total_area_m2 = sum(w.length.value * w.height.value for r in plan.rooms for w in r.walls)
        results[cap] = {
            "n_false_positive_regions": len(regions),
            "total_wall_area_m2": round(total_area_m2, 2),
            "fp_per_m2": round(len(regions) / max(total_area_m2, 1e-6), 4),
            "regions": regions,
        }
    return results


def run_synthetic_benchmark(data_dir: Path, capture: str, cache_dir: str, max_frames: int = 200,
                            n_composite_frames: int = 40, progress=None) -> dict:
    fs = load_stray(data_dir / capture)
    plan = build_fixture_plan(fs)
    wall = max(plan.rooms[0].walls, key=lambda w: w.length.value)
    wall_id = wall.id
    surface = next(s for s in plan.surfaces() if s["id"] == wall_id)
    origin, u_hat, v_hat, uv_corners = _surface_basis(surface)
    extent_uv = (float(uv_corners[:, 0].min()), float(uv_corners[:, 0].max()),
                float(uv_corners[:, 1].min()), float(uv_corners[:, 1].max()))
    instances = make_instances(extent_uv)
    fs2, ground_truth = composite_frameset(fs, plan, wall_id, instances, n_frames=n_composite_frames)
    regions = detect_damage(fs2, plan, cache_dir=cache_dir, max_frames=max_frames, progress=progress)
    report = evaluate_detections(regions, ground_truth, wall_id)
    report["capture"] = capture
    report["wall_id"] = wall_id
    report["ground_truth"] = ground_truth
    report["all_detections_on_wall"] = [r for r in regions if r["surface_id"] == wall_id]
    report["disclosure"] = ("SYNTHETIC BENCHMARK: damage textures composited into real camera frames "
                            "of a real (undamaged) apartment wall. Not a real staged-damage capture.")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache-dir", default=".cache")
    ap.add_argument("--out-dir", default="bench/out")
    ap.add_argument("--capture", default="c00a170fe1")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--skip-clean", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    def progress(msg):
        print(msg, flush=True)

    if not args.skip_clean:
        print("=== clean-capture false-positive rate ===")
        fp = measure_clean_fp_rate(data_dir, args.cache_dir, max_frames=args.max_frames, progress=progress)
        (out_dir / "clean_fp_report.json").write_text(json.dumps(fp, indent=2, default=str))
        for cap, r in fp.items():
            print(f"  {cap}: {r['n_false_positive_regions']} regions over {r['total_wall_area_m2']} m2 "
                 f"({r['fp_per_m2']} / m2)")

    print("=== synthetic staged-damage benchmark ===")
    report = run_synthetic_benchmark(data_dir, args.capture, args.cache_dir, max_frames=args.max_frames,
                                     progress=progress)
    (out_dir / "synthetic_damage_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"capture={report['capture']} wall={report['wall_id']}")
    print(f"recall={report['recall']:.2f} precision={report['precision']:.2f} "
         f"n_gt={report['n_gt']} n_det={report['n_det']} n_matched={report['n_matched']}")
    print(f"mean_abs_extent_err_m={report['mean_abs_extent_err_m']}")
    print(report["disclosure"])


if __name__ == "__main__":
    main()
