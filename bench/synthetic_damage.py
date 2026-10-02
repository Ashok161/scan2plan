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
     scan2plan.manhattan / scan2plan.walls / scan2plan.fusion), written
     before the coordinator's scan2plan/layout.py existed, to develop
     damage.py against. layout.build_plan is now the real implementation
     and is what `_build_plan()` below uses everywhere (multi-room,
     real door/gap segmentation); build_fixture_plan is kept only as a
     dependency-free fallback (`_build_plan` falls back to it if
     scan2plan.layout cannot be imported) and because it is still what
     tests/test_damage.py's clustering-math tests build surfaces against
     by hand -- it is no longer on the path any real benchmark number
     here comes from.

  2. The synthetic-damage harness: `make_instances`, `composite_frameset`,
     `run_synthetic_benchmark` (recall/precision/extent error against known
     ground truth, on a real Plan from layout.build_plan), and
     `measure_clean_fp_rate` (false-positive rate on the three *unmodified*
     clean captures, also against real Plans -- the honesty check required
     alongside the synthetic recall numbers).

Run directly: `.venv/bin/python bench/synthetic_damage.py --data-dir data`
"""
from __future__ import annotations

import argparse
from collections import Counter
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
from scan2plan.damage import (DAMAGE_CLASSES, DEFAULT_OPERATING_POINT, detect_damage,
                              _detect_damage_raw, _passes_operating_point, _surface_basis)

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
        # the single longest contiguous segment, not the min/max across all
        # segments: merging across a real gap (doorway, corner into another
        # room) would silently stitch two different physical walls into one
        # "flat" rectangle, which breaks the planarity assumption everything
        # downstream (damage projection, synthetic compositing) relies on.
        a, b = max(ln.segments, key=lambda s: s[1] - s[0])
        covered = b - a
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


def render_peeling_paint(width_m: float, height_m: float, seed: int) -> np.ndarray:
    """Flaky, lighter-undercoat patches with ragged edges and strong local texture."""
    rng = np.random.default_rng(seed)
    w, h = int(width_m * PX_PER_M), int(height_m * PX_PER_M)
    base = _smooth_noise(h, w, rng, blur=max(w // 6, 3) | 1)
    base = (base - base.min()) / (base.max() - base.min() + 1e-6)
    flakes = rng.random((h, w)) > 0.75
    flakes = cv2.GaussianBlur(flakes.astype(np.float32), (3, 3), 0)
    yy, xx = np.mgrid[0:h, 0:w]
    rad = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    envelope = np.clip(1.15 - rad, 0, 1)
    alpha = np.clip((0.5 * base + 0.6 * flakes) * envelope, 0, 1)
    rgba = np.zeros((h, w, 4), np.uint8)
    undercoat = np.array([225, 220, 205])   # lighter exposed undercoat, BGR-ish
    rgba[..., 0] = undercoat[0] * (0.85 + 0.15 * base)
    rgba[..., 1] = undercoat[1] * (0.85 + 0.15 * base)
    rgba[..., 2] = undercoat[2] * (0.85 + 0.15 * base)
    rgba[..., 3] = (alpha * 170).astype(np.uint8)
    return rgba


RENDERERS = {
    "water_stain": lambda w, h, seed: render_water_stain(w, h, seed),
    "mould": lambda w, h, seed: render_mould(w, h, seed),
    "crack": lambda w, h, seed: render_crack(max(w, h), min(w, h), seed),
    "hole_or_impact": lambda w, h, seed: render_hole(w, h, seed),
    "peeling_paint": lambda w, h, seed: render_peeling_paint(w, h, seed),
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
    # Kept close to the wall's horizontal and vertical centre: real captures
    # walk at roughly eye height, so coverage/visibility is best in a
    # central band -- the wall's nominal top/bottom (especially the top,
    # when the ceiling was never observed and the wall height is a fallback
    # guess) is frequently never seen frontally at all.
    specs = [
        {"id": "syn.water_stain.1", "class": "water_stain", "width": 0.4, "height": 0.3,
         "center_uv": (uc - 0.26 * span_u, vc)},
        {"id": "syn.mould.1", "class": "mould", "width": 0.25, "height": 0.2,
         "center_uv": (uc + 0.26 * span_u, vc)},
        {"id": "syn.crack.1", "class": "crack", "width": 0.5, "height": 0.03,
         "center_uv": (uc - 0.06 * span_u, vc + 0.14 * span_v)},
        {"id": "syn.hole_or_impact.1", "class": "hole_or_impact", "width": 0.15, "height": 0.15,
         "center_uv": (uc + 0.06 * span_u, vc - 0.14 * span_v)},
        {"id": "syn.peeling_paint.1", "class": "peeling_paint", "width": 0.3, "height": 0.2,
         "center_uv": (uc + 0.16 * span_u, vc + 0.2 * span_v)},
    ]
    out = []
    for s in specs:
        cu, cv = s["center_uv"]
        if not (u0 + 0.08 < cu < u1 - 0.08 and v0 + 0.08 < cv < v1 - 0.08):
            continue   # keep instances inside the actually-observed wall patch
        out.append(s)
    return out


# physically plausible (width, height) ranges per class, metres -- spans the
# 5-60 cm range the coordinator asked for, per class (crack is thin x long).
_SIZE_RANGES_M = {
    "water_stain": ((0.15, 0.6), (0.1, 0.5)),
    "mould": ((0.08, 0.4), (0.08, 0.35)),
    "crack": ((0.2, 0.6), (0.01, 0.03)),
    "hole_or_impact": ((0.05, 0.3), (0.05, 0.3)),
    "peeling_paint": ((0.1, 0.5), (0.08, 0.4)),
}
# fixed relative grid of (du, dv) offsets (as a fraction of the half-span)
# so instances spread across the wall instead of piling at the centre;
# jitter is added per-instance on top of this.
_GRID = [(-0.32, 0.18), (0.32, 0.18), (-0.32, -0.18), (0.32, -0.18), (0.0, 0.3),
        (0.0, -0.3), (-0.16, 0.0), (0.16, 0.0)]


def make_instances_random(wall_extent_uv: tuple[float, float, float, float], rng: np.random.Generator,
                          tag: str, n_per_class: int = 2) -> list[dict]:
    """Randomised ground-truth instances for the enlarged benchmark: varied
    size (5-60 cm per class range above), varied position across the wall
    (grid + jitter), all 5 classes including the disabled-by-default
    peeling_paint (still worth measuring its raw recall/precision even
    though it is off by default -- that is the number that justifies
    disabling it). Deterministic for a given `rng` state.
    """
    u0, u1, v0, v1 = wall_extent_uv
    uc, vc = (u0 + u1) / 2, (v0 + v1) / 2
    span_u, span_v = (u1 - u0), (v1 - v0)
    out = []
    slot = 0
    for cls in DAMAGE_CLASSES:
        (wlo, whi), (hlo, hhi) = _SIZE_RANGES_M[cls]
        for k in range(n_per_class):
            w = float(rng.uniform(wlo, whi))
            h = float(rng.uniform(hlo, hhi))
            du, dv = _GRID[slot % len(_GRID)]
            slot += 1
            jitter_u = float(rng.uniform(-0.06, 0.06))
            jitter_v = float(rng.uniform(-0.06, 0.06))
            cu = uc + du * span_u * 0.5 + jitter_u
            cv = vc + dv * span_v * 0.5 + jitter_v
            margin = max(0.08, 0.6 * max(w, h) / 2)
            if not (u0 + margin < cu < u1 - margin and v0 + margin < cv < v1 - margin):
                continue
            out.append({"id": f"syn.{tag}.{cls}.{k + 1}", "class": cls,
                       "width": round(w, 3), "height": round(h, 3), "center_uv": (cu, cv)})
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

    # Pick frames per *instance*, not per whole wall: a 2-6 m wall is rarely
    # frontally visible in its entirety from any single pose in a small
    # apartment room, but each individual damage patch (tens of cm) usually
    # is. A frame qualifies for an instance if all 4 of that instance's own
    # corners project inside the image with reasonable range and the camera
    # is roughly facing the wall plane.
    n = surface["normal"] / np.linalg.norm(surface["normal"])
    margin = 15
    per_instance_frames: dict[str, list[int]] = {}
    for inst in instances:
        cu, cv = inst["center_uv"]
        hw, hh = inst["width"] / 2, inst["height"] / 2
        corners_uv = [(cu - hw, cv - hh), (cu + hw, cv - hh), (cu + hw, cv + hh), (cu - hw, cv + hh)]
        valid = []
        for fr in fs.frames:
            if fr.load_rgb is None:
                continue
            view_dir = fr.T_wc[:3, 2]
            facing = float(-view_dir @ (T_align_inv[:3, :3] @ n))
            if facing < 0.5:
                continue
            ok = True
            for (uu, vv) in corners_uv:
                p_plan = origin + uu * u_hat + vv * v_hat
                proj = _project_plan_point(p_plan, T_align_inv, fr.T_wc, fr.K_rgb)
                if proj is None:
                    ok = False
                    break
                xy, z = proj
                if not (0.4 <= z <= 4.5 and margin <= xy[0] < img_size[0] - margin
                       and margin <= xy[1] < img_size[1] - margin):
                    ok = False
                    break
            if ok:
                valid.append(fr.index)
        if not valid:
            continue   # this instance's placement is never frontally visible on this wall; drop it, don't abort the whole wall
        per_instance_frames[inst["id"]] = valid

    instances = [i for i in instances if i["id"] in per_instance_frames]
    if not instances:
        raise RuntimeError(f"synthetic bench: no instance on {wall_id} is frontally visible from any frame")

    # Composite onto the *full* union of geometrically-valid frames per
    # instance, not a small curated subsample: `detect_damage` runs its own
    # `select_keyframes` + `max_frames` subsampling over the *whole*
    # capture, which has no reason to land exactly on a small hand-picked
    # set of frame indices. The union needs to be wide enough that whatever
    # subset the real keyframe selector lands on, it still overlaps with
    # frames that actually carry the synthetic damage. Compositing is lazy
    # (only computed if `load_rgb()` is actually called), so a wide union
    # costs nothing extra at frames detect_damage never touches.
    sel_set: set[int] = set()
    for inst in instances:
        sel_set.update(per_instance_frames[inst["id"]])
    sel = sorted(sel_set)
    cap = max(n_frames, 600)
    if len(sel) > cap:
        idxs = np.linspace(0, len(sel) - 1, cap).astype(int)
        sel = sorted({sel[i] for i in idxs})

    composite_cache: dict[int, np.ndarray] = {}

    def make_loader(frame_idx: int, orig_loader):
        def _load():
            if frame_idx in composite_cache:
                return composite_cache[frame_idx]
            rgb = orig_loader().copy()
            h, w = rgb.shape[:2]
            fr = next(f for f in fs.frames if f.index == frame_idx)
            canvas = rgb.copy()
            # Lighting variation: deterministically darken ~1/3 of composited
            # frames (seeded on frame index, not random per-run) so recall is
            # also measured under the low-light CLAHE path in damage.py, not
            # just well-lit frontal shots.
            if frame_idx % 3 == 0:
                gamma = 0.45 + 0.1 * ((frame_idx // 3) % 3)   # ~0.45-0.65 multiplicative darkening
                canvas = np.clip(canvas.astype(np.float32) * gamma, 0, 255).astype(np.uint8)
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
    # The disk cache in damage.py keys solely on (source string, frame index);
    # fold a hash of the instance spec into the source so that editing
    # instance placement/size/class during development invalidates stale
    # cached box detections instead of silently reusing them.
    import hashlib as _hashlib
    inst_hash = _hashlib.sha1(repr(instances).encode()).hexdigest()[:10]
    fs2 = FrameSet(tier=fs.tier, frames=new_frames, errors=fs.errors,
                  source=f"{fs.source}+synthetic:{inst_hash}",
                  meta={**fs.meta, "synthetic_frames": sel})

    ground_truth = []
    for inst in instances:
        n_this = len(set(per_instance_frames[inst["id"]]) & set(sel))
        ground_truth.append({**inst, "surface_id": wall_id, "n_frames_composited": n_this})
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


def _build_plan(fs: FrameSet, progress=None):
    """Real layout.build_plan when available (scan2plan.layout, built by the
    coordinator in parallel), else the hand-made fixture above. Both return
    a Plan with the same `.surfaces()` / `.rooms[].walls` shape, so
    everything downstream (detect_damage, the synthetic harness) is
    agnostic to which one produced it.
    """
    try:
        from scan2plan.layout import build_plan
        return build_plan(fs, progress=progress)
    except ImportError:
        return build_fixture_plan(fs)


def measure_clean_fp_rate(data_dir: Path, cache_dir: str, max_frames: int = 200, progress=None,
                          out_path: str | Path | None = None) -> dict:
    from scan2plan.io.stray import is_stray_capture
    results = {}
    captures = sorted(p.name for p in data_dir.iterdir() if p.is_dir() and is_stray_capture(p))
    for cap in captures:
        fs = load_stray(data_dir / cap)
        plan = _build_plan(fs, progress=(lambda m, c=cap: progress(f"[{c}] {m}")) if progress else None)
        regions = detect_damage(fs, plan, cache_dir=cache_dir, max_frames=max_frames,
                                progress=(lambda m, c=cap: progress(f"[{c}] {m}")) if progress else None)
        total_area_m2 = sum(w.length.value * w.height.value for r in plan.rooms for w in r.walls)
        results[cap] = {
            "n_rooms": len(plan.rooms),
            "n_false_positive_regions": len(regions),
            "total_wall_area_m2": round(total_area_m2, 2),
            "fp_per_m2": round(len(regions) / max(total_area_m2, 1e-6), 4),
            "regions": regions,
        }
        if out_path:   # write incrementally: a later capture crashing shouldn't lose earlier results
            Path(out_path).write_text(json.dumps(results, indent=2, default=str))
    return results


def _pick_synthetic_wall(plan: Plan, min_len=1.2, max_len=4.5):
    """Walls most likely to let a frontal view of a human-scale patch exist:
    well covered (few/no gaps) and not so long that no pose steps back
    far enough to see a patch on it frontally within range, ordered by
    coverage then by how close length is to a "comfortable" ~2.2 m.
    """
    cands = [w for r in plan.rooms for w in r.walls if min_len <= w.length.value <= max_len]
    cands.sort(key=lambda w: (-w.coverage, abs(w.length.value - 2.2)))
    return cands


def run_synthetic_benchmark(data_dir: Path, capture: str, cache_dir: str, max_frames: int = 200,
                            n_composite_frames: int = 40, progress=None) -> dict:
    fs = load_stray(data_dir / capture)
    plan = _build_plan(fs, progress=progress)
    wall_candidates = _pick_synthetic_wall(plan)
    if not wall_candidates:
        wall_candidates = [max(w for r in plan.rooms for w in r.walls)]
    last_err = None
    for wall in wall_candidates:
        wall_id = wall.id
        surface = next(s for s in plan.surfaces() if s["id"] == wall_id)
        origin, u_hat, v_hat, uv_corners = _surface_basis(surface)
        extent_uv = (float(uv_corners[:, 0].min()), float(uv_corners[:, 0].max()),
                    float(uv_corners[:, 1].min()), float(uv_corners[:, 1].max()))
        instances = make_instances(extent_uv)
        if len(instances) < 2:
            continue
        try:
            fs2, ground_truth = composite_frameset(fs, plan, wall_id, instances, n_frames=n_composite_frames)
            break
        except RuntimeError as e:
            last_err = e
            continue
    else:
        raise RuntimeError(f"synthetic bench: no candidate wall could be composited ({last_err})")

    regions = detect_damage(fs2, plan, cache_dir=cache_dir, max_frames=max_frames, progress=progress)
    report = evaluate_detections(regions, ground_truth, wall_id)
    report["capture"] = capture
    report["wall_id"] = wall_id
    report["wall_length_m"] = wall.length.value
    report["ground_truth"] = ground_truth
    report["all_detections_on_wall"] = [r for r in regions if r["surface_id"] == wall_id]
    report["disclosure"] = ("SYNTHETIC BENCHMARK: damage textures composited into real camera frames "
                            "of a real (undamaged) apartment wall. Not a real staged-damage capture.")
    return report


# --------------------------------------------------------------------------
# 5. Enlarged multi-wall / multi-capture synthetic suite + PR / FP-per-m2 sweep
# --------------------------------------------------------------------------
def run_synthetic_suite(data_dir: Path, cache_dir: str, max_frames: int = 200,
                        walls_per_capture: int = 2, n_per_class: int = 2, n_composite_frames: int = 40,
                        seed: int = 20260101, progress=None) -> dict:
    """Composite randomised instances (all 5 classes, 5-60cm, varied position/
    lighting) onto `walls_per_capture` walls in each of the 3 captures, run
    the *raw* (unfiltered, all-classes-enabled) detector on each, and return
    everything needed to sweep operating points without re-running the model:
    `raw_detections` (tagged with "capture"/"wall_id") and `ground_truth`
    (same tagging). Disclosed as synthetic in every report that uses this.
    """
    from scan2plan.io.stray import is_stray_capture
    captures = sorted(p.name for p in data_dir.iterdir() if p.is_dir() and is_stray_capture(p))
    rng = np.random.default_rng(seed)
    all_raw: list[dict] = []
    all_gt: list[dict] = []
    wall_info = []
    for cap in captures:
        fs = load_stray(data_dir / cap)
        plan = _build_plan(fs, progress=(lambda m, c=cap: progress(f"[{c}] {m}")) if progress else None)
        cands = _pick_synthetic_wall(plan)
        chosen = 0
        for wall in cands:
            if chosen >= walls_per_capture:
                break
            wall_id = wall.id
            surface = next(s for s in plan.surfaces() if s["id"] == wall_id)
            origin, u_hat, v_hat, uv_corners = _surface_basis(surface)
            extent_uv = (float(uv_corners[:, 0].min()), float(uv_corners[:, 0].max()),
                        float(uv_corners[:, 1].min()), float(uv_corners[:, 1].max()))
            tag = f"{cap}.{wall_id}"
            instances = make_instances_random(extent_uv, rng, tag=tag, n_per_class=n_per_class)
            if len(instances) < 3:
                continue
            try:
                fs2, ground_truth = composite_frameset(fs, plan, wall_id, instances, n_frames=n_composite_frames)
            except RuntimeError as e:
                if progress:
                    progress(f"[{cap}] {wall_id}: skipped ({e})")
                continue
            regions = _detect_damage_raw(fs2, plan, cache_dir=cache_dir, max_frames=max_frames,
                                         progress=(lambda m, c=cap, w=wall_id: progress(f"[{c}:{w}] {m}")) if progress else None,
                                         enabled_classes=DAMAGE_CLASSES)
            for r in regions:
                r["capture"] = cap
            for g in ground_truth:
                g["capture"] = cap
            all_raw.extend(regions)
            all_gt.extend(ground_truth)
            wall_info.append({"capture": cap, "wall_id": wall_id, "wall_length_m": wall.length.value,
                              "n_instances": len(ground_truth),
                              "n_instances_by_class": dict(Counter(g["class"] for g in ground_truth))})
            chosen += 1
            if progress:
                progress(f"[{cap}] {wall_id}: {len(ground_truth)} instances composited, {len(regions)} raw detections")
    return {"raw_detections": all_raw, "ground_truth": all_gt, "walls": wall_info}


def evaluate_suite(raw_detections: list[dict], ground_truth: list[dict], op: dict | None = None,
                   dist_thresh: float = 0.35) -> dict:
    """Per-class + overall recall/precision/extent-error, matching within
    (capture, surface_id, class) groups. If `op` is given, filters
    raw_detections through `_passes_operating_point` first -- this is the
    function the sweep calls once per threshold combination, over data
    that was computed exactly once.
    """
    dets = [d for d in raw_detections if op is None or _passes_operating_point(d, op)]
    by_key: dict[tuple, dict] = {}
    for d in dets:
        by_key.setdefault((d.get("capture"), d["surface_id"]), {"det": [], "gt": []})["det"].append(d)
    for g in ground_truth:
        by_key.setdefault((g.get("capture"), g["surface_id"]), {"det": [], "gt": []})["gt"].append(g)

    per_class: dict[str, dict] = {c: {"n_gt": 0, "n_det": 0, "n_matched": 0, "extent_errs": []}
                                  for c in DAMAGE_CLASSES}
    matches = []
    for (cap, sid), group in by_key.items():
        gts, ds = group["gt"], group["det"]
        matched_gt, matched_det = set(), set()
        for gi, g in enumerate(gts):
            best_j, best_d = None, 1e9
            for dj, d in enumerate(ds):
                if dj in matched_det or d["class"] != g["class"]:
                    continue
                dc = np.array(d["polygon_surface"]).mean(axis=0) if d["polygon_surface"] else np.array([1e9, 1e9])
                dist = float(np.linalg.norm(dc - np.array(g["center_uv"])))
                if dist < dist_thresh and dist < best_d:
                    best_j, best_d = dj, dist
            if best_j is not None:
                matched_gt.add(gi)
                matched_det.add(best_j)
                d = ds[best_j]
                werr = abs(d["extent"]["width"]["value"] - g["width"])
                herr = abs(d["extent"]["height"]["value"] - g["height"])
                per_class[g["class"]]["extent_errs"].extend([werr, herr])
                matches.append({"capture": cap, "surface_id": sid, "class": g["class"],
                               "gt_id": g["id"], "det_id": d["id"]})
        for gi, g in enumerate(gts):
            per_class[g["class"]]["n_gt"] += 1
        for dj, d in enumerate(ds):
            per_class[d["class"]]["n_det"] += 1
        for gi in matched_gt:
            per_class[gts[gi]["class"]]["n_matched"] += 1

    out_per_class = {}
    for c, v in per_class.items():
        out_per_class[c] = {
            "n_gt": v["n_gt"], "n_det": v["n_det"], "n_matched": v["n_matched"],
            "recall": v["n_matched"] / v["n_gt"] if v["n_gt"] else None,
            "precision": v["n_matched"] / v["n_det"] if v["n_det"] else None,
            "mean_abs_extent_err_m": float(np.mean(v["extent_errs"])) if v["extent_errs"] else None,
        }
    n_gt = sum(v["n_gt"] for v in per_class.values())
    n_det = sum(v["n_det"] for v in per_class.values())
    n_matched = sum(v["n_matched"] for v in per_class.values())
    return {
        "n_gt": n_gt, "n_det": n_det, "n_matched": n_matched,
        "recall": n_matched / n_gt if n_gt else None,
        "precision": n_matched / n_det if n_det else None,
        "per_class": out_per_class,
        "matches": matches,
    }


def sweep_operating_points(clean_raw: dict, suite: dict, candidates: list[dict]) -> list[dict]:
    """clean_raw: {capture: {"regions": [...raw...], "total_wall_area_m2": ...}}
    suite: {"raw_detections": [...], "ground_truth": [...]} from run_synthetic_suite.
    Returns one row per candidate operating point: FP count/rate (clean) +
    recall/precision (synthetic), with no re-running of the model -- every
    row is a cheap pure-Python filter + match over data computed once.
    """
    rows = []
    total_area = sum(d["total_wall_area_m2"] for d in clean_raw.values())
    for op in candidates:
        fp_total = sum(sum(1 for r in d["regions"] if _passes_operating_point(r, op)) for d in clean_raw.values())
        fp_by_capture = {cap: sum(1 for r in d["regions"] if _passes_operating_point(r, op))
                         for cap, d in clean_raw.items()}
        res = evaluate_suite(suite["raw_detections"], suite["ground_truth"], op=op)
        rows.append({
            "op": op, "fp_total": fp_total, "fp_per_m2": round(fp_total / total_area, 5),
            "fp_by_capture": fp_by_capture,
            "recall": res["recall"], "precision": res["precision"],
            "n_gt": res["n_gt"], "n_det": res["n_det"], "n_matched": res["n_matched"],
            "per_class_recall": {c: v["recall"] for c, v in res["per_class"].items()},
        })
    return rows


def collect_raw_clean(data_dir: Path, cache_dir: str, max_frames: int = 200,
                      enabled_classes=DAMAGE_CLASSES, progress=None,
                      out_path: str | Path | None = None) -> dict:
    """Raw (unfiltered, all-requested-classes) detections on the 3 clean
    captures -- the FP side of the operating-point sweep. Separate from
    `measure_clean_fp_rate` (which calls the *filtered* `detect_damage`,
    i.e. reports the actual default-pipeline FP count) because the sweep
    needs the unfiltered pool to try thresholds without re-running the model.
    """
    from scan2plan.io.stray import is_stray_capture
    out = {}
    captures = sorted(p.name for p in data_dir.iterdir() if p.is_dir() and is_stray_capture(p))
    for cap in captures:
        fs = load_stray(data_dir / cap)
        plan = _build_plan(fs, progress=(lambda m, c=cap: progress(f"[{c}] {m}")) if progress else None)
        regions = _detect_damage_raw(fs, plan, cache_dir=cache_dir, max_frames=max_frames,
                                     progress=(lambda m, c=cap: progress(f"[{c}] {m}")) if progress else None,
                                     enabled_classes=enabled_classes)
        total_area = sum(w.length.value * w.height.value for r in plan.rooms for w in r.walls)
        out[cap] = {"regions": regions, "total_wall_area_m2": total_area, "n_rooms": len(plan.rooms)}
        if out_path:
            Path(out_path).write_text(json.dumps(out, indent=2, default=str))
    return out


def generate_candidate_grid() -> list[dict]:
    """The threshold grid actually swept to pick DEFAULT_OPERATING_POINT in
    damage.py. Kept here (not just inline in a notebook) so `--full-sweep`
    regenerates the exact same table that produced the shipped default.
    """
    grid = []
    for s in (0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.46, 0.50):
        for nv in (1, 2, 3):
            for cc in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                for ct in (0.0, 3, 6, 9, 12, 15, 18, 22):
                    grid.append({"min_score": s, "min_views": nv, "min_color_consistency": cc, "min_contrast": ct})
    return grid


def choose_operating_point(rows: list[dict], max_fp_per_m2: float = 0.01, max_fp_per_capture: int = 2) -> dict | None:
    qualifying = [r for r in rows if r["fp_per_m2"] <= max_fp_per_m2
                 and all(v <= max_fp_per_capture for v in r["fp_by_capture"].values())]
    if not qualifying:
        return None
    qualifying.sort(key=lambda r: (-(r["recall"] or 0.0), r["fp_per_m2"]))
    return qualifying[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache-dir", default=".cache")
    ap.add_argument("--out-dir", default="reports/damage")
    ap.add_argument("--capture", default="c00a170fe1")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--skip-clean", action="store_true")
    ap.add_argument("--full-sweep", action="store_true",
                    help="enlarged multi-wall/multi-capture synthetic suite + PR/FP-per-m2 threshold "
                         "sweep (requirement: ~15-20 min, reruns the model with all 5 classes enabled)")
    ap.add_argument("--walls-per-capture", type=int, default=2)
    ap.add_argument("--n-per-class", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    def progress(msg):
        print(msg, flush=True)

    if args.full_sweep:
        print("=== DISCLOSURE: synthetic staged-damage benchmark below. No real staged-damage capture "
             "exists (no iPhone); damage textures are composited into real camera frames of the "
             "undamaged apartment captures with known physical size on a known wall plane. ===")
        print("=== raw (unfiltered, all 5 classes) detections on the 3 clean captures ===")
        clean = collect_raw_clean(data_dir, args.cache_dir, max_frames=args.max_frames,
                                  enabled_classes=DAMAGE_CLASSES, progress=progress,
                                  out_path=out_dir / "raw_sweep_clean.json")
        print("=== enlarged synthetic suite: 5 classes, 5-60cm, several walls, all 3 captures ===")
        suite = run_synthetic_suite(data_dir, args.cache_dir, max_frames=args.max_frames,
                                    walls_per_capture=args.walls_per_capture, n_per_class=args.n_per_class,
                                    progress=progress)
        (out_dir / "synthetic_suite_raw.json").write_text(json.dumps(suite, indent=2, default=str))
        print(f"n_ground_truth_instances={len(suite['ground_truth'])} across {len(suite['walls'])} walls")

        print("=== PR / FP-per-m2 sweep over (score, n_views, colour-consistency, mask-contrast) ===")
        rows = sweep_operating_points(clean, suite, generate_candidate_grid())
        (out_dir / "operating_point_sweep.json").write_text(json.dumps(rows, indent=2, default=str))
        chosen = choose_operating_point(rows)
        print(f"chosen operating point (FP<=0.01/m2, FP<=2/capture, max synthetic recall): {chosen}")

        print("=== default (filtered) detect_damage on the clean captures, for comparison ===")
        fp = measure_clean_fp_rate(data_dir, args.cache_dir, max_frames=args.max_frames, progress=progress,
                                   out_path=out_dir / "clean_fp_report.json")
        for cap, r in fp.items():
            print(f"  {cap}: {r['n_false_positive_regions']} regions over {r['total_wall_area_m2']} m2 "
                 f"({r['fp_per_m2']} / m2)")
        return

    if not args.skip_clean:
        print("=== clean-capture false-positive rate (default operating point) ===")
        fp = measure_clean_fp_rate(data_dir, args.cache_dir, max_frames=args.max_frames, progress=progress,
                                   out_path=out_dir / "clean_fp_report.json")
        for cap, r in fp.items():
            print(f"  {cap}: {r['n_false_positive_regions']} regions over {r['total_wall_area_m2']} m2 "
                 f"({r['fp_per_m2']} / m2)")

    print("=== synthetic staged-damage benchmark (single wall; use --full-sweep for the full suite) ===")
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
