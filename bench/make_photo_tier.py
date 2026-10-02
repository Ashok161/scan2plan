"""Generate the photo-tier benchmark set from LiDAR captures.

We have no real iPhone photos (see the project README for the disclosure).
This script builds a disclosed stand-in: it picks a handful of sharp,
viewpoint-diverse frames per room out of a StrayScanner LiDAR capture's
rgb.mp4, following the SAME capture protocol a human is asked to follow
(the Photos section of docs/capture_protocol.md) -- including the doorway-photo duplication
step -- and saves them as plain JPEGs with EXIF FocalLengthIn35mmFilm set
from the capture's real intrinsics (what an iPhone still would carry).

Output `data/photo_tier/<capture_id>/<room_name>/*.jpg` contains ONLY
images (+ EXIF focal/orientation metadata); no depth, no pose, no LiDAR
data is written into the photo folders. A `_manifest.json` sidecar is
written next to the room folders (not inside any of them) purely so
`bench/eval_photo_tier.py` can score the result against the LiDAR plan --
`scan2plan.tiers.photo.load_photo_property` never reads it.

Room membership of a frame:
  --plan <path/to/plan.json>  (or auto-discovered at
  out/bench/<capture_id>/lidar/plan.json): assign each frame's camera
  position to the LiDAR plan's room polygons (point-in-polygon, in the
  plan frame via T_align) -- this is the coordinator's actual output.
  Falls back to a self-contained room segmentation (fuse -> Manhattan
  align -> floor/ceiling -> wall lines/gaps -> segment_rooms_by_walls,
  the same primitives layout.py itself is built from) when no plan.json
  is available yet.

Either way this is LiDAR-depth-and-pose used ONLY to build the benchmark
offline; none of it reaches the photo folders or the photo pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon

from scan2plan.frames import backproject, to_world
from scan2plan.fusion import fuse, select_keyframes
from scan2plan.io.stray import load_stray
from scan2plan.manhattan import dominant_yaw, floor_and_ceiling, yaw_rotation
from scan2plan.maps import build_maps, segment_rooms_by_walls
from scan2plan.video_io import VideoFrames
from scan2plan.walls import extract_wall_lines, find_gaps

DEFAULT_CAPTURES = ["c00a170fe1", "1a8384c3f6", "c7d28f72c6"]


# --------------------------------------------------------------- room assignment


def _assign_from_lidar_plan(fs, plan_json: dict):
    """Point-in-polygon room assignment using a real coordinator plan.json."""
    T_align = np.array(plan_json["frame"]["T_align"], dtype=float)
    T_plan_from_world = T_align
    T_world_from_plan = np.linalg.inv(T_align)
    rooms = plan_json["rooms"]
    polys = {r["id"]: Polygon(r["polygon"]).buffer(0.05) for r in rooms if len(r["polygon"]) >= 3}
    names = {r["id"]: (r.get("name") or r["id"]) for r in rooms}

    labels: list[str | None] = []
    for fr in fs.frames:
        p_world = np.append(fr.position, 1.0)
        p_plan = (T_plan_from_world @ p_world)[[0, 2]]
        pt = Point(p_plan)
        lbl = next((rid for rid, poly in polys.items() if poly.contains(pt)), None)
        labels.append(lbl)

    opening_by_id = {}
    for r in rooms:
        for o in r.get("openings", []):
            opening_by_id[o["id"]] = o

    doors = []
    for adj in plan_json.get("adjacency", []):
        via = adj.get("via") or []
        oid = via[0] if via else None
        o = opening_by_id.get(oid) if oid else None
        if o is None:
            continue
        cx, cz = o["center"]
        p_plan_h = np.array([cx, 0.0, cz, 1.0])
        p_world = (T_world_from_plan @ p_plan_h)[:3]
        doors.append({"rooms": adj["rooms"], "midpoint_world": p_world, "via": via})
    return labels, names, doors


def _assign_from_scratch(fs, progress=print):
    """Fallback room segmentation (no plan.json yet): the same primitives
    layout.build_plan uses, kept intentionally simpler (grid labels only,
    no polygon/opening refinement -- we only need a frame->room label and
    an approximate door location here)."""
    kf = select_keyframes(fs, min_trans=0.08, min_rot_deg=8.0, max_frames=800)
    progress(f"fallback segmentation: fusing {len(kf)} keyframes")
    cloud = fuse(fs, kf, pixel_stride=3)
    yaw, mscore = dominant_yaw(cloud.normals)
    T_align = yaw_rotation(yaw)
    R = T_align[:3, :3]
    cloud.points[:] = cloud.points @ R.T
    cloud.normals[:] = cloud.normals @ R.T
    cloud.cam_pos[:] = cloud.cam_pos @ R.T

    (floor_y, _, _), ceil = floor_and_ceiling(cloud.points, cloud.normals)
    ceil_y = ceil[0] if ceil is not None else None
    top = (ceil_y - floor_y) if ceil_y is not None else 2.6
    maps = build_maps(cloud, floor_y, ceil_y)

    h = cloud.points[:, 1] - floor_y
    band = (np.abs(cloud.normals[:, 1]) < 0.3) & (h > 0.25) & (h < min(1.95, top - 0.1))
    lines = extract_wall_lines(cloud.points[band][:, [0, 2]], cloud.normals[band][:, [0, 2]])
    gaps = find_gaps(lines)
    labels_grid, n_rooms, closures = segment_rooms_by_walls(maps, gaps)
    progress(f"fallback segmentation: {n_rooms} room(s), {len(closures)} door closure(s)")

    g = maps.grid
    labels: list[str | None] = []
    for fr in fs.frames:
        p = (R @ fr.position)[[0, 2]]
        ij = g.ij(p[None])[0]
        lbl = None
        if g.inside(ij[None])[0]:
            v = int(labels_grid[ij[0], ij[1]])
            if v > 0:
                lbl = f"R{v}"
        labels.append(lbl)
    names = {f"R{i}": f"room{i}" for i in range(1, n_rooms + 1)}

    doors = []
    for gp, kind in closures:
        if kind != "door":
            continue
        e = gp.endpoints()
        mid_rot = e.mean(axis=0)   # (x,z) in the yaw-aligned frame
        mid_world3 = R.T @ np.array([mid_rot[0], 0.0, mid_rot[1]])
        ring = _ring_labels(labels_grid, g, mid_rot)
        if len(ring) != 2:
            continue
        doors.append({"rooms": [f"R{a}" for a in ring], "midpoint_world": mid_world3, "via": []})
    return labels, names, doors


def _ring_labels(labels_grid, grid, xz, radius_cells: int = 4):
    ij = grid.ij(xz[None])[0]
    i0, i1 = max(0, ij[0] - radius_cells), ij[0] + radius_cells + 1
    j0, j1 = max(0, ij[1] - radius_cells), ij[1] + radius_cells + 1
    win = labels_grid[i0:i1, j0:j1]
    vals = sorted(set(int(v) for v in np.unique(win) if v > 0))
    return vals


# ------------------------------------------------------------------- selection


def _rotate_upright(rgb: np.ndarray) -> np.ndarray:
    """StrayScanner stores raw ARKit sensor-landscape frames even when the
    phone was held portrait (checked visually on all 3 captures: rotating
    90 deg clockwise makes them upright). A real iPhone still is upright
    via EXIF orientation; this makes the bench photos match that, pixel
    for pixel, rather than relying on an orientation tag to fix a sideways
    image a real phone would never produce."""
    return cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)


def _laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _edge_density(gray: np.ndarray) -> float:
    return float((cv2.Canny(gray, 50, 150) > 0).mean())


def _global_floor_ceiling(fs, progress=print):
    """One quick whole-capture fuse, purely to get a floor/ceiling height
    for scoring candidate frames by "does this shot actually show floor
    and ceiling lines" (per the protocol) -- LiDAR used only to pick which
    frames become bench photos, never fed into the photo pipeline itself.
    """
    kf = select_keyframes(fs, min_trans=0.1, min_rot_deg=10.0, max_frames=400)
    cloud = fuse(fs, kf, pixel_stride=4)
    (floor_y, _, _), ceil = floor_and_ceiling(cloud.points, cloud.normals)
    ceil_y = ceil[0] if ceil is not None else None
    progress(f"  global floor_y={floor_y:.2f} ceiling={'%.2f' % ceil_y if ceil_y is not None else 'not observed'}")
    return floor_y, ceil_y


def _floor_ceiling_fraction(fr, floor_y: float, ceil_y: float | None, tol: float = 0.15):
    """Fraction of a frame's own (real LiDAR) depth points lying near the
    global floor / ceiling height -- a direct, geometric stand-in for "can
    you see the floor line / ceiling line in this shot", instead of
    guessing from pixels."""
    d, v = fr.load_depth()
    pts, _, _ = backproject(d, v, fr.K_depth, stride=4)
    if len(pts) == 0:
        return 0.0, 0.0
    pw = to_world(fr.T_wc, pts)
    hy = pw[:, 1]
    floor_frac = float((np.abs(hy - floor_y) < tol).mean())
    ceil_frac = float((np.abs(hy - ceil_y) < tol).mean()) if ceil_y is not None else 0.0
    return floor_frac, ceil_frac


def _rank01(values: dict[int, float], idxs: list[int]) -> dict[int, float]:
    """0..1 percentile rank within `idxs` (ties broken by index order)."""
    order = sorted(idxs, key=lambda i: values[i])
    n = len(order)
    return {i: (k / max(1, n - 1)) for k, i in enumerate(order)}


def _quality_score(idxs: list[int], sharpness: dict[int, float], edges: dict[int, float],
                   floor_frac: dict[int, float], ceil_frac: dict[int, float],
                   ceiling_observed: bool) -> dict[int, float]:
    """Combine sharpness, edge density and floor/ceiling visibility into one
    0..1-ish per-frame quality score (percentile ranks, so the different
    raw scales/units don't need hand-tuned normalisation). A blank wall
    (per the protocol: every room photo should show floor+ceiling lines
    and some structure) scores low on both edge density and floor/ceiling
    visibility even if it happens to be perfectly sharp.
    """
    s_rank = _rank01(sharpness, idxs)
    e_rank = _rank01(edges, idxs)
    f_rank = _rank01(floor_frac, idxs)
    c_rank = _rank01(ceil_frac, idxs) if ceiling_observed else {i: 0.5 for i in idxs}   # neutral if never observed
    w_floor = 0.25
    w_ceil = 0.20 if ceiling_observed else 0.0
    w_sharp = 0.30
    w_edge = 1.0 - w_floor - w_ceil - w_sharp
    return {i: w_sharp * s_rank[i] + w_edge * e_rank[i] + w_floor * f_rank[i] + w_ceil * c_rank[i] for i in idxs}


def _select_diverse_sharp(cand_idx: list[int], positions: dict[int, np.ndarray],
                          quality: dict[int, float], n_min: int, n_max: int,
                          quality_floor_pct: float = 25.0) -> list[int]:
    """Protocol step 'turn in place, one photo every 60-90 deg': approximate
    viewpoint diversity by spacing picks evenly along *cumulative camera
    travel distance* through the room (not evenly in frame count / time,
    which would under-sample a room the camera lingered in and over-sample
    one it walked straight through), discarding the quarter with the worst
    combined quality score (sharpness + edge density + floor/ceiling
    visibility; see `_quality_score`) first.
    """
    if len(cand_idx) <= n_min:
        return sorted(cand_idx)
    thr = np.percentile([quality[i] for i in cand_idx], quality_floor_pct)
    ok = [i for i in cand_idx if quality[i] >= thr] or list(cand_idx)
    ok = sorted(ok)
    step = np.r_[0.0, np.linalg.norm(np.diff([positions[i] for i in ok], axis=0), axis=1)]
    arc = np.cumsum(step)
    n = int(np.clip(len(ok), n_min, n_max))
    targets = np.linspace(0.0, arc[-1], n) if arc[-1] > 1e-6 else np.zeros(n)
    sel = {ok[int(np.argmin(np.abs(arc - t)))] for t in targets}
    return sorted(sel)


# ----------------------------------------------------------------------- EXIF


def _focal35_from_fx(fx_px: float, image_w_px: float) -> float:
    return float(fx_px / image_w_px * 36.0)


def _save_jpeg_with_exif(rgb: np.ndarray, path: Path, focal35: float, capture_id: str, frame_idx: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(rgb)
    exif = Image.Exif()
    exif[274] = 1                              # Orientation: already upright
    exif[41989] = int(round(focal35))          # FocalLengthIn35mmFilm
    exif[37386] = (int(round(focal35 * 10)), 10)   # FocalLength (rational, mm): approx, informational only
    exif[271] = "scan2plan-bench (SYNTHETIC)"   # Make
    exif[272] = "frame-from-LiDAR, not a camera"  # Model
    exif[305] = f"derived from LiDAR capture {capture_id} frame {frame_idx:06d}; no real camera used"  # Software
    img.save(path, format="JPEG", quality=92, exif=exif)


# --------------------------------------------------------------------- driver


def process_capture(capture_dir: Path, out_root: Path, plan_path: Path | None = None,
                    n_photos: tuple[int, int] = (5, 6), seed: int = 0, progress=print):
    rng = np.random.default_rng(seed)
    capture_id = capture_dir.name
    progress(f"[{capture_id}] loading StrayScanner capture")
    fs = load_stray(capture_dir)
    K_rgb = fs.frames[0].K_rgb
    fx_px = float(K_rgb[0, 0])

    vf = VideoFrames(capture_dir / "rgb.mp4")
    image_w = vf.size[0]   # sensor-landscape long edge; unaffected by the upright rotation below
    focal35 = _focal35_from_fx(fx_px, image_w)
    progress(f"[{capture_id}] fx={fx_px:.1f}px, image_w={image_w}px -> FocalLengthIn35mmFilm={focal35:.1f}")
    floor_y, ceil_y = _global_floor_ceiling(fs, progress)

    plan_json = None
    if plan_path is None:
        default_plan = Path("out") / "bench" / capture_id / "lidar" / "plan.json"
        if default_plan.exists():
            plan_path = default_plan
    if plan_path is not None and Path(plan_path).exists():
        progress(f"[{capture_id}] using LiDAR plan {plan_path}")
        plan_json = json.loads(Path(plan_path).read_text())
        labels, names, doors = _assign_from_lidar_plan(fs, plan_json)
    else:
        progress(f"[{capture_id}] no plan.json found; using built-in fallback room segmentation")
        labels, names, doors = _assign_from_scratch(fs, progress=progress)

    by_room: dict[str, list[int]] = {}
    for i, lbl in enumerate(labels):
        if lbl is not None:
            by_room.setdefault(lbl, []).append(i)
    by_room = {rid: idxs for rid, idxs in by_room.items() if len(idxs) >= 3}
    if not by_room:
        progress(f"[{capture_id}] WARNING: no room got >=3 candidate frames; skipping capture")
        return None
    progress(f"[{capture_id}] {len(by_room)} room(s) with camera coverage: "
            f"{[(names.get(r, r), len(v)) for r, v in by_room.items()]}")

    # A long multi-room walkthrough can label *every* frame of a 10k-frame
    # video into some room; decoding and scoring (depth backproject, Canny,
    # Laplacian) all of them is wasted work we don't need only ~6-8 picks
    # per room from, and was slow/memory-heavy enough under concurrent load
    # to get OOM-killed on the longest capture. Pre-subsample each room's
    # candidate pool to a bounded size (evenly spaced by index -> still
    # spans the whole time the camera spent in that room) before doing any
    # per-frame decode/analysis.
    max_candidates_per_room = 400
    for rid in list(by_room):
        idxs = sorted(by_room[rid])
        if len(idxs) > max_candidates_per_room:
            pick = np.linspace(0, len(idxs) - 1, max_candidates_per_room).astype(int)
            by_room[rid] = sorted({idxs[k] for k in pick})

    # quality signals over the union of all candidates, decoded in one
    # forward pass. A handful of frames near the end of an HEVC clip can
    # fail to decode (a StrayScanner/ffmpeg quirk, not something a capture
    # protocol can avoid); skip those rather than aborting the whole capture.
    all_idx = sorted({i for v in by_room.values() for i in v} |
                     {i for d in doors for i in (_nearest(fs, d, by_room, side) for side in (0, 1)) if i is not None})
    sharpness: dict[int, float] = {}
    edge_density: dict[int, float] = {}
    floor_frac: dict[int, float] = {}
    ceil_frac: dict[int, float] = {}
    rgb_cache: dict[int, np.ndarray] = {}
    n_failed = 0
    for i in all_idx:
        try:
            rgb = _rotate_upright(vf.get(i))   # StrayScanner frames are sensor-landscape; phone was portrait
        except Exception:
            n_failed += 1
            continue
        rgb_cache[i] = rgb
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        sharpness[i] = _laplacian_sharpness(gray)
        edge_density[i] = _edge_density(gray)
        floor_frac[i], ceil_frac[i] = _floor_ceiling_fraction(fs.frames[i], floor_y, ceil_y)
    if n_failed:
        progress(f"[{capture_id}] {n_failed}/{len(all_idx)} candidate frame(s) failed to decode; skipped")
    good = set(sharpness)
    by_room = {rid: [i for i in idxs if i in good] for rid, idxs in by_room.items()}
    by_room = {rid: idxs for rid, idxs in by_room.items() if len(idxs) >= 2}
    if not by_room:
        progress(f"[{capture_id}] WARNING: no room has decodable frames after filtering; skipping capture")
        return None
    positions = {i: fs.frames[i].position for i in good}
    ceiling_observed = ceil_y is not None
    quality: dict[int, float] = {}
    for idxs in by_room.values():
        quality.update(_quality_score(idxs, sharpness, edge_density, floor_frac, ceil_frac, ceiling_observed))

    n_min, n_max = n_photos
    room_sel: dict[str, set[int]] = {}
    for rid, idxs in by_room.items():
        room_sel[rid] = set(_select_diverse_sharp(idxs, positions, quality, n_min, n_max))

    door_pairs = []
    for d in doors:
        ra, rb = d["rooms"]
        if ra not in by_room or rb not in by_room:
            continue
        ia = _nearest_in_room(fs, d["midpoint_world"], by_room[ra])
        ib = _nearest_in_room(fs, d["midpoint_world"], by_room[rb])
        if ia is None or ib is None:
            continue
        door_pairs.append((ra, rb, ia, ib))
        room_sel[ra].add(ia)
        room_sel[ra].add(ib)   # doorway-photo duplication: both doorway shots in both folders
        room_sel[rb].add(ia)
        room_sel[rb].add(ib)

    # keep within the 2-8 photo contract, protecting doorway frames
    protected = {i for (_, _, ia, ib) in door_pairs for i in (ia, ib)}
    for rid in room_sel:
        sel = room_sel[rid]
        if len(sel) > 8:
            keep_protected = sel & protected
            rest = sorted(sel - protected, key=lambda i: -quality.get(i, 0.0))
            sel2 = set(keep_protected) | set(rest[:max(0, 8 - len(keep_protected))])
            room_sel[rid] = sel2
        if len(room_sel[rid]) < 2 and len(by_room[rid]) >= 2:
            extra = sorted(by_room[rid], key=lambda i: -quality.get(i, 0.0))
            for i in extra:
                room_sel[rid].add(i)
                if len(room_sel[rid]) >= 2:
                    break

    out_dir = out_root / capture_id
    manifest = {"capture_id": capture_id, "source": str(capture_dir), "focal_35mm": focal35,
               "room_names": {}, "rooms": {}, "doors": []}
    for rid, sel in room_sel.items():
        name = _slug(names.get(rid, rid))
        manifest["room_names"][rid] = name
        room_dir = out_dir / name
        frame_list = []
        for i in sorted(sel):
            rgb = rgb_cache.get(i)
            if rgb is None:
                rgb = _rotate_upright(vf.get(i))
            fname = f"{i:06d}.jpg"
            _save_jpeg_with_exif(rgb, room_dir / fname, focal35, capture_id, i)
            frame_list.append({"frame": i, "file": fname, "sharpness": round(sharpness.get(i, 0.0), 1),
                               "edge_density": round(edge_density.get(i, 0.0), 3),
                               "floor_frac": round(floor_frac.get(i, 0.0), 3),
                               "ceil_frac": round(ceil_frac.get(i, 0.0), 3),
                               "quality": round(quality.get(i, 0.0), 3)})
        manifest["rooms"][rid] = {"name": name, "n_photos": len(frame_list), "frames": frame_list}
        progress(f"[{capture_id}] room '{name}' ({rid}): {len(frame_list)} photos -> {room_dir}")
    for (ra, rb, ia, ib) in door_pairs:
        manifest["doors"].append({"rooms": [manifest["room_names"][ra], manifest["room_names"][rb]],
                                  "room_ids": [ra, rb], "frames": [ia, ib]})

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    progress(f"[{capture_id}] wrote manifest -> {out_dir / '_manifest.json'}")
    return manifest


def _nearest(fs, door, by_room, side):
    rid = door["rooms"][side] if side < len(door["rooms"]) else None
    if rid is None or rid not in by_room:
        return None
    return _nearest_in_room(fs, door["midpoint_world"], by_room[rid])


def _nearest_in_room(fs, midpoint_world: np.ndarray, idxs: list[int]):
    if not idxs:
        return None
    d = [np.linalg.norm(fs.frames[i].position - midpoint_world) for i in idxs]
    return idxs[int(np.argmin(d))]


def _slug(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_") or "room"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", action="append", default=None,
                    help="capture dir or id under data/; repeatable. Default: all three bundled captures.")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="data/photo_tier")
    ap.add_argument("--plan", default=None, help="path to a specific plan.json (overrides auto-discovery)")
    ap.add_argument("--n-min", type=int, default=5)
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    captures = a.capture or DEFAULT_CAPTURES
    out_root = Path(a.out)
    for c in captures:
        cdir = Path(c)
        if not cdir.is_absolute() and not cdir.exists():
            cdir = Path(a.data_root) / c
        if not cdir.exists():
            print(f"skip {c}: not found at {cdir}")
            continue
        process_capture(cdir, out_root, plan_path=Path(a.plan) if a.plan else None,
                        n_photos=(a.n_min, a.n_max), seed=a.seed)


if __name__ == "__main__":
    main()
