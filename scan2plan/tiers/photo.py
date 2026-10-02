"""Photo tier: 2-8 stills per room, no depth, no poses, any iPhone 15+.

This is the floor of the three input tiers: every piece of 3D structure has
to be invented from pixels. The pipeline per room folder is:

  1. Load each image (JPG/HEIC/PNG), bake EXIF orientation into the pixels,
     and build a camera intrinsics matrix from EXIF FocalLength /
     FocalLengthIn35mmFilm when present, else from the depth model's own
     focal estimate, else a fixed-FOV heuristic (see `_photo_intrinsics`).
  2. Run a monocular *metric* depth model on every photo
     (`scan2plan.tiers.mono_depth.predict_depth`, the wrapper shared with the
     video tier, disk-cached). `photo_depth.py` was this module's private,
     short-lived copy of that wrapper while mono_depth.py did not exist yet;
     it is now unused here and kept only as a documented fallback (see its
     own docstring) in case mono_depth.py is ever unavailable.
  3. Match SIFT features between every pair of photos in the room, lift the
     matched 2D points to camera-frame 3D points with each photo's own depth
     map, and fit a similarity transform (rotation + translation + a
     per-photo-pair scale ratio) with RANSAC. This is the "feature matching +
     depth-lifted 3D-3D RANSAC with per-photo scale refinement" required by
     the spec: the scale ratio absorbs the fact that a monocular depth model
     has no reason to agree on absolute scale between two different photos
     of the same room.
  4. Build a pose graph over the room's photos (edges = registrations,
     weight = inlier count) and take a maximum-spanning forest. The largest
     component becomes "the room"; smaller components are kept as separate,
     unplaced fragments (never dropped, never allowed to crash the room) and
     reported in `PhotoProperty.warnings` / `meta`.
  5. The room's own point cloud (now posed) is used to estimate gravity
     (`manhattan.gravity_from_normals`) and a dominant Manhattan yaw
     (`manhattan.dominant_yaw`); the whole room is re-expressed in that
     gravity-and-Manhattan-aligned local frame, independently per room, as
     required ("each room in its own gravity-aligned metric frame").

Cross-room links (consumed by `scan2plan.stitch`) come from two sources:
  - "shared_photo": the capture protocol asks for a doorway photo to be
    copied, byte-for-byte, into both of the rooms it connects. If that one
    image's pose was resolved inside both rooms' pose graphs, the relative
    SE(3) between the two rooms drops straight out of the two local poses.
  - "feature_match": any photo in room A is matched against any photo in
    room B exactly as in step 3; a resolved transform (with its own
    rotation+scale, i.e. a similarity transform) and inlier count becomes a
    link, independent of whether the protocol's doorway-duplicate step was
    followed.

Everything here only ever touches the input photos. No LiDAR depth, no
LiDAR pose, is used by this module (see bench/eval_photo_tier.py for the
LiDAR-only *evaluation* path, which is a separate, offline script).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from ..frames import Frame, FrameSet, PHOTO_ERRORS, backproject, to_world
from ..manhattan import dominant_yaw, gravity_from_normals, rotation_aligning, yaw_rotation
try:
    from . import mono_depth as _depth_backend
    # Model choice: bench/eval_depth_scale.py measured apple/DepthPro-hf given
    # the photo's own EXIF-derived focal length (see `_load_photo` below) as
    # having by far the smallest *per-image metric-depth scale bias* against
    # LiDAR of every model compared (median +9-18% vs +27-44% for the
    # Depth-Anything variants; see reports/depth_scale.json). It does NOT win
    # on the metric that actually matters for this tier, though: re-running
    # the full photo-tier pipeline (bench/eval_photo_tier.py) with DepthPro as
    # the dense-depth model made footprint error WORSE on all 3 bench
    # captures (-36/-32/-63 % -> -78/-54/+45 %), because this tier's
    # registration step (_register_pair) backprojects individual SIFT
    # keypoints with per-pixel depth and fits a 3D-3D RANSAC similarity
    # transform: that step is sensitive to *local* depth noise (AbsRel, which
    # DepthPro does not clearly improve: 21-45% vs 30-48% for Depth-Anything),
    # not to the *global median* scale bias the step-1 table measures. A
    # locally-noisier-but-less-biased depth map can and did produce far worse
    # keypoint-level 3D points. Depth-Anything-V2-Metric-Indoor-Small (the
    # original default, Apache-2.0) is therefore kept as DEFAULT_MODEL_ID;
    # DepthPro and Depth-Anything-Base remain selectable via
    # `load_photo_property(..., model_id=...)` for anyone who wants to
    # reproduce or build on this comparison (see also LEGACY_MODEL_ID, kept as
    # an alias of the same default for API clarity/back-compat).
    DEFAULT_MODEL_ID = _depth_backend.DEPTH_ANYTHING_SMALL
    LEGACY_MODEL_ID = _depth_backend.DEPTH_ANYTHING_SMALL
except ImportError:
    from . import photo_depth as _depth_backend
    DEFAULT_MODEL_ID = _depth_backend.DEFAULT_MODEL_ID
    LEGACY_MODEL_ID = _depth_backend.DEFAULT_MODEL_ID
DEPTH_MODEL_ID = DEFAULT_MODEL_ID   # kept for backward compatibility

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
# synthetic "inliers" weight for shared_photo links (see _cross_room_links):
# large enough to always win a max-spanning-forest tie against any
# realistic cross-folder feature_match inlier count.
SHARED_PHOTO_WEIGHT = 1000

# ---------------------------------------------------------------- data model


@dataclass
class PhotoProperty:
    """Output of `load_photo_property`: per-room FrameSets plus cross-room links."""
    rooms: dict[str, FrameSet]                 # room name -> main (largest) component
    fragments: dict[str, list[FrameSet]]       # room name -> unregistered leftover components
    links: list[dict]                          # cross-room link dicts, see `_cross_room_links`
    warnings: list[str]
    meta: dict = field(default_factory=dict)


@dataclass
class _Photo:
    path: Path
    rgb: np.ndarray            # HxWx3 uint8, orientation-corrected
    K: np.ndarray              # 3x3, matches rgb/depth resolution
    depth_raw: np.ndarray      # HxW float32 metres, UNSCALED model output
    kp: list
    desc: np.ndarray | None
    hash: str                  # sha1 of the raw file bytes
    focal_src: str


@dataclass
class _RoomPhotoRef:
    room: str
    path: Path
    hash: str
    pose: np.ndarray | None    # 4x4, this room's local gravity-aligned frame; None if unregistered
    scale: float | None        # this room's resolved depth-scale factor for this photo (post median-recentre)
    K: np.ndarray
    depth_raw: np.ndarray
    kp: list
    desc: np.ndarray | None


# ------------------------------------------------------------------ loading


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS)


def _fov_focal_px(w: int, h: int, fov_deg: float = 70.0) -> float:
    """Fallback intrinsics: assume a ~70 deg horizontal FOV (typical phone main camera)."""
    return (max(w, h) / 2.0) / np.tan(np.radians(fov_deg) / 2.0)


def _photo_intrinsics(path: Path):
    """Load one image, oriented upright, with a best-effort camera matrix.

    Returns (rgb_uint8, K, focal_source_str).
    """
    img = Image.open(path)
    exif = img.getexif()
    try:
        exif_ifd = exif.get_ifd(0x8769)
    except Exception:
        exif_ifd = {}
    raw_w, raw_h = img.size   # native sensor-order size, before orientation is applied
    oriented = ImageOps.exif_transpose(img).convert("RGB")
    rgb = np.array(oriented)
    h, w = rgb.shape[:2]

    focal_px, src = None, None
    f35 = exif.get(41989) or exif_ifd.get(41989)
    if f35:
        try:
            f35v = float(f35)
            if f35v > 0:
                # FocalLengthIn35mmFilm is relative to the 36 mm width of 35mm
                # film; the phone sensor's native long edge is that reference.
                focal_px = f35v / 36.0 * max(raw_w, raw_h)
                src = "exif_focal_length_35mm"
        except (TypeError, ValueError):
            pass
    if focal_px is None:
        f_mm = exif.get(37386) or exif_ifd.get(37386)
        if f_mm:
            try:
                f_mm_v = float(f_mm)
                if f_mm_v > 0:
                    # No 35mm-equivalent tag: approximate with a typical phone
                    # main-camera active sensor width (~7.0 mm). Coarser than
                    # the 35mm-equivalent path; flagged via focal_src.
                    focal_px = f_mm_v / 7.0 * max(raw_w, raw_h)
                    src = "exif_focal_length_mm_approx_sensor"
            except (TypeError, ValueError):
                pass
    if focal_px is None:
        focal_px = _fov_focal_px(w, h)
        src = "fov_heuristic_70deg"

    K = np.array([[focal_px, 0.0, w / 2.0], [0.0, focal_px, h / 2.0], [0.0, 0.0, 1.0]])
    return rgb, K, src


_sift = None


def _get_sift():
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create()
    return _sift


def _load_photo(path: Path, cache_dir, device, model_id: str = DEFAULT_MODEL_ID, progress=None) -> _Photo:
    rgb, K, src = _photo_intrinsics(path)
    file_hash = hashlib.sha1(path.read_bytes()).hexdigest()
    cache_key = f"{file_hash[:24]}_{rgb.shape[1]}x{rgb.shape[0]}"
    # When the photo's own focal length is known (EXIF, not the FOV
    # heuristic), hand it to the depth model as `focal_hint_px`: for models
    # that predict their own focal (DepthPro), this makes the metric depth
    # output exactly what the model would have produced given the TRUE
    # focal length instead of its own FOV-head estimate (see
    # mono_depth.predict_depth's docstring for why the rescale is exact, not
    # an approximation). Models that don't predict a focal (Depth-Anything)
    # silently ignore the hint.
    focal_hint = None if src == "fov_heuristic_70deg" else float(K[0, 0])
    depth_raw, model_focal = _depth_backend.predict_depth(rgb, model_id, cache_key,
                                                           cache_dir=cache_dir, device=device,
                                                           focal_hint_px=focal_hint)
    if src == "fov_heuristic_70deg" and model_focal:
        K = np.array([[model_focal, 0.0, rgb.shape[1] / 2.0],
                      [0.0, model_focal, rgb.shape[0] / 2.0], [0.0, 0.0, 1.0]])
        src = "depth_model_focal_estimate"
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kp, desc = _get_sift().detectAndCompute(gray, None)
    if progress:
        progress(f"  {path.name}: depth + {len(kp or [])} SIFT keypoints ({src})")
    return _Photo(path=path, rgb=rgb, K=K, depth_raw=depth_raw, kp=list(kp or []), desc=desc,
                  hash=file_hash, focal_src=src)


# --------------------------------------------------------- rigid/similarity fit


def _umeyama(P: np.ndarray, Q: np.ndarray):
    """Similarity transform minimising ||Q - (s R P + t)||. P, Q: Nx3."""
    muP, muQ = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - muP, Q - muQ
    n = len(P)
    cov = (Qc.T @ Pc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_p = (Pc ** 2).sum() / n
    s = float(np.sum(D * np.diag(S)) / var_p) if var_p > 1e-12 else 1.0
    t = muQ - s * (R @ muP)
    return R, s, t


def _umeyama_ransac(P: np.ndarray, Q: np.ndarray, thresh: float = 0.12, iters: int = 2000,
                    min_inliers: int = 10, scale_bounds: tuple[float, float] = (0.3, 3.0),
                    seed: int = 0):
    """RANSAC similarity fit of Q ~= s R P + t. Returns dict or None."""
    n = len(P)
    if n < 4:
        return None
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    best_inliers, best_score = None, -1
    for _ in range(iters):
        sample = rng.choice(idx, size=3, replace=False)
        if np.linalg.matrix_rank(P[sample] - P[sample].mean(0), tol=1e-6) < 2:
            continue
        try:
            R, s, t = _umeyama(P[sample], Q[sample])
        except np.linalg.LinAlgError:
            continue
        if not (scale_bounds[0] <= s <= scale_bounds[1]):
            continue
        pred = s * (P @ R.T) + t
        err = np.linalg.norm(pred - Q, axis=1)
        inliers = err < thresh
        score = int(inliers.sum())
        if score > best_score:
            best_score, best_inliers = score, inliers
    if best_inliers is None or best_score < min_inliers:
        return None
    for _ in range(2):   # refit on inliers, then re-threshold once
        R, s, t = _umeyama(P[best_inliers], Q[best_inliers])
        pred = s * (P @ R.T) + t
        err = np.linalg.norm(pred - Q, axis=1)
        best_inliers = err < thresh
    if best_inliers.sum() < min_inliers:
        return None
    R, s, t = _umeyama(P[best_inliers], Q[best_inliers])
    pred = s * (P[best_inliers] @ R.T) + t
    rmse = float(np.sqrt(np.mean(np.sum((pred - Q[best_inliers]) ** 2, axis=1))))
    return {"R": R, "s": float(s), "t": t, "inliers": int(best_inliers.sum()), "n": n, "rmse": rmse}


def _backproject_uv(uv: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    u = np.clip(uv[:, 0], 0, w - 1)
    v = np.clip(uv[:, 1], 0, h - 1)
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    z = depth[vi, ui].astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=1)
    bad = ~np.isfinite(z) | (z <= 0.15) | (z > 8.0)
    pts[bad] = np.nan
    return pts


def _register_pair(a: _Photo, b: _Photo, ratio: float = 0.75, ransac_thresh: float = 0.15,
                   iters: int = 2000, min_inliers: int = 8):
    """Register b onto a: fit A ~= s R B + t. Returns dict(R,s,t,inliers,n,rmse) or None.

    Thresholds measured empirically on the bench photo-tier captures: with
    only 5-8 photos sweeping a whole room, non-adjacent viewpoints
    legitimately share only 10-30 raw matches (not the hundreds a video
    tier gets), and monocular depth error on far points easily exceeds
    10 cm. min_inliers=8 / thresh=0.15 m is the loosest setting that still
    rejects pure-noise pairs (checked against zero-overlap pairs scoring
    <=6 inliers) while keeping the handful of real but weak overlaps that
    are often the only edge connecting two halves of a room's photo set.

    A kornia LoFTR (`pretrained="indoor_new"`) dense-matching fallback for
    SIFT-starved pairs was tried here and measured end-to-end (see
    bench/eval_photo_tier.py and the task report): it reliably raised the
    registration rate (57% -> 62-71% of bench photos, depending on
    threshold) but did not reliably improve footprint/wall-length accuracy
    against LiDAR -- across every inlier/confidence threshold tried, it
    improved footprint error on one bench capture (-35.9% -> -8.6% best
    case) while making it worse, sometimes much worse (up to -81.6%), on
    the other two, because dense matches on repeated/low-texture surfaces
    (tile floors, blank walls) can be RANSAC-self-consistent yet
    geometrically wrong -- monocular depth is itself noisiest exactly
    there, and the max-spanning-forest pose chain is sensitive to any
    single changed edge. Reverted; SIFT-only is kept as measurably more
    stable on the 3-capture bench.
    """
    if a.desc is None or b.desc is None or len(a.kp) < 4 or len(b.kp) < 4:
        return None
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(a.desc, b.desc, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
    if len(good) < 8:
        return None
    uvA = np.array([a.kp[m.queryIdx].pt for m in good])
    uvB = np.array([b.kp[m.trainIdx].pt for m in good])
    PA = _backproject_uv(uvA, a.depth_raw, a.K)
    PB = _backproject_uv(uvB, b.depth_raw, b.K)
    ok = np.isfinite(PA).all(axis=1) & np.isfinite(PB).all(axis=1)
    PA, PB = PA[ok], PB[ok]
    if len(PA) < 8:
        return None
    return _umeyama_ransac(PB, PA, thresh=ransac_thresh, iters=iters, min_inliers=min_inliers)


# ------------------------------------------------------------- pose chaining


def _max_spanning_forest(n: int, edges: list[dict]):
    """Kruskal max-weight spanning forest. edges: list of dict with i,j,inliers.

    Returns (tree_edges, components) where components maps an arbitrary root
    index to the sorted list of node indices in that component.
    """
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    tree = []
    for e in sorted(edges, key=lambda e: -e["inliers"]):
        if union(e["i"], e["j"]):
            tree.append(e)
    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return tree, comps


def _apply_edge(known_pose: np.ndarray, known_scale: float, edge: dict, known_idx: int):
    """Pose+scale of the *other* endpoint of `edge`, given the known endpoint's.

    Edge was fit on RAW (unscaled) depth as A_raw ~= s R B_raw + t with
    i = index(A) (target), j = index(B) (source). Because the fit never saw
    either photo's true metric scale, its translation `t` is expressed in
    the TARGET photo's raw units; it must be multiplied by the target's
    *resolved* scale factor before it is a metric offset usable as a plain
    SE(3) translation (see derivation in the module docstring / design
    notes). Returns (other_idx, other_pose, other_scale).
    """
    R, s, t = edge["R"], edge["s"], edge["t"]
    if known_idx == edge["i"]:
        other_idx = edge["j"]
        scale_i = known_scale                 # i is known here
        other_scale = known_scale * s          # scale[j] = scale[i] * s
    else:
        other_idx = edge["i"]
        other_scale = known_scale / s           # scale[i] = scale[j] / s
        scale_i = other_scale
    t_metric = t * scale_i                      # now a true SE(3) translation
    T_i_from_j = np.eye(4)
    T_i_from_j[:3, :3] = R
    T_i_from_j[:3, 3] = t_metric
    if known_idx == edge["i"]:
        other_pose = known_pose @ T_i_from_j
    else:
        T_j_from_i = np.eye(4)
        T_j_from_i[:3, :3] = R.T
        T_j_from_i[:3, 3] = -R.T @ t_metric
        other_pose = known_pose @ T_j_from_i
    return other_idx, other_pose, other_scale


def _chain_poses(comp: list[int], tree_edges: list[dict], root: int):
    """BFS pose/scale propagation over the tree edges restricted to `comp`.

    Returns (pose, scale) dicts keyed by global photo index.
    """
    members = set(comp)
    local_edges = [e for e in tree_edges if e["i"] in members and e["j"] in members]
    adj: dict[int, list[dict]] = {g: [] for g in comp}
    for e in local_edges:
        adj[e["i"]].append(e)
        adj[e["j"]].append(e)
    pose = {root: np.eye(4)}
    scale = {root: 1.0}
    stack = [root]
    visited = {root}
    while stack:
        cur = stack.pop()
        for e in adj[cur]:
            other = e["j"] if e["i"] == cur else e["i"]
            if other in visited:
                continue
            visited.add(other)
            _, other_pose, other_scale = _apply_edge(pose[cur], scale[cur], e, cur)
            pose[other] = other_pose
            scale[other] = other_scale
            stack.append(other)
    for g in comp:
        pose.setdefault(g, np.eye(4))
        scale.setdefault(g, 1.0)
    return pose, scale


# --------------------------------------------------------------- room assembly


def _scale_translation(T: np.ndarray, k: float) -> np.ndarray:
    T2 = T.copy()
    T2[:3, 3] = T[:3, 3] * k
    return T2


def _make_depth_loader(depth_raw: np.ndarray, scale: float):
    d = (depth_raw * scale).astype(np.float32)

    def load():
        valid = np.isfinite(d) & (d > 0.2) & (d < 8.0)
        return d, valid
    return load


def _make_rgb_loader(rgb: np.ndarray):
    def load():
        return rgb
    return load


def _build_room_frameset(name: str, photos: list[_Photo], comp: list[int], tree_edges: list[dict],
                         main: bool, model_id: str = DEFAULT_MODEL_ID):
    comp = sorted(comp)
    warnings: list[str] = []
    deg = {g: 0 for g in comp}
    for e in tree_edges:
        if e["i"] in deg and e["j"] in deg:
            deg[e["i"]] += 1
            deg[e["j"]] += 1
    root = max(comp, key=lambda g: (deg[g], -g))
    pose_map, scale_map = _chain_poses(comp, tree_edges, root)

    # Robust per-room scale: anchoring the whole room to one arbitrarily-
    # chosen root photo's own (possibly biased) depth estimate is fragile
    # with only 2-8 photos. Recentre on the *median* resolved scale across
    # every registered photo instead -- same relative calibration between
    # photos (unaffected), different, more robust choice of which value
    # counts as "1.0". Camera translations must be rescaled by the same
    # factor so poses stay consistent with the rescaled depth.
    if len(scale_map) > 1:
        med = float(np.median(list(scale_map.values())))
        if med > 1e-9:
            k = 1.0 / med
            scale_map = {g: s * k for g, s in scale_map.items()}
            pose_map = {g: _scale_translation(T, k) for g, T in pose_map.items()}

    pts_list, nrm_list, rel_list = [], [], []
    for g in comp:
        ph = photos[g]
        depth = ph.depth_raw * scale_map[g]
        valid = np.isfinite(depth) & (depth > 0.2) & (depth < 8.0)
        pts, nrm, _ = backproject(depth, valid, ph.K, stride=4)
        if len(pts):
            pw, nw = to_world(pose_map[g], pts, nrm)
            pts_list.append(pw)
            nrm_list.append(nw)
            rel_list.append(pw - pose_map[g][:3, 3])   # points relative to that photo's own camera

    R_align = np.eye(3)
    if nrm_list:
        all_n = np.concatenate(nrm_list)
        horizontal_evidence = int((np.abs(all_n @ np.array([0.0, 1.0, 0.0])) > 0.85).sum())
        if horizontal_evidence >= 50:
            g_up = gravity_from_normals(all_n)
            # gravity_from_normals only recovers the vertical AXIS, not which
            # end is "up": disambiguate the sign the same way the video tier
            # does (manhattan.gravity_from_normals's docstring), using the
            # prior that a room holds more floor+lower-wall volume below eye
            # height than ceiling volume above it, so most backprojected
            # points should sit *below* their own camera along the true up
            # axis (median relative-point projection on g_up < 0).
            all_rel = np.concatenate(rel_list)
            if np.median(all_rel @ g_up) > 0:
                g_up = -g_up
            R_align = rotation_aligning(g_up, np.array([0.0, 1.0, 0.0]))
            aligned_n = all_n @ R_align.T
            yaw, score = dominant_yaw(aligned_n)
            if score > 0.2:
                R_align = yaw_rotation(yaw)[:3, :3] @ R_align
            else:
                warnings.append(f"room '{name}': walls not clearly Manhattan (yaw score {score:.2f}); "
                                 f"gravity-aligned only, wall directions left as reconstructed")
        else:
            warnings.append(f"room '{name}': too little floor/ceiling evidence "
                             f"({horizontal_evidence} pts) to align gravity; frame left in raw "
                             f"camera-derived orientation (wider intervals apply downstream)")

    A = np.eye(4)
    A[:3, :3] = R_align
    aligned_pose_map = {g: A @ pose_map[g] for g in comp}
    frames = []
    for g in comp:
        ph = photos[g]
        frames.append(Frame(index=g, timestamp=float(g), T_wc=aligned_pose_map[g], K_depth=ph.K, K_rgb=ph.K,
                            load_depth=_make_depth_loader(ph.depth_raw, scale_map[g]),
                            load_rgb=_make_rgb_loader(ph.rgb), group=name))
    fs = FrameSet("photo", frames, PHOTO_ERRORS, source=name,
                 meta={"n_photos": len(comp), "main_component": main,
                       "photo_scales": {photos[g].path.name: float(scale_map[g]) for g in comp},
                       "paths": [str(photos[g].path) for g in comp],
                       "focal_sources": {photos[g].path.name: photos[g].focal_src for g in comp},
                       "depth_model": model_id,
                       "depth_model_license": _depth_backend.MODEL_LICENSES.get(model_id, "unknown")
                       if hasattr(_depth_backend, "MODEL_LICENSES") else "unknown"})
    # pose_map returned to the caller is the gravity-aligned (A @ ...) pose,
    # i.e. exactly the Frame.T_wc above: _RoomPhotoRef.pose (used by
    # _cross_room_links to compute cross-room SE(3) transforms) must match
    # what the FrameSet actually carries, not the pre-alignment pose.
    return fs, warnings, aligned_pose_map, scale_map


def _process_room(name: str, paths: list[Path], cache_dir, device, model_id: str = DEFAULT_MODEL_ID,
                  progress=None):
    warnings: list[str] = []
    photos: list[_Photo] = []
    for p in paths:
        try:
            photos.append(_load_photo(p, cache_dir, device, model_id=model_id, progress=progress))
        except Exception as exc:
            warnings.append(f"room '{name}': could not read '{p.name}' ({exc}); skipped")
    n = len(photos)
    if n == 0:
        raise ValueError(f"room '{name}': no readable photos (checked {len(paths)} file(s))")
    if not (2 <= len(paths) <= 8):
        warnings.append(f"room '{name}': {len(paths)} photo file(s) found, outside the 2-8 "
                         f"contract; proceeding with what is available")

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            fit = _register_pair(photos[i], photos[j])
            if fit is not None:
                fit = dict(fit, i=i, j=j)
                edges.append(fit)
    tree_edges, comps = _max_spanning_forest(n, edges)
    comp_list = sorted(comps.values(), key=lambda c: -len(c))
    main_comp = comp_list[0]
    frag_comps = comp_list[1:]

    main_fs, main_warn, main_poses, main_scales = _build_room_frameset(name, photos, main_comp, tree_edges,
                                                                        main=True, model_id=model_id)
    warnings.extend(main_warn)
    frag_fs_list = []
    for comp in frag_comps:
        fs, fw, _, _ = _build_room_frameset(name, photos, comp, tree_edges, main=False, model_id=model_id)
        frag_fs_list.append(fs)
        names = [photos[g].path.name for g in comp]
        warnings.append(f"room '{name}': {len(comp)} photo(s) {names} did not register against the "
                         f"main reconstruction; kept as a separate, unplaced fragment")
    if n == 1:
        warnings.append(f"room '{name}': single photo, no intra-room registration possible; "
                         f"scale and orientation rely entirely on the depth model (unverified)")

    refs = []
    for g, p in enumerate(photos):
        in_main = g in main_comp
        pose = main_poses.get(g) if in_main else None
        scale = main_scales.get(g) if in_main else None
        refs.append(_RoomPhotoRef(room=name, path=p.path, hash=p.hash, pose=pose, scale=scale, K=p.K,
                                  depth_raw=p.depth_raw, kp=p.kp, desc=p.desc))
    return main_fs, frag_fs_list, refs, warnings


# ---------------------------------------------------------------- cross-room


def _apply_global_scale(fs: FrameSet, c: float):
    """Rescale every frame in `fs` by `c`: camera translations and the
    metric depth each frame loads, in place. Used only for the
    cross-room scale-anchoring correction below -- a uniform rescale of a
    whole room about its own origin leaves every relative measurement
    inside that room (wall lengths, door widths, areas up to c^2) exactly
    as internally self-consistent as before, it just changes which
    absolute metres they are expressed in.
    """
    if abs(c - 1.0) < 1e-9:
        return
    for f in fs.frames:
        f.T_wc = _scale_translation(f.T_wc, c)
        old_load = f.load_depth

        def new_load(old=old_load, c=c):
            d, v = old()
            return d * c, v
        f.load_depth = new_load
    fs.meta["global_scale_correction"] = round(float(c), 4)


def _cross_room_scale_corrections(registry: list[_RoomPhotoRef], progress=None) -> dict[str, float]:
    """Shared doorway photos as scale anchors.

    A doorway photo copied byte-for-byte into two room folders has exactly
    one underlying depth map, but each room's own registration resolves an
    independent scale for it (relative to that room's own, separately
    median-recentred, photo set). Any disagreement between the two rooms'
    resolved scale for *that one photo* is therefore a direct, measured
    estimate of their relative absolute-scale error -- not a modelling
    assumption, unlike the feature_match path's "scale=1 since we can't
    tell" default. This chains those pairwise ratios into one global
    per-room correction via a spanning tree over the shared-photo graph.
    Rooms with no shared-photo path to anything else get no correction:
    there is nothing measured to correct them against.
    """
    hash_rooms: dict[str, dict[str, _RoomPhotoRef]] = {}
    for r in registry:
        if r.pose is None or r.scale is None or r.scale <= 1e-9:
            continue
        hash_rooms.setdefault(r.hash, {}).setdefault(r.room, r)

    # A single doorway photo is only ever one noisy measurement (each
    # room's own resolved scale for it rests on however well that one
    # photo happened to register against its own 1-7 neighbours). Chaining
    # a wild single-edge disagreement multiplicatively through several
    # rooms can amplify rather than correct error (measured: an
    # unclamped version of this produced a >2000% area error on one
    # capture). A per-edge ratio outside [0.25, 4] is treated as the
    # anchor photo having registered badly in at least one of the two
    # rooms, not as a real 4x+ scale disagreement, and is dropped rather
    # than trusted.
    edge_bounds = (0.25, 4.0)
    adj: dict[str, list[tuple[str, float]]] = {}
    dropped = 0
    for h, by_r in hash_rooms.items():
        rn = sorted(by_r)
        for a_i in range(len(rn)):
            for b_i in range(a_i + 1, len(rn)):
                ra, rb = by_r[rn[a_i]], by_r[rn[b_i]]
                ratio_ab = ra.scale / rb.scale   # c_b = c_a * ratio_ab
                if not (edge_bounds[0] <= ratio_ab <= edge_bounds[1]):
                    dropped += 1
                    continue
                adj.setdefault(ra.room, []).append((rb.room, ratio_ab))
                adj.setdefault(rb.room, []).append((ra.room, 1.0 / ratio_ab))
    if dropped and progress:
        progress(f"  scale anchor: ignored {dropped} shared-photo scale ratio(s) outside "
                f"[{edge_bounds[0]}, {edge_bounds[1]}]x (unreliable single-photo measurement)")
    if not adj:
        return {}

    correction: dict[str, float] = {}
    for start in sorted(adj):
        if start in correction:
            continue
        correction[start] = 1.0
        stack = [start]
        while stack:
            cur = stack.pop()
            for other, ratio in adj[cur]:
                if other in correction:
                    continue
                correction[other] = correction[cur] * ratio
                stack.append(other)

    # Same guard on the final chained value (a couple of borderline-ok
    # edges can still compound past a sane range over a longer chain).
    chain_bounds = (0.3, 3.0)
    for name in list(correction):
        c = correction[name]
        if not (chain_bounds[0] <= c <= chain_bounds[1]):
            if progress:
                progress(f"  scale anchor: room '{name}' chained correction {c:.3f}x outside "
                        f"[{chain_bounds[0]}, {chain_bounds[1]}]x; not applying it")
            correction[name] = 1.0

    if progress:
        for name, c in sorted(correction.items()):
            progress(f"  scale anchor: room '{name}' global correction = {c:.3f}x")
    return correction



def _cross_room_links(registry: list[_RoomPhotoRef], progress=None, min_inliers: int = 12):
    links: list[dict] = []
    by_room: dict[str, list[_RoomPhotoRef]] = {}
    for r in registry:
        by_room.setdefault(r.room, []).append(r)
    rooms = sorted(by_room)

    hash_rooms: dict[str, dict[str, _RoomPhotoRef]] = {}
    for r in registry:
        if r.pose is None:
            continue
        hash_rooms.setdefault(r.hash, {}).setdefault(r.room, r)

    done_pairs: set[tuple[str, str]] = set()
    for h, by_r in hash_rooms.items():
        rn = sorted(by_r)
        if len(rn) < 2:
            continue
        for a_i in range(len(rn)):
            for b_i in range(a_i + 1, len(rn)):
                key = (rn[a_i], rn[b_i])
                if key in done_pairs:
                    continue
                ra, rb = by_r[rn[a_i]], by_r[rn[b_i]]
                T_a_from_b = ra.pose @ np.linalg.inv(rb.pose)
                # "inliers" here is really a placement-confidence weight fed
                # into stitch.py's max-spanning-forest edge selection: a
                # byte-identical doorway photo registered independently in
                # both rooms' own pose graphs is far more trustworthy than
                # any generic cross-folder SIFT match (which can and does
                # false-positive between visually similar rooms, e.g. same
                # flooring/paint, in the same apartment -- see the eval
                # report), so it must outrank feature_match whenever both
                # exist for the same room pair.
                links.append({"rooms": [ra.room, rb.room], "T_a_from_b": T_a_from_b, "scale": 1.0,
                             "inliers": SHARED_PHOTO_WEIGHT, "evidence": "shared_photo",
                             "via": [ra.path.name]})
                done_pairs.add(key)
                if progress:
                    progress(f"  link {ra.room} <-> {rb.room}: shared photo {ra.path.name}")

    for ai in range(len(rooms)):
        for bi in range(ai + 1, len(rooms)):
            key = (rooms[ai], rooms[bi])
            if key in done_pairs:
                continue
            best = None
            for pa in by_room[rooms[ai]]:
                if pa.pose is None:
                    continue
                for pb in by_room[rooms[bi]]:
                    if pb.pose is None:
                        continue
                    fit = _register_pair(pa, pb, min_inliers=min_inliers)
                    if fit is not None and (best is None or fit["inliers"] > best[0]["inliers"]):
                        best = (fit, pa, pb)
            if best is None:
                continue
            fit, pa, pb = best
            T_cam = np.eye(4)
            T_cam[:3, :3] = fit["s"] * fit["R"]
            T_cam[:3, 3] = fit["t"]
            T_a_from_b = pa.pose @ T_cam @ np.linalg.inv(pb.pose)
            links.append({"rooms": [rooms[ai], rooms[bi]], "T_a_from_b": T_a_from_b, "scale": fit["s"],
                         "inliers": fit["inliers"], "evidence": "feature_match",
                         "via": [pa.path.name, pb.path.name]})
            if progress:
                progress(f"  link {rooms[ai]} <-> {rooms[bi]}: feature match "
                        f"{pa.path.name}/{pb.path.name} ({fit['inliers']} inliers)")
    return links


# --------------------------------------------------------------------- public


def load_photo_property(root: str | Path, cache_dir=".cache", device=None, model_id: str = DEFAULT_MODEL_ID,
                        progress=None) -> PhotoProperty:
    """Build a PhotoProperty from `root`/<room>/*.{jpg,heic,png}.

    `root` must contain one sub-folder per room (any names), each holding
    2-8 photos. Returns per-room FrameSets (tier="photo") plus cross-room
    links derived from shared doorway photos and cross-folder feature
    matches. Never raises on a registration failure inside a room (bad
    photos become fragments); only raises if a room folder has zero
    readable images or the root has no room sub-folders at all.

    `model_id`: dense per-photo depth model (see mono_depth.py / this
    module's `DEFAULT_MODEL_ID` / `LEGACY_MODEL_ID`). Defaults to the model
    bench/eval_depth_scale.py measured as having the smallest metric-depth
    scale bias against LiDAR; pass `LEGACY_MODEL_ID` to restore the
    previous Depth-Anything-V2-Metric-Indoor-Small behaviour.
    """
    root = Path(root)
    room_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not room_dirs:
        raise ValueError(f"{root}: no room sub-folders found")

    rooms: dict[str, FrameSet] = {}
    fragments: dict[str, list[FrameSet]] = {}
    warnings: list[str] = []
    registry: list[_RoomPhotoRef] = []
    meta: dict = {"rooms": {}}

    for room_dir in room_dirs:
        name = room_dir.name
        paths = _list_images(room_dir)
        if progress:
            progress(f"[{name}] {len(paths)} photo file(s)")
        main_fs, frag_fs_list, refs, room_warnings = _process_room(name, paths, cache_dir, device,
                                                                   model_id=model_id, progress=progress)
        rooms[name] = main_fs
        if frag_fs_list:
            fragments[name] = frag_fs_list
        warnings.extend(room_warnings)
        registry.extend(refs)
        meta["rooms"][name] = {
            "n_photo_files": len(paths),
            "n_registered_main": len(main_fs.frames),
            "n_fragment_components": len(frag_fs_list),
            "n_fragment_photos": sum(len(f.frames) for f in frag_fs_list),
        }

    corrections = _cross_room_scale_corrections(registry, progress)
    if corrections:
        for name, c in corrections.items():
            if abs(c - 1.0) < 1e-6:
                continue
            _apply_global_scale(rooms[name], c)
            for fs in fragments.get(name, []):
                _apply_global_scale(fs, c)
            meta["rooms"][name]["global_scale_correction"] = round(c, 4)
            warnings.append(f"room '{name}': rescaled by {c:.3f}x using shared-doorway-photo scale "
                            f"anchoring (its own depth-model scale disagreed with a neighbour's "
                            f"independent estimate of the same photo)")
        # the registry's cached poses/scales predate this correction; refresh
        # them so _cross_room_links' feature_match path (which still reads
        # registry poses, not the FrameSets) sees the corrected geometry too
        for r in registry:
            c = corrections.get(r.room)
            if c and r.pose is not None:
                r.pose = _scale_translation(r.pose, c)
                r.scale = r.scale * c if r.scale is not None else None

    links = _cross_room_links(registry, progress)
    meta["n_links"] = len(links)
    meta["n_rooms"] = len(room_dirs)
    meta["n_scale_corrections"] = sum(1 for c in corrections.values() if abs(c - 1.0) >= 1e-6)
    if len(links) < len(room_dirs) - 1:
        warnings.append(f"only {len(links)} cross-room link(s) found for {len(room_dirs)} rooms; "
                        f"the stitched plan may be unable to place every room with evidence "
                        f"(see the Photos section of docs/capture_protocol.md for the doorway-photo step)")

    return PhotoProperty(rooms=rooms, fragments=fragments, links=links, warnings=warnings, meta=meta)
