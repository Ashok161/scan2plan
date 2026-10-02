"""Per-surface damage detection, projected and fused across views.

Pipeline (documented in detail in the technical report, "Damage detection"):

  1. Keyframe selection reuses `fusion.select_keyframes` (same translation /
     rotation thresholds as plan building) so damage keyframes line up with
     the frames the plan was built from, capped at `max_frames`.
  2. Open-vocabulary box proposals: OWLv2 (google/owlv2-base-patch16-ensemble,
     Apache-2.0, via `transformers`) scores each keyframe against a short
     text prompt per damage class. This is the only learned model in the
     stage; everything downstream is classical and deterministic.
  3. Each box gets a pixel mask from a class-specific classical segmentation
     rule (colour/texture contrast against a clean-surface reference sampled
     from the box margin). We deliberately do not run SAM/SAM2: at the
     resolution the LiDAR depth map actually supports (256x192, re-upsampled
     to RGB res), a learned mask's boundary precision is below the
     backprojection's own pixel footprint, so it would add model weight and
     latency without moving the measured extent error (see bench report).
  4. Masked pixels are backprojected with the frame's own depth + pose,
     mapped into the Plan frame via `plan.T_align`, and assigned to the
     nearest matching planar surface, subject to a point-to-plane residual
     gate. Views whose masked pixels do not actually sit on the candidate
     surface's plane are dropped as `geometry_mismatch` -- this is the
     primary defence against mirrors/glass (LiDAR confidence already drops
     out most glass/mirror returns upstream in `io/stray.py`; the residual
     gate catches what gets through, e.g. deep furniture, open doorways).
  5. Per-surface, per-class candidates from different views are clustered
     by 2D overlap in the surface's own (u, v) metres frame and fused: the
     polygon is the union of the per-view masks, confidence blends detector
     score, view count and cross-view colour consistency. Reflections and
     specular glare are view-dependent in appearance even when the backing
     geometry is static (a real wall), so a low colour-consistency score
     kills them even if they survive the geometry gate.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .frames import FrameSet
from .fusion import select_keyframes
from .measure import Measurement, combine
from .plan_types import Plan

# --------------------------------------------------------------------------
# Taxonomy
# --------------------------------------------------------------------------
# Five classes chosen to cover what a visual inspection of a residential
# interior can evidence without invasive testing (no moisture meter, no
# borescope -- those live downstream as "recommended investigation" in
# concealed.py). Each maps to one or more plain-language prompts for the
# open-vocabulary detector and one classical segmentation rule below.
DAMAGE_CLASSES = ["water_stain", "mould", "crack", "hole_or_impact", "peeling_paint"]

PROMPTS = {
    "water_stain": ["a brown water stain on the wall", "a damp patch on the ceiling",
                    "a yellow water stain"],
    "mould": ["black mould spots on the wall", "a mildew patch", "dark mold speckles"],
    "crack": ["a crack in the wall", "a crack in the ceiling", "a split in the plaster"],
    "hole_or_impact": ["a hole in the wall", "a broken hole in the drywall",
                       "impact damage on a wall"],
    "peeling_paint": ["peeling paint on the wall", "flaking paint", "paint bubbling off a wall"],
}

MODEL_ID = "google/owlv2-base-patch16-ensemble"   # Apache-2.0, Google (OWLv2)
MODEL_LICENSE = "apache-2.0"

# Surfaces where each class is even considered. Water stains / mould on a
# floor are overwhelmingly wet-look tile glare or a rug in these captures;
# excluding floor for those two classes is a deliberate false-positive
# control, documented in the bench report's failure-mode section.
CLASS_SURFACE_KINDS = {
    "water_stain": {"wall", "ceiling"},
    "mould": {"wall", "ceiling"},
    "crack": {"wall", "ceiling"},
    "hole_or_impact": {"wall"},
    "peeling_paint": {"wall", "ceiling"},
}

_BOX_SCORE_THRESHOLD = 0.22      # tuned against the measured FP rate, see bench report
_GEOM_RESIDUAL_MAX = 0.06        # m, point-to-plane gate (mirror/glass/furniture reject)
_GEOM_VALID_FRAC_MIN = 0.35      # min fraction of masked px with usable depth
_MIN_VIEWS_CONFIRMED = 2         # cross-view agreement required for "confirmed"
_CLUSTER_DIST = 0.35             # m, surface-uv centroid distance to merge candidates across views
_COLOR_CONSISTENCY_SCALE = 18.0  # Lab distance scale for the consistency score


# --------------------------------------------------------------------------
# Model loading / prefetch
# --------------------------------------------------------------------------
_MODEL_CACHE: dict[str, tuple] = {}


def prefetch_damage_models() -> dict:
    """Download (or verify cached) weights for the damage stage. No inference.

    Wired into scripts/fetch_weights.py by the coordinator. Safe to call
    repeatedly; `from_pretrained` is a no-op once the HF cache is warm.
    """
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    Owlv2Processor.from_pretrained(MODEL_ID)
    Owlv2ForObjectDetection.from_pretrained(MODEL_ID)
    return {"model": MODEL_ID, "license": MODEL_LICENSE, "task": "zero-shot object detection"}


def _get_model(device: str):
    if device not in _MODEL_CACHE:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        proc = Owlv2Processor.from_pretrained(MODEL_ID)
        model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(device).eval()
        _MODEL_CACHE[device] = (proc, model, torch)
    return _MODEL_CACHE[device]


def _pick_device(device: str | None) -> str:
    if device:
        return device
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --------------------------------------------------------------------------
# Detector cache (disk) -- keyed on source path + frame index + model id so
# reruns are deterministic and fast; the live path (uncached) is identical
# code, just without the json round-trip.
# --------------------------------------------------------------------------
def _cache_path(cache_dir: Path, source: str, frame_idx: int, enabled_classes: tuple[str, ...]) -> Path:
    # the class tag is part of the key: a frame detected with a different
    # set of enabled prompts is a genuinely different model call, not a
    # cache hit (disabling peeling_paint by default must not silently
    # replay a stale 5-class result, nor vice versa).
    tag = ",".join(enabled_classes)
    h = hashlib.sha1(f"{source}|{tag}".encode()).hexdigest()[:12]
    d = cache_dir / "damage_boxes" / h
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{frame_idx:06d}.json"


def _detect_boxes_cached(rgb: np.ndarray, cache_dir: Path, source: str, frame_idx: int,
                          device: str, enabled_classes: tuple[str, ...] = tuple(DAMAGE_CLASSES)) -> list[dict]:
    cp = _cache_path(cache_dir, source, frame_idx, enabled_classes)
    if cp.exists():
        return json.loads(cp.read_text())
    boxes = _detect_boxes(rgb, device, enabled_classes)
    txt = json.dumps(boxes)
    cp.write_text(txt)
    return json.loads(txt)          # live run returns exactly what a cache replay returns


def _detect_boxes(rgb: np.ndarray, device: str, enabled_classes: tuple[str, ...] = tuple(DAMAGE_CLASSES)) -> list[dict]:
    """OWLv2 zero-shot boxes for every *enabled* damage class on one RGB frame."""
    from PIL import Image
    proc, model, torch = _get_model(device)
    img = Image.fromarray(rgb)
    classes = [c for c in DAMAGE_CLASSES if c in enabled_classes]
    all_prompts = [p for cls in classes for p in PROMPTS[cls]]
    prompt_cls = [cls for cls in classes for _ in PROMPTS[cls]]
    inputs = proc(text=[all_prompts], images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    target_sizes = torch.tensor([img.size[::-1]])
    res = proc.post_process_grounded_object_detection(
        out, target_sizes=target_sizes, threshold=_BOX_SCORE_THRESHOLD)[0]
    boxes = []
    for box, score, label in zip(res["boxes"].tolist(), res["scores"].tolist(), res["labels"].tolist()):
        boxes.append({"class": prompt_cls[int(label)], "score": float(score),
                     "box": [float(v) for v in box]})
    return boxes


# --------------------------------------------------------------------------
# Classical mask segmentation inside a detector box
# --------------------------------------------------------------------------
def _reference_color(rgb_lab: np.ndarray, box: tuple[int, int, int, int], margin: int, shape):
    """Median Lab colour of a ring just outside the box: the 'clean surface' baseline."""
    h, w = shape
    x0, y0, x1, y1 = box
    xo0, yo0 = max(0, x0 - margin), max(0, y0 - margin)
    xo1, yo1 = min(w, x1 + margin), min(h, y1 + margin)
    ring = rgb_lab[yo0:yo1, xo0:xo1].reshape(-1, 3)
    inner = rgb_lab[y0:y1, x0:x1].reshape(-1, 3)
    if len(ring) <= len(inner) + 4:
        return np.median(inner, axis=0)
    # subtract the inner box's contribution by just using the border strip
    mask = np.ones((yo1 - yo0, xo1 - xo0), bool)
    mask[y0 - yo0:y1 - yo0, x0 - xo0:x1 - xo0] = False
    strip = rgb_lab[yo0:yo1, xo0:xo1][mask]
    if len(strip) < 20:
        return np.median(inner, axis=0)
    return np.median(strip, axis=0)


def _segment_mask(crop_bgr: np.ndarray, cls: str, ref_lab: np.ndarray) -> tuple[np.ndarray, dict]:
    """Boolean mask of damaged pixels inside a crop, plus evidence stats."""
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    diff = np.linalg.norm(lab - ref_lab[None, None, :], axis=-1)
    stats = {"ref_L": float(ref_lab[0]), "ref_a": float(ref_lab[1]), "ref_b": float(ref_lab[2])}

    if cls == "water_stain":
        # darker + yellow/brown shift (lower L*, higher b*) vs. the clean baseline
        score = np.clip(ref_lab[0] - lab[..., 0], 0, None) + np.clip(lab[..., 2] - ref_lab[2], 0, None)
        mask = _otsu_mask(score)
    elif cls == "mould":
        # dark, low-saturation speckle: darker + desaturated vs. baseline, with local texture
        tex = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        tex = cv2.GaussianBlur(np.abs(tex), (5, 5), 0)
        darker = np.clip(ref_lab[0] - lab[..., 0], 0, None)
        score = darker * (0.4 + np.clip(tex, 0, 60) / 60.0)
        mask = _otsu_mask(score)
    elif cls == "crack":
        edges = cv2.Canny(gray, 40, 120)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        mask = edges > 0
        mask = _keep_elongated(mask)
    elif cls == "hole_or_impact":
        darker = np.clip(ref_lab[0] - lab[..., 0], 0, None)
        mask = _otsu_mask(darker)
        mask = _keep_compact(mask)
    elif cls == "peeling_paint":
        tex = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        tex = cv2.GaussianBlur(np.abs(tex), (5, 5), 0)
        mask = _otsu_mask(tex)
    else:
        mask = diff > diff.mean()

    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    stats["mask_frac"] = float(mask.mean())
    return mask, stats


def _otsu_mask(score: np.ndarray) -> np.ndarray:
    s = score.astype(np.float32)
    lo, hi = np.percentile(s, 1), np.percentile(s, 99)
    if hi <= lo:
        return np.zeros_like(s, bool)
    norm = np.clip((s - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    thr, m = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m > 0


def _keep_elongated(mask: np.ndarray, min_aspect: float = 2.5, min_area: int = 20) -> np.ndarray:
    n, lbl = cv2.connectedComponents(mask.astype(np.uint8))
    out = np.zeros_like(mask)
    for i in range(1, n):
        comp = lbl == i
        ys, xs = np.where(comp)
        if len(xs) < min_area:
            continue
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        (_, _), (w, h), _ = cv2.minAreaRect(pts)
        major, minor = max(w, h), max(min(w, h), 1e-3)
        if major / minor >= min_aspect:
            out |= comp
    return out


def _keep_compact(mask: np.ndarray, min_compactness: float = 0.35, min_area: int = 25) -> np.ndarray:
    n, lbl = cv2.connectedComponents(mask.astype(np.uint8))
    out = np.zeros_like(mask)
    for i in range(1, n):
        comp = lbl == i
        area = int(comp.sum())
        if area < min_area:
            continue
        ys, xs = np.where(comp)
        bbox_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
        if area / max(bbox_area, 1) >= min_compactness:
            out |= comp
    return out


# --------------------------------------------------------------------------
# Surface geometry helpers
# --------------------------------------------------------------------------
def _surface_basis(surface: dict):
    """Orthonormal (u_hat, v_hat) in-plane basis + a stable origin for a surface."""
    n = surface["normal"] / np.linalg.norm(surface["normal"])
    origin = surface["corners"][0]
    if abs(n[1]) > 0.9:   # floor / ceiling: plan (x, z) is already the in-plane basis
        u_hat = np.array([1.0, 0.0, 0.0])
        v_hat = np.array([0.0, 0.0, 1.0])
    else:                 # wall: horizontal along-wall axis, vertical = world up
        v_hat = np.array([0.0, 1.0, 0.0])
        u_hat = np.cross(v_hat, n)
        u_hat /= max(np.linalg.norm(u_hat), 1e-9)
    uv_corners = np.stack([
        np.array([(c - origin) @ u_hat, (c - origin) @ v_hat]) for c in surface["corners"]
    ])
    return origin, u_hat, v_hat, uv_corners


def _polygon_contains(uv_poly: np.ndarray, pts_uv: np.ndarray, margin: float = 0.08) -> np.ndarray:
    from shapely.geometry import Point, Polygon
    poly = Polygon(uv_poly).buffer(margin)
    return np.array([poly.contains(Point(p)) for p in pts_uv])


def _best_surface(point_plan: np.ndarray, surfaces: list[dict], bases: list) -> int | None:
    """Index of the surface whose plane the point is closest to AND inside (with margin)."""
    best_i, best_d = None, 1e9
    for i, (s, (origin, u_hat, v_hat, uv_corners)) in enumerate(zip(surfaces, bases)):
        n = s["normal"] / np.linalg.norm(s["normal"])
        d = abs(n @ point_plan - s["offset"])
        if d > _GEOM_RESIDUAL_MAX or d >= best_d:
            continue
        uv = np.array([(point_plan - origin) @ u_hat, (point_plan - origin) @ v_hat])
        from shapely.geometry import Point, Polygon
        if Polygon(uv_corners).buffer(0.1).contains(Point(uv)):
            best_i, best_d = i, d
    return best_i


# --------------------------------------------------------------------------
# Candidate = one class detection on one surface from one view, in uv metres
# --------------------------------------------------------------------------
class _Candidate:
    __slots__ = ("surface_idx", "cls", "score", "frame", "uv_pts", "mean_lab",
                "n_pts", "residual_mean", "valid_frac", "range_m", "contrast")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


_WORK_MAX_SIDE = 960   # classical-CV / backprojection working resolution cap (runtime; see module docstring)


def _process_view(fr, fs: FrameSet, plan: Plan, surfaces: list[dict], bases: list,
                  surf_kind: list[str], boxes: list[dict], device: str) -> list[_Candidate]:
    depth, valid = fr.load_depth()
    rgb_full = fr.load_rgb()
    if rgb_full is None:
        return []
    h_full, w_full = rgb_full.shape[:2]
    # Work at a capped resolution for everything *after* detection: the box
    # coordinates below are rescaled into this frame. OWLv2 itself always
    # sees the frame at full resolution (box detection is cached keyed on
    # full-res coordinates), only the per-box classical segmentation,
    # Lab conversion and depth backprojection -- the actual per-frame cost --
    # run at the capped size. Depth's native grid is 256x192, so even at
    # _WORK_MAX_SIDE=960 we oversample it ~4-5x; there is no detail to lose.
    scale = min(1.0, _WORK_MAX_SIDE / max(h_full, w_full))
    if scale < 1.0:
        w, h = int(round(w_full * scale)), int(round(h_full * scale))
        rgb = cv2.resize(rgb_full, (w, h), interpolation=cv2.INTER_AREA)
    else:
        w, h = w_full, h_full
        rgb = rgb_full

    depth_up = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    valid_up = cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # mild low-light compensation for the *detector/segmenter* input only
    v_mean = rgb.mean()
    if v_mean < 60:
        lab_eq = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab_eq[..., 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab_eq[..., 0])
        bgr = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    lab_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)   # once per frame, not per box

    fx, fy, cx, cy = fr.K_rgb[0, 0] * scale, fr.K_rgb[1, 1] * scale, fr.K_rgb[0, 2] * scale, fr.K_rgb[1, 2] * scale
    R_wc, t_wc = fr.T_wc[:3, :3], fr.T_wc[:3, 3]
    T_align = plan.T_align

    out = []
    for b in boxes:
        cls = b["class"]
        x0, y0, x1, y1 = [int(round(v * scale)) for v in b["box"]]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        crop = bgr[y0:y1, x0:x1]
        ref = _reference_color(lab_full, (x0, y0, x1, y1), margin=max(6, (x1 - x0) // 4), shape=(h, w))
        mask, _ = _segment_mask(crop, cls, ref)
        if mask.sum() < 10:
            continue

        ys, xs = np.where(mask)
        ys, xs = ys + y0, xs + x0
        # segmentation contrast: how far the masked pixels sit from the local
        # "clean surface" reference colour -- a cheap, model-free confidence
        # signal (noise/texture FPs tend to be low-contrast; real damage,
        # even subtle, is a locally distinct patch, that's the whole premise
        # of the Otsu split in _segment_mask).
        contrast = float(np.linalg.norm(lab_full[ys, xs] - ref[None, :], axis=1).mean())
        d = depth_up[ys, xs]
        v = valid_up[ys, xs]
        valid_frac = float(v.mean())
        if valid_frac < _GEOM_VALID_FRAC_MIN:
            continue
        xs_v, ys_v, d_v = xs[v], ys[v], d[v]
        if len(d_v) < 10:
            continue
        pts_cam = np.stack([(xs_v - cx) / fx * d_v, (ys_v - cy) / fy * d_v, d_v], axis=1)
        pts_world = pts_cam @ R_wc.T + t_wc
        pts_plan = (T_align[:3, :3] @ pts_world.T).T + T_align[:3, 3]

        # candidate surfaces of the right kind only
        cand_idx = [i for i, k in enumerate(surf_kind) if k in CLASS_SURFACE_KINDS[cls]]
        if not cand_idx:
            continue
        # vectorised distance to each candidate plane; pick per-point best, then majority surface
        dmat = np.stack([
            np.abs(surfaces[i]["normal"] / np.linalg.norm(surfaces[i]["normal"]) @ pts_plan.T
                   - surfaces[i]["offset"]) for i in cand_idx
        ], axis=0)
        best_local = np.argmin(dmat, axis=0)
        best_d = dmat[best_local, np.arange(len(pts_plan))]
        ok = best_d < _GEOM_RESIDUAL_MAX
        if ok.mean() < 0.5:
            continue   # geometry_mismatch: likely glass/mirror/occluder, not this plane
        votes = np.bincount(best_local[ok])
        chosen_local = int(np.argmax(votes))
        surf_i = cand_idx[chosen_local]
        sel = ok & (best_local == chosen_local)
        if sel.sum() < 10:
            continue

        origin, u_hat, v_hat, uv_corners = bases[surf_i]
        pp = pts_plan[sel]
        uv = np.stack([(pp - origin) @ u_hat, (pp - origin) @ v_hat], axis=1)
        inside = _polygon_contains(uv_corners, uv, margin=0.1)
        if inside.sum() < 8:
            continue
        uv = uv[inside]
        range_m = float(np.linalg.norm(pts_cam[sel][inside], axis=1).mean())
        lab_px = lab_full[ys_v, xs_v][sel][inside]

        out.append(_Candidate(
            surface_idx=surf_i, cls=cls, score=b["score"], frame=fr.index,
            uv_pts=uv, mean_lab=lab_px.mean(axis=0), n_pts=int(len(uv)),
            residual_mean=float(best_d[sel][inside].mean()), valid_frac=valid_frac,
            range_m=range_m, contrast=contrast,
        ))
    return out


# --------------------------------------------------------------------------
# Fusion across views
# --------------------------------------------------------------------------
def _cluster_candidates(cands: list[_Candidate]) -> list[list[_Candidate]]:
    """Greedy clustering by (surface, class) and uv-centroid proximity."""
    groups: dict[tuple, list[_Candidate]] = {}
    for c in cands:
        groups.setdefault((c.surface_idx, c.cls), []).append(c)
    clusters = []
    for _, items in groups.items():
        items = sorted(items, key=lambda c: -c.n_pts)
        used = [False] * len(items)
        centroids = [c.uv_pts.mean(axis=0) for c in items]
        for i, c in enumerate(items):
            if used[i]:
                continue
            cluster = [c]
            used[i] = True
            for j in range(i + 1, len(items)):
                if used[j]:
                    continue
                if np.linalg.norm(centroids[i] - centroids[j]) < _CLUSTER_DIST:
                    cluster.append(items[j])
                    used[j] = True
            clusters.append(cluster)
    return clusters


def _hull_polygon(pts: np.ndarray):
    from shapely.geometry import MultiPoint
    mp = MultiPoint(pts)
    hull = mp.convex_hull
    if hull.geom_type == "Point":
        return hull.buffer(0.01)
    if hull.geom_type == "LineString":
        return hull.buffer(0.01)
    return hull


def _fuse_cluster(cluster: list[_Candidate], fs: FrameSet, plan_tier: str) -> dict | None:
    all_frames = sorted(set(c.frame for c in cluster))
    n_views = len(all_frames)
    all_pts = np.concatenate([c.uv_pts for c in cluster], axis=0)
    poly = _hull_polygon(all_pts)
    area_val = float(poly.area)
    if area_val < 1e-4:
        return None
    minx, miny, maxx, maxy = poly.bounds
    width_val = maxx - minx
    height_val = maxy - miny

    # colour consistency across views (kills reflections / specular glare)
    view_colors = np.stack([c.mean_lab for c in cluster])
    if len(cluster) > 1:
        color_spread = float(np.linalg.norm(view_colors - view_colors.mean(axis=0), axis=1).mean())
    else:
        color_spread = 0.0
    color_consistency = float(math.exp(-color_spread / _COLOR_CONSISTENCY_SCALE))

    det_score = float(np.mean([c.score for c in cluster]))
    view_bonus = min(n_views / _MIN_VIEWS_CONFIRMED, 1.0)
    confidence = float(np.clip(0.5 * det_score + 0.3 * view_bonus + 0.2 * color_consistency, 0.0, 1.0))
    if n_views < _MIN_VIEWS_CONFIRMED:
        confidence *= 0.6   # single-view: down-weighted, still reported (recall over silence)

    contrast_mean = float(np.mean([c.contrast for c in cluster]))
    # Stricter multi-view geometric consistency: cluster membership above is
    # just centroid proximity (<_CLUSTER_DIST), which tolerates views whose
    # footprints barely overlap at all -- real persistent damage should
    # paint roughly the *same* patch of wall from every view, not just
    # nearby ones. Mean pairwise IoU of the per-view hulls (in surface uv)
    # is a free-standing diagnostic (shapely, no extra model) kept in
    # evidence and used by the operating point below; a mean IoU near 0
    # with n_views>=2 is a decent tell for "two unrelated noise boxes
    # happened to land near each other", not one real damage patch.
    if n_views > 1:
        from shapely.geometry import MultiPoint
        per_view_polys = [MultiPoint(c.uv_pts).convex_hull.buffer(0.02) for c in cluster]
        ious = []
        for i in range(len(per_view_polys)):
            for j in range(i + 1, len(per_view_polys)):
                a, b = per_view_polys[i], per_view_polys[j]
                u = a.union(b).area
                if u > 1e-9:
                    ious.append(a.intersection(b).area / u)
        mean_iou = float(np.mean(ious)) if ious else 0.0
    else:
        mean_iou = 1.0   # single view: nothing to disagree with; not penalised here

    err = fs.errors
    range_m = float(np.mean([c.range_m for c in cluster]))
    depth_sigma = err.depth_sigma_abs + err.depth_sigma_rel * range_m
    boundary_sigma = max(0.015, 0.03 * min(width_val, height_val))   # mask-boundary uncertainty
    multiview_sigma = 0.0
    if n_views > 1:
        per_view_w = []
        for c in cluster:
            vx0, vx1 = c.uv_pts[:, 0].min(), c.uv_pts[:, 0].max()
            per_view_w.append(vx1 - vx0)
        multiview_sigma = float(np.std(per_view_w))

    width = Measurement(width_val, combine(depth_sigma, boundary_sigma, multiview_sigma,
                                           err.scale_sigma_rel * width_val), "m",
                        method="owlv2+classical-mask, surface-uv bbox",
                        budget={"depth": depth_sigma, "boundary": boundary_sigma,
                                "multiview": multiview_sigma, "scale": err.scale_sigma_rel * width_val})
    height = Measurement(height_val, combine(depth_sigma, boundary_sigma, multiview_sigma,
                                             err.scale_sigma_rel * height_val), "m",
                         method="owlv2+classical-mask, surface-uv bbox",
                         budget={"depth": depth_sigma, "boundary": boundary_sigma,
                                 "multiview": multiview_sigma, "scale": err.scale_sigma_rel * height_val})
    perim = 2 * (width_val + height_val)
    area_edge_sigma = perim * (depth_sigma + boundary_sigma) / math.sqrt(2.0)
    area = Measurement(area_val, combine(area_edge_sigma, 2 * area_val * err.scale_sigma_rel), "m2",
                       method="owlv2+classical-mask, surface-uv hull",
                       budget={"edges": area_edge_sigma, "scale": 2 * area_val * err.scale_sigma_rel})

    return {
        "cls": cluster[0].cls,
        "surface_idx": cluster[0].surface_idx,
        "confidence": confidence,
        "n_views": n_views,
        "frames": all_frames,
        "area": area,
        "width": width,
        "height": height,
        "centroid_uv": all_pts.mean(axis=0),
        "evidence": {
            "model": MODEL_ID,
            "detector_score_mean": det_score,
            "color_consistency": color_consistency,
            "color_spread_lab": color_spread,
            "valid_depth_frac_mean": float(np.mean([c.valid_frac for c in cluster])),
            "geom_residual_mean_m": float(np.mean([c.residual_mean for c in cluster])),
            "n_points": int(sum(c.n_pts for c in cluster)),
            "mask_contrast_mean": contrast_mean,
            "mean_pairwise_iou": mean_iou,
        },
    }


# --------------------------------------------------------------------------
# Operating point
# --------------------------------------------------------------------------
# Chosen from a measured precision/recall/FP-per-m2 grid sweep (810 points:
# score x n_views x colour-consistency x segmentation-contrast) over the 3
# clean captures (FP side, ~178 raw regions / 490 m2) and the enlarged
# 55-instance / 5-class / 6-wall / 3-capture synthetic staged-damage suite
# (recall side), both reproducible from bench/synthetic_damage.py
# (`sweep_operating_points`); the full table is in
# reports/damage/operating_point_sweep.json and the technical report's
# "damage operating point" section. Selection rule, applied in this order:
# (1) FP rate on the clean captures <= 0.01/m2 AND <= 2 per capture;
# (2) *then* maximise synthetic recall. A `min_pairwise_iou` (stricter
# multi-view geometric consistency, computed in evidence) was swept too and
# did not move the frontier at any FP-qualifying point, so it is not part
# of the gate (kept only as an evidence diagnostic -- requirement 4's
# "keep it only if it improves the curve").
#
# Measured result at this point: 2 false regions / 490 m2 (0.0041/m2) on
# the clean captures; on the synthetic suite, 5/55 instances matched
# (recall 0.091), 10 detections survived the filter (precision 0.5). This
# is low, and it is the honest number: at a false-positive budget tight
# enough to not put phantom repaint line items in every report, this
# detector mostly goes quiet rather than finding real damage. See the
# per-class breakdown in the report -- hole_or_impact (high-contrast,
# compact, geometrically crisp) is the only class with non-trivial recall
# (4/12, 33%) at this point; water_stain/mould/crack/peeling_paint are
# ~0 because 3-view confirmation at score>=0.34 is a high bar for a
# single staged patch when `max_frames=200` keyframes only sees that patch
# from 1-2 well-placed views much of the time.
DEFAULT_OPERATING_POINT = {
    "min_score": 0.34,
    "min_views": 3,
    "min_color_consistency": 0.0,
    "min_contrast": 0.0,
}

# peeling_paint is excluded from the default run: on real (undamaged)
# textured walls it was the single largest false-positive source (89/174
# false regions, 51%, in the clean-capture measurement) and no combination
# of the four thresholds above gets its false-positive rate near the
# target without also erasing essentially all of its true-positive
# recall (the class is intrinsically a texture/lighting-variation
# classifier, and real painted walls already have plenty of both). It is
# never queried by default (saves a detector prompt + its classical
# segmentation pass, which also helps runtime). Passing it explicitly via
# `enabled_classes` re-enables detection, but every peeling_paint region is
# tagged `low_confidence: true` and scope.py skips tagged regions when
# building line items, so it can never silently produce a phantom repaint
# quote even if re-enabled.
ENABLED_CLASSES_DEFAULT = ("water_stain", "mould", "crack", "hole_or_impact")
LOW_CONFIDENCE_CLASSES = frozenset({"peeling_paint"})


def _passes_operating_point(region: dict, op: dict) -> bool:
    ev = region["evidence"]
    return (ev["detector_score_mean"] >= op["min_score"]
            and region["n_views"] >= op["min_views"]
            and ev["color_consistency"] >= op["min_color_consistency"]
            and ev["mask_contrast_mean"] >= op["min_contrast"])


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def _detect_damage_raw(fs: FrameSet, plan: Plan, cache_dir: str = ".cache", device: str | None = None,
                       max_frames: int = 200, progress=None,
                       enabled_classes=DAMAGE_CLASSES) -> list[dict]:
    """Every fused cluster with full evidence, *no* operating-point filter
    and no low_confidence tagging -- only `enabled_classes` gates which
    detector prompts are even queried (so disabling a class also saves
    runtime, not just precision). This is what the PR / FP-per-m2 sweep in
    bench/synthetic_damage.py calls directly so thresholds can be swept in
    plain Python without re-running the model. `detect_damage` below is a
    thin operating-point filter on top of this.
    """
    device = _pick_device(device)
    cache_path = Path(cache_dir)
    surfaces = plan.surfaces()
    if not surfaces:
        return []
    bases = [_surface_basis(s) for s in surfaces]
    surf_kind = [s["kind"] for s in surfaces]
    enabled = tuple(sorted(enabled_classes))

    keyframes = select_keyframes(fs, max_frames=max_frames)
    all_cands: list[_Candidate] = []
    for n, fi in enumerate(keyframes):
        fr = fs.frames[fi]
        if fr.load_rgb is None:
            continue
        try:
            rgb = fr.load_rgb()
        except Exception:
            continue
        boxes = _detect_boxes_cached(rgb, cache_path, fs.source or "fs", fr.index, device, enabled)
        if boxes:
            all_cands.extend(_process_view(fr, fs, plan, surfaces, bases, surf_kind, boxes, device))
        if progress and n % 20 == 0:
            progress(f"damage: scanned {n}/{len(keyframes)} keyframes, "
                     f"{len(all_cands)} raw candidates")

    clusters = _cluster_candidates(all_cands)
    regions = []
    counters: dict[str, int] = {}
    for cl in clusters:
        fused = _fuse_cluster(cl, fs, plan.tier)
        if fused is None:
            continue
        surface = surfaces[fused["surface_idx"]]
        origin, u_hat, v_hat, _ = bases[fused["surface_idx"]]
        cu, cv = fused["centroid_uv"]
        centroid_plan = origin + cu * u_hat + cv * v_hat
        sid = surface["id"]
        counters[sid] = counters.get(sid, 0) + 1
        region_id = f"{sid}.D{counters[sid]}"
        all_uv = np.concatenate([c.uv_pts for c in cl], axis=0)
        poly = _hull_polygon(all_uv)
        polygon_uv = [[round(float(x), 4), round(float(y), 4)] for x, y in list(poly.exterior.coords)] \
            if poly.geom_type == "Polygon" else []
        regions.append({
            "id": region_id,
            "surface_id": sid,
            "room_id": surface["room_id"],
            "class": fused["cls"],
            "confidence": round(fused["confidence"], 3),
            "area": fused["area"].to_json(),
            "extent": {"width": fused["width"].to_json(), "height": fused["height"].to_json()},
            "centroid_plan": [round(float(v), 4) for v in centroid_plan],
            "polygon_surface": polygon_uv,
            "n_views": fused["n_views"],
            "frames": fused["frames"],
            "evidence": fused["evidence"],
            "low_confidence": False,
        })
    return regions


def detect_damage(fs: FrameSet, plan: Plan, cache_dir: str = ".cache", device: str | None = None,
                  max_frames: int = 200, progress=None, enabled_classes=ENABLED_CLASSES_DEFAULT,
                  operating_point: dict | None = None) -> list[dict]:
    """Default entry point: `enabled_classes` + `operating_point` default to
    the measured, conservative operating point above (FP <= 0.01/m2 on the
    clean captures). Pass `enabled_classes=damage.DAMAGE_CLASSES` to also
    run the disabled low-confidence classes (peeling_paint); they come back
    tagged `low_confidence: true` and `scope.build_scope` ignores them.
    """
    op = operating_point or DEFAULT_OPERATING_POINT
    raw = _detect_damage_raw(fs, plan, cache_dir=cache_dir, device=device, max_frames=max_frames,
                             progress=progress, enabled_classes=enabled_classes)
    out = []
    for r in raw:
        if not _passes_operating_point(r, op):
            continue
        r["low_confidence"] = r["class"] in LOW_CONFIDENCE_CLASSES
        out.append(r)
    return out
