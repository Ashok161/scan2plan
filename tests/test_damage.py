"""Fast tests for scan2plan.damage.

Model-dependent paths (OWLv2 forward pass) are skipped unless the weights
are already present in the local HF cache -- these tests must stay fast and
must not require network access to run in CI. The clustering/fusion math
(the actual novel code: multi-view clustering, geometric fusion, uncertainty
propagation) is tested directly against hand-built `_Candidate` objects, so
it is covered regardless of model availability.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from scan2plan import damage as dmg


def _hf_weights_cached() -> bool:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = home / "hub"
    if not hub.exists():
        return False
    return any("owlv2" in p.name.lower() for p in hub.iterdir())


requires_weights = pytest.mark.skipif(not _hf_weights_cached(), reason="OWLv2 weights not cached locally")


def test_taxonomy_is_consistent():
    assert set(dmg.PROMPTS) == set(dmg.DAMAGE_CLASSES)
    assert set(dmg.CLASS_SURFACE_KINDS) == set(dmg.DAMAGE_CLASSES)
    for cls, prompts in dmg.PROMPTS.items():
        assert len(prompts) >= 1
        assert all(isinstance(p, str) and p for p in prompts)


def _wall_surface(normal=(1.0, 0.0, 0.0), offset=0.0, floor_y=0.0, top=2.4, length=3.0):
    s3 = np.array([0.0, floor_y, 0.0])
    e3 = np.array([0.0, floor_y, length])
    corners = np.array([s3, e3, e3 + [0, top - floor_y, 0], s3 + [0, top - floor_y, 0]])
    return {"id": "r.W0", "room_id": "r", "kind": "wall", "normal": np.array(normal),
           "offset": offset, "corners": corners}


def _floor_surface():
    corners = np.array([[0, 0, 0], [3, 0, 0], [3, 0, 3], [0, 0, 3]], dtype=float)
    return {"id": "r.floor", "room_id": "r", "kind": "floor", "normal": np.array([0.0, 1.0, 0.0]),
           "offset": 0.0, "corners": corners}


def test_surface_basis_wall_is_orthonormal_and_matches_corners():
    s = _wall_surface()
    origin, u_hat, v_hat, uv_corners = dmg._surface_basis(s)
    assert np.isclose(np.linalg.norm(u_hat), 1.0)
    assert np.isclose(np.linalg.norm(v_hat), 1.0)
    assert np.isclose(u_hat @ v_hat, 0.0, atol=1e-9)
    # corner 1 is "length" metres along u from corner 0 (sign depends on the
    # normal's orientation, cross-product convention; magnitude is what
    # downstream extent/area math actually uses)
    assert np.isclose(abs(uv_corners[1, 0]), 3.0, atol=1e-6)
    assert np.isclose(uv_corners[1, 1], 0.0, atol=1e-6)
    # corner 3 is "top-floor" metres along v from corner 0
    assert np.isclose(uv_corners[3, 1], 2.4, atol=1e-6)


def test_surface_basis_floor_uses_plan_xz():
    s = _floor_surface()
    origin, u_hat, v_hat, uv_corners = dmg._surface_basis(s)
    assert np.allclose(u_hat, [1, 0, 0])
    assert np.allclose(v_hat, [0, 0, 1])


def _candidate(surface_idx, cls, score, frame, uv_pts, mean_lab=(50.0, 0.0, 0.0), range_m=1.5,
              residual=0.01, valid_frac=0.9, contrast=15.0):
    return dmg._Candidate(surface_idx=surface_idx, cls=cls, score=score, frame=frame,
                          uv_pts=np.asarray(uv_pts, dtype=float), mean_lab=np.array(mean_lab),
                          n_pts=len(uv_pts), residual_mean=residual, valid_frac=valid_frac,
                          range_m=range_m, contrast=contrast)


def test_cluster_candidates_groups_by_surface_class_and_proximity():
    near_a = _candidate(0, "water_stain", 0.3, 1, [[0.0, 0.0], [0.05, 0.05]])
    near_b = _candidate(0, "water_stain", 0.3, 2, [[0.02, 0.03], [0.04, 0.01]])
    far = _candidate(0, "water_stain", 0.3, 3, [[5.0, 5.0], [5.1, 5.1]])
    other_class = _candidate(0, "crack", 0.3, 4, [[0.01, 0.01]])
    other_surface = _candidate(1, "water_stain", 0.3, 5, [[0.0, 0.0]])

    clusters = dmg._cluster_candidates([near_a, near_b, far, other_class, other_surface])
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 1, 1, 2]   # near_a+near_b merge; the other three stay singleton


def test_fuse_cluster_single_view_is_downweighted_vs_multi_view():
    from scan2plan.frames import LIDAR_ERRORS, FrameSet
    fs = FrameSet(tier="lidar", frames=[], errors=LIDAR_ERRORS, source="unit-test")

    square = [[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]]
    single = [_candidate(0, "water_stain", 0.4, 1, square)]
    multi = [_candidate(0, "water_stain", 0.4, 1, square),
            _candidate(0, "water_stain", 0.4, 2, square)]

    fused_single = dmg._fuse_cluster(single, fs, "lidar")
    fused_multi = dmg._fuse_cluster(multi, fs, "lidar")
    assert fused_single["n_views"] == 1
    assert fused_multi["n_views"] == 2
    assert fused_multi["confidence"] > fused_single["confidence"]
    # extent is in metres and roughly matches the 0.3 x 0.3 synthetic hull
    assert 0.2 < fused_single["width"].value < 0.4
    assert fused_single["area"].sigma > 0


def test_fuse_cluster_color_inconsistency_lowers_confidence():
    square = [[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]]
    from scan2plan.frames import LIDAR_ERRORS, FrameSet
    fs = FrameSet(tier="lidar", frames=[], errors=LIDAR_ERRORS, source="unit-test")

    consistent = [_candidate(0, "water_stain", 0.4, 1, square, mean_lab=(40, 10, 10)),
                 _candidate(0, "water_stain", 0.4, 2, square, mean_lab=(41, 11, 9))]
    inconsistent = [_candidate(0, "water_stain", 0.4, 1, square, mean_lab=(40, 10, 10)),
                   _candidate(0, "water_stain", 0.4, 2, square, mean_lab=(90, -40, 60))]
    fc = dmg._fuse_cluster(consistent, fs, "lidar")
    fi = dmg._fuse_cluster(inconsistent, fs, "lidar")
    assert fc["evidence"]["color_consistency"] > fi["evidence"]["color_consistency"]
    assert fc["confidence"] > fi["confidence"]


def test_otsu_mask_finds_a_contrasting_blob():
    score = np.zeros((40, 40), np.float32)
    score[10:25, 10:25] = 50.0
    mask = dmg._otsu_mask(score)
    assert mask[15:20, 15:20].all()
    assert not mask[0:5, 0:5].any()


def test_keep_elongated_filters_blobs_but_keeps_lines():
    mask = np.zeros((50, 50), bool)
    mask[20:22, 5:45] = True     # a long thin line: aspect ratio ~20
    mask[2:10, 2:10] = True      # a compact blob: aspect ratio ~1
    out = dmg._keep_elongated(mask)
    assert out[21, 20]
    assert not out[5, 5]


@requires_weights
def test_detect_boxes_smoke(tmp_path):
    """Only runs if OWLv2 weights are already cached; exercises the real forward pass."""
    rgb = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)
    boxes = dmg._detect_boxes(rgb, device="cpu")
    assert isinstance(boxes, list)
    for b in boxes:
        assert b["class"] in dmg.DAMAGE_CLASSES
        assert 0.0 <= b["score"] <= 1.0
        assert len(b["box"]) == 4


def _data_dir():
    return Path(__file__).resolve().parent.parent / "data"


requires_capture = pytest.mark.skipif(not (_data_dir() / "c00a170fe1").exists(),
                                      reason="data/c00a170fe1 capture not present")


@requires_weights
@requires_capture
def test_detect_damage_on_real_plan_smoke(tmp_path):
    """Integration smoke test against the coordinator's real layout.build_plan,
    not the dev fixture: a few keyframes only, just to confirm the surface
    projection / clustering / Measurement plumbing works end to end and the
    returned dicts have the right shape. Full-length runs (max_frames=200)
    are exercised by bench/synthetic_damage.py, not here, to keep this fast.
    """
    from scan2plan.io.stray import load_stray
    from scan2plan.layout import build_plan

    fs = load_stray(_data_dir() / "c00a170fe1")
    plan = build_plan(fs)
    regions = dmg.detect_damage(fs, plan, cache_dir=str(tmp_path / ".cache"), max_frames=10)
    assert isinstance(regions, list)
    surface_ids = {s["id"] for s in plan.surfaces()}
    for r in regions:
        assert r["surface_id"] in surface_ids
        assert r["class"] in dmg.DAMAGE_CLASSES
        assert 0.0 <= r["confidence"] <= 1.0
        assert r["area"]["value"] >= 0
        assert r["area"]["unit"] == "m2"
        assert r["extent"]["width"]["unit"] == "m"
        assert len(r["centroid_plan"]) == 3
        assert r["n_views"] == len(set(r["frames"]))


# --------------------------------------------------------------------------
# Operating point: FP <= 0.01/m2 on the clean captures, then max synthetic
# recall (see bench/synthetic_damage.py's sweep + damage.py's
# DEFAULT_OPERATING_POINT docstring for how these numbers were chosen).
# --------------------------------------------------------------------------
def _region(score, n_views, color_consistency, contrast, cls="water_stain"):
    return {"class": cls, "n_views": n_views,
           "evidence": {"detector_score_mean": score, "color_consistency": color_consistency,
                       "mask_contrast_mean": contrast}}


def test_passes_operating_point_requires_all_four_thresholds():
    op = dmg.DEFAULT_OPERATING_POINT
    good = _region(0.9, 10, 1.0, 50.0)
    assert dmg._passes_operating_point(good, op)
    too_weak_score = _region(op["min_score"] - 0.01, 10, 1.0, 50.0)
    assert not dmg._passes_operating_point(too_weak_score, op)
    too_few_views = _region(0.9, max(op["min_views"] - 1, 0), 1.0, 50.0)
    assert not dmg._passes_operating_point(too_few_views, op)


def test_peeling_paint_excluded_by_default_but_tagged_if_enabled():
    assert "peeling_paint" not in dmg.ENABLED_CLASSES_DEFAULT
    assert "peeling_paint" in dmg.LOW_CONFIDENCE_CLASSES
    assert "peeling_paint" in dmg.DAMAGE_CLASSES   # taxonomy still documents it


def test_scope_skips_low_confidence_regions(monkeypatch=None):
    from scan2plan.measure import Measurement
    from scan2plan.plan_types import Plan, Room, Wall
    from scan2plan.scope import generate_scope
    import numpy as np

    def m(v, s=0.01, u="m"):
        return Measurement(v, s, u)

    wall = Wall(id="r.W0", start=np.array([0.0, 0.0]), end=np.array([3.0, 0.0]),
               normal_in=np.array([0.0, 1.0]), length=m(3.0), height=m(2.4),
               offset_sigma=0.01, coverage=1.0)
    room = Room(id="r", name="r", kind="room", polygon=np.array([[0, 0], [3, 0], [3, 3], [0, 3]], dtype=float),
               floor_y=0.0, ceiling_y=2.4, area=m(9.0, 0.05, "m2"), perimeter=m(12.0),
               ceiling_height=m(2.4), walls=[wall])
    plan = Plan(tier="lidar", rooms=[room], adjacency=[], T_align=np.eye(4), floor_y=0.0)

    region = {"id": "d.1", "surface_id": "r.W0", "room_id": "r", "class": "peeling_paint",
             "confidence": 0.5, "low_confidence": True,
             "area": {"value": 0.1, "sigma": 0.01, "ci95": [0, 1], "unit": "m2"},
             "extent": {"width": {"value": 0.3, "sigma": 0.01, "ci95": [0, 1], "unit": "m"},
                       "height": {"value": 0.3, "sigma": 0.01, "ci95": [0, 1], "unit": "m"}}}
    items = generate_scope(plan, [region], [])
    assert items == []   # a low_confidence region must never produce a scope line item
