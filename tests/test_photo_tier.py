"""Fast tests for the photo tier.

Model-dependent paths (actual monocular depth inference, full
load_photo_property on real images) are skipped unless the weights are
already present in the local HF cache -- these tests must stay fast and
must not require network access to run in CI. The actual novel math here
(similarity-transform RANSAC, pose+scale chaining over a photo pose graph,
EXIF-based intrinsics) is tested directly, independent of model
availability, mirroring tests/test_video_tier.py's split.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scan2plan.tiers import photo


def _hf_weights_cached(substr: str) -> bool:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = home / "hub"
    if not hub.exists():
        return False
    return any(substr in p.name.lower() for p in hub.iterdir())


requires_depth_weights = pytest.mark.skipif(
    not _hf_weights_cached("depth-anything"), reason="Depth-Anything weights not cached locally")

BENCH_CAPTURE = Path(__file__).resolve().parents[1] / "data" / "photo_tier" / "c00a170fe1"


# ---------------------------------------------------------------------------
# Pure math: similarity-transform RANSAC
# ---------------------------------------------------------------------------

def test_umeyama_recovers_known_similarity():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(30, 3))
    R_true, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R_true) < 0:
        R_true[:, 0] *= -1
    s_true, t_true = 1.7, np.array([0.3, -0.2, 1.1])
    Q = s_true * (P @ R_true.T) + t_true

    R, s, t = photo._umeyama(P, Q)
    assert np.isclose(s, s_true, atol=1e-6)
    np.testing.assert_allclose(R, R_true, atol=1e-6)
    np.testing.assert_allclose(t, t_true, atol=1e-6)


def test_umeyama_ransac_rejects_outliers():
    rng = np.random.default_rng(1)
    P = rng.normal(size=(60, 3))
    R_true, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R_true) < 0:
        R_true[:, 0] *= -1
    s_true, t_true = 0.8, np.array([-0.5, 1.0, 0.2])
    Q = s_true * (P @ R_true.T) + t_true + rng.normal(scale=0.005, size=(60, 3))
    Q[:15] += rng.normal(scale=3.0, size=(15, 3))   # 25% gross outliers

    fit = photo._umeyama_ransac(P, Q, thresh=0.05, iters=1000, min_inliers=10, seed=0)
    assert fit is not None
    assert fit["inliers"] >= 40
    assert np.isclose(fit["s"], s_true, atol=0.05)
    assert fit["rmse"] < 0.05


def test_umeyama_ransac_returns_none_on_pure_noise():
    rng = np.random.default_rng(2)
    P = rng.normal(size=(20, 3))
    Q = rng.normal(size=(20, 3))   # unrelated point sets
    fit = photo._umeyama_ransac(P, Q, thresh=0.02, iters=500, min_inliers=12, seed=0)
    assert fit is None


# ---------------------------------------------------------------------------
# Pure math: pose+scale chaining over a photo pose graph
# ---------------------------------------------------------------------------

def _random_T(rng):
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = rng.normal(size=3)
    return T


def test_chain_poses_mixed_direction_edges_recover_correct_rotations_and_relative_scale():
    """4 cameras, each with its own true (unknown-in-practice) absolute
    monocular scale; edges fit purely from RAW (unscaled) point
    correspondences, exactly as `_register_pair` would produce. Mixing
    which endpoint is the fit's "target" (i) vs "source" (j) exercises both
    the forward and reverse branch of `_apply_edge`."""
    rng = np.random.default_rng(3)
    T_true = [_random_T(rng) for _ in range(4)]
    scale_true = [1.0, 1.3, 0.8, 2.1]

    def edge_between(i, j):
        T_ij = np.linalg.inv(T_true[i]) @ T_true[j]
        return {"R": T_ij[:3, :3], "s": scale_true[j] / scale_true[i],
               "t": T_ij[:3, 3] / scale_true[i], "i": i, "j": j, "inliers": 100}

    edges = [edge_between(0, 1), edge_between(2, 1), edge_between(2, 3)]
    root = 2
    pose_map, scale_map = photo._chain_poses([0, 1, 2, 3], edges, root)

    for k in range(4):
        expected_rot = (np.linalg.inv(T_true[root]) @ T_true[k])[:3, :3]
        np.testing.assert_allclose(pose_map[k][:3, :3], expected_rot, atol=1e-8)
        assert np.isclose(scale_map[k], scale_true[k] / scale_true[root], atol=1e-8)
    assert scale_map[root] == 1.0


def test_max_spanning_forest_prefers_higher_weight_edges_and_splits_components():
    edges = [
        {"i": 0, "j": 1, "inliers": 5},
        {"i": 1, "j": 2, "inliers": 50},
        {"i": 0, "j": 2, "inliers": 3},   # would create a cycle, lower weight: dropped
        {"i": 3, "j": 4, "inliers": 10},
    ]
    tree, comps = photo._max_spanning_forest(5, edges)
    assert len(tree) == 3   # one dropped (cycle), 5 nodes -> 2 components -> 5-2=3 tree edges
    comp_sets = sorted(sorted(c) for c in comps.values())
    assert comp_sets == [[0, 1, 2], [3, 4]]
    # the cycle-forming low-weight edge (0,2,inliers=3) must not survive
    assert not any(e["i"] == 0 and e["j"] == 2 for e in tree)


# ---------------------------------------------------------------------------
# EXIF / intrinsics
# ---------------------------------------------------------------------------

def _make_jpeg(tmp_path, name, size=(640, 480), focal35=None, orientation=None) -> Path:
    img = Image.new("RGB", size, color=(120, 140, 160))
    exif = Image.Exif()
    if orientation is not None:
        exif[274] = orientation
    if focal35 is not None:
        exif[41989] = int(focal35)
    path = tmp_path / name
    img.save(path, format="JPEG", exif=exif)
    return path


def test_photo_intrinsics_reads_exif_focal_35mm(tmp_path):
    path = _make_jpeg(tmp_path, "a.jpg", size=(1920, 1440), focal35=30, orientation=1)
    rgb, K, src = photo._photo_intrinsics(path)
    assert rgb.shape[:2] == (1440, 1920)
    assert src == "exif_focal_length_35mm"
    expected_focal_px = 30 / 36.0 * 1920
    assert np.isclose(K[0, 0], expected_focal_px, rtol=0.01)
    assert np.isclose(K[0, 2], 1920 / 2.0)
    assert np.isclose(K[1, 2], 1440 / 2.0)


def test_photo_intrinsics_falls_back_to_fov_heuristic_without_exif(tmp_path):
    path = _make_jpeg(tmp_path, "b.jpg", size=(800, 600))
    rgb, K, src = photo._photo_intrinsics(path)
    assert src == "fov_heuristic_70deg"
    assert K[0, 0] > 0
    assert rgb.shape[:2] == (600, 800)


def test_list_images_filters_by_extension(tmp_path):
    for name in ["a.jpg", "b.JPEG", "c.png", "d.heic", "e.txt", "f.gif"]:
        (tmp_path / name).write_bytes(b"\x00")
    found = {p.name for p in photo._list_images(tmp_path)}
    assert found == {"a.jpg", "b.JPEG", "c.png", "d.heic"}


# ---------------------------------------------------------------------------
# load_photo_property error handling (no model required)
# ---------------------------------------------------------------------------

def test_load_photo_property_raises_on_no_room_folders(tmp_path):
    with pytest.raises(ValueError, match="no room sub-folders"):
        photo.load_photo_property(tmp_path)


def test_load_photo_property_raises_on_empty_room_folder(tmp_path):
    (tmp_path / "kitchen").mkdir()
    with pytest.raises(ValueError, match="no readable photos"):
        photo.load_photo_property(tmp_path)


# ---------------------------------------------------------------------------
# End-to-end: only runs if weights are cached AND the bench photo set exists
# (generated by `python -m bench.make_photo_tier`).
# ---------------------------------------------------------------------------

@requires_depth_weights
def test_load_photo_property_end_to_end_smoke(tmp_path):
    if not BENCH_CAPTURE.exists():
        pytest.skip("bench photo-tier data not present (run bench/make_photo_tier.py first)")

    prop = photo.load_photo_property(BENCH_CAPTURE, cache_dir=tmp_path)
    assert set(prop.rooms) == {p.name for p in BENCH_CAPTURE.iterdir() if p.is_dir()}
    for name, fs in prop.rooms.items():
        assert fs.tier == "photo"
        assert len(fs.frames) >= 1
        for f in fs.frames:
            assert f.group == name
            d, v = f.load_depth()
            assert d.shape == v.shape
            assert f.T_wc.shape == (4, 4)
            d2, _ = f.load_depth()
            np.testing.assert_array_equal(d, d2)   # deterministic across calls
    assert isinstance(prop.warnings, list)
    assert "rooms" in prop.meta


@requires_depth_weights
def test_load_photo_property_shared_doorway_photo_yields_link(tmp_path):
    if not BENCH_CAPTURE.exists():
        pytest.skip("bench photo-tier data not present (run bench/make_photo_tier.py first)")

    prop = photo.load_photo_property(BENCH_CAPTURE, cache_dir=tmp_path)
    assert len(prop.rooms) >= 2
    assert len(prop.links) >= 1
    evidences = {l["evidence"] for l in prop.links}
    assert evidences <= {"shared_photo", "feature_match"}
    for link in prop.links:
        assert link["T_a_from_b"].shape == (4, 4)
        assert len(link["rooms"]) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
