"""Fast tests for the video tier.

Model-dependent paths (actual monocular depth inference, full load_video) are
skipped unless the weights are already present in the local HF cache -- these
tests must stay fast and must not require network access to run in CI. The VO
math (PnP-from-depth, pose-graph optimisation) and caching logic are the
actual novel code here, so they're tested directly with synthetic data /
monkeypatched models, independent of model availability.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scan2plan.tiers import mono_depth
from scan2plan.tiers.video import _select_indices
from scan2plan.tiers.video_vo import (
    Edge,
    chain_sequential,
    detect_and_describe,
    match_descriptors,
    optimize_pose_graph,
    relative_pose_pnp,
)


def _hf_weights_cached(substr: str) -> bool:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = home / "hub"
    if not hub.exists():
        return False
    return any(substr in p.name.lower() for p in hub.iterdir())


requires_depth_weights = pytest.mark.skipif(
    not _hf_weights_cached("depth-anything"), reason="Depth-Anything weights not cached locally")
requires_depthpro_weights = pytest.mark.skipif(
    not _hf_weights_cached("depthpro"), reason="DepthPro weights not cached locally")


# ---------------------------------------------------------------------------
# Pure logic: keyframe selection
# ---------------------------------------------------------------------------

def test_select_indices_respects_cap():
    idx = _select_indices(n_frames=10000, fps=46.0, max_keyframes=400, target_fps=3.0)
    assert len(idx) <= 400
    assert idx[0] == 0
    assert all(a < b for a, b in zip(idx, idx[1:]))


def test_select_indices_short_video_no_cap_needed():
    idx = _select_indices(n_frames=100, fps=30.0, max_keyframes=400, target_fps=3.0)
    assert len(idx) <= 400
    assert idx[-1] < 100


# ---------------------------------------------------------------------------
# Synthetic VO: a textured plane viewed from two known camera poses should
# yield a metric relative pose close to ground truth, with no model needed.
# ---------------------------------------------------------------------------

def _synthetic_textured_depth(K, w=320, h=240, plane_z=2.0, seed=0):
    rng = np.random.default_rng(seed)
    img = (rng.random((h, w)) * 255).astype(np.uint8)
    img = np.stack([img] * 3, axis=-1)
    depth = np.full((h, w), plane_z, dtype=np.float32)
    return img, depth


def test_relative_pose_pnp_recovers_known_translation():
    K = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])
    img_a, depth_a = _synthetic_textured_depth(K)
    gray_a = img_a[:, :, 0]

    # Known relative motion: camera b is translated +0.2m in x relative to a.
    R_true = np.eye(3)
    t_true = np.array([0.2, 0.0, 0.0])

    kp_a, des_a = detect_and_describe(gray_a, n_features=500)
    assert len(kp_a) > 50
    pts_a = np.array([k.pt for k in kp_a])
    ui = np.clip(pts_a[:, 0].astype(int), 0, 319)
    vi = np.clip(pts_a[:, 1].astype(int), 0, 239)
    z = depth_a[vi, ui]
    X = (pts_a[:, 0] - K[0, 2]) / K[0, 0] * z
    Y = (pts_a[:, 1] - K[1, 2]) / K[1, 1] * z
    obj = np.stack([X, Y, z], axis=1)
    proj = (R_true @ obj.T + t_true[:, None])
    pts_b = np.stack([proj[0] / proj[2] * K[0, 0] + K[0, 2],
                       proj[1] / proj[2] * K[1, 1] + K[1, 2]], axis=1)

    r = relative_pose_pnp(pts_a, pts_b, depth_a, K, min_inliers=10)
    assert r is not None
    assert r["n_inliers"] >= 10
    np.testing.assert_allclose(r["t"], t_true, atol=0.01)
    np.testing.assert_allclose(r["R"], R_true, atol=0.01)


# ---------------------------------------------------------------------------
# Pose-graph optimisation on synthetic edges
# ---------------------------------------------------------------------------

def test_pose_graph_noop_without_loop_edges():
    poses = np.tile(np.eye(4), (5, 1, 1))
    for i in range(1, 5):
        poses[i, :3, 3] = [i * 0.5, 0, 0]
    edges = [Edge(i, i + 1, np.eye(3), np.array([-0.5, 0.0, 0.0]), 50, "sequential")
              for i in range(4)]
    out = optimize_pose_graph(poses, edges)
    np.testing.assert_array_equal(out, poses)


def test_pose_graph_corrects_drift_with_loop_closure():
    # A noisy chain that should have returned to the origin, plus a loop-closure
    # edge asserting frame 4 ~= frame 0: optimisation should pull it back.
    n = 5
    poses = np.tile(np.eye(4), (n, 1, 1))
    drift = np.array([0.1, 0.0, 0.0])
    for i in range(1, n):
        poses[i, :3, 3] = poses[i - 1, :3, 3] + drift
    edges = [Edge(i, i + 1, np.eye(3), -drift, 50, "sequential") for i in range(n - 1)]
    edges.append(Edge(0, n - 1, np.eye(3), np.zeros(3), 100, "loop"))
    out = optimize_pose_graph(poses, edges, max_nfev=200)
    err_before = np.linalg.norm(poses[n - 1, :3, 3] - poses[0, :3, 3])
    err_after = np.linalg.norm(out[n - 1, :3, 3] - out[0, :3, 3])
    assert err_after < err_before


# ---------------------------------------------------------------------------
# Caching: predict_depth should hit the on-disk cache on the second call and
# not re-invoke the model. Model loading itself is monkeypatched out so this
# needs no weights / network access.
# ---------------------------------------------------------------------------

def test_predict_depth_caches_to_disk(tmp_path, monkeypatch):
    calls = {"n": 0}

    class FakeOutputs:
        pass

    def fake_get_model(model_id, device):
        calls["n"] += 1

        class FakeInputs(dict):
            def to(self, device):
                return self

        class FakeProcessor:
            def __call__(self, images, return_tensors):
                return FakeInputs()

            def post_process_depth_estimation(self, outputs, target_sizes=None):
                h, w = (8, 8) if target_sizes is None else target_sizes[0]
                import torch
                return [{"predicted_depth": torch.ones(h, w) * 1.5, "focal_length": torch.tensor(100.0)}]

        class FakeModel:
            def __call__(self, **kw):
                return FakeOutputs()

        return FakeProcessor(), FakeModel()

    monkeypatch.setattr(mono_depth, "_get_model", fake_get_model)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    d1, f1 = mono_depth.predict_depth(rgb, "fake/model", "testkey_000000",
                                       cache_dir=tmp_path, resize_to_input=True)
    assert calls["n"] == 1
    assert np.allclose(d1, 1.5)
    assert f1 == 100.0

    d2, f2 = mono_depth.predict_depth(rgb, "fake/model", "testkey_000000",
                                       cache_dir=tmp_path, resize_to_input=True)
    assert calls["n"] == 1  # cache hit: model not called again
    np.testing.assert_allclose(d1, d2)
    assert f2 == 100.0

    cache_file = tmp_path / "mono_depth" / "fake__model" / "testkey_000000_full.npz"
    assert cache_file.exists()


def test_video_content_hash_stable(tmp_path):
    p = tmp_path / "v.bin"
    p.write_bytes(os.urandom(1000))
    h1 = mono_depth.video_content_hash(p)
    h2 = mono_depth.video_content_hash(p)
    assert h1 == h2
    assert len(h1) == 16


# ---------------------------------------------------------------------------
# End-to-end: only runs if both model weights are already cached locally.
# ---------------------------------------------------------------------------

@requires_depth_weights
@requires_depthpro_weights
def test_load_video_end_to_end_smoke(tmp_path):
    from scan2plan.tiers.video import load_video

    data_dir = Path(__file__).resolve().parents[1] / "data" / "c00a170fe1"
    video = data_dir / "rgb.mp4"
    if not video.exists():
        pytest.skip("dev data not present")

    fs = load_video(video, cache_dir=tmp_path, max_keyframes=8)
    assert fs.tier == "video"
    assert len(fs.frames) > 0
    for f in fs.frames:
        d, v = f.load_depth()
        assert d.shape == v.shape
        assert f.T_wc.shape == (4, 4)
        assert f.K_depth.shape == (3, 3)
    assert "depth_model" in fs.meta
    assert "focal_px_estimate" in fs.meta
