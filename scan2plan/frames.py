"""Tier-agnostic posed RGB-D frames.

Every input tier (LiDAR, video, photos) is reduced to a FrameSet: a list of
frames with metric depth, a camera-to-world pose and intrinsics, in a world
frame whose +Y axis points up (against gravity). The layout backend only ever
sees FrameSets, which is what makes the output contract identical across tiers;
the tiers differ only in where depth/poses come from and in their error model.

Camera convention: OpenCV (x right, y down, z forward).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class TierErrorModel:
    """1-sigma error terms that feed every Measurement (see measure.py)."""
    name: str
    depth_sigma_abs: float      # per-point depth noise floor [m]
    depth_sigma_rel: float      # per-point depth noise proportional to range
    plane_bias: float           # systematic per-plane offset error [m]
    scale_sigma_rel: float      # global metric scale uncertainty (relative)
    drift_per_m: float          # residual pose drift after correction, per metre walked
    coverage_penalty: float     # extra sigma for walls only partially observed [m]


LIDAR_ERRORS = TierErrorModel("lidar", 0.008, 0.004, 0.004, 0.002, 0.0004, 0.02)
VIDEO_ERRORS = TierErrorModel("video", 0.03, 0.02, 0.01, 0.012, 0.002, 0.05)
PHOTO_ERRORS = TierErrorModel("photo", 0.05, 0.03, 0.02, 0.03, 0.0, 0.10)


@dataclass
class Frame:
    index: int
    timestamp: float
    T_wc: np.ndarray                       # 4x4 camera-to-world
    K_depth: np.ndarray                    # 3x3 intrinsics at depth resolution
    load_depth: Callable[[], tuple[np.ndarray, np.ndarray]]   # -> (depth[m] HxW float32, valid HxW bool)
    load_rgb: Callable[[], np.ndarray] | None = None          # -> HxWx3 uint8 RGB
    K_rgb: np.ndarray | None = None
    group: str = ""                        # photo tier: room folder name

    @property
    def position(self) -> np.ndarray:
        return self.T_wc[:3, 3]


@dataclass
class FrameSet:
    tier: str
    frames: list[Frame]
    errors: TierErrorModel
    source: str = ""
    meta: dict = field(default_factory=dict)

    def positions(self) -> np.ndarray:
        return np.array([f.position for f in self.frames])


def backproject(depth: np.ndarray, valid: np.ndarray, K: np.ndarray, stride: int = 1):
    """Depth image -> camera-frame points and normals (both Nx3) for valid pixels.

    Normals come from central differences on the depth grid; pixels on depth
    discontinuities get no normal and are dropped.
    """
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth.astype(np.float32)
    P = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=-1)

    du = np.zeros_like(P)
    dv = np.zeros_like(P)
    du[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dv[1:-1, :] = P[2:, :] - P[:-2, :]
    n = np.cross(dv, du)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(nn, 1e-9)

    ok = valid.copy()
    ok[:, [0, -1]] = False
    ok[[0, -1], :] = False
    # reject normals across depth edges: neighbour spacing must be small vs range
    step = np.maximum(np.linalg.norm(du, axis=-1), np.linalg.norm(dv, axis=-1))
    ok &= step < 0.06 * np.maximum(z, 0.3)
    # validity of neighbours used in the differences
    vn = np.zeros_like(valid)
    vn[1:-1, 1:-1] = valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1]
    ok &= vn
    if stride > 1:
        sub = np.zeros_like(ok)
        sub[::stride, ::stride] = True
        ok &= sub

    pts = P[ok]
    nrm = n[ok]
    # orient normals towards the camera
    flip = np.einsum("ij,ij->i", nrm, pts) > 0
    nrm[flip] *= -1
    return pts, nrm, ok


def to_world(T_wc: np.ndarray, pts: np.ndarray, nrm: np.ndarray | None = None):
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    pw = pts @ R.T + t
    if nrm is None:
        return pw
    return pw, nrm @ R.T
