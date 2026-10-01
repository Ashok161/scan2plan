"""Keyframe selection and point-cloud fusion."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import FrameSet, backproject, to_world


@dataclass
class Cloud:
    points: np.ndarray      # Nx3 world
    normals: np.ndarray     # Nx3 world, unit
    frame: np.ndarray       # N   keyframe ordinal (index into keyframes)
    cam_pos: np.ndarray     # Kx3 keyframe camera centres
    keyframes: list[int]    # indices into FrameSet.frames
    ranges: np.ndarray      # N   range from camera [m] (drives per-point noise)

    def subset(self, mask: np.ndarray) -> "Cloud":
        return Cloud(self.points[mask], self.normals[mask], self.frame[mask],
                     self.cam_pos, self.keyframes, self.ranges[mask])

    def apply_frame_transforms(self, T: np.ndarray) -> "Cloud":
        """Left-multiply each keyframe's points by its own 4x4 world correction T[k]."""
        R = T[self.frame, :3, :3]
        t = T[self.frame, :3, 3]
        p = np.einsum("nij,nj->ni", R, self.points) + t
        n = np.einsum("nij,nj->ni", R, self.normals)
        cp = np.einsum("kij,kj->ki", T[:, :3, :3], self.cam_pos) + T[:, :3, 3]
        return Cloud(p, n, self.frame, cp, self.keyframes, self.ranges)


def select_keyframes(fs: FrameSet, min_trans: float = 0.05, min_rot_deg: float = 5.0,
                     max_frames: int = 1500) -> list[int]:
    keep = [0]
    last = fs.frames[0].T_wc
    cos_thr = np.cos(np.deg2rad(min_rot_deg))
    for i, f in enumerate(fs.frames[1:], start=1):
        T = f.T_wc
        dt = np.linalg.norm(T[:3, 3] - last[:3, 3])
        # angle between optical axes
        c = float(np.dot(T[:3, 2], last[:3, 2]))
        if dt > min_trans or c < cos_thr:
            keep.append(i)
            last = T
    if len(keep) > max_frames:
        sel = np.linspace(0, len(keep) - 1, max_frames).astype(int)
        keep = [keep[i] for i in sel]
    return keep


def fuse(fs: FrameSet, keyframes: list[int] | None = None, pixel_stride: int = 2,
         progress=None) -> Cloud:
    if keyframes is None:
        keyframes = select_keyframes(fs)
    P, N, F, Rg = [], [], [], []
    cam = np.zeros((len(keyframes), 3))
    for k, fi in enumerate(keyframes):
        fr = fs.frames[fi]
        depth, valid = fr.load_depth()
        pts, nrm, _ = backproject(depth, valid, fr.K_depth, stride=pixel_stride)
        if len(pts) == 0:
            cam[k] = fr.position
            continue
        pw, nw = to_world(fr.T_wc, pts, nrm)
        P.append(pw.astype(np.float32))
        N.append(nw.astype(np.float32))
        F.append(np.full(len(pw), k, dtype=np.int32))
        Rg.append(np.linalg.norm(pts, axis=1).astype(np.float32))
        cam[k] = fr.position
        if progress and k % 200 == 0:
            progress(f"fused {k}/{len(keyframes)} keyframes")
    if not P:
        raise RuntimeError("no valid depth in capture")
    return Cloud(np.concatenate(P), np.concatenate(N), np.concatenate(F), cam, keyframes,
                 np.concatenate(Rg))


def voxel_downsample(points: np.ndarray, voxel: float, *extra: np.ndarray):
    """Average points (and any per-point arrays) within voxels. Returns (pts, extras..., counts)."""
    key = np.floor(points / voxel).astype(np.int64)
    key -= key.min(axis=0)
    dims = key.max(axis=0) + 1
    lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    uniq, inv, counts = np.unique(lin, return_inverse=True, return_counts=True)
    out = []
    for arr in (points,) + extra:
        a2 = arr.reshape(len(arr), -1).astype(np.float64)
        acc = np.stack([np.bincount(inv, weights=a2[:, j], minlength=len(uniq))
                        for j in range(a2.shape[1])], axis=1)
        acc /= counts[:, None]
        out.append(acc.reshape((len(uniq),) + arr.shape[1:]).astype(np.float32))
    return (*out, counts)
