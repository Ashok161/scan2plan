"""Loader for StrayScanner exports (LiDAR tier).

Layout of one capture directory (as exported by the StrayScanner iOS app):
  rgb.mp4               1920x1440 colour video, one video frame per odometry row
  depth/NNNNNN.png      256x192 uint16 depth in millimetres (LiDAR)
  confidence/NNNNNN.png 256x192 uint8 ARKit confidence {0,1,2}
  odometry.csv          timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, ...
  camera_matrix.csv     3x3 RGB intrinsics
  imu.csv               accelerometer / gyro

Poses are ARKit camera-to-world in a gravity-aligned (+Y up) world. Verified
empirically: back-projecting with the OpenCV camera convention puts 5% of all
points in a single 1 cm floor bin; the OpenGL convention smears them over 4 m.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from ..frames import Frame, FrameSet, LIDAR_ERRORS

DEPTH_W, DEPTH_H = 256, 192


def is_stray_capture(path: Path) -> bool:
    return (path / "odometry.csv").exists() and (path / "depth").is_dir()


def read_odometry(path: Path) -> np.ndarray:
    return np.genfromtxt(path / "odometry.csv", delimiter=",", skip_header=1, usecols=range(13))


def load_stray(path: str | Path, min_confidence: int = 2) -> FrameSet:
    path = Path(path)
    od = read_odometry(path)
    K_rgb_file = np.loadtxt(path / "camera_matrix.csv", delimiter=",")
    frames = []
    for row in od:
        idx = int(row[1])
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(row[5:9]).as_matrix()
        T[:3, 3] = row[2:5]
        # per-frame intrinsics from odometry (ARKit refines focal length online)
        K_rgb = np.array([[row[9], 0, row[11]], [0, row[10], row[12]], [0, 0, 1.0]])
        if not np.all(np.isfinite(K_rgb)):
            K_rgb = K_rgb_file
        s = DEPTH_W / 1920.0
        K_d = K_rgb.copy()
        K_d[:2] *= s
        frames.append(Frame(
            index=idx,
            timestamp=float(row[0]),
            T_wc=T,
            K_depth=K_d,
            K_rgb=K_rgb,
            load_depth=_depth_loader(path, idx, min_confidence),
        ))
    fs = FrameSet("lidar", frames, LIDAR_ERRORS, source=str(path),
                  meta={"device_format": "StrayScanner", "n_frames": len(frames)})
    attach_video(fs, path / "rgb.mp4")
    return fs


def _depth_loader(path: Path, idx: int, min_conf: int):
    def load():
        d = np.asarray(Image.open(path / "depth" / f"{idx:06d}.png"), dtype=np.float32) / 1000.0
        c = np.asarray(Image.open(path / "confidence" / f"{idx:06d}.png"))
        valid = (c >= min_conf) & (d > 0.15) & (d < 5.0)
        return d, valid
    return load


def attach_video(fs: FrameSet, video: Path):
    """Give frames lazy RGB access through a shared sequential decoder cache."""
    if not video.exists():
        return
    from ..video_io import VideoFrames
    vf = VideoFrames(video)
    for f in fs.frames:
        f.load_rgb = (lambda i=f.index: vf.get(i))
    fs.meta["video"] = str(video)
