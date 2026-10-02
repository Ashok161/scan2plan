"""Build a synthetic NeRFCapture-format export from a StrayScanner capture.

We have no NeRFCapture-capable iPhone to produce a real export, so this script
converts the StrayScanner sample captures (`data/<id>/`: rgb.mp4, odometry.csv,
camera_matrix.csv) into the exact folder/JSON layout the real NeRFCapture app
(github.com/jc211/NeRFCapture) writes for its "Offline" mode, so
scan2plan.tiers.posed_video can be exercised against the real export format
instead of only StrayScanner's.

Verified NeRFCapture export facts (DatasetWriter.swift / Manifest.swift /
Utils.swift), reproduced here:
  * folder = transforms.json + images/0.png, images/1.png, ... (no depth for
    non-LiDAR phones -- this script always omits depth, see below).
  * transforms.json keys are snake_case: top level w, h, fl_x, fl_y, cx, cy,
    depth_integer_scale, depth_source, frames. Each frame has file_path
    ("images/0", NO extension), depth_path (null here), transform_matrix
    (4x4 row-major camera-to-world, OpenGL/ARKit convention: x right, y up,
    z backward, world +Y up / gravity-aligned), timestamp, and its own
    fl_x/fl_y/cx/cy/w/h.
  * Images are frame.capturedImage: sensor-native landscape. A portrait hold
    (what the StrayScanner sample captures used) gives sideways images, and
    the intrinsics refer to that sideways image -- exactly like StrayScanner's
    own rgb.mp4, which lets us reuse the StrayScanner frame directly for the
    "portrait hold" variant.

StrayScanner's odometry.csv already stores camera-to-world poses in OpenCV
convention (x right, y down, z forward; see scan2plan/io/stray.py docstring).
To write them back out as NeRFCapture would, we apply the OpenCV->OpenGL
camera-basis flip, i.e. the exact inverse of the flip
scan2plan.tiers.posed_video._nerfcapture_frames applies on load (the flip
matrix diag(1,-1,-1,1) is its own inverse).

Usage:
    .venv/bin/python -m bench.make_nerfcapture data/<id> data/nerfcapture/<id>
    .venv/bin/python -m bench.make_nerfcapture data/<id> data/nerfcapture/<id>_landscape --landscape
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from scan2plan.io.stray import read_odometry
from scan2plan.video_io import VideoFrames

STRIDE = 10
MAX_FRAMES = 200

# OpenCV camera (x right, y down, z forward) <-> OpenGL/ARKit camera (x right,
# y up, z backward): flipping the y and z camera axes is its own inverse.
_CV_GL_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])

# A 90 deg physical roll of the phone about the lens axis, expressed in OpenCV
# camera-local axes: (x, y, z) -> (-y, x, z). Applied as R_new = R_old @ ROLL
# (post-multiply: redefines the camera's own local x/y axes), together with
# rotating the raw image +90 deg CW (np.rot90(img, -1)) and swapping w/h and
# fx/fy, cx/cy -- this is the "hold the phone in landscape instead of
# portrait" transform, derived to match the gravity-projection rule
# scan2plan.tiers.posed_video._rot_k_from_poses uses to recover orientation.
_ROLL_90 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def convert(src: Path, dst: Path, stride: int = STRIDE, max_frames: int = MAX_FRAMES,
            landscape: bool = False) -> Path:
    od = read_odometry(src)
    vf = VideoFrames(src / "rgb.mp4")
    K_file = np.loadtxt(src / "camera_matrix.csv", delimiter=",")
    images_dir = dst / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = list(od)[::stride]
    if len(rows) > max_frames:
        sel = np.linspace(0, len(rows) - 1, max_frames).astype(int)
        rows = [rows[i] for i in sel]

    frames_meta = []
    w0 = h0 = fx0 = fy0 = cx0 = cy0 = None
    n = 0
    for row in rows:
        idx = int(row[1])
        try:
            rgb = vf.get(idx)   # sensor-landscape RGB, exactly as ARKit/StrayScanner capture it
        except IndexError:
            # cv2's reported frame count can overshoot the actually decodable
            # stream on iPhone HEVC files (same quirk tiers/video.py works
            # around); just stop rather than crash.
            break
        fx, fy, cx, cy = float(row[9]), float(row[10]), float(row[11]), float(row[12])
        if not np.all(np.isfinite([fx, fy, cx, cy])):
            fx, fy, cx, cy = float(K_file[0, 0]), float(K_file[1, 1]), float(K_file[0, 2]), float(K_file[1, 2])

        T_wc_cv = np.eye(4)
        T_wc_cv[:3, :3] = Rotation.from_quat(row[5:9]).as_matrix()
        T_wc_cv[:3, 3] = row[2:5]

        h, w = rgb.shape[:2]
        if landscape:
            rgb = np.ascontiguousarray(np.rot90(rgb, -1))          # phone physically rolled 90 deg
            T_wc_cv[:3, :3] = T_wc_cv[:3, :3] @ _ROLL_90             # pose rolled the same way
            w, h = h, w
            fx, fy, cx, cy = fy, fx, cy, cx

        T_wc_gl = T_wc_cv @ _CV_GL_FLIP   # -> OpenGL/ARKit convention, as NeRFCapture writes it

        fname = f"images/{n}"
        Image.fromarray(rgb).save(dst / f"{fname}.png")
        frames_meta.append({
            "file_path": fname,
            "depth_path": None,
            "transform_matrix": T_wc_gl.tolist(),
            "timestamp": float(row[0]),
            "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
            "w": int(w), "h": int(h),
        })
        if w0 is None:
            w0, h0, fx0, fy0, cx0, cy0 = w, h, fx, fy, cx, cy
        n += 1

    manifest = {
        "w": w0, "h": h0, "fl_x": fx0, "fl_y": fy0, "cx": cx0, "cy": cy0,
        "depth_integer_scale": 1.0, "depth_source": None,
        "frames": frames_meta,
    }
    (dst / "transforms.json").write_text(json.dumps(manifest, indent=2))
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="StrayScanner capture directory, e.g. data/c00a170fe1")
    ap.add_argument("dst", help="output NeRFCapture-format directory, e.g. data/nerfcapture/c00a170fe1")
    ap.add_argument("--stride", type=int, default=STRIDE, help="take every Nth odometry row")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--landscape", action="store_true",
                    help="simulate a landscape phone hold instead of portrait (image + pose both rolled 90 deg)")
    a = ap.parse_args(argv)
    dst = Path(a.dst)
    convert(Path(a.src), dst, stride=a.stride, max_frames=a.max_frames, landscape=a.landscape)
    n = len(json.loads((dst / "transforms.json").read_text())["frames"])
    print(f"wrote {n} frames to {dst}")


if __name__ == "__main__":
    main()
