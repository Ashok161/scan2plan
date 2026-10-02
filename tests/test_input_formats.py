"""Fast tests for real-iPhone input-format robustness (no model inference, no network):

  * NeRFCapture folder/zip handling and pose-derived upright-rotation detection
    (scan2plan/tiers/posed_video.py), including a tiny synthetic 10-frame export.
  * CLI tier detection for NeRFCapture folders/zips (scan2plan/cli.py).
  * HEIC + EXIF Orientation=6 decoding and FocalLengthIn35mmFilm intrinsics
    (scan2plan/tiers/photo.py).

These must stay fast and must not require model weights or network access.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from scan2plan.cli import detect_tier
from scan2plan.tiers.photo import _photo_intrinsics
from scan2plan.tiers.posed_video import (
    _nerfcapture_frames,
    _resolve_input,
    _rot_k_from_poses,
    is_nerfcapture,
)

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HAVE_HEIF = True
except Exception:
    HAVE_HEIF = False


# --------------------------------------------------------------- rot_k_from_poses

def _T(roll_deg: float = 0.0) -> np.ndarray:
    """Camera-to-world pose (OpenCV convention: x right, y down, z forward) for a
    phone held level and right-side-up (roll_deg=0, "portrait" by this test's
    convention) in a +Y-up world, then rolled roll_deg about its own optical axis.

    Base case: camera forward (local +z) looks along world -Z, camera "down"
    (local +y) points along world -Y (i.e. right-side up) -- that is a 180 deg
    rotation about the world/camera X axis, R_base = diag(1, -1, -1).
    """
    T = np.eye(4)
    base = Rotation.from_euler("x", 180, degrees=True)
    roll = Rotation.from_euler("z", roll_deg, degrees=True)
    T[:3, :3] = (base * roll).as_matrix()
    return T


def test_rot_k_portrait_is_identity_up():
    # camera roll=0 (as built by stray.py / NeRFCapture for a portrait hold): gravity
    # should already point image-down, i.e. rot_k == 0 for this synthetic family.
    Ts = [_T(roll_deg=0.0) for _ in range(5)]
    assert _rot_k_from_poses(Ts) == 0


def test_rot_k_landscape_is_different_from_portrait():
    k_portrait = _rot_k_from_poses([_T(roll_deg=0.0) for _ in range(5)])
    k_landscape = _rot_k_from_poses([_T(roll_deg=90.0) for _ in range(5)])
    assert k_landscape != k_portrait


def test_rot_k_mode_is_robust_to_a_few_noisy_frames():
    Ts = [_T(roll_deg=0.0)] * 9 + [_T(roll_deg=90.0)]   # one outlier frame
    assert _rot_k_from_poses(Ts) == 0


# --------------------------------------------------------------- NeRFCapture format

def _write_tiny_nerfcapture(dst: Path, n_frames: int = 10) -> Path:
    """A minimal but format-correct NeRFCapture offline export: transforms.json +
    images/0.png.. (no extension in file_path, per the real exporter)."""
    images = dst / "images"
    images.mkdir(parents=True, exist_ok=True)
    w, h = 64, 48
    frames = []
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    for i in range(n_frames):
        Image.fromarray((np.random.default_rng(i).integers(0, 255, (h, w, 3))).astype(np.uint8)).save(
            images / f"{i}.png")
        T_cv = np.eye(4)
        T_cv[:3, 3] = [0.1 * i, 0.0, 0.0]
        T_gl = T_cv @ flip
        frames.append({
            "file_path": f"images/{i}", "depth_path": None,
            "transform_matrix": T_gl.tolist(), "timestamp": float(i),
            "fl_x": 50.0, "fl_y": 50.0, "cx": w / 2, "cy": h / 2, "w": w, "h": h,
        })
    manifest = {"w": w, "h": h, "fl_x": 50.0, "fl_y": 50.0, "cx": w / 2, "cy": h / 2,
                "depth_integer_scale": 1.0, "depth_source": None, "frames": frames}
    (dst / "transforms.json").write_text(json.dumps(manifest))
    return dst


def test_is_nerfcapture_folder(tmp_path):
    folder = _write_tiny_nerfcapture(tmp_path / "export")
    assert is_nerfcapture(folder)


def test_nerfcapture_frames_round_trip(tmp_path):
    folder = _write_tiny_nerfcapture(tmp_path / "export", n_frames=10)
    out, size = _nerfcapture_frames(folder)
    assert len(out) == 10
    assert size == (64, 48)
    idx, ts, T, K, load_rgb = out[3]
    assert idx == 3
    assert T.shape == (4, 4)
    rgb = load_rgb()
    assert rgb.shape == (48, 64, 3)
    # OpenCV convention camera-to-world round trip of a pure-flip-no-rotation pose:
    # translation must survive the OpenGL<->OpenCV conversion unchanged.
    assert np.allclose(T[:3, 3], [0.3, 0.0, 0.0])


def test_nerfcapture_folder_detects_as_video_tier(tmp_path):
    folder = _write_tiny_nerfcapture(tmp_path / "export", n_frames=10)
    assert detect_tier(folder) == "video"


def test_nerfcapture_nested_folder_in_zip_is_resolved(tmp_path):
    """Real NeRFCapture AirDrop zips unzip into a dated subfolder, not the top level."""
    export = _write_tiny_nerfcapture(tmp_path / "staging" / "251002143000", n_frames=10)
    zpath = tmp_path / "capture.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for f in export.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(tmp_path / "staging"))
    assert detect_tier(zpath) == "video"
    resolved = _resolve_input(zpath)
    assert (resolved / "transforms.json").exists()
    out, size = _nerfcapture_frames(resolved)
    assert len(out) == 10


def test_nerfcapture_nested_folder_without_zip_is_resolved(tmp_path):
    outer = tmp_path / "export_parent"
    _write_tiny_nerfcapture(outer / "251002143000", n_frames=10)
    resolved = _resolve_input(outer)
    assert (resolved / "transforms.json").exists()
    assert detect_tier(outer) == "video"


# --------------------------------------------------------------- HEIC / EXIF orientation

def _make_oriented_image(path: Path, upright_size=(300, 400), f35mm=26, as_heic=False):
    """An upright `upright_size` (w, h) image written as a raw sensor-landscape
    buffer + EXIF Orientation=6, exactly as a portrait-held iPhone photo stores it."""
    w, h = upright_size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[: h // 10, :, 0] = 255   # red stripe along the top of the UPRIGHT image
    upright = Image.fromarray(arr)
    raw = upright.transpose(Image.ROTATE_90)   # sensor-native landscape raw buffer
    exif = Image.Exif()
    exif[274] = 6
    ifd = exif.get_ifd(0x8769)
    ifd[41989] = f35mm
    exif[0x8769] = ifd
    if as_heic:
        raw.save(path, format="HEIF", exif=exif.tobytes())
    else:
        raw.save(path, format="JPEG", exif=exif.tobytes())


def test_jpeg_exif_orientation6_decodes_upright():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "photo.jpg"
        _make_oriented_image(p, upright_size=(300, 400), as_heic=False)
        rgb, K, src = _photo_intrinsics(p)
        assert rgb.shape == (400, 300, 3)          # upright (h, w, 3)
        assert rgb[:40, :, 0].mean() > 200          # red stripe landed at the TOP, i.e. upright
        assert src == "exif_focal_length_35mm"
        # raw buffer is landscape (400x300 -> transposed to 300x400): max(raw dims) == 400,
        # same either way, so the 35mm-equivalent focal_px math is orientation-independent.
        assert np.isclose(K[0, 0], 26 / 36.0 * 400)


@pytest.mark.skipif(not HAVE_HEIF, reason="pillow-heif not installed")
def test_heic_decodes_and_keeps_exif_focal_length():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "photo.heic"
        _make_oriented_image(p, upright_size=(300, 400), as_heic=True)
        rgb, K, src = _photo_intrinsics(p)
        # pillow-heif's writer bakes EXIF-rotated pixels into the HEIC on save and
        # resets Orientation to 1 (measured; see bench/make_iphone_photos.py docstring),
        # so the file loads already upright either way -- what matters here is that it
        # decodes at all and the FocalLengthIn35mmFilm tag survives the HEIC round trip.
        assert rgb.ndim == 3 and rgb.shape[2] == 3
        assert rgb[:40, :, 0].mean() > 200
        assert src == "exif_focal_length_35mm"
