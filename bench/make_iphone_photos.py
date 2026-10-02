"""Build a synthetic "real iPhone stills" photo-tier test set from the existing
bench/make_photo_tier.py output, since we have no iPhone to shoot real stills.

Real iPhone photos differ from `data/photo_tier/<id>/` (plain upright JPEGs,
1440x1920, no EXIF) in several ways this script reproduces:
  * resolution: 4032x3024 (main wide camera) or 5712x4284 (48MP mode) -- this
    script targets 4032x3024 (raw, sensor-native landscape) / 3024x4032
    (upright display size), the common case.
  * EXIF Orientation: a portrait-held photo is very often stored as a raw
    LANDSCAPE buffer plus Orientation=6 ("rotate 90 CW to display"), not as
    already-upright pixels. `PIL.ImageOps.exif_transpose` undoes this; the
    inverse used to construct the raw buffer here is `.transpose(Image.
    ROTATE_90)` (verified empirically: round-tripping raw.transpose(ROTATE_90)
    through exif_transpose with Orientation=6 reproduces the original upright
    image exactly).
  * EXIF FocalLengthIn35mmFilm (tag 41989), which scan2plan/tiers/photo.py
    reads from the Exif sub-IFD (0x8769) to compute real intrinsics; kept at a
    constant value (26mm, the iPhone main-camera 35mm-equivalent) across the
    whole set, as a real capture would.
  * container: HEIC (iOS default, via pillow-heif) for most files, plain JPEG
    for the rest, mirroring a mixed real hand-off (AirDrop/Files can yield
    either depending on the phone's camera format setting).

Measured caveat (see bench run notes / final report): pillow-heif's HEIC
writer bakes the EXIF-rotated orientation into the stored pixels on save and
resets Orientation to 1 -- i.e. the Orientation=6 "sideways raw buffer" case
is only exercised through the JPEG files this script also writes, not the
HEIC ones. This does not affect correctness of the focal-length math in
scan2plan/tiers/photo.py (it uses max(raw_w, raw_h), which is identical either
way), but it means HEIC specifically only tests decode-and-FocalLength, not
the orientation-transpose path.

Usage:
    .venv/bin/python -m bench.make_iphone_photos data/photo_tier/c00a170fe1 data/photo_tier_iphone/c00a170fe1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HAVE_HEIF = True
except Exception:
    _HAVE_HEIF = False

TARGET_UPRIGHT_SIZE = (3024, 4032)   # (w, h), displayed/upright
FOCAL_LENGTH_35MM = 26               # iPhone main-camera 35mm-equivalent, kept constant
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _exif_with_orientation6(focal_35mm: int = FOCAL_LENGTH_35MM) -> bytes:
    exif = Image.Exif()
    exif[274] = 6                              # Orientation: rotate 90 CW to display
    ifd = exif.get_ifd(0x8769)                 # Exif sub-IFD
    ifd[41989] = focal_35mm                    # FocalLengthIn35mmFilm
    ifd[37386] = (6, 1)                         # FocalLength (mm), rational (num, den)
    exif[0x8769] = ifd
    return exif.tobytes()


def _convert_one(src: Path, dst: Path, as_heic: bool) -> Path:
    upright = Image.open(src).convert("RGB").resize(TARGET_UPRIGHT_SIZE, Image.LANCZOS)
    raw = upright.transpose(Image.ROTATE_90)   # sensor-native landscape raw buffer
    exif_bytes = _exif_with_orientation6()
    ext = ".heic" if (as_heic and _HAVE_HEIF) else ".jpg"
    out = dst.with_suffix(ext)
    if ext == ".heic":
        raw.save(out, format="HEIF", exif=exif_bytes, quality=90)
    else:
        raw.save(out, format="JPEG", exif=exif_bytes, quality=92)
    return out


def convert(src: Path, dst: Path) -> list[Path]:
    written = []
    room_dirs = sorted(d for d in src.iterdir() if d.is_dir())
    n = 0
    for room in room_dirs:
        out_room = dst / room.name
        out_room.mkdir(parents=True, exist_ok=True)
        for f in sorted(p for p in room.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS):
            as_heic = (n % 2 == 0)   # alternate HEIC / JPEG, like a mixed real hand-off
            out = _convert_one(f, out_room / f.stem, as_heic)
            written.append(out)
            n += 1
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="existing photo_tier room-folder parent, e.g. data/photo_tier/c00a170fe1")
    ap.add_argument("dst", help="output parent folder, e.g. data/photo_tier_iphone/c00a170fe1")
    a = ap.parse_args(argv)
    written = convert(Path(a.src), Path(a.dst))
    n_heic = sum(1 for p in written if p.suffix == ".heic")
    print(f"wrote {len(written)} photos ({n_heic} heic, {len(written) - n_heic} jpg) to {a.dst}")


if __name__ == "__main__":
    main()
