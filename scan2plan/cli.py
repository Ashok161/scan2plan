"""One command per capture:

    scan2plan run <capture> [--tier auto|lidar|video|photo] [--out DIR]

<capture> is
  * a StrayScanner export directory (LiDAR tier; `--tier video` uses only its rgb.mp4),
  * a video file (.mov / .mp4) for the video tier,
  * a directory of per-room photo folders for the photo tier.
Outputs in DIR: plan.json (schema/scan2plan_output.schema.json), plan.png, plan.svg, run.log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
VID_EXT = {".mov", ".mp4", ".m4v"}


def detect_tier(path: Path) -> str:
    from .io.stray import is_stray_capture
    if path.is_file() and path.suffix.lower() in VID_EXT:
        return "video"
    if path.is_file() and path.suffix.lower() == ".zip":
        import zipfile
        try:
            with zipfile.ZipFile(path) as zf:
                if any(n.endswith("transforms.json") for n in zf.namelist()):
                    return "video"      # NeRFCapture AirDrop zip hand-off (any iPhone)
        except zipfile.BadZipFile:
            pass
        raise SystemExit(f"cannot detect input tier for zip {path}: expected a NeRFCapture "
                         f"export (transforms.json + images/)")
    if path.is_dir():
        if is_stray_capture(path):
            return "lidar"
        # NeRFCapture export: RGB + ARKit poses (any iPhone). transforms.json may sit at the
        # top level, or one level down (a real export unzips into a dated subfolder).
        if (path / "transforms.json").exists() or any(path.rglob("transforms.json")):
            return "video"
        subs = [d for d in path.iterdir() if d.is_dir()]
        if subs and any(f.suffix.lower() in IMG_EXT for d in subs for f in d.iterdir()):
            return "photo"
        if any(f.suffix.lower() in IMG_EXT for f in path.iterdir()):
            return "photo"
    raise SystemExit(f"cannot detect input tier for {path}: expected a StrayScanner folder, a video file, "
                     f"a NeRFCapture folder/zip, or a folder of per-room photo folders")


def capture_id(path: Path) -> str:
    return path.stem if path.is_file() else path.name


def run(path: str | Path, tier: str = "auto", out: str | Path | None = None, drift: bool = True,
        damage: bool = True, cache_dir: str = ".cache", quiet: bool = False, plain_video: bool = False) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"input not found: {path}")
    if tier == "auto":
        tier = detect_tier(path)
    cid = capture_id(path)
    out = Path(out) if out else Path("out") / cid / tier
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "run.log", "w")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        log_f.write(line + "\n")
        log_f.flush()
        if not quiet:
            print(line, flush=True)

    t0 = time.time()
    timings = {}
    log(f"scan2plan {tier} tier on {path}")
    from .layout import build_plan
    fs = None
    if tier == "lidar":
        from .io.stray import load_stray
        fs = load_stray(path)
        timings["load"] = round(time.time() - t0, 2)
        plan = build_plan(fs, drift_correction=drift, progress=log)
    elif tier == "video":
        is_posed_input = path.is_dir() or path.suffix.lower() == ".zip"
        if is_posed_input and not plain_video:
            # video + phone motion: RGB + ARKit metric poses (StrayScanner / NeRFCapture
            # folder or AirDrop zip), no depth sensor
            from .tiers.posed_video import load_posed_video
            fs = load_posed_video(path, cache_dir=cache_dir, progress=log)
        else:
            # plain clip: monocular depth + visual odometry, scale from the depth model only
            from .tiers.video import load_video
            video = path / "rgb.mp4" if path.is_dir() else path
            # StrayScanner stores ARKit frames sensor-landscape without a rotation tag (portrait
            # hold -> 90 deg CW); clips from the Camera app carry rotation metadata instead
            rot = 1 if path.is_dir() else None
            fs = load_video(video, cache_dir=cache_dir, progress=log, rotation_k=rot)
        timings["load_and_reconstruct"] = round(time.time() - t0, 2)
        plan = build_plan(fs, drift_correction=drift, progress=log)
    elif tier == "photo":
        from .tiers.photo import load_photo_property
        from .stitch import stitch_rooms
        prop = load_photo_property(path, cache_dir=cache_dir, progress=log)
        timings["load_and_reconstruct"] = round(time.time() - t0, 2)
        room_plans = []
        for name, rfs in prop.rooms.items():
            log(f"room folder '{name}': {len(rfs.frames)} photos")
            rp = build_plan(rfs, drift_correction=False, progress=log, single_room=True, room_name=name)
            room_plans.append(rp)
        plan = stitch_rooms(room_plans, prop.links, progress=log)
        fs = prop
    else:
        raise SystemExit(f"unknown tier {tier}")
    timings["layout"] = round(time.time() - t0 - sum(timings.values()), 2)

    dmg, flags, scope = [], [], []
    if damage:
        t1 = time.time()
        try:
            from .damage import detect_damage
            from .concealed import evaluate_flags
            from .scope import build_scope
            frame_sets = list(fs.rooms.values()) if tier == "photo" else [fs]
            for f in frame_sets:
                dmg += detect_damage(f, plan, cache_dir=cache_dir, progress=log)
            flags = evaluate_flags(dmg, plan)
            scope = build_scope(dmg, flags, plan)
        except ImportError as e:
            log(f"damage stage unavailable: {e}")
            plan.warnings.append(f"damage stage unavailable: {e}")
        timings["damage"] = round(time.time() - t1, 2)
    timings["total"] = round(time.time() - t0, 2)

    from .output import plan_to_json, validate, write_json
    from .render import render_plan
    capture = {"id": cid, "tier": tier, "source": str(path), "input_sha1": _fingerprint(path)}
    if fs is not None and hasattr(fs, "meta"):
        capture["meta"] = {k: v for k, v in fs.meta.items() if isinstance(v, (str, int, float, bool, list, dict))}
    doc = plan_to_json(plan, capture, dmg, flags, scope, timings)
    errs = validate(doc)
    if errs:
        log(f"SCHEMA VALIDATION FAILED ({len(errs)} errors): {errs[:5]}")
        doc["warnings"].append(f"schema validation failed: {errs[:5]}")
    write_json(doc, out / "plan.json")
    render_plan(plan, str(out / "plan.png"), title=f"{cid} ({tier})", damage=doc["damage"])
    render_plan(plan, str(out / "plan.svg"), title=f"{cid} ({tier})", damage=doc["damage"])
    log(f"done in {timings['total']:.1f}s: {len(plan.rooms)} rooms, "
        f"{sum(len(r.openings) for r in plan.rooms)} openings, {len(dmg)} damage regions -> {out}")
    log_f.close()
    return doc


def _fingerprint(path: Path) -> str:
    """Cheap content fingerprint: sizes + first/last MB of the main files."""
    h = hashlib.sha1()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file()
                                                 and p.suffix.lower() in VID_EXT | IMG_EXT | {".csv"})[:2000]
    for f in files:
        st = f.stat()
        h.update(f"{f.name}:{st.st_size}".encode())
        if f.suffix.lower() in VID_EXT | {".csv"}:
            with open(f, "rb") as fh:
                h.update(fh.read(1 << 20))
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="scan2plan", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process one capture")
    r.add_argument("capture")
    r.add_argument("--tier", default="auto", choices=["auto", "lidar", "video", "photo"])
    r.add_argument("--out", default=None)
    r.add_argument("--no-drift", action="store_true", help="ablation: use poses as captured")
    r.add_argument("--no-damage", action="store_true")
    r.add_argument("--plain-video", action="store_true",
                   help="video tier from RGB only (ignore phone motion data even if present)")
    r.add_argument("--cache", default=".cache")
    r.add_argument("--quiet", action="store_true")
    v = sub.add_parser("validate", help="validate a plan.json against the schema")
    v.add_argument("json")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.capture, a.tier, a.out, drift=not a.no_drift, damage=not a.no_damage, cache_dir=a.cache,
            quiet=a.quiet, plain_video=a.plain_video)
    elif a.cmd == "validate":
        from .output import validate
        errs = validate(json.loads(Path(a.json).read_text()))
        print("valid" if not errs else "\n".join(errs))
        sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
