"""Score the photo tier against the LiDAR plan for the same capture.

LiDAR depth/pose is used here ONLY as an offline ground truth to measure
accuracy; `scan2plan.tiers.photo.load_photo_property` and `scan2plan.stitch`
never see it (see the Photos section of docs/capture_protocol.md and the module docstrings
of photo.py / stitch.py for the actual photo-only pipeline).

For one capture, this:
  1. Runs the real photo-tier pipeline in-process (load_photo_property ->
     per-room build_plan -> stitch_rooms), the same calls `scan2plan.cli.run`
     makes, so registration-success-rate metadata (PhotoProperty.meta) is
     available alongside the stitched Plan.
  2. Loads the LiDAR plan.json for the same capture (ground truth) and the
     `data/photo_tier/<id>/_manifest.json` this bench's `make_photo_tier.py`
     wrote (purely to map "photo folder name" <-> "LiDAR room id"; nothing
     from it reaches the photo pipeline above).
  3. Reports, per room and overall: area ratio (photo/LiDAR), footprint
     ratio, perimeter ("wall length") ratio, registration success rate
     (photos that made it into the main reconstruction vs total), and
     photo<->LiDAR adjacency agreement.

No ground truth beyond the LiDAR plan is available in this project (see the
project README's disclosure); this script reports measured numbers only,
never a fabricated accuracy claim.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scan2plan.layout import build_plan
from scan2plan.stitch import stitch_rooms
from scan2plan.tiers.photo import load_photo_property


def _pct(a: float, b: float) -> float:
    """Signed relative error of a vs b, as a percentage of b."""
    if b == 0:
        return float("nan")
    return 100.0 * (a - b) / b


def evaluate_capture(photo_root: Path, lidar_plan_path: Path, manifest_path: Path | None = None,
                     cache_dir: str = ".cache", model_id: str | None = None, progress=print) -> dict:
    capture_id = photo_root.name
    manifest_path = manifest_path or (photo_root / "_manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    lidar = json.loads(lidar_plan_path.read_text())
    lidar_by_id = {r["id"]: r for r in lidar["rooms"]}
    folder_to_lidar_id = {}
    if manifest:
        folder_to_lidar_id = {v: k for k, v in manifest.get("room_names", {}).items()}

    kwargs = {} if model_id is None else {"model_id": model_id}
    t0 = time.time()
    prop = load_photo_property(photo_root, cache_dir=cache_dir, progress=lambda m: progress(f"  {m}"), **kwargs)
    t_load = time.time() - t0

    room_plans = []
    names = []
    for name, rfs in prop.rooms.items():
        rp = build_plan(rfs, drift_correction=False, single_room=True, room_name=name)
        room_plans.append(rp)
        names.append(name)
    t_layout = time.time() - t0 - t_load

    plan = stitch_rooms(room_plans, prop.links, names=names)
    t_stitch = time.time() - t0 - t_load - t_layout

    # registration success rate
    reg_total = sum(m["n_photo_files"] for m in prop.meta["rooms"].values())
    reg_main = sum(m["n_registered_main"] for m in prop.meta["rooms"].values())
    per_room_reg = {r: (m["n_registered_main"], m["n_photo_files"]) for r, m in prop.meta["rooms"].items()}

    rooms_report = []
    for r in plan.rooms:
        lidar_id = folder_to_lidar_id.get(r.name)
        lroom = lidar_by_id.get(lidar_id) if lidar_id else None
        entry = {"photo_room": r.id, "name": r.name, "photo_area_m2": round(r.area.value, 3),
                "photo_perimeter_m": round(r.perimeter.value, 3),
                "n_registered": per_room_reg.get(r.name)}
        if lroom:
            entry.update({
                "lidar_room": lidar_id,
                "lidar_area_m2": round(lroom["floor_area"]["value"], 3),
                "lidar_perimeter_m": round(lroom["perimeter"]["value"], 3),
                "area_error_pct": round(_pct(r.area.value, lroom["floor_area"]["value"]), 1),
                "perimeter_error_pct": round(_pct(r.perimeter.value, lroom["perimeter"]["value"]), 1),
            })
        rooms_report.append(entry)

    lidar_footprint = lidar["stitched"]["footprint_area"]["value"] if lidar["stitched"]["footprint_area"] else None
    photo_footprint = plan.footprint_area.value if plan.footprint_area else None
    footprint_error_pct = _pct(photo_footprint, lidar_footprint) if (photo_footprint and lidar_footprint) else None

    def adjacency_pairs(adj, rooms_by_id):
        out = set()
        for e in adj:
            a, b = e["rooms"]
            if a in rooms_by_id and b in rooms_by_id:
                out.add(tuple(sorted((rooms_by_id[a], rooms_by_id[b]))))
        return out

    photo_room_name_by_id = {r.id: r.name for r in plan.rooms}
    photo_lidar_id_by_photo_id = {rid: folder_to_lidar_id.get(name) for rid, name in photo_room_name_by_id.items()}
    photo_adj = adjacency_pairs(plan.adjacency, photo_lidar_id_by_photo_id)
    lidar_room_name_by_id = {r["id"]: r["id"] for r in lidar["rooms"]}
    lidar_adj = adjacency_pairs(lidar["adjacency"], lidar_room_name_by_id)

    report = {
        "capture_id": capture_id,
        "n_rooms_photo": len(plan.rooms),
        "n_rooms_lidar": len(lidar["rooms"]),
        "registration_success_rate": round(reg_main / reg_total, 3) if reg_total else None,
        "n_registered_main": reg_main,
        "n_photos_total": reg_total,
        "rooms": rooms_report,
        "footprint_photo_m2": round(photo_footprint, 3) if photo_footprint else None,
        "footprint_lidar_m2": round(lidar_footprint, 3) if lidar_footprint else None,
        "footprint_error_pct": round(footprint_error_pct, 1) if footprint_error_pct is not None else None,
        "adjacency_photo": sorted(photo_adj),
        "adjacency_lidar": sorted(lidar_adj),
        "adjacency_matches_lidar": photo_adj == lidar_adj,
        "n_links": prop.meta.get("n_links"),
        "n_warnings_photo_property": len(prop.warnings),
        "n_warnings_stitched_plan": len(plan.warnings),
        "runtime_s": {"load_and_register": round(t_load, 2), "per_room_layout": round(t_layout, 2),
                     "stitch": round(t_stitch, 2), "total": round(time.time() - t0, 2)},
    }
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", action="append", default=None)
    ap.add_argument("--photo-root", default="data/photo_tier")
    ap.add_argument("--lidar-root", default="out/bench")
    ap.add_argument("--out", default=None, help="write the JSON report here (per capture if multiple)")
    ap.add_argument("--cache", default=".cache")
    ap.add_argument("--model-id", default=None,
                    help="override photo-tier dense depth model (default: photo.DEFAULT_MODEL_ID)")
    a = ap.parse_args(argv)

    captures = a.capture or sorted(p.name for p in Path(a.photo_root).iterdir() if p.is_dir())
    reports = []
    for cid in captures:
        photo_root = Path(a.photo_root) / cid
        lidar_plan = Path(a.lidar_root) / cid / "lidar" / "plan.json"
        if not photo_root.exists() or not lidar_plan.exists():
            print(f"skip {cid}: missing photo folder or LiDAR plan.json")
            continue
        print(f"=== {cid} ===")
        report = evaluate_capture(photo_root, lidar_plan, cache_dir=a.cache, model_id=a.model_id)
        print(json.dumps(report, indent=2))
        reports.append(report)
        if a.out:
            out_path = Path(a.out)
            target = out_path / f"{cid}.json" if out_path.suffix == "" else out_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2))
    return reports


if __name__ == "__main__":
    main()
