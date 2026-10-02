"""Benchmark harness: every reported number is regenerated from raw captures here.

    python -m bench.run_benchmark            # run pipeline on all captures/tiers, then score
    python -m bench.run_benchmark --score-only

GROUND TRUTH DISCLOSURE: no tape/laser measurements exist for these captures
(the author had no access to an iPhone or the property; captures were supplied
as sample data). Gates that need tape ground truth are therefore scored with
the strongest available proxy and labelled as such:
  * repeatability / opening agreement: capture-vs-capture at the same tier
  * video / photo accuracy: against the LiDAR-tier plan of the same rooms
  * ceiling repeatability: split-half of a single capture (disclosed)
Fill bench/ground_truth/<capture>.yaml with tape numbers and re-run to score
against real ground truth (see bench/ground_truth/README.md).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bench.compare import load, match_openings, match_rooms, match_walls, register

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "bench"
REPORTS = ROOT / "reports"

CAPTURES = {
    "c00a170fe1": {"desc": "single room + bathroom (StrayScanner, 37 s)", "multi_room": False},
    "1a8384c3f6": {"desc": "whole apartment, floor-focused walk (StrayScanner, 115 s)", "multi_room": True},
    "c7d28f72c6": {"desc": "whole apartment incl. ceiling (StrayScanner, 215 s)", "multi_room": True},
}
REPEAT_PAIRS = [("1a8384c3f6", "c7d28f72c6"), ("c00a170fe1", "c7d28f72c6"), ("c00a170fe1", "1a8384c3f6")]
REFERENCE = "c7d28f72c6"           # best-covered LiDAR capture = proxy reference for thinner tiers


def run_all(tiers, captures, nodrift=True):
    from scan2plan.cli import run
    timing = {}
    for cid in captures:
        src = ROOT / "data" / cid
        for tier in tiers:
            if tier == "photo":
                src_t = ROOT / "data" / "photo_tier" / cid
                if not src_t.exists():
                    # build the photo-tier stills from this capture's video, using the LiDAR plan
                    # (just written above) only to assign frames to rooms; see bench/make_photo_tier.py
                    from bench.make_photo_tier import main as make_photo_tier
                    plan = OUT / cid / "lidar" / "plan.json"
                    make_photo_tier(["--capture", cid] + (["--plan", str(plan)] if plan.exists() else []))
            else:
                src_t = src
            out = OUT / cid / tier
            t = time.time()
            try:
                run(src_t, tier=tier, out=out, quiet=True)
                timing[f"{cid}/{tier}"] = round(time.time() - t, 1)
            except Exception as e:     # a failing tier must not hide the others
                timing[f"{cid}/{tier}"] = f"FAILED: {type(e).__name__}: {e}"
                print(f"{cid}/{tier} failed: {e}")
            if nodrift and tier in ("lidar", "video") and CAPTURES[cid]["multi_room"]:
                try:
                    run(src_t, tier=tier, out=OUT / cid / f"{tier}_nodrift", drift=False, damage=False, quiet=True)
                except Exception as e:
                    print(f"{cid}/{tier} nodrift failed: {e}")
    return timing


# ------------------------------------------------------------------ scoring helpers

def _plan(cid, tier):
    p = OUT / cid / tier / "plan.json"
    return load(p) if p.exists() else None


def compare_plans(A, B, rel_tol=None, abs_tol=None):
    """Wall/opening/area agreement of plan A against plan B (B = reference)."""
    reg = register(A, B)
    k, t, iou = reg
    ra = {r["id"]: r for r in A["rooms"]}
    rb = {r["id"]: r for r in B["rooms"]}
    rooms = match_rooms(A, B, reg)
    walls, opens = [], []
    for a, b, riou in rooms:
        for wa, wb in match_walls(ra[a], rb[b], k, t):
            la, lb = wa["length"]["value"], wb["length"]["value"]
            sa, sb = wa["length"]["sigma"], wb["length"]["sigma"]
            walls.append({"room_a": a, "room_b": b, "wall_a": wa["id"], "wall_b": wb["id"],
                          "len_a": la, "len_b": lb, "diff": la - lb, "sigma_a": sa, "sigma_b": sb,
                          "cov_a": wa["coverage"], "cov_b": wb["coverage"]})
        for oa, ob in match_openings(ra[a], rb[b], k, t):
            opens.append({"a": oa["id"], "b": ob["id"], "w_a": oa["width"]["value"], "w_b": ob["width"]["value"],
                          "diff": oa["width"]["value"] - ob["width"]["value"],
                          "sigma_a": oa["width"]["sigma"], "sigma_b": ob["width"]["sigma"]})
    n_open_a = sum(len(r["openings"]) for r in A["rooms"])
    n_open_b = sum(len(r["openings"]) for r in B["rooms"])
    fa = A["stitched"]["footprint_area"]
    fb = B["stitched"]["footprint_area"]
    return {"registration": {"quarter_turns": int(k), "t": [round(float(x), 3) for x in t], "iou": round(iou, 3)},
            "rooms_a": len(A["rooms"]), "rooms_b": len(B["rooms"]),
            "rooms_matched": [{"a": a, "b": b, "iou": round(i, 3),
                               "area_a": ra[a]["floor_area"]["value"], "area_b": rb[b]["floor_area"]["value"]}
                              for a, b, i in rooms],
            "walls": walls, "openings": opens, "n_open_a": n_open_a, "n_open_b": n_open_b,
            "footprint_a": fa, "footprint_b": fb}


def wall_stats(walls, rel=None, abs_=None, both=False):
    if not walls:
        return {"n": 0}
    d = np.array([w["diff"] for w in walls])
    L = np.array([w["len_b"] for w in walls])
    s = np.array([np.hypot(w["sigma_a"], w["sigma_b"]) if both else w["sigma_a"] for w in walls])
    e = np.abs(d)
    out = {"n": len(walls), "median_abs_mm": round(float(np.median(e)) * 1000, 1),
           "p90_abs_mm": round(float(np.percentile(e, 90)) * 1000, 1),
           "median_abs_pct": round(float(np.median(e / L)) * 100, 2),
           "z_rms": round(float(np.sqrt(np.mean((d / s) ** 2))), 2),
           "ci95_coverage": round(float(np.mean(e <= 1.96 * s)), 3)}
    if rel is not None or abs_ is not None:
        ok = np.zeros(len(e), bool)
        if abs_ is not None:
            ok |= e <= abs_
        if rel is not None:
            ok |= e <= rel * L
        out["pass_rate"] = round(float(ok.mean()), 3)
    return out


def opening_stats(cmp, tol=0.02):
    m = cmp["openings"]
    ok = sum(1 for o in m if abs(o["diff"]) <= tol)
    # misses: unmatched openings on either side count against
    total = max(cmp["n_open_a"], cmp["n_open_b"])
    return {"matched": len(m), "within_2cm": ok, "openings_a": cmp["n_open_a"], "openings_b": cmp["n_open_b"],
            "score": round(ok / total, 3) if total else None,
            "median_abs_mm": round(float(np.median([abs(o["diff"]) for o in m])) * 1000, 1) if m else None}


def drift_ablation(cid, tier):
    on, off = _plan(cid, tier), _plan(cid, f"{tier}_nodrift")
    if on is None or off is None:
        return None
    ref = _plan(REFERENCE, "lidar") if cid != REFERENCE else None
    out = {"footprint_on": on["stitched"]["footprint_area"]["value"],
           "footprint_off": off["stitched"]["footprint_area"]["value"],
           "rooms_on": len(on["rooms"]), "rooms_off": len(off["rooms"]),
           "wall_plane_spread_mm": [on["drift"].get("wall_plane_spread_before_mm"),
                                    on["drift"].get("wall_plane_spread_after_mm")],
           "corrections": on["drift"].get("iterations")}
    # self-consistency: the same physical walls in the two runs
    c = compare_plans(on, off)
    out["walls_changed_mm_median"] = wall_stats(c["walls"]).get("median_abs_mm")
    return out


def ceiling_stats(doc):
    rows = []
    for r in doc["rooms"]:
        ch = r["ceiling_height"]
        rows.append({"room": r["id"], "name": r["name"], "observed": r["ceiling_observed"],
                     "value": ch["value"], "ci95": ch["ci95"], "notes": r.get("notes", [])})
    return rows


def score():
    res = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "captures": CAPTURES,
           "ground_truth": "NONE (sample data only; no tape/laser available). Proxies used; see docstring."}
    tiers = ["lidar", "video", "photo"]
    res["outputs"] = {f"{c}/{t}": (_plan(c, t) is not None) for c in CAPTURES for t in tiers}

    # repeatability (same tier, two captures of the same rooms)
    rep = {}
    for a, b in REPEAT_PAIRS:
        for t in tiers:
            A, B = _plan(a, t), _plan(b, t)
            if A is None or B is None:
                continue
            c = compare_plans(A, B)
            rep[f"{a}_vs_{b}/{t}"] = {"walls": wall_stats(c["walls"], rel=0.005, abs_=0.01, both=True),
                                      "rooms": c["rooms_matched"], "rooms_a": c["rooms_a"], "rooms_b": c["rooms_b"],
                                      "registration": c["registration"], "openings": opening_stats(c),
                                      "wall_pairs": c["walls"]}
    res["repeatability"] = rep

    # thinner tiers vs LiDAR reference of the same capture
    acc = {}
    for cid in CAPTURES:
        ref = _plan(cid, "lidar")
        for t, tol in (("video", 0.03), ("photo", 0.08)):
            P = _plan(cid, t)
            if P is None or ref is None:
                continue
            c = compare_plans(P, ref)
            if not c["footprint_a"] or not P["rooms"]:
                acc[f"{cid}/{t}"] = {"walls": {"n": 0}, "footprint_err_pct": -100.0, "footprint_in_ci95": False,
                                     "rooms": [], "rooms_a": 0, "rooms_b": c["rooms_b"], "adjacency_a": 0,
                                     "openings": opening_stats(c), "note": "tier produced no rooms"}
                continue
            fa, fb = c["footprint_a"]["value"], c["footprint_b"]["value"]
            sfa = c["footprint_a"]["sigma"]
            acc[f"{cid}/{t}"] = {"walls": wall_stats(c["walls"], rel=tol),
                                 "footprint_err_pct": round((fa - fb) / fb * 100, 2),
                                 "footprint_in_ci95": abs(fa - fb) <= 1.96 * sfa,
                                 "rooms": c["rooms_matched"], "rooms_a": c["rooms_a"], "rooms_b": c["rooms_b"],
                                 "adjacency_a": len(P["adjacency"]), "openings": opening_stats(c)}
    res["tier_vs_lidar"] = acc

    res["drift_ablation"] = {f"{c}/{t}": drift_ablation(c, t) for c in CAPTURES if CAPTURES[c]["multi_room"]
                             for t in ("lidar", "video")}
    res["ceilings"] = {f"{c}/lidar": ceiling_stats(_plan(c, "lidar")) for c in CAPTURES if _plan(c, "lidar")}
    split = REPORTS / "ceiling_split_half.json"
    if split.exists():
        res["ceiling_split_half"] = json.loads(split.read_text())
    tf = REPORTS / "timing.json"
    if tf.exists():
        res["timing_s"] = json.loads(tf.read_text())
    return res


def write_reports(res):
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "benchmark.json").write_text(json.dumps(res, indent=2, default=float))
    from bench.report_md import render_markdown
    (REPORTS / "benchmark.md").write_text(render_markdown(res))
    print(f"wrote {REPORTS / 'benchmark.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--tiers", default="lidar,video,photo")
    ap.add_argument("--captures", default=",".join(CAPTURES))
    a = ap.parse_args()
    if not a.score_only:
        timing = run_all(a.tiers.split(","), a.captures.split(","))
        REPORTS.mkdir(exist_ok=True)
        tf = REPORTS / "timing.json"
        old = json.loads(tf.read_text()) if tf.exists() else {}
        old.update(timing)
        tf.write_text(json.dumps(old, indent=2))
        from bench.ceiling_split import run as split_run
        split_run()
    write_reports(score())


if __name__ == "__main__":
    main()
