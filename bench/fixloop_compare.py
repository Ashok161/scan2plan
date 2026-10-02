"""Score two output trees (before / after) with the same scorer.

    python -m bench.fixloop_compare out/fixloop/before out/fixloop/after
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from bench.compare import load
from bench.run_benchmark import REPEAT_PAIRS, compare_plans, opening_stats, wall_stats

ROOT = Path(__file__).resolve().parent.parent


def score_tree(root: Path) -> dict:
    out = {}
    for a, b in REPEAT_PAIRS:
        pa, pb = root / a / "lidar" / "plan.json", root / b / "lidar" / "plan.json"
        if not (pa.exists() and pb.exists()):
            continue
        c = compare_plans(load(pa), load(pb))
        out[f"{a}_vs_{b}"] = {"walls": wall_stats(c["walls"], rel=0.005, abs_=0.01, both=True),
                              "openings": opening_stats(c),
                              "rooms_matched": len(c["rooms_matched"]),
                              "rooms": [c["rooms_a"], c["rooms_b"]]}
    return out


def main(before: str, after: str):
    res = {"before": score_tree(Path(before)), "after": score_tree(Path(after))}
    d = ROOT / "reports" / "fixloop"
    d.mkdir(parents=True, exist_ok=True)
    (d / "before_after.json").write_text(json.dumps(res, indent=2))
    L = ["# Fix loop: before vs after (same scorer, regenerated from raw captures)\n",
         "| pair | metric | before | after |", "|---|---|---|---|"]
    for k in res["after"]:
        b, a = res["before"].get(k, {}), res["after"][k]
        for m, lab in (("pass_rate", "walls passing 1 cm / 0.5%"), ("median_abs_mm", "median wall diff (mm)"),
                       ("ci95_coverage", "CI95 covers the difference"), ("z_rms", "z rms (1 = calibrated)"),
                       ("n", "matched walls")):
            L.append(f"| {k} | {lab} | {b.get('walls', {}).get(m, '–')} | {a['walls'].get(m, '–')} |")
        L.append(f"| {k} | openings within 2 cm | {b.get('openings', {}).get('score', '–')} | {a['openings'].get('score', '–')} |")
    (d / "before_after.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
