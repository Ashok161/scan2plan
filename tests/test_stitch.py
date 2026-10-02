"""Unit tests for scan2plan.stitch, with synthetic rectangular rooms.

Layout (ground truth, metres, x/z plan): room A and room B flank the north
end of a corridor; room C hangs off the corridor's south wall. Three doors,
all 0.15 m wall thickness apart (matching stitch.wall_thickness's default):

    A (4x3) --door-- corridor (1.2x6) --door-- B (4x3)
                            |
                           door
                            |
                           C (3.2x3)

Two independent paths through stitch_rooms are exercised:
  * no links at all -> pure door-opening-matching fallback must discover
    the same adjacency from scratch, with no overlaps.
  * exact links for all three doors -> the link/pose-graph path must
    reproduce the correct relative placement without touching the door
    matcher at all.
"""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from scan2plan.measure import Measurement
from scan2plan.plan_types import Opening, Plan, Room, Wall
from scan2plan.stitch import stitch_rooms



def _rot2(yaw_deg):
    th = np.radians(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def _apply_se2(xz, yaw_deg, t):
    return xz @ _rot2(yaw_deg).T + t


def _se2_to_T4(yaw_deg, t):
    th = np.radians(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    T = np.eye(4)
    T[0, 0], T[0, 2], T[2, 0], T[2, 2] = c, -s, s, c
    T[0, 3], T[2, 3] = t[0], t[1]
    return T


_SIDES = {
    # side -> (start_corner_fn, end_corner_fn, normal_in) given (x0,z0,x1,z1)
    "S": (lambda x0, z0, x1, z1: (x0, z0), lambda x0, z0, x1, z1: (x1, z0), np.array([0.0, 1.0])),
    "E": (lambda x0, z0, x1, z1: (x1, z0), lambda x0, z0, x1, z1: (x1, z1), np.array([-1.0, 0.0])),
    "N": (lambda x0, z0, x1, z1: (x1, z1), lambda x0, z0, x1, z1: (x0, z1), np.array([0.0, -1.0])),
    "W": (lambda x0, z0, x1, z1: (x0, z1), lambda x0, z0, x1, z1: (x0, z0), np.array([1.0, 0.0])),
}


def _rect_room(rid: str, x0: float, z0: float, x1: float, z1: float, doors=None,
              ceiling_h: float = 2.5) -> Room:
    """doors: list of dict(side, t0, t1, kind='door')."""
    poly = np.array([[x0, z0], [x1, z0], [x1, z1], [x0, z1]])
    walls = []
    for side, (sf, ef, n) in _SIDES.items():
        s, e = np.array(sf(x0, z0, x1, z1)), np.array(ef(x0, z0, x1, z1))
        length = float(np.linalg.norm(e - s))
        walls.append(Wall(id=f"{rid}.W{side}", start=s, end=e, normal_in=n,
                          length=Measurement(length, 0.01), height=Measurement(ceiling_h, 0.02),
                          offset_sigma=0.01, coverage=1.0))
    openings = []
    for i, d in enumerate(doors or []):
        sf, ef, n = _SIDES[d["side"]]
        s, e = np.array(sf(x0, z0, x1, z1)), np.array(ef(x0, z0, x1, z1))
        along = (e - s) / np.linalg.norm(e - s)
        t0, t1 = d["t0"], d["t1"]
        center = s + along * ((t0 + t1) / 2.0)
        width = abs(t1 - t0)
        openings.append(Opening(id=f"{rid}.D{i}", kind=d.get("kind", "door"),
                                wall_id=f"{rid}.W{d['side']}", center=center, along=along,
                                width=Measurement(width, 0.01)))
    area = abs((x1 - x0) * (z1 - z0))
    perim = 2 * (abs(x1 - x0) + abs(z1 - z0))
    return Room(id=rid, name=rid, kind="room", polygon=poly, floor_y=0.0, ceiling_y=ceiling_h,
               area=Measurement(area, 0.05), perimeter=Measurement(perim, 0.05),
               ceiling_height=Measurement(ceiling_h, 0.02), walls=walls, openings=openings)


def _ground_truth_rooms():
    # Door widths are deliberately distinct (0.9 / 0.8 / 0.6 m) so the
    # width-only door matcher has a unique best candidate everywhere; with
    # identical widths on every door, a wrong-but-equally-scored pairing
    # (e.g. room B docking straight onto room A's door) is a real, expected
    # ambiguity of width-only matching, not a bug -- real doors vary in size.
    room_a = _rect_room("roomA", 0.0, 0.0, 4.0, 3.0, doors=[{"side": "E", "t0": 1.0, "t1": 1.9}])
    corridor = _rect_room("corridor", 4.15, 0.0, 5.35, 6.0,
                          doors=[{"side": "W", "t0": 1.0, "t1": 1.9},
                                 {"side": "E", "t0": 1.05, "t1": 1.85},
                                 {"side": "N", "t0": 0.3, "t1": 0.9}])
    room_b = _rect_room("roomB", 5.5, 0.0, 9.5, 3.0, doors=[{"side": "W", "t0": 1.05, "t1": 1.85}])
    room_c = _rect_room("roomC", 4.0, 6.15, 7.2, 9.15, doors=[{"side": "S", "t0": 0.45, "t1": 1.05}])
    return {"roomA": room_a, "corridor": corridor, "roomB": room_b, "roomC": room_c}


GROUND_TRUTH_AREA = 4 * 3 + 1.2 * 6 + 4 * 3 + 3.2 * 3   # 40.8 m2


def _scrambled_plans(rooms: dict, scrambles: dict):
    """Wrap each room in its own Plan after an independent SE(2) 'arrived in
    its own local frame' scramble. Returns (plans, names) aligned by name."""
    plans, names = [], []
    for name, room in rooms.items():
        yaw, t = scrambles[name]
        local = Room(**{**room.__dict__, "polygon": _apply_se2(room.polygon, yaw, t),
                        "walls": [Wall(**{**w.__dict__, "start": _rot2(yaw) @ w.start + t,
                                          "end": _rot2(yaw) @ w.end + t,
                                          "normal_in": _rot2(yaw) @ w.normal_in})
                                  for w in room.walls],
                        "openings": [Opening(**{**o.__dict__, "center": _rot2(yaw) @ o.center + t,
                                               "along": _rot2(yaw) @ o.along})
                                    for o in room.openings]})
        plans.append(Plan(tier="photo", rooms=[local], adjacency=[], T_align=np.eye(4), floor_y=0.0))
        names.append(name)
    return plans, names


def _no_overlaps(plan: Plan, tol: float = 1e-5) -> bool:
    polys = [Polygon(r.polygon) for r in plan.rooms]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i].buffer(-1e-6).intersection(polys[j].buffer(-1e-6)).area > tol:
                return False
    return True


def _adjacency_name_pairs(plan: Plan):
    # Room.name survives _transform_room untouched even when Room.id collides
    # across plans (as it does for real single-room Plans from build_plan).
    name_of = {r.id: r.name for r in plan.rooms}
    return {tuple(sorted((name_of[a], name_of[b]))) for a, b in (e["rooms"] for e in plan.adjacency)}


EXPECTED_PAIRS = {("corridor", "roomA"), ("corridor", "roomB"), ("corridor", "roomC")}


def _random_scrambles(names, seed: int):
    """Independent per-call RNG: each test gets its own fixed, reproducible
    scramble regardless of test execution order (no shared mutable state)."""
    rng = np.random.default_rng(seed)
    return {n: (int(rng.choice([0, 90, 180, 270])), rng.uniform(-5, 5, size=2)) for n in names}


# --------------------------------------------------------------------- tests


def test_door_matching_fallback_recovers_layout_with_no_links():
    rooms = _ground_truth_rooms()
    scrambles = _random_scrambles(rooms, seed=1)
    plans, names = _scrambled_plans(rooms, scrambles)

    merged = stitch_rooms(plans, links=[], names=names, seed=0)

    assert not any("overlap detected" in w for w in merged.warnings)
    assert _no_overlaps(merged)
    assert _adjacency_name_pairs(merged) == EXPECTED_PAIRS
    assert merged.footprint_area is not None
    rel_err = abs(merged.footprint_area.value - GROUND_TRUTH_AREA) / GROUND_TRUTH_AREA
    assert rel_err < 0.08, f"footprint relative error {rel_err:.3f} exceeds 8%"
    assert merged.drift["n_door_matched_attachments"] == 3
    assert merged.drift["n_arbitrary_placements"] == 0


def test_link_pose_graph_recovers_layout_without_door_matching():
    rooms = _ground_truth_rooms()
    # strip every opening: the door matcher physically cannot do anything here
    for name, room in rooms.items():
        rooms[name] = Room(**{**room.__dict__, "openings": []})
    scrambles = _random_scrambles(rooms, seed=2)
    plans, names = _scrambled_plans(rooms, scrambles)

    def T4(name):
        yaw, t = scrambles[name]
        return _se2_to_T4(yaw, t)

    links = []
    for a, b in [("roomA", "corridor"), ("roomB", "corridor"), ("corridor", "roomC")]:
        T_rel = T4(a) @ np.linalg.inv(T4(b))
        links.append({"rooms": [a, b], "T_a_from_b": T_rel, "inliers": 40, "evidence": "feature_match"})

    merged = stitch_rooms(plans, links=links, names=names, seed=0)

    assert not any("overlap detected" in w for w in merged.warnings)
    assert _no_overlaps(merged)
    assert _adjacency_name_pairs(merged) == EXPECTED_PAIRS
    assert all(e["evidence"] == "feature_match" for e in merged.adjacency)
    assert merged.drift["n_door_matched_attachments"] == 0
    assert merged.drift["n_arbitrary_placements"] == 0
    rel_err = abs(merged.footprint_area.value - GROUND_TRUTH_AREA) / GROUND_TRUTH_AREA
    assert rel_err < 0.02, f"footprint relative error {rel_err:.3f} (link path should be near-exact)"


def test_default_names_use_room_name_not_id():
    """Mirrors cli.run's real wiring: layout.build_plan(single_room=True,
    room_name=<folder>) sets every single-room Plan's Room.id to the generic
    "R1" and only Room.name to the photo folder name, and cli.run calls
    stitch_rooms(room_plans, prop.links, progress=log) with no `names` arg.
    stitch_rooms must therefore default to matching on `.name`, not `.id`."""
    rooms = _ground_truth_rooms()
    for name, room in rooms.items():
        rooms[name] = Room(**{**room.__dict__, "id": "R1", "name": name})
    scrambles = _random_scrambles(rooms, seed=3)
    plans, _names = _scrambled_plans(rooms, scrambles)
    assert len({p.rooms[0].id for p in plans}) == 1   # every id collides, as in real build_plan output

    seen = []

    def progress(msg):
        seen.append(msg)

    merged = stitch_rooms(plans, links=[], progress=progress)   # no names= passed, as cli.py does

    assert seen, "progress callback should be invoked"
    assert _no_overlaps(merged)
    assert _adjacency_name_pairs(merged) == EXPECTED_PAIRS


def test_unreachable_room_is_placed_without_overlap_and_warns():
    rooms = _ground_truth_rooms()
    # drop roomC's door so it can neither link nor door-match to anything
    rooms["roomC"] = Room(**{**rooms["roomC"].__dict__, "openings": []})
    scrambles = _random_scrambles(rooms, seed=4)
    plans, names = _scrambled_plans(rooms, scrambles)
    links = []
    for a, b in [("roomA", "corridor"), ("roomB", "corridor")]:
        yaw, t = scrambles[a]
        Ta = _se2_to_T4(yaw, t)
        yaw, t = scrambles[b]
        Tb = _se2_to_T4(yaw, t)
        links.append({"rooms": [a, b], "T_a_from_b": Ta @ np.linalg.inv(Tb),
                     "inliers": 40, "evidence": "feature_match"})

    merged = stitch_rooms(plans, links=links, names=names, seed=0)

    assert _no_overlaps(merged)
    assert merged.drift["n_arbitrary_placements"] == 1
    assert any("no link and no matching door" in w for w in merged.warnings)
    pairs = _adjacency_name_pairs(merged)
    assert ("corridor", "roomC") not in pairs
    assert {("corridor", "roomA"), ("corridor", "roomB")} <= pairs


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
