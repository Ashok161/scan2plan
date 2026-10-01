"""Fast, model-free tests for scan2plan.concealed and scan2plan.scope.

Builds a tiny hand-made two-room Plan (no model inference, no captures
needed) and hand-made damage regions shaped exactly like
`damage.detect_damage`'s output dicts, then checks that the rule engines in
concealed.py / scope.py fire the rules we expect and propagate quantities
and uncertainty sensibly.
"""
from __future__ import annotations

import numpy as np
import pytest

from scan2plan.concealed import evaluate_concealed, load_rules
from scan2plan.measure import Measurement
from scan2plan.plan_types import Opening, Plan, Room, Wall
from scan2plan.scope import generate_scope, load_scope_rules


def _m(value, sigma=0.01, unit="m"):
    return Measurement(value, sigma, unit)


def build_test_plan() -> Plan:
    opening = Opening(id="room0.op0", kind="door", wall_id="room0.W1",
                      center=np.array([3.0, 1.5]), along=np.array([0.0, 1.0]),
                      width=_m(1.0), connects=["room0", "room1"])
    w0 = Wall(id="room0.W0", start=np.array([0.0, 0.0]), end=np.array([3.0, 0.0]),
             normal_in=np.array([0.0, 1.0]), length=_m(3.0), height=_m(2.4),
             offset_sigma=0.01, coverage=1.0)   # exterior: no opening references it
    w1 = Wall(id="room0.W1", start=np.array([3.0, 0.0]), end=np.array([3.0, 3.0]),
             normal_in=np.array([-1.0, 0.0]), length=_m(3.0), height=_m(2.4),
             offset_sigma=0.01, coverage=1.0)   # interior: has a door to room1
    room0 = Room(id="room0", name="Living Room", kind="room",
                polygon=np.array([[0, 0], [3, 0], [3, 3], [0, 3]], dtype=float),
                floor_y=0.0, ceiling_y=2.4, area=_m(9.0, 0.05, "m2"),
                perimeter=_m(12.0, 0.05), ceiling_height=_m(2.4, 0.02),
                walls=[w0, w1], openings=[opening])
    bath = Room(id="bath", name="Bathroom", kind="room",
               polygon=np.array([[4, 0], [5, 0], [5, 1], [4, 1]], dtype=float),
               floor_y=0.0, ceiling_y=2.4, area=_m(1.0, 0.02, "m2"),
               perimeter=_m(4.0, 0.02), ceiling_height=_m(2.4, 0.02), walls=[])
    return Plan(tier="lidar", rooms=[room0, bath], adjacency=[], T_align=np.eye(4), floor_y=0.0)


def _region(rid, cls, surface_id, room_id, centroid_plan, width=0.3, height=0.3, confidence=0.8):
    return {
        "id": rid, "surface_id": surface_id, "room_id": room_id, "class": cls,
        "confidence": confidence,
        "area": {"value": width * height, "sigma": 0.001, "ci95": [0, 1], "unit": "m2"},
        "extent": {"width": {"value": width, "sigma": 0.01, "ci95": [0, 1], "unit": "m"},
                  "height": {"value": height, "sigma": 0.01, "ci95": [0, 1], "unit": "m"}},
        "centroid_plan": list(centroid_plan),
        "polygon_surface": [[0, 0], [width, 0], [width, height], [0, height]],
        "n_views": 3, "frames": [1, 2, 3], "evidence": {},
    }


@pytest.fixture
def plan():
    return build_test_plan()


@pytest.fixture
def regions():
    return [
        _region("d.ceiling", "water_stain", "room0.ceiling", "room0", (1.5, 2.4, 0.2)),
        _region("d.wall_base", "water_stain", "room0.W0", "room0", (1.5, 0.1, 0.0)),
        _region("d.mould_ext", "mould", "room0.W0", "room0", (1.0, 1.0, 0.0)),
        _region("d.wall_top", "water_stain", "room0.W0", "room0", (1.5, 2.3, 0.2)),
        _region("d.crack", "crack", "room0.W1", "room0", (3.0, 1.2, 1.9), width=0.6, height=0.02),
        _region("d.near_wet", "mould", "room0.W0", "room0", (2.9, 1.0, 0.0)),
    ]


def test_rules_file_loads_and_compiles():
    rules = load_rules()
    ids = {r["id"] for r in rules}
    assert {"ceiling_water_stain", "wall_base_moisture", "near_wet_room_moisture",
           "exterior_wall_mould", "structural_crack", "ceiling_wall_junction_stain"} <= ids


def test_ceiling_water_stain_fires(plan, regions):
    flags = evaluate_concealed(plan, regions)
    by_rule = {}
    for f in flags:
        by_rule.setdefault(f["rule_id"], []).append(f)
    assert "ceiling_water_stain" in by_rule
    assert "d.ceiling" in by_rule["ceiling_water_stain"][0]["triggered_by"]


def test_wall_base_moisture_fires_only_near_floor(plan, regions):
    flags = evaluate_concealed(plan, regions)
    triggered = [f for f in flags if f["rule_id"] == "wall_base_moisture"]
    assert len(triggered) == 1
    assert triggered[0]["triggered_by"] == ["d.wall_base"]


def test_exterior_wall_mould_fires_on_wall_without_opening(plan, regions):
    flags = evaluate_concealed(plan, regions)
    triggered_ids = {tuple(f["triggered_by"]) for f in flags if f["rule_id"] == "exterior_wall_mould"}
    assert ("d.mould_ext",) in triggered_ids or ("d.near_wet",) in triggered_ids


def test_structural_crack_fires_near_opening_corner(plan, regions):
    flags = evaluate_concealed(plan, regions)
    triggered = [f for f in flags if f["rule_id"] == "structural_crack"]
    assert len(triggered) == 1
    assert triggered[0]["triggered_by"] == ["d.crack"]


def test_near_wet_room_moisture_fires_for_region_near_bathroom(plan, regions):
    flags = evaluate_concealed(plan, regions)
    triggered = [f for f in flags if f["rule_id"] == "near_wet_room_moisture"]
    assert any(f["triggered_by"] == ["d.near_wet"] for f in triggered)


def test_ceiling_wall_junction_pair_rule_fires(plan, regions):
    flags = evaluate_concealed(plan, regions)
    triggered = [f for f in flags if f["rule_id"] == "ceiling_wall_junction_stain"]
    assert len(triggered) >= 1
    assert set(triggered[0]["triggered_by"]) == {"d.ceiling", "d.wall_top"}


def test_no_rules_fire_for_an_isolated_benign_region(plan):
    far_crack = _region("d.benign", "crack", "room0.W1", "room0", (3.0, 1.2, 0.1), width=0.1, height=0.01)
    flags = evaluate_concealed(plan, [far_crack])
    assert flags == []


def test_concealed_flag_schema(plan, regions):
    flags = evaluate_concealed(plan, regions)
    for f in flags:
        assert set(f) == {"id", "surface_id", "rule_id", "rule", "flag", "severity",
                          "triggered_by", "recommendation"}
        assert f["severity"] in ("low", "medium", "high")


# --------------------------------------------------------------------------
# scope.py
# --------------------------------------------------------------------------
def test_scope_rules_file_loads():
    rules = load_scope_rules()
    ids = {r["id"] for r in rules}
    assert {"stain_mould_repaint", "mould_remediation", "crack_fill", "hole_patch",
           "peeling_paint_prep_repaint", "concealed_investigation"} <= ids


def test_scope_groups_stain_and_mould_on_same_wall_into_one_repaint_item(plan, regions):
    items = generate_scope(plan, regions, [])
    repaint = [i for i in items if i["code"] == "09.91-REPAINT-SURF" and i["surface_id"] == "room0.W0"]
    assert len(repaint) == 1
    assert set(repaint[0]["triggered_by"]) == {"d.wall_base", "d.mould_ext", "d.wall_top", "d.near_wet"}
    # quantity is the whole wall's area (3.0 x 2.4), not the damage footprint
    assert repaint[0]["quantity"]["value"] == pytest.approx(3.0 * 2.4, rel=1e-6)


def test_scope_mould_remediation_uses_damage_area_with_margin(plan, regions):
    items = generate_scope(plan, regions, [])
    mould_items = [i for i in items if i["code"] == "01.74-MOLD-REMED" and i["surface_id"] == "room0.W0"]
    assert len(mould_items) == 1
    expected_area = sum(r["area"]["value"] for r in regions if r["class"] == "mould" and r["surface_id"] == "room0.W0")
    assert mould_items[0]["quantity"]["value"] == pytest.approx(expected_area * 1.3, rel=1e-6)


def test_scope_crack_fill_is_linear(plan, regions):
    items = generate_scope(plan, regions, [])
    crack_items = [i for i in items if i["code"] == "09.21-CRACK-FILL"]
    assert len(crack_items) == 1
    assert crack_items[0]["unit"] == "m"
    assert crack_items[0]["quantity"]["value"] == pytest.approx(0.6, rel=1e-6)


def test_scope_no_hole_items_when_no_hole_regions(plan, regions):
    items = generate_scope(plan, regions, [])
    assert not [i for i in items if i["code"] == "09.21-PATCH"]


def test_scope_concealed_investigation_one_per_flag(plan, regions):
    flags = evaluate_concealed(plan, regions)
    items = generate_scope(plan, regions, flags)
    inspect_items = [i for i in items if i["code"] == "01.32-INVASIVE-INSPECT"]
    assert len(inspect_items) == len(flags)
    for item in inspect_items:
        assert item["quantity"]["value"] == 1.0


def test_scope_item_schema(plan, regions):
    flags = evaluate_concealed(plan, regions)
    items = generate_scope(plan, regions, flags)
    for item in items:
        assert set(item) == {"id", "surface_id", "room_id", "code", "description",
                             "quantity", "unit", "basis", "triggered_by"}
        assert "ci95" in item["quantity"]
