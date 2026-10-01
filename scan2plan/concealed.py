"""Concealed-damage rule engine.

Takes the visible damage regions from `damage.detect_damage` plus the Plan
they were measured against, and fires the rule table in
`scan2plan/rules/concealed_rules.yaml` to produce concealed-damage flags:
things a visual-only scan cannot confirm (a leak behind a wall, wicking
under a floor, condensation inside a cavity) but can reasonably infer from
where and what kind of visible damage shows up.

Rule conditions are plain Python boolean expressions evaluated with
`eval(expr, {"__builtins__": {}}, context)` -- no builtins are exposed, so a
rule can only read the documented context keys below; it cannot import,
call arbitrary functions, or touch the filesystem. A malformed condition
raises at rule-load time (fail loud, not fail open).

Two rule scopes:
  - "region": evaluated once per damage region, context keys documented in
    `_build_region_context`.
  - "pair": evaluated once per *unordered* pair of regions in the same room
    (capped, see `_MAX_PAIR_REGIONS`), context is {"a": ctx_a, "b": ctx_b,
    "centroid_dist_m": ...} where ctx_a/ctx_b expose the same keys as the
    region context via attribute access.

Heuristics called out explicitly because the Plan type does not yet carry
an authoritative flag for them:
  - exterior-facing wall: a wall with no opening that connects this room to
    another interior room is treated as exterior-facing. A window-only or
    blank exterior wall satisfies this; an interior partition with a door
    does not. This is wrong for a blank interior partition (no door at all)
    -- flagged as a known limitation in the report.
  - wet room: a room whose name/id contains a wet-room keyword
    (bathroom/toilet/wc/ensuite/kitchen/laundry/utility). Depends entirely
    on upstream room naming; if the Plan uses generic names ("Room 2") this
    heuristic never fires and the corresponding rule simply does not apply.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .plan_types import Plan

_RULES_PATH = Path(__file__).parent / "rules" / "concealed_rules.yaml"
_WET_ROOM_KEYWORDS = ("bath", "toilet", "wc", "ensuite", "en-suite", "kitchen", "laundry", "utility", "shower")
_NEAR_WET_ROOM_M = 2.0
_MAX_PAIR_REGIONS = 400   # guard against O(n^2) blowup; benchmark captures are far below this


def load_rules(path: str | Path = _RULES_PATH) -> list[dict]:
    rules = yaml.safe_load(Path(path).read_text())
    for r in rules:
        compile(r["condition"], f"<rule {r['id']}>", "eval")   # fail loud on a bad rule
    return rules


class _Ctx:
    """Attribute-access view of a context dict, for pair-rule a.foo / b.foo."""
    def __init__(self, d: dict):
        self._d = d

    def __getattr__(self, k):
        try:
            return self._d[k]
        except KeyError as e:
            raise AttributeError(k) from e


def _is_exterior_wall(plan: Plan, room, wall) -> bool:
    for op in room.openings:
        if op.wall_id == wall.id and len(op.connects) >= 2:
            return False
    return True


def _is_wet_room(room) -> bool:
    name = f"{room.id} {room.name}".lower()
    return any(k in name for k in _WET_ROOM_KEYWORDS)


def _nearest_wet_room_dist(plan: Plan, point_xz: np.ndarray, this_room_id: str) -> float:
    best = float("inf")
    for r in plan.rooms:
        if r.id == this_room_id or not _is_wet_room(r):
            continue
        c = r.polygon.mean(axis=0)
        best = min(best, float(np.linalg.norm(point_xz - c)))
    return best


def _nearest_opening_corner_dist(room, point_xz: np.ndarray) -> float:
    best = float("inf")
    for op in room.openings:
        half = op.width.value / 2.0
        for sign in (-1, 1):
            corner = op.center + sign * half * op.along
            best = min(best, float(np.linalg.norm(point_xz - corner)))
    return best


def _build_region_context(region: dict, plan: Plan) -> dict:
    room = plan.room(region["room_id"])
    surface = next(s for s in plan.surfaces() if s["id"] == region["surface_id"])
    kind = surface["kind"]
    cx, cy, cz = region["centroid_plan"]
    point_xz = np.array([cx, cz])

    height_above_floor = cy - room.floor_y
    is_exterior = False
    diag_from_corner = False
    if kind == "wall":
        wall = next((w for w in room.walls if w.id == region["surface_id"]), None)
        if wall is not None:
            is_exterior = _is_exterior_wall(plan, room, wall)
        dcorner = _nearest_opening_corner_dist(room, point_xz)
        diag_from_corner = dcorner < 0.5

    dwet = _nearest_wet_room_dist(plan, point_xz, room.id)
    width_m = region["extent"]["width"]["value"] if isinstance(region["extent"]["width"], dict) else region["extent"]["width"].value
    height_m = region["extent"]["height"]["value"] if isinstance(region["extent"]["height"], dict) else region["extent"]["height"].value

    return {
        "cls": region["class"],
        "confidence": region["confidence"],
        "surface_kind": kind,
        "surface_id": region["surface_id"],
        "room_id": region["room_id"],
        "width_m": width_m,
        "height_m": height_m,
        "length_m": max(width_m, height_m),
        "height_above_floor_m": height_above_floor,
        "is_exterior_wall": is_exterior,
        "near_wet_room": dwet < _NEAR_WET_ROOM_M,
        "wet_room_dist_m": dwet,
        "diagonal_from_opening_corner": diag_from_corner,
        "centroid_xz": point_xz,
    }


def evaluate_concealed(plan: Plan, damage_regions: list[dict], rules: list[dict] | None = None) -> list[dict]:
    rules = rules if rules is not None else load_rules()
    contexts = {r["id"]: _build_region_context(r, plan) for r in damage_regions}
    flags = []
    counter = 0

    region_rules = [r for r in rules if r.get("scope", "region") == "region"]
    pair_rules = [r for r in rules if r.get("scope") == "pair"]

    for region in damage_regions:
        ctx = contexts[region["id"]]
        for rule in region_rules:
            if eval(rule["condition"], {"__builtins__": {}}, ctx):   # noqa: S307 (no builtins exposed)
                counter += 1
                flags.append(_make_flag(counter, rule, [region["id"]], region["surface_id"]))

    if pair_rules:
        by_room: dict[str, list[dict]] = {}
        for region in damage_regions:
            by_room.setdefault(region["room_id"], []).append(region)
        for room_id, regions in by_room.items():
            if len(regions) > _MAX_PAIR_REGIONS:
                regions = regions[:_MAX_PAIR_REGIONS]
            for i in range(len(regions)):
                for j in range(i + 1, len(regions)):
                    a_region, b_region = regions[i], regions[j]
                    a_ctx, b_ctx = contexts[a_region["id"]], contexts[b_region["id"]]
                    dist = float(np.linalg.norm(a_ctx["centroid_xz"] - b_ctx["centroid_xz"]))
                    pctx = {"a": _Ctx(a_ctx), "b": _Ctx(b_ctx), "centroid_dist_m": dist}
                    for rule in pair_rules:
                        if eval(rule["condition"], {"__builtins__": {}}, pctx):   # noqa: S307
                            counter += 1
                            flags.append(_make_flag(counter, rule, [a_region["id"], b_region["id"]],
                                                    a_region["surface_id"]))
    return flags


def _make_flag(counter: int, rule: dict, triggered_by: list[str], surface_id: str) -> dict:
    return {
        "id": f"concealed.{counter:04d}",
        "surface_id": surface_id,
        "rule_id": rule["id"],
        "rule": rule["description"],
        "flag": rule["flag"],
        "severity": rule["severity"],
        "triggered_by": triggered_by,
        "recommendation": rule["recommended_investigation"].strip(),
    }
