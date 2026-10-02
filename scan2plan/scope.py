"""Scope-of-work line items keyed to surfaces.

Turns damage regions (`damage.detect_damage`) and concealed-damage flags
(`concealed.evaluate_concealed`) into quantified repair line items, driven
by the rule table in `scan2plan/rules/scope_rules.yaml`. One line item is
emitted per (surface, rule) pair rather than per damage region: a wall with
three separate mould blobs gets one "mould remediation" line covering all
three, which is how a contractor actually quotes it. `triggered_by` lists
every damage id (or concealed-flag id) folded into that line.

Quantities are Measurements. For "surface_area" quantities the value comes
straight from the Plan surface's own dimensions (Wall.length x Wall.height,
or Room.area for floor/ceiling), independent of the damage detector's own
extent uncertainty -- the repair covers the whole surface, so its quantity
error is the Plan's wall/floor measurement error, not the damage mask's.
For damage-derived quantities (area_margin, length, count) the uncertainty
is propagated in quadrature from the contributing Measurement objects.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .measure import Measurement, combine
from .plan_types import Plan

_RULES_PATH = Path(__file__).parent / "rules" / "scope_rules.yaml"


def load_scope_rules(path: str | Path = _RULES_PATH) -> list[dict]:
    return yaml.safe_load(Path(path).read_text())


def _measurement_from_json(d) -> Measurement:
    if isinstance(d, Measurement):
        return d
    lo, hi = d["ci95"]
    return Measurement(d["value"], d["sigma"], d.get("unit", "m"), d.get("method", ""))


def _surface_area_measurement(plan: Plan, surface: dict) -> Measurement:
    room = plan.room(surface["room_id"])
    if surface["kind"] == "wall":
        wall = next(w for w in room.walls if w.id == surface["id"])
        L, H = wall.length, wall.height
        value = L.value * H.value
        rel = combine(L.sigma / max(L.value, 1e-6), H.sigma / max(H.value, 1e-6))
        return Measurement(value, value * rel, "m2", method="wall.length * wall.height")
    # floor and ceiling: Room.area is measured on the floor polygon; the
    # ceiling is assumed to share the same footprint (flagged as an
    # assumption -- a stepped ceiling would break this).
    a = room.area
    return Measurement(a.value, a.sigma, "m2",
                       method="room.area" + ("" if surface["kind"] == "floor" else " (ceiling := floor footprint)"))


def _sum_measurements(ms: list[Measurement], scale: float = 1.0, unit: str = "m2") -> Measurement:
    value = scale * sum(m.value for m in ms)
    sigma = scale * combine(*[m.sigma for m in ms]) if ms else 0.0
    return Measurement(value, sigma, unit, method=f"sum of {len(ms)} region measurement(s) x {scale}")


def build_scope(dmg: list[dict], flags: list[dict], plan: Plan,
                rules_path: str | Path = _RULES_PATH) -> list[dict]:
    """CLI entry point: `from .scope import build_scope; build_scope(dmg, flags, plan)`.

    Thin argument-order wrapper around `generate_scope` (plan-first, matching
    this module's other helpers, which need `plan` to resolve surface
    dimensions).
    """
    return generate_scope(plan, dmg, flags, rules_path)


def generate_scope(plan: Plan, damage_regions: list[dict], concealed_flags: list[dict],
                   rules_path: str | Path = _RULES_PATH) -> list[dict]:
    rules = load_scope_rules(rules_path)
    surfaces_by_id = {s["id"]: s for s in plan.surfaces()}
    items = []
    counter = 0

    for rule in rules:
        if rule["applies_to"] == "damage":
            matched = [d for d in damage_regions
                      if d["class"] in rule["classes"]
                      and surfaces_by_id[d["surface_id"]]["kind"] in rule["surface_kinds"]
                      and not d.get("low_confidence", False)]
            by_surface: dict[str, list[dict]] = {}
            for d in matched:
                by_surface.setdefault(d["surface_id"], []).append(d)
            for surface_id, regions in by_surface.items():
                surface = surfaces_by_id[surface_id]
                qty = _quantity(rule, plan, surface, regions)
                counter += 1
                items.append({
                    "id": f"scope.{counter:04d}",
                    "surface_id": surface_id,
                    "room_id": surface["room_id"],
                    "code": rule["code"],
                    "description": rule["description"],
                    "quantity": qty.to_json(),
                    "unit": rule["unit"],
                    "basis": rule["basis"],
                    "triggered_by": [d["id"] for d in regions],
                })
        elif rule["applies_to"] == "concealed":
            for flag in concealed_flags:
                counter += 1
                items.append({
                    "id": f"scope.{counter:04d}",
                    "surface_id": flag["surface_id"],
                    "room_id": next((s["room_id"] for s in surfaces_by_id.values()
                                    if s["id"] == flag["surface_id"]), None),
                    "code": rule["code"],
                    "description": rule["description"],
                    "quantity": Measurement(1.0, 0.0, "count", method="one allowance per concealed flag").to_json(),
                    "unit": rule["unit"],
                    "basis": rule["basis"],
                    "triggered_by": [flag["id"]],
                })
    return items


def _quantity(rule: dict, plan: Plan, surface: dict, regions: list[dict]) -> Measurement:
    basis = rule["quantity_basis"]
    if basis == "surface_area":
        return _surface_area_measurement(plan, surface)
    if basis == "damage_area_margin":
        areas = [_measurement_from_json(d["area"]) for d in regions]
        return _sum_measurements(areas, scale=rule.get("margin", 1.0), unit="m2")
    if basis == "damage_length":
        lengths = []
        for d in regions:
            w = _measurement_from_json(d["extent"]["width"])
            h = _measurement_from_json(d["extent"]["height"])
            lengths.append(w if w.value >= h.value else h)
        return _sum_measurements(lengths, unit="m")
    if basis == "count":
        return Measurement(float(len(regions)), 0.0, "count", method="count of matched damage regions")
    raise ValueError(f"unknown quantity_basis {basis!r}")
