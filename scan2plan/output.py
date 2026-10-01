"""Plan (+ damage, flags, scope) -> JSON matching schema/scan2plan_output.schema.json."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from . import __version__
from .measure import Measurement
from .plan_types import Plan

SCHEMA_VERSION = "1.0.0"


def _m(x):
    if x is None:
        return None
    if isinstance(x, Measurement):
        return x.to_json()
    return x


def _r(a, nd=4):
    return [round(float(v), nd) for v in np.asarray(a).ravel()]


def plan_to_json(plan: Plan, capture: dict, damage=None, flags=None, scope=None,
                 timings: dict | None = None) -> dict:
    rooms = []
    for r in plan.rooms:
        rooms.append({
            "id": r.id,
            "name": r.name,
            "kind": r.kind,
            "polygon": [_r(v) for v in r.polygon],
            "floor_area": _m(r.area),
            "perimeter": _m(r.perimeter),
            "ceiling_height": _m(r.ceiling_height),
            "ceiling_observed": "NOT OBSERVED" not in r.ceiling_height.method,
            "floor_elevation": round(float(r.floor_y - plan.floor_y), 4),
            "walls": [{
                "id": w.id,
                "start": _r(w.start),
                "end": _r(w.end),
                "normal_in": _r(w.normal_in),
                "length": _m(w.length),
                "height": _m(w.height),
                "area": _m(Measurement(w.length.value * w.height.value,
                                       float(np.hypot(w.length.sigma * w.height.value,
                                                      w.height.sigma * w.length.value)), "m2",
                                       "length x height (gross, openings not deducted)")),
                "coverage": round(float(w.coverage), 3),
                "observed": bool(w.observed),
                "openings": [o.id for o in r.openings if o.wall_id == w.id],
            } for w in r.walls],
            "openings": [{
                "id": o.id,
                "kind": o.kind,
                "wall_id": o.wall_id,
                "center": _r(o.center),
                "width": _m(o.width),
                "height": _m(o.height),
                "sill_height": _m(o.sill),
                "connects": o.connects,
                "evidence": o.evidence,
            } for o in r.openings],
            "surfaces": r.surface_ids(),
            "notes": r.notes,
        })
    out = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": __version__,
        "capture": capture,
        "tier": plan.tier,
        "units": {"length": "m", "area": "m2", "ci": "95% two-sided, value +- 1.96 sigma"},
        "frame": {
            "description": "plan frame: +Y up (gravity), dominant walls along X and Z; plan coordinates are (x, z); "
                           "drawn top-down as (x, -z)",
            "T_align": [_r(row, 6) for row in plan.T_align],
        },
        "rooms": rooms,
        "adjacency": plan.adjacency,
        "stitched": {
            "n_rooms": len(plan.rooms),
            "footprint_area": _m(plan.footprint_area),
            "bbox": _bbox(plan),
        },
        "drift": _jsonable(plan.drift),
        "damage": _jsonable(damage or []),
        "concealed_damage_flags": _jsonable(flags or []),
        "scope": _jsonable(scope or []),
        "warnings": plan.warnings,
        "timings_s": timings or {},
        "environment": {"python": platform.python_version(), "machine": platform.machine(),
                        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
    }
    return out


def _bbox(plan: Plan):
    if not plan.rooms:
        return None
    v = np.vstack([r.polygon for r in plan.rooms])
    return {"min": _r(v.min(0)), "max": _r(v.max(0))}


def _jsonable(o):
    if isinstance(o, Measurement):
        return o.to_json()
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float):
        return round(o, 6)
    return o


def write_json(obj: dict, path: str | Path):
    Path(path).write_text(json.dumps(obj, indent=2))


def validate(obj: dict) -> list[str]:
    """Validate against the published schema; returns a list of error messages."""
    import jsonschema
    schema_path = Path(__file__).resolve().parent.parent / "schema" / "scan2plan_output.schema.json"
    schema = json.loads(schema_path.read_text())
    v = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(map(str, e.path))}: {e.message}" for e in v.iter_errors(obj)]
