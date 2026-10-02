"""Layout backend against exact ground truth (ray-cast synthetic apartment, tests/synthetic.py)."""
import numpy as np
import pytest

from scan2plan.layout import build_plan
from scan2plan.output import plan_to_json, validate
from tests.synthetic import make_capture


def _wall_errors(plan):
    errs = []
    for r in plan.rooms:
        L = sorted(w.length.value for w in r.walls)
        ref = sorted([4.0, 3.0] * 2) if r.area.value > 10.5 else sorted([3.0] * 4)
        assert len(L) == 4, f"{r.id} should be a rectangle, got {len(L)} walls"
        errs += list(np.abs(np.array(L) - np.array(ref)))
    return np.array(errs)


@pytest.fixture(scope="module")
def clean():
    fs, gt = make_capture(noise=0.005)
    return build_plan(fs, drift_correction=False), gt


def test_two_rooms_and_door(clean):
    plan, gt = clean
    assert len(plan.rooms) == 2
    assert len(plan.adjacency) == 1 and plan.adjacency[0]["kind"] == "door"


def test_wall_lengths_mm_accurate(clean):
    plan, _ = clean
    assert _wall_errors(plan).max() < 0.005


def test_door_width_and_ceiling(clean):
    plan, gt = clean
    doors = [o for r in plan.rooms for o in r.openings if o.kind == "door"]
    assert doors and all(abs(o.width.value - gt["door"]) < 0.01 for o in doors)
    for r in plan.rooms:
        assert abs(r.ceiling_height.value - gt["ceiling"]) < 0.005


def test_intervals_cover_truth(clean):
    plan, gt = clean
    for r in plan.rooms:
        lo, hi = r.ceiling_height.ci95
        assert lo <= gt["ceiling"] <= hi


def test_json_validates(clean):
    plan, _ = clean
    doc = plan_to_json(plan, {"id": "synthetic", "tier": "lidar", "source": "synthetic"})
    assert validate(doc) == []


def test_drift_correction_recovers_yaw_drift():
    fs, _ = make_capture(yaw_drift_deg_per_m=0.25)
    off = _wall_errors(build_plan(fs, drift_correction=False)).max()
    on = _wall_errors(build_plan(fs, drift_correction=True)).max()
    assert off > 0.03 and on < 0.01, (off, on)
