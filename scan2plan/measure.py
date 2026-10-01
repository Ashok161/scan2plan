"""Measurements with calibrated confidence intervals.

Every number the pipeline emits is a Measurement: a point value, a 1-sigma
standard error, and a 95% interval. Sigma is built from an explicit error
budget (see docs/technical_report.md, "Error budget") and then multiplied by a
per-tier calibration factor fitted on the benchmark (scan2plan/calibration.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

Z95 = 1.959964


@dataclass
class Measurement:
    value: float
    sigma: float
    unit: str = "m"
    method: str = ""
    # extra terms kept for the error-budget breakdown in the report
    budget: dict = field(default_factory=dict)

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.value - Z95 * self.sigma, self.value + Z95 * self.sigma)

    def to_json(self) -> dict:
        lo, hi = self.ci95
        d = {
            "value": round(self.value, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "sigma": round(self.sigma, 5),
            "unit": self.unit,
        }
        if self.method:
            d["method"] = self.method
        if self.budget:
            d["budget"] = {k: round(v, 5) for k, v in self.budget.items()}
        return d


def combine(*sigmas: float) -> float:
    """Root-sum-square of independent error terms."""
    return math.sqrt(sum(s * s for s in sigmas))


def length_measurement(value: float, offset_sigmas: tuple[float, float], scale_sigma_rel: float,
                       extra: dict | None = None, method: str = "") -> Measurement:
    """Distance between two independently located planes, plus a relative scale term."""
    s_a, s_b = offset_sigmas
    s_scale = abs(value) * scale_sigma_rel
    budget = {"plane_a": s_a, "plane_b": s_b, "scale": s_scale}
    if extra:
        budget.update(extra)
    return Measurement(value, combine(*budget.values()), "m", method, budget)


def area_measurement(value: float, perimeter: float, edge_sigma: float, scale_sigma_rel: float,
                     method: str = "") -> Measurement:
    """Polygon area. Each edge offset error sweeps roughly edge_length*sigma of area."""
    s_edges = perimeter * edge_sigma / math.sqrt(2.0)
    s_scale = 2.0 * abs(value) * scale_sigma_rel
    budget = {"edges": s_edges, "scale": s_scale}
    return Measurement(value, combine(s_edges, s_scale), "m2", method, budget)
