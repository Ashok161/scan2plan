"""Plan data model shared by every tier and every output stage.

Coordinates: the *plan frame* is gravity-aligned (+Y up) and Manhattan-aligned
(dominant walls along X and Z). Plan coordinates are (x, z) in metres.
`Plan.T_align` maps FrameSet world coordinates into the plan frame.

Surface ids are stable strings used by damage, concealed-damage flags and
scope line items:  "<room_id>.W<k>" (walls), "<room_id>.floor", "<room_id>.ceiling".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .measure import Measurement


@dataclass
class Opening:
    id: str
    kind: str                       # "door" | "passage" | "window"
    wall_id: str
    center: np.ndarray              # (x, z) plan frame
    along: np.ndarray               # unit (x, z) direction along the wall
    width: Measurement
    height: Measurement | None = None
    sill: Measurement | None = None
    connects: list[str] = field(default_factory=list)   # room ids
    evidence: str = ""


@dataclass
class Wall:
    id: str                          # "<room_id>.W<k>"
    start: np.ndarray                # (x, z)
    end: np.ndarray                  # (x, z)
    normal_in: np.ndarray            # (x, z) unit, pointing into the room
    length: Measurement
    height: Measurement
    offset_sigma: float              # 1-sigma of the wall plane position
    coverage: float                  # fraction of the wall length with observed points
    observed: bool = True


@dataclass
class Room:
    id: str
    name: str
    kind: str                        # "room" | "connector"
    polygon: np.ndarray              # Kx2 (x, z), counter-clockwise, interior faces
    floor_y: float
    ceiling_y: float | None
    area: Measurement
    perimeter: Measurement
    ceiling_height: Measurement
    walls: list[Wall] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def surface_ids(self) -> list[str]:
        return [w.id for w in self.walls] + [f"{self.id}.floor", f"{self.id}.ceiling"]


@dataclass
class Plan:
    tier: str
    rooms: list[Room]
    adjacency: list[dict]            # {"rooms": [a, b], "via": opening_id, "kind": ...}
    T_align: np.ndarray              # 4x4 FrameSet world -> plan frame
    floor_y: float
    footprint_area: Measurement | None = None
    drift: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def room(self, rid: str) -> Room:
        return next(r for r in self.rooms if r.id == rid)

    def surfaces(self) -> list[dict]:
        """Planar surfaces in the plan frame, for damage projection.

        Each: {id, room_id, kind, normal (3,), offset d (n.x = d), corners (4x3)}.
        """
        out = []
        for r in self.rooms:
            top = r.ceiling_y if r.ceiling_y is not None else r.floor_y + r.ceiling_height.value
            for w in r.walls:
                n3 = np.array([w.normal_in[0], 0.0, w.normal_in[1]])
                s3 = np.array([w.start[0], 0.0, w.start[1]])
                e3 = np.array([w.end[0], 0.0, w.end[1]])
                corners = np.array([s3 + [0, r.floor_y, 0], e3 + [0, r.floor_y, 0],
                                    e3 + [0, top, 0], s3 + [0, top, 0]])
                out.append({"id": w.id, "room_id": r.id, "kind": "wall", "normal": n3,
                            "offset": float(n3 @ corners[0]), "corners": corners})
            poly3 = np.c_[r.polygon[:, 0], np.full(len(r.polygon), r.floor_y), r.polygon[:, 1]]
            out.append({"id": f"{r.id}.floor", "room_id": r.id, "kind": "floor",
                        "normal": np.array([0, 1.0, 0]), "offset": r.floor_y, "corners": poly3})
            polyc = poly3.copy()
            polyc[:, 1] = top
            out.append({"id": f"{r.id}.ceiling", "room_id": r.id, "kind": "ceiling",
                        "normal": np.array([0, -1.0, 0]), "offset": -top, "corners": polyc})
        return out
