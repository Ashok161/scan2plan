"""Stitch per-room Plans into one whole-property Plan.

`stitch_rooms` is tier-agnostic: it only consumes `Plan` objects (see
`plan_types.py`) and a list of cross-room `links`. It is written primarily
for the photo tier (where no pose/depth continuity exists between rooms at
all, so links are weak and the door-matching fallback matters most), but
nothing here assumes photos specifically.

Placement model: every input Plan is treated as one rigid body (all its
rooms keep whatever relative arrangement they already have). Each Plan gets
exactly one global placement: a translation plus a yaw snapped to a
multiple of 90 degrees (every Plan handed to this function is already
gravity + Manhattan aligned in its own frame, per `plan_types.py`, so a
clean building only ever rotates one room relative to another by a multiple
of a right angle). Two sources of placement, used in this priority order:

  1. **Links.** Each link in `links` is `{"rooms": [name_a, name_b],
     "T_a_from_b": 4x4, "scale": float, "inliers": int, "evidence": str}`,
     exactly what `scan2plan.tiers.photo.load_photo_property` emits, with
     `T_a_from_b` expressed in each room's own FrameSet-*world* frame (not
     the plan frame yet). `names` maps each `room_plans[i]` to the room name
     used in `links` (defaults to `room_plans[i].rooms[0].id`). Links are
     converted into plan-frame SE(2) edges via each Plan's own `T_align`,
     snapped to the nearest right angle, and a maximum-spanning forest
     (edge weight = inlier count) is used to chain poses within each
     connected group of rooms -- exactly as `tiers/photo.py` chains camera
     poses within a room, one level up.
  2. **Door matching (fallback).** Any room/component not reachable through
     a link is attached to the growing placed set by searching over
     (door-sized opening in the placed set) x (door-sized opening in the
     candidate) x (0/90/180/270 degree yaw) for a placement that: faces the
     two matched walls at each other (opposite-pointing normals once
     rotated), offsets them by `wall_thickness`, and produces **zero**
     polygon overlap with everything already placed (shapely, hard
     constraint). Width similarity scores candidates; the best valid one
     wins. A room with no usable door match anywhere is placed at a clear
     offset next to the current footprint (still overlap-free) and flagged
     with a large placement sigma plus a warning -- it is never dropped.

Every placement's uncertainty becomes a per-room entry in
`Plan.drift["placement_sigma_m"]`, folded (as `coverage_penalty`-style
inflation) is left to the caller/report; this module only reports it.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .measure import Measurement, area_measurement
from .plan_types import Opening, Plan, Room, Wall

_YAWS = (0, 90, 180, 270)


# --------------------------------------------------------------- SE(2) helpers


def _rot2(yaw_deg: float) -> np.ndarray:
    th = np.radians(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def _apply_se2(xz: np.ndarray, yaw_deg: float, t: np.ndarray) -> np.ndarray:
    return xz @ _rot2(yaw_deg).T + t


def _compose(yaw1: float, t1: np.ndarray, yaw2: float, t2: np.ndarray):
    """SE(2) composition: apply (yaw2,t2) first, then (yaw1,t1)."""
    yaw = (yaw1 + yaw2) % 360
    t = _rot2(yaw1) @ t2 + t1
    return yaw, t


def _invert(yaw: float, t: np.ndarray):
    yaw_inv = (-yaw) % 360
    t_inv = -_rot2(yaw_inv) @ t
    return yaw_inv, t_inv


def _snap90(yaw_deg: float) -> int:
    return int(round(yaw_deg / 90.0)) % 4 * 90


def _yaw_t_from_T4(T: np.ndarray):
    """Decompose a 4x4 FrameSet-world/plan-frame transform into a snapped
    planar yaw (about +Y) and an (x, z) translation."""
    R = T[:3, :3]
    yaw = np.degrees(np.arctan2(R[2, 0], R[0, 0]))
    t = np.array([T[0, 3], T[2, 3]])
    return _snap90(yaw), t


# ------------------------------------------------------------ geometry transforms


def _transform_room(room: Room, yaw_deg: float, t: np.ndarray, new_id: str) -> Room:
    Rt = _rot2(yaw_deg)
    poly = room.polygon @ Rt.T + t
    walls = []
    for w in room.walls:
        nw = replace(w,
                     id=f"{new_id}.{w.id.split('.')[-1]}",
                     start=Rt @ w.start + t,
                     end=Rt @ w.end + t,
                     normal_in=Rt @ w.normal_in)
        walls.append(nw)
    openings = []
    for o in room.openings:
        no = replace(o,
                     id=f"{new_id}.{o.id.split('.')[-1]}",
                     wall_id=f"{new_id}.{o.wall_id.split('.')[-1]}",
                     center=Rt @ o.center + t,
                     along=Rt @ o.along,
                     connects=list(o.connects))
        openings.append(no)
    return replace(room, id=new_id, polygon=poly, walls=walls, openings=openings)


def _room_polygon2d(room: Room) -> Polygon:
    return Polygon(room.polygon)


# --------------------------------------------------------------- link graph


def _link_edges(room_plans: list[Plan], links: list[dict], names: list[str]):
    name_to_idx = {n: i for i, n in enumerate(names)}
    best: dict[tuple[int, int], dict] = {}
    for link in links:
        rn = link.get("rooms")
        if not rn or len(rn) != 2 or rn[0] not in name_to_idx or rn[1] not in name_to_idx:
            continue
        a, b = name_to_idx[rn[0]], name_to_idx[rn[1]]
        if a == b:
            continue
        T_world = np.asarray(link["T_a_from_b"], dtype=float)
        T_plan = room_plans[a].T_align @ T_world @ np.linalg.inv(room_plans[b].T_align)
        yaw, t = _yaw_t_from_T4(T_plan)
        inliers = int(link.get("inliers", 1))
        sigma = max(0.03, 0.5 / np.sqrt(max(inliers, 1)))
        key = (min(a, b), max(a, b))
        edge = {"i": a, "j": b, "yaw": yaw, "t": t, "inliers": inliers, "sigma": sigma,
               "evidence": link.get("evidence", "link")}
        if key not in best or edge["inliers"] > best[key]["inliers"]:
            best[key] = edge
    return list(best.values())


def _max_spanning_forest(n: int, edges: list[dict]):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    tree = []
    for e in sorted(edges, key=lambda e: -e["inliers"]):
        if union(e["i"], e["j"]):
            tree.append(e)
    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return tree, list(comps.values())


def _chain_se2(comp: list[int], tree_edges: list[dict], root: int):
    members = set(comp)
    local = [e for e in tree_edges if e["i"] in members and e["j"] in members]
    adj: dict[int, list[dict]] = {g: [] for g in comp}
    for e in local:
        adj[e["i"]].append(e)
        adj[e["j"]].append(e)
    yaw = {root: 0.0}
    t = {root: np.zeros(2)}
    sigma = {root: 0.0}
    stack, visited = [root], {root}
    while stack:
        cur = stack.pop()
        for e in adj[cur]:
            other = e["j"] if e["i"] == cur else e["i"]
            if other in visited:
                continue
            visited.add(other)
            if cur == e["i"]:
                oy, ot = _compose(yaw[cur], t[cur], e["yaw"], e["t"])
            else:
                yi, ti = _invert(e["yaw"], e["t"])
                oy, ot = _compose(yaw[cur], t[cur], yi, ti)
            yaw[other] = _snap90(oy)
            t[other] = ot
            sigma[other] = float(np.hypot(sigma[cur], e["sigma"]))
            stack.append(other)
    for g in comp:
        yaw.setdefault(g, 0.0)
        t.setdefault(g, np.zeros(2))
        sigma.setdefault(g, 0.0)
    return yaw, t, sigma


# ---------------------------------------------------------- door-matching fallback


def _door_openings(room: Room):
    return [o for o in room.openings if o.kind in ("door", "passage")]


def _try_attach_component(comp: list[int], base_yaw: dict, base_t: dict,
                          room_plans: list[Plan], placed_union, placed_openings,
                          wall_thickness: float, width_tol: float):
    """Try to attach `comp` (already internally rigid via base_yaw/base_t,
    relative to comp's own root) to the current placed set.

    Returns (score, extra_yaw, extra_t, s_open, c_open) or None. `extra_*` is
    the additional SE(2) to compose on top of each room's base placement.
    """
    best = None
    cand_openings = []
    for gi in comp:
        plan = room_plans[gi]
        for room in plan.rooms:
            for o in _door_openings(room):
                cand_openings.append((gi, room, o))
    if not cand_openings or not placed_openings:
        return None
    for extra_yaw in _YAWS:
        Rt = _rot2(extra_yaw)
        for (gi, c_room, c_open) in cand_openings:
            # candidate opening position in the *base* (pre-attach) frame of comp
            by, bt = base_yaw[gi], base_t[gi]
            c_center = _rot2(by) @ c_open.center + bt
            c_along = _rot2(by) @ c_open.along
            c_center_r = Rt @ c_center
            c_normal = _rot2(by) @ c_room.walls[_wall_index(c_room, c_open)].normal_in
            c_normal_r = Rt @ c_normal
            for (s_room_poly, s_open, s_normal, s_room_id) in placed_openings:
                if abs(s_open.width.value - c_open.width.value) > width_tol:
                    continue
                if float(np.dot(s_normal, c_normal_r)) > -0.6:
                    continue   # must face each other once rotated
                # the candidate room sits on the far side of the placed room's
                # wall from that wall's interior, i.e. *against* s_normal
                target = s_open.center - s_normal * wall_thickness
                extra_t = target - c_center_r
                # build trial polygons for the whole component
                polys = []
                ok = True
                for gj in comp:
                    yj, tj = _compose(extra_yaw, extra_t, base_yaw[gj], base_t[gj])
                    for room in room_plans[gj].rooms:
                        poly = Polygon(_apply_se2(room.polygon, yj, tj))
                        if not poly.is_valid or poly.area < 1e-6:
                            ok = False
                            break
                        polys.append(poly)
                    if not ok:
                        break
                if not ok:
                    continue
                if placed_union is not None:
                    if any(poly.buffer(-0.02).intersects(placed_union) for poly in polys):
                        continue
                score = abs(s_open.width.value - c_open.width.value)
                if best is None or score < best[0]:
                    best = (score, extra_yaw, extra_t, s_room_id, c_room.id)
    return best


def _wall_index(room: Room, opening) -> int:
    for i, w in enumerate(room.walls):
        if w.id == opening.wall_id:
            return i
    return 0


# --------------------------------------------------------------------- public


def stitch_rooms(room_plans: list[Plan], links: list[dict], names: list[str] | None = None,
                 wall_thickness: float = 0.15, width_tol: float = 0.35,
                 room_gap: float = 1.0, seed: int = 0, progress=None) -> Plan:
    """Place `room_plans` into one whole-property Plan.

    `names[i]` is the room name used to match `links[*]["rooms"]` against
    `room_plans[i]`. Defaults to `room_plans[i].rooms[0].name` -- this is the
    integration point with `cli.run`: `layout.build_plan(..., single_room=True,
    room_name=<photo folder name>)` sets `Room.name` to the folder name while
    `Room.id` stays the generic "R1" for every single-room Plan, so `.id`
    cannot be used to recover which photo folder a Plan came from, only
    `.name` can. Every input Plan is kept rigid (its own rooms' relative
    arrangement is untouched); only one global SE(2) placement is solved per
    Plan.
    """
    log = progress or (lambda *_: None)
    n = len(room_plans)
    if n == 0:
        raise ValueError("stitch_rooms: no room plans given")
    if names is None:
        names = [p.rooms[0].name if p.rooms else f"plan{i}" for i, p in enumerate(room_plans)]
    if len(names) != n:
        raise ValueError("stitch_rooms: len(names) must match len(room_plans)")

    warnings: list[str] = []
    edges = _link_edges(room_plans, links, names)
    log(f"stitch: {n} room plan(s), {len(edges)} usable link edge(s) out of {len(links)} link(s)")
    tree, comps = _max_spanning_forest(n, edges)
    comps = [sorted(c) for c in comps]

    base_yaw: dict[int, float] = {}
    base_t: dict[int, np.ndarray] = {}
    sigma_within: dict[int, float] = {}
    for comp in comps:
        root = comp[0]
        yaw, t, sigma = _chain_se2(comp, tree, root)
        base_yaw.update(yaw)
        base_t.update(t)
        sigma_within.update(sigma)

    comps_sorted = sorted(comps, key=lambda c: -len(c))
    placed_global_yaw: dict[int, float] = {}
    placed_global_t: dict[int, np.ndarray] = {}
    placed_sigma: dict[int, float] = {}
    attach_log: list[dict] = []

    seed_comp = comps_sorted[0]
    for gi in seed_comp:
        placed_global_yaw[gi] = base_yaw[gi]
        placed_global_t[gi] = base_t[gi]
        placed_sigma[gi] = sigma_within[gi]
    pending = [c for c in comps_sorted[1:]]

    def placed_union_and_openings():
        polys = []
        openings = []
        for gi, gyaw in placed_global_yaw.items():
            gt = placed_global_t[gi]
            for room in room_plans[gi].rooms:
                xz = _apply_se2(room.polygon, gyaw, gt)
                polys.append(Polygon(xz))
                for o in _door_openings(room):
                    center = _rot2(gyaw) @ o.center + gt
                    normal = _rot2(gyaw) @ room.walls[_wall_index(room, o)].normal_in
                    openings.append((Polygon(xz), o, normal, f"p{gi}.{room.id}"))
        union = unary_union(polys) if polys else None
        return union, openings

    progressed = True
    while pending and progressed:
        progressed = False
        union, openings = placed_union_and_openings()
        attempts = []
        for comp in pending:
            res = _try_attach_component(comp, base_yaw, base_t, room_plans, union, openings,
                                        wall_thickness, width_tol)
            if res is not None:
                attempts.append((res[0], comp, res))
        if not attempts:
            break
        attempts.sort(key=lambda x: x[0])
        _, comp, (score, extra_yaw, extra_t, s_room_id, c_room_id) = attempts[0]
        for gi in comp:
            gy, gt = _compose(extra_yaw, extra_t, base_yaw[gi], base_t[gi])
            placed_global_yaw[gi] = gy
            placed_global_t[gi] = gt
            placed_sigma[gi] = float(np.hypot(sigma_within[gi], 0.5 * width_tol))
        attach_log.append({"component": comp, "via_rooms": [s_room_id, c_room_id],
                           "score": score, "evidence": "door_matching"})
        log(f"stitch: door-matched {[room_plans[gi].rooms[0].name for gi in comp if room_plans[gi].rooms]} "
            f"onto {s_room_id} via {c_room_id} (width diff {score:.3f} m)")
        pending.remove(comp)
        progressed = True

    if pending:
        union, _ = placed_union_and_openings()
        minx, miny, maxx, maxy = (union.bounds if union is not None else (0, 0, 0, 0))
        cursor = maxx + room_gap
        for comp in pending:
            for gi in comp:
                placed_global_yaw[gi] = base_yaw[gi]
                placed_global_t[gi] = base_t[gi] + np.array([cursor, 0.0])
                placed_sigma[gi] = 0.75   # arbitrary placement: large, explicit uncertainty
            names_here = [room_plans[gi].rooms[0].name for gi in comp if room_plans[gi].rooms]
            warnings.append(f"room(s) {names_here} had no link and no matching door opening to the "
                            f"rest of the property; placed at an arbitrary offset with no claimed adjacency")
            log(f"stitch: WARNING room(s) {names_here} placed arbitrarily (no link, no door match)")
            u, _ = placed_union_and_openings()
            cursor = u.bounds[2] + room_gap if u is not None else cursor + room_gap

    merged_rooms: list[Room] = []
    id_map: dict[tuple[int, str], str] = {}
    for gi in range(n):
        gyaw, gt = placed_global_yaw[gi], placed_global_t[gi]
        for room in room_plans[gi].rooms:
            new_id = f"p{gi}.{room.id}"
            id_map[(gi, room.id)] = new_id
            merged_rooms.append(_transform_room(room, gyaw, gt, new_id))

    polys = {r.id: _room_polygon2d(r) for r in merged_rooms}
    ids = list(polys)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            inter = polys[ids[a]].buffer(-1e-6).intersection(polys[ids[b]].buffer(-1e-6))
            if inter.area > 1e-6:
                warnings.append(f"overlap detected between {ids[a]} and {ids[b]} "
                                f"(area {inter.area:.4f} m2) after placement")

    adjacency: list[dict] = []
    seen_adj: set[tuple[str, str]] = set()
    for e in edges:
        a, b = e["i"], e["j"]
        if room_plans[a].rooms and room_plans[b].rooms:
            ra, rb = id_map.get((a, room_plans[a].rooms[0].id)), id_map.get((b, room_plans[b].rooms[0].id))
        else:
            continue
        key = tuple(sorted((ra, rb)))
        if key in seen_adj:
            continue
        seen_adj.add(key)
        via = []
        ra_room = next((r for r in merged_rooms if r.id == ra), None)
        if ra_room and _door_openings(ra_room):
            via = [_door_openings(ra_room)[0].id]
        adjacency.append({"rooms": [ra, rb], "via": via,
                          "kind": "door" if via else "passage", "evidence": e["evidence"]})
    for att in attach_log:
        s_room, c_room = att["via_rooms"]
        key = tuple(sorted((s_room, c_room)))
        if key in seen_adj:
            continue
        seen_adj.add(key)
        c_room_obj = next((r for r in merged_rooms if r.id == c_room), None)
        via = [_door_openings(c_room_obj)[0].id] if c_room_obj and _door_openings(c_room_obj) else []
        adjacency.append({"rooms": [s_room, c_room], "via": via,
                          "kind": "door" if via else "passage", "evidence": "door_matching"})

    footprint = None
    if polys:
        try:
            u = unary_union(list(polys.values()))
            perim = sum(r.perimeter.value for r in merged_rooms)
            footprint = area_measurement(u.area, perim, edge_sigma=0.02, scale_sigma_rel=0.03,
                                         method="union_of_placed_rooms")
        except Exception:
            footprint = None

    floor_y = float(np.mean([r.floor_y for r in merged_rooms])) if merged_rooms else 0.0
    drift = {
        "enabled": True,
        "method": "link_pose_graph+door_matching_fallback",
        "placement_sigma_m": {id_map.get((gi, room_plans[gi].rooms[0].id), f"plan{gi}"): placed_sigma.get(gi, 0.0)
                              for gi in range(n) if room_plans[gi].rooms},
        "n_link_components": len(comps),
        "n_link_edges": len(edges),
        "n_door_matched_attachments": len(attach_log),
        "n_arbitrary_placements": sum(len(c) for c in pending) if pending else 0,
    }

    return Plan(tier=room_plans[0].tier if room_plans else "photo", rooms=merged_rooms,
               adjacency=adjacency, T_align=np.eye(4), floor_y=floor_y, footprint_area=footprint,
               drift=drift, warnings=warnings)
