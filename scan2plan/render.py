"""Rendered floor plan (PNG/SVG) in the style of consumer scanning apps."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Polygon as MplPolygon

from .plan_types import Plan

WALL = "#2b2b2b"
FILL = "#f4efe6"
CONN = "#e8eef4"
DMG = {"water_stain": "#3b7dd8", "mould": "#2e8b57", "mold": "#2e8b57", "crack": "#d62728",
       "hole_or_impact": "#9467bd", "peeling_paint": "#ff7f0e"}


def _to_view(plan: Plan) -> Plan:
    """Plan (x, z) -> drawing (x, -z).

    The plan frame is right-handed with +Y up, so seen from above +Z points
    towards the viewer's bottom; drawing z upwards would mirror the plan.
    """
    import copy
    v = copy.deepcopy(plan)
    flip = np.array([1.0, -1.0])
    for r in v.rooms:
        r.polygon = r.polygon * flip
        for w in r.walls:
            w.start, w.end, w.normal_in = w.start * flip, w.end * flip, w.normal_in * flip
        for op in r.openings:
            op.center, op.along = op.center * flip, op.along * flip
    return v


def render_plan(plan: Plan, path: str, title: str = "", underlay=None, damage: list | None = None,
                dpi: int = 160):
    plan = _to_view(plan)
    rooms = plan.rooms
    if not rooms:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no rooms", ha="center")
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return
    allv = np.vstack([r.polygon for r in rooms])
    lo, hi = allv.min(0) - 0.8, allv.max(0) + 0.8
    span = hi - lo
    fig, ax = plt.subplots(figsize=(max(6, span[0] * 1.3), max(5, span[1] * 1.3)))
    if underlay is not None:
        img, extent = underlay
        extent = (extent[0], extent[1], -extent[3], -extent[2])
        ax.imshow(img[::-1], extent=extent, origin="lower", cmap="Greys", alpha=0.35, zorder=0)
    for r in rooms:
        ax.add_patch(MplPolygon(r.polygon, closed=True, fc=CONN if r.kind == "connector" else FILL,
                                ec="none", zorder=1))
        P = np.vstack([r.polygon, r.polygon[:1]])
        ax.plot(P[:, 0], P[:, 1], color=WALL, lw=3.2, solid_capstyle="projecting", zorder=3)
        c = np.asarray(_label_point(r.polygon))
        ch = r.ceiling_height
        ch_txt = (f"H {ch.value:.2f} m" if "NOT OBSERVED" not in ch.method else "H n/o")
        ax.text(c[0], c[1], f"{r.name}\n{r.area.value:.2f} m²\n±{1.96 * r.area.sigma:.2f}\n{ch_txt}",
                ha="center", va="center", fontsize=8, zorder=6, color="#333")
        for w in r.walls:
            if w.length.value < 0.25:
                continue
            m = 0.5 * (w.start + w.end)
            d = (w.end - w.start) / max(w.length.value, 1e-9)
            pos = m - w.normal_in * 0.22
            ang = np.degrees(np.arctan2(d[1], d[0]))
            if ang > 90 or ang < -90:
                ang += 180
            ax.text(pos[0], pos[1], f"{w.length.value * 100:.0f}", rotation=ang, ha="center", va="center",
                    fontsize=6.5, color="#1f4e79" if w.observed else "#b05050", zorder=6)
        for op in r.openings:
            hw = op.width.value / 2
            a = op.center - op.along * hw
            b = op.center + op.along * hw
            col = "#5aa0d8" if op.kind == "window" else "white"
            ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=4.5, zorder=4, solid_capstyle="butt")
            if op.kind == "door":
                wall = next((w for w in r.walls if w.id == op.wall_id), None)
                if wall is not None:
                    nin = wall.normal_in
                    ang0 = np.degrees(np.arctan2(op.along[1], op.along[0]))
                    side = np.degrees(np.arctan2(nin[1], nin[0]))
                    t1 = side - ang0
                    t1 = (t1 + 180) % 360 - 180
                    th = (ang0, ang0 + t1) if t1 > 0 else (ang0 + t1, ang0)
                    ax.add_patch(Arc(a, 2 * op.width.value, 2 * op.width.value, theta1=th[0], theta2=th[1],
                                     color="#888", lw=0.7, zorder=4))
                    ax.plot([a[0], a[0] + nin[0] * op.width.value], [a[1], a[1] + nin[1] * op.width.value],
                            color="#888", lw=0.7, zorder=4)
            ax.text(op.center[0] + op.along[0] * 0, op.center[1], f"{op.width.value * 100:.0f}",
                    fontsize=5.5, color="#0a6", ha="center", va="center", zorder=7)
    for dmg in damage or []:
        cp = dmg.get("centroid_plan")
        if cp is None:
            continue
        ax.scatter([cp[0]], [-cp[2]], s=60, marker="X", color=DMG.get(dmg.get("class"), "red"), zorder=8)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.axis("off")
    fp = plan.footprint_area
    sub = f"tier: {plan.tier}   rooms: {len(rooms)}"
    if fp is not None:
        sub += f"   net floor area: {fp.value:.2f} m² (95% CI {fp.ci95[0]:.2f}-{fp.ci95[1]:.2f})"
    ax.set_title((title + "\n" if title else "") + sub + "\nwall dimensions in cm (red = wall not observed)",
                 fontsize=9)
    # 1 m scale bar
    ax.plot([lo[0] + 0.3, lo[0] + 1.3], [lo[1] + 0.3, lo[1] + 0.3], color="k", lw=2)
    ax.text(lo[0] + 0.8, lo[1] + 0.4, "1 m", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _label_point(poly: np.ndarray):
    from shapely.geometry import Polygon
    p = Polygon(poly)
    if not p.is_valid:
        p = p.buffer(0)
    pt = p.representative_point() if p.geom_type == "Polygon" else p.centroid
    c = p.centroid
    return (c.x, c.y) if p.contains(c) else (pt.x, pt.y)
