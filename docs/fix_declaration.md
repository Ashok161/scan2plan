# Fix declaration (Part 4) — written before the fix was implemented

Baseline: git tag `fixloop-before`. Regenerate: `bash scripts/fixloop.sh` (runs both tags on the raw captures).

## 1. Worst-performing gate

**Repeatability (LiDAR tier, `1a8384c3f6` vs `c7d28f72c6`, same rooms, same tier).**
Gate: two captures agree within 1 cm or 0.5% per wall.

| metric (before) | value |
|---|---|
| matched walls passing | **2 / 17 = 12%** |
| median per-wall length difference | **234 mm** |
| rooms matched (IoU ≥ 0.4) | 4 of 6 (IoU 0.81, 0.80, 0.60, 0.46) |

(The opening-width proxy, 0/23, fails for the same reason: openings are only compared inside matched rooms.)

## 2. Root-cause hypothesis and evidence

**Hypothesis:** the gross failures (0.1–1.6 m) come from *room topology*, not measurement: the two captures
partition the same apartment into different rooms, so "the same wall" is a different polygon edge.
Two mechanisms:

1. **Missed door splits in the floor-focused capture.** Door closures need collinear wall segments in the
   1.25–1.95 m band. `1a8384c3f6` rarely looks above 1.25 m, so wall lines break and door-sized gaps never form:
   its region R3 is 32.3 m² where `c7d28f72c6` splits the same space into a 15.7 m² room plus neighbours.
2. **Slivers leaking through unclosed gaps.** Room R2 of `1a8384c3f6` has 14 walls vs 4 in `c7d28f72c6`.
   Its bounding planes agree with `c7d28f72c6` within 1–10 mm on three sides (x −0.762 vs −0.765,
   x 2.244 vs 2.234, z −1.100 vs −1.102), but it extends 1.6 m further along z through a ~0.3 m-wide strip of the
   adjacent corridor.

**Evidence that measurement is not the gross error:** room-level wall planes of matched rooms differ by a median
of 31 mm (18 planes, `reports/fixloop_evidence_planes.json`): far below the 234 mm length error.
A plane-based similarity fit between the captures gives 0.13° rotation and −0.07% scale, so it is
not a registration or scale problem either.

## 3. Fix and prediction

**Fix:** make room partition independent of upper-band wall evidence:
- *neck splitting*: inside each segmented region, distance-transform cores (> 0.45 m from any obstacle) are grown
  geodesically; a split between two cores is accepted only when their shared boundary is door-like
  (≤ 1.3 m) and both new rooms are ≥ 1.5 m²; otherwise the cores merge back;
- *sliver removal*: each room mask is morphologically opened with a 0.5 m disk; strips thinner than that are
  re-assigned to the adjacent room they belong to geometrically, or dropped.

**Predicted after the fix:**
- rooms matched at IoU ≥ 0.7: from 2 to ≥ 4 of 6;
- median per-wall length difference: from 234 mm to **≤ 45 mm**;
- pass rate: from 12% to **20–35%**.

**The gate is predicted to still FAIL.** The remaining ~3 cm plane-level disagreement (median 31 mm) comes from the two
captures observing different surfaces (skirting / furniture faces in the floor-focused walk vs. upper wall in the
other). This fix does not address it. The post-fix report states the measured numbers against this prediction.
