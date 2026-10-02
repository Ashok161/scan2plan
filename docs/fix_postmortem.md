# Fix loop post-mortem (Part 4)

Declaration: `docs/fix_declaration.md` (committed before any fix, git `45beed3`).
Regenerate both runs from raw captures: `bash scripts/fixloop.sh` → `reports/fixloop/before_after.md`.
Readable diff: `git diff fixloop-before -- scan2plan/layout.py scan2plan/maps.py bench/compare.py`. The shipped fix is
the occlusion test in `layout.py` (`FURNITURE_H`, `OCCLUDED_SIGMA`, `UNSEEN_SIGMA`, `_refine_lines`) plus the
registration change in `bench/compare.py`. The same diff also contains later, separately documented changes:
drift correction (technical report §3), the opening tests and ceiling-level rule (§2), and the attempt-1/2 code behind
flags.

## Outcome against the prediction

| | declared prediction | measured after (final run) |
|---|---|---|
| Repeatability pass rate (`1a8384c3f6` vs `c7d28f72c6`) | 12% → 20-35% | **12% → 0%** |
| Median per-wall difference | 234 mm → ≤ 45 mm | **234 mm → 731 mm** |
| Gate | still FAIL | **FAIL** |

**The prediction was badly wrong, because the root-cause hypothesis was wrong.** The declaration blamed
room topology (an algorithm problem: missed door splits, slivers). The evidence gathered while trying to
ship that fix shows the dominant cause is **missing observations in the floor-focused capture**.

## What happened, in order

1. **Attempt 1, as declared** (neck splitting + sliver removal): median got *worse*, 234 → 929 mm.
   A debug view of the segmentation (`python -m bench.debug_seg`) showed a bug in the new code: after the
   morphological opening it kept only the largest piece of each region, deleting whole rooms joined to it
   through a door.
2. **Attempt 2** (bug fixed, plus long low walls into the barrier map): 341-954 mm, still worse than baseline.
3. **Stopped tuning and tested the hypothesis directly** on the simplest matched room, a plain rectangle in
   `c7d28f72c6`. Two of its walls agree across captures within 2 cm. The far wall does not: counting raw
   LiDAR points on the true wall plane (x = 8.83 m), by height band:

   | capture | 0-0.5 | 0.5-1 | 1-1.5 | 1.5-2 | 2-2.5 | 2.5-3 m |
   |---|---|---|---|---|---|---|
   | `1a8384c3f6` (floor-focused) | 0 | 0 | 0 | 0 | 0 | 0 |
   | `c7d28f72c6` | 1,739 | 40,999 | 78,813 | 44,008 | 36,308 | 36,617 |

   The floor-focused capture **never observed that wall**. What it reported as the wall is a furniture face
   at x = 8.55 m, seen only up to 1.5 m. No partitioning algorithm can recover a wall that was not seen.
   The single-room capture `c00a170fe1` shows the same pattern (walls 9-73 cm short, ceiling never observed).

## What was shipped

The real defect in the pipeline was not that it missed those walls (the data cannot support them). It was that it
reported furniture faces as walls with **centimetre intervals**: confident garbage. Shipped:

- **Occlusion-aware wall evidence** (`scan2plan/layout.py`, `FURNITURE_H`, `OCCLUDED_SIGMA`, `UNSEEN_SIGMA`):
  a wall plane whose supporting points never reach 1.6 m is reported `observed: false`, with a 0.25 m sigma
  (true wall at or behind it). An edge with no points at all gets a 0.30 m sigma. Rooms note how many walls are affected.
- **Benchmark fix (disclosed):** registration of two captures used IoU, which cannot register a *partial*
  capture (single room) against a whole-home capture. It now uses the overlap coefficient. Before, the
  `c00a170fe1` vs `c7d28f72c6` pair did not register at all. Both before and after are scored with the new
  scorer in `reports/fixloop/before_after.md`.
- Segmentation changes from attempts 1-2 are kept behind flags (`neck_split`, `low_walls`), **off by default**.

## Measured effect (same scorer, both runs regenerated from raw data, `reports/fixloop/before_after.md`)

| pair | metric | before | after |
|---|---|---|---|
| `1a8384c3f6` vs `c7d28f72c6` | walls within 1 cm / 0.5% | 12% (2/17) | 0% (0/13) |
| | median wall difference | 234 mm | 731 mm |
| | CI95 covers the difference | 18% | 31% |
| | z rms (1.0 = calibrated) | 63.0 | **3.7** |
| `c00a170fe1` vs `c7d28f72c6` | walls within 1 cm / 0.5% | 0% | 0% |
| | CI95 covers the difference | 0% | **67%** |
| | z rms | 30.9 | **1.8** |
| `c00a170fe1` vs `1a8384c3f6` | matched walls | 0 (did not register) | 4, z rms 1.35, CI95 coverage 75% |

**Interpretation.** Interval honesty improved by an order of magnitude on every pair (z rms 63 → 3.7, 31 → 1.8).
The per-wall agreement on the first pair got *worse* in the final run (234 → 731 mm). That change is not from the
fix itself: a later, separately validated drift-correction improvement (synthetic 0.25°/m drift: 47 → 2.7 mm;
real wall-plane spread 19.1 → 14.1 mm) shifted the geometry by millimetres, and room segmentation then split this
apartment differently (6 → 4 rooms in `c7d28f72c6`). That sensitivity is the main open problem (technical report
§8), and it is why the gate is far from passing.

## Why it fell short of the gate, and what would close it

- The gate compares walls the floor-focused and single-room captures **did not observe**. The fix makes the
  output honest about that (z rms 63 → 3.7 and 31 → 1.8; CI95 coverage 0-18% → 31-75%), but it cannot
  create the missing measurements.
- Remaining overconfidence (z rms 3.7 on the first pair) comes from room-topology differences: the same space
  split into different rooms, so "the same wall" is a different polygon edge.
- The remedy is in the capture, not the code: the protocol now requires sweeping every wall floor to ceiling
  and pointing at every ceiling (`docs/capture_protocol.md`, "How to walk", step 2). Two protocol-compliant captures of the same
  rooms would test the gate fairly. The author had no iPhone or site access to make them.
