# scan2plan technical report

*≤ 6 pages. All numbers regenerate with `python -m bench.run_benchmark` and `bash scripts/fixloop.sh`.*

> **Data disclosure.** No iPhone and no site access were available (device constraints), so no tape/laser ground
> truth or new captures exist. Everything below uses the three supplied StrayScanner captures of one apartment plus
> a ray-cast synthetic apartment with exact ground truth (`tests/synthetic.py`). Proxy metrics are labelled as such.

## 1. Architecture

```
            ┌ LiDAR: StrayScanner export → depth + ARKit poses + intrinsics ──────────────┐
capture ───>┤ Video: frames + ARKit poses → mono depth scaled by multi-view triangulation  ├──> FrameSet
            └ Photos: per-room folders → mono depth + intra-room registration + room links ┘   (metric depth,
                                                                                               pose, K, +Y up)
FrameSet ──> fuse (keyframes, depth-gradient normals) ──> Manhattan frame (4-fold normal histogram)
         ──> drift correction (§3) ──> floor / ceiling planes ──> 2D evidence maps (wall-band counts,
             tall-structure slabs, ray-carved free space, trajectory) ──> Manhattan wall lines + gaps
         ──> rooms: close door-sized gaps, keep visited components (§2) ──> per room: rectilinear polygon
             snapped to wall lines, every edge re-fitted on raw points, occlusion test, openings
             (jamb-to-jamb), ceiling plane ──> Plan ──> damage → concealed flags → scope ──> JSON + render
```

Design rule: **every tier reduces to the same `FrameSet`** (metric depth, camera-to-world pose, intrinsics, gravity-up
world). One layout backend therefore produces one output contract for all tiers. Tiers differ only in where depth
and poses come from, and in the error model (`frames.TierErrorModel`) that sizes every interval.

## 2. Rooms, walls, openings (layout backend)

- **Wall evidence.** Vertical points (|n_y| < 0.3) in the 1.25–1.95 m band (above furniture, below door heads), plus
  "tall structure" cells with points in ≥ 3 of 5 height slabs, form a barrier map. Manhattan wall lines are 1 cm
  histogram peaks of plane offset per (axis, facing side), split into solid segments along the line.
- **Rooms.** Gaps between collinear segments, or between a segment and a perpendicular wall, are opening candidates.
  Gaps ≤ 1.3 m are closed as doors, but a closure is kept only if it separates two regions. Wider gaps are closed only when
  that cuts off space the camera never entered (glass, windows, views through doorways). Rooms are the visited connected components.
- **Geometry.** The room mask is converted to a rectilinear polygon on a cell complex built from nearby wall lines, so
  edges only appear where walls were seen. Each edge plane is then re-fitted on raw 1 cm-voxel points: outermost
  well-supported plane facing the room, robust mean. Wall length is the distance between the two adjacent planes.
  Ceiling height is a robust ceiling-plane fit minus a floor-plane fit inside the room. Opening widths are measured
  jamb face to jamb face.
- **Openings must be seen through.** A gap in a wall line is an opening only if camera rays actually crossed the
  plane inside it: points beyond the plane whose camera was on the room side, with the ray hitting the gap
  rectangle. This is normalised by the capture's density of points on observed walls. Calibrated against openings
  labelled by eye in the camera frames of `c7d28f72c6`:

  | group | through-ratio |
  |---|---|
  | real walk-through doors | 2.8–6.0 |
  | glass doors / stair voids | 0.4–1.7 |
  | phantoms on unswept or furniture-hidden wall | 0–0.28 |

  Thresholds: opening ≥ 0.3, door ≥ 2.0. **Mirror test:** points seen "through" a mirror are a reflection.
  Reflecting the points on surfaces parallel to the plane back across it lands them on real room points
  (score > 0.7 = mirror). Floor, ceiling and perpendicular walls are excluded, because any vertical reflection
  maps them onto themselves. A duplicate filter merges the same opening found on both faces of one wall.
  Effect on `c7d28f72c6`: 21 reported openings → 13 kept, all blank-wall phantoms labelled by eye removed, 3 doors seen from
  both rooms agreeing within 5–34 mm. One bathroom mirror above a sink scored only 0.28 and survives as a "window".
- **Ceiling level.** With bulkheads or dropped ceilings, the reported height is the level covering the largest plan
  area, not the one with the most points (point count depends on where the camera looked). All levels are listed in
  the room notes.
- **Occlusion test (fix loop, §6).** An edge whose supporting points never reach 1.6 m is `observed: false`, with a
  0.25 m sigma: it is probably a furniture face in front of the wall. An edge with no points gets 0.30 m.

**Exact-ground-truth check** (synthetic two-room apartment, 5 mm depth noise, `tests/test_layout_synthetic.py`):
wall lengths within **0.2 mm**, door width 0.9008 m vs 0.900 m, ceiling 2.5000 m vs 2.500 m, correct door adjacency.
The geometry is sound when the walls are observed; the real-data errors below come from coverage (§6).

**Video tier design.** A plain clip has no scale sensor. Its scale came only from the monocular model, and that
was 26-88% wrong on the sample data (footprint −46 to −76%). The video tier therefore uses the phone's own motion
tracking: every iPhone runs ARKit visual-inertial odometry, whose poses are metric, and NeRFCapture (free, any iPhone)
records them. Monocular depth (Depth Anything V2 Small) is rescaled per keyframe by triangulating SIFT matches between
keyframes with the metric poses (median z_tri / z_mono). Depth pixels that disagree with reprojected neighbours by
more than 6% are dropped. No depth sensor is used. Result: footprint +1 / +14 / +21%, median wall error 11-14%.
Tried and rejected, all measured: COLMAP SfM (registered 9-17% of frames), joint pairwise scale chaining (worse:
31-59%), and Depth Pro (2/5 walls within 3% but footprint +12.6%, 30x slower, non-commercial licence). The plain-RGB
path remains as a fallback.

**Photo tier, tried and rejected (measured):** Depth Pro with the EXIF focal length has the best raw scale (bias +9 to +18% vs
+27 to +43%), but made footprints worse (−78 / −54 / +45%), because registration needs locally consistent depth.
LoFTR dense matching (kornia, Apache-2.0) raised registration from 57% to 62–71% of photos, but footprint error swung
between −8.6% and +84% across captures and thresholds: dense matches on blank walls and tiles are self-consistent
but wrong. Both reverted; the SIFT + Depth Anything V2 Small baseline ships. Final run from raw data with an empty cache:
footprint +9.3 / −48.8 / −52.4%, so the ±8% gate is not met on any capture. Regenerating the photo set is
bit-identical, and live inference now returns exactly what a cache replay returns (float16), so first and repeat runs agree.

## 3. Drift handling

`scan2plan/drift.py`, plane-anchored correction. The trajectory is cut into ~1 m chunks. Each chunk measures four
offsets against globally consistent structure:
- yaw from the 4-fold circular mean of its wall normals, against the Manhattan frame;
- height from its floor points, against the global floor plane;
- x and z from its wall points, against consensus wall planes extracted from the whole capture.

Measurements are weighted by support and solved per degree of freedom as a 1D pose graph with a **curvature**
(second-difference) prior. Accumulated drift is a ramp, which a first-difference prior shrinks; that was a measured
bug (35.6 mm residual vs 2.7 mm). Corrections are interpolated per keyframe along the path, and yaw rotates each
keyframe about its own camera centre. Revisiting a wall from a distant part of the walk acts as a loop closure
through the shared anchor plane. Three iterations.

| check | without | with |
|---|---|---|
| synthetic, injected yaw drift 0.25°/m: max wall error | 47.3 mm | **2.7 mm** |
| synthetic, 0.5°/m | 338 mm | 134 mm (fails: limit) |
| real `c7d28f72c6`: wall-plane spread (robust σ of wall points about their planes) | 20.9 mm | 13.7 mm |
| real `1a8384c3f6`: wall-plane spread | 19.1 mm | 14.1 mm |

The stitched-footprint ablation (on vs off) is in `reports/benchmark.md` (`--no-drift`). "Poses used as-is" is never the default.

## 4. Error budget and intervals

Each wall plane's sigma combines:
- plane-fit standard error, with n_eff = points/25;
- per-plane bias (LiDAR 4 mm: range bias + mixed pixels);
- a coverage penalty × (1 − coverage);
- for occluded or unseen planes, 0.25 m or 0.30 m.

Wall length combines the two adjacent planes and a relative scale term (LiDAR 0.2%). Area and perimeter propagate
edge sigmas. Ceiling height combines both plane fits, √2 × bias and scale. When the ceiling is not observed, the
pipeline does not invent a measurement: it reports a bounded prior (lower bound from the highest wall points, 2.5–3.2 m)
with method text `NOT OBSERVED`. Tier error models (`frames.py`, `tiers/posed_video.py`), so intervals widen as
sensor data thins:

| tier | per-plane bias | length-proportional sigma | source of that sigma |
|---|---|---|---|
| LiDAR | 4 mm | 0.2% | ARKit / LiDAR range scale |
| Video (phone motion) | 20 mm | 18% | measured wall-length error of mono-depth shape (11-14% median) |
| Video (plain clip) | 10 mm | 55% | measured trajectory scale error (26-88%) |
| Photos | 20 mm | 55% | measured per-room scale error (median 59%) |

## 5. Calibration analysis

Calibration is measured by whether the 95% interval of a difference covers the difference between two captures of
the same walls (z = Δ/√(σa² + σb²); calibrated → z_rms ≈ 1, coverage ≈ 95%).

| pair (LiDAR) | CI95 coverage before → after fix | z_rms before → after |
|---|---|---|
| `1a8384c3f6` vs `c7d28f72c6` | 18% → 31% | 63 → 3.7 |
| `c00a170fe1` vs `c7d28f72c6` | 0% → 67% | 31 → 1.8 |
| `c00a170fe1` vs `1a8384c3f6` | not registrable → 75% | – → 1.35 |

Video and photo intervals are calibrated to measured error (video length sigma 18%, photo scale sigma 55%). For the
video tier, the 95% CI covers the LiDAR reference for 75% of matched walls and for all 3 footprints. For photos,
it covers every matched wall, and the footprint on `c00a170fe1` (+9.3%), but not on the two multi-room captures
(−49, −52%), where unregistered photos leave rooms partial: photo footprint intervals there are still
overconfident. The remaining LiDAR overconfidence comes from room-topology differences (§8).

## 6. Fix loop

Full story: `docs/fix_declaration.md` (committed before the fix) and `docs/fix_postmortem.md`.
- **Worst gate:** repeatability, 12% of walls (234 mm median).
- **Declared cause:** room topology. **Prediction:** ≤ 45 mm. **Measured:** 731 mm in the final run (the gate got
  worse after a later drift change re-partitioned the rooms); the prediction was badly wrong.
- **What the evidence showed instead:** the floor-focused capture has **0 points at any height** on the disputed wall
  (the other capture has 238k), and its "wall" is a furniture face seen up to 1.5 m.
- **Shipped:** the occlusion test, which makes intervals honest (z rms 63 → 3.7 and 31 → 1.8), plus a benchmark
  registration fix (disclosed). The real remedy is in the protocol: sweep every wall floor to ceiling.

## 7. Damage, concealed-damage flags, scope

- **Detection.** `google/owlv2-base-patch16-ensemble` (Apache-2.0) proposes zero-shot boxes per class prompt
  (`water_stain`, `mould`, `crack`, `hole_or_impact`; `peeling_paint` off by default). A classical contrast mask refines
  each box against a clean-surface ring.
- **Projection and fusion.** Mask pixels are back-projected through depth and pose onto the nearest plan surface,
  gated at a 6 cm point-to-plane residual. That gate is the defence against mirrors, glass and occluders, on top of
  ARKit confidence. Regions are clustered across views in surface (u, v) metres, and colour consistency across views
  rejects view-dependent reflections and glare. Stains and mould are only allowed on walls and ceilings, so wet-look
  floor tiles are ignored. Extents are Measurements.
- **Operating point.** Chosen on a 1,183-point sweep (`reports/damage/operating_point_sweep.json`): FP ≤ 0.01/m² on the
  clean captures, then maximum recall on 55 **synthetic** staged-damage instances composited into real posed frames
  (the sample data has no staged damage). Result: **1 false region over 490 m² of wall**, synthetic **recall 9.1%,
  precision 62.5%**; holes 33%, mould 8%, stains / cracks 0%. Low recall is reported, not tuned away: phantom repaint
  items in a homeowner report are worse than misses.
- **Concealed-damage flags and scope.** Six YAML rules (`scan2plan/rules/concealed_rules.yaml`), e.g. ceiling stain →
  possible leak above; stain at wall base or near a wet room → wicking / subfloor moisture. Each flag names the rule
  that fired and its triggers. Six scope rules map damage and flags to line items keyed to surface ids, with
  quantities propagated from surface and damage Measurements.

## 8. Known failure modes

- **Room partition is sensitive to geometry.** Small drift-induced shifts change which door gaps exist, so the same
  apartment can come out as 4-8 rooms; even dropping a random 10% of points flips it between 4 and 6. Three
  stabilisers were tried and measured, none helped: a threshold-consensus vote (`maps.segment_consensus`),
  bootstrap bagging of wall-line extraction, and bagging over rebuilt evidence maps (`build_plan(bagging=...)`, off by
  default). This is the main open problem; it hurts repeatability, not per-wall accuracy.
- **Unobserved walls / ceilings** (floor-focused walks): flagged and widened, not recovered.
- **Drift > ~0.5°/m** is not fully corrected.
- **Non-Manhattan walls** are snapped to the dominant axes (warning when the Manhattan score is < 0.5).
- **Closed doors** are not detected (protocol: open all doors). **Curtains** in front of windows hide them.
- **Mirrors / glass**: LiDAR sees through glass and into mirror reflections. The mirror test (§2) removes most
  mirror "openings"; one above a sink, with little parallel structure in its reflection, is missed. Space the camera
  never entered is dropped, and wide gaps into unvisited space are closed as glass/window.
- **Curtained windows** are not detected: the curtain is opaque to LiDAR, so nothing is seen through.
- **Multi-level homes**: `c7d28f72c6` is a duplex with a staircase. A stair void is reported as a window-type
  opening; floors are not separated per storey.
- **Low light**: ARKit tracking and depth confidence degrade. Confidence-2 depth only is used, so coverage falls and
  intervals widen.

**Walk-in readiness (inputs verified end to end, `tests/test_input_formats.py`):** Stray Scanner folders; NeRFCapture
exports as a folder, nested folder or `.zip`, with image orientation derived from ARKit gravity (portrait and landscape
holds); HEIC / JPEG stills at 12 MP with EXIF orientation; `.mov` with a rotation tag. Two real bugs were found this way
and fixed: orientation re-estimated from camera motion, and a fixed 4:3 resize that halved the area of non-4:3 frames.
A from-scratch reproduction found a third: the depth cache stored float16 while the live path returned float32, so a
machine's first run could disagree with its replays (1 vs 3 rooms on `c00a170fe1`). The live path now returns the
quantised value, and live and replayed outputs are verified bit-identical for the video, photo and damage stages.
