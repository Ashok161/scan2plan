# Compliance matrix

Requirement → file path → artifact → status. Status key: **Done**, **Partial** (built, gate not met or proxy only),
**Blocked** (needs an iPhone / site access the author did not have, see README disclosure), **Fail** (measured, gate not met).

## Part 1: capture

| requirement | file | artifact | status |
|---|---|---|---|
| Capture route (Route 2: stock apps + one-page protocol) | `docs/capture_protocol.md` | protocol for all three tiers | Done |
| Photo tier: 2-8 stills per room, no depth/poses, per-room folders | `scan2plan/tiers/photo.py` | `scan2plan run <parent folder>` | Partial (footprint +9.3 / −49 / −52%) |
| Photo folders → one stitched whole-property plan | `scan2plan/stitch.py` | `reports/plans/*/photo/plan.json` | Partial |
| Video tier: handheld walkthrough, any iPhone 15+ | `scan2plan/tiers/posed_video.py` (RGB + ARKit poses via NeRFCapture); fallback `video.py` | `scan2plan run <NeRFCapture folder>` / `clip.mov` | Partial (footprint +1 to +21%, walls 11-14%; gate ±3% not met) |
| LiDAR tier: depth + poses + intrinsics | `scan2plan/io/stray.py` | `scan2plan run <StrayScanner folder>` | Done |
| Same output contract from every tier, intervals widen | `scan2plan/frames.py` (tier error models), `scan2plan/layout.py` | one schema for all tiers | Done |
| Device matrix | `docs/device_matrix.md` | tier × hardware × accuracy | Done (accuracy from proxies) |

## Part 2: output contract and gates

| requirement | file | artifact | status |
|---|---|---|---|
| Per-room plan: walls, ceiling height, floor area, openings | `scan2plan/layout.py` | `rooms[]` in plan.json | Done |
| Stitched multi-room plan with adjacency | `scan2plan/layout.py`, `scan2plan/stitch.py` | `adjacency[]`, `stitched` | Done (LiDAR), Partial (photo/video) |
| Per-surface damage regions with class and metric extent | `scan2plan/damage.py` | `damage[]` | Done (validated on synthetic damage only) |
| Concealed-damage flags with the rule that fired | `scan2plan/concealed.py`, `scan2plan/rules/concealed_rules.yaml` | `concealed_damage_flags[]` | Done |
| Scope line items keyed to surfaces | `scan2plan/scope.py`, `scan2plan/rules/scope_rules.yaml` | `scope[]` | Done |
| Confidence interval on every measurement | `scan2plan/measure.py` | every number is `{value, ci95, sigma}` | Done |
| One command per capture | `scan2plan/cli.py` | `scan2plan run <capture>` | Done |
| JSON to the published schema | `schema/scan2plan_output.schema.json`, `scan2plan/output.py` | validated on every run | Done |
| Rendered plan | `scan2plan/render.py` | `plan.png`, `plan.svg`; final renders in `reports/plans/<capture>/<tier>/` | Done |
| Benchmark: multi-room ≥ 3 rooms + connector | `data/1a8384c3f6`, `data/c7d28f72c6` | supplied sample captures | Done (sample data) |
| Benchmark: furnished room with staged damage, 2 classes | `bench/synthetic_damage.py` | synthetic damage on real frames | Blocked (no staged capture), synthetic substitute |
| Benchmark: same rooms at all three tiers | `bench/make_photo_tier.py`; video = RGB + ARKit poses of the captures, LiDAR depth ignored | `data/photo_tier/`, `reports/plans/*/{video,photo}` | Partial (derived from LiDAR captures) |
| Benchmark: one room captured twice, same tier | `c00a170fe1` vs `c7d28f72c6` | `reports/benchmark.md` repeatability | Done |
| Laser / tape ground truth | `bench/ground_truth/TEMPLATE.yaml` | template + scorer | Blocked |
| Gate: opening widths ≤ 2 cm on ≥ 85% | `scan2plan/layout.py` (`ThroughTester`, mirror test), `bench/run_benchmark.py` | proxy: capture-vs-capture | Fail (proxy). Synthetic door 0.8 mm; phantom openings cut 21 → 13 on `c7d28f72c6`; the same door seen from both sides agrees within 5-34 mm |
| Gate: ceiling height ≤ 1.5 cm; spread ≤ 1 cm | `bench/ceiling_split.py` | split-half proxy | Fail (proxy) / tape Blocked |
| Gate: repeatability 1 cm / 0.5% per wall | `bench/run_benchmark.py` | repeatability table | Fail (see fix loop) |
| Gate: drift accountability + on/off ablation | `scan2plan/drift.py`, `--no-drift` | ablation rows in `reports/benchmark.md`; renders `reports/plans/*/lidar_nodrift/` | Done |
| Gate: photo-tier whole-property stitch | `scan2plan/stitch.py` | `reports/plans/*/photo` | Partial: one stitched plan per capture, room counts 3/4/4 match LiDAR; footprint +9.3 / −49 / −52%, outside ±8% on all three |
| Photo ±8% / video ±3% wall lengths, calibrated | `bench/run_benchmark.py` | tier vs LiDAR rows | Fail (proxy) |

## Part 3: head-to-head

| requirement | file | artifact | status |
|---|---|---|---|
| LiDAR tier vs consumer app on 2 rooms, export submitted | `bench/compare.py` (plan comparison ready) | – | Blocked (no iPhone to produce a Polycam/magicplan export) |

## Part 4: fix loop

| requirement | file | artifact | status |
|---|---|---|---|
| Fix declaration: worst gate, root cause + evidence, fix + prediction | `docs/fix_declaration.md` | committed before the fix | Done |
| Shipped fix, before/after regenerable, readable diff | `scripts/fixloop.sh`, `bench/fixloop_compare.py`, tag `fixloop-before` | `reports/fixloop/before_after.md` | Done |
| Post-mortem | `docs/fix_postmortem.md` | prediction badly wrong; calibration fixed, gate not | Done |

## Part 5 and deliverables

| requirement | file | artifact | status |
|---|---|---|---|
| Commit as you work | git history | 8 commits + tag `fixloop-before` | Partial: work after the fix-loop attempts is not yet committed |
| README: fresh capture running in < 15 min | `README.md`, `scripts/setup.sh` | | Done |
| Reproduction bundle: regenerate every number from raw inputs | `bench/run_benchmark.py`, `scripts/*.sh`, `.cache/` replay | | Done |
| Benchmark report: gates at all tiers, repeatability, head-to-head, timing | `reports/benchmark.md` | | Partial (head-to-head blocked) |
| Technical report ≤ 6 pages | `docs/technical_report.md` | | Done |
| Raw benchmark data: sensor logs, ground truth, app exports | `scripts/fetch_data.sh` (sample zips) | | Partial (no GT, no app export) |
| Weights fetched by script; no own infrastructure | `scripts/fetch_weights.py` | Hugging Face hub downloads | Done |
| Walk-in readiness: all three tiers run cold on real iPhone formats | `tests/test_input_formats.py`, `bench/make_nerfcapture.py`, `bench/make_iphone_photos.py` | Stray Scanner folder, NeRFCapture folder / zip (portrait and landscape), HEIC / JPEG 12 MP with EXIF rotation, rotation-tagged `.mov` all run end to end | Done |
| Mirrors, glass, wet-look surfaces, low light covered | `docs/technical_report.md` §2 (mirror test), §7 (damage), §8 (failure modes); protocol "Avoid" | | Done (handled and documented) |
