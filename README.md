# scan2plan: phone capture → dimensioned, stitched floor plan

One command per capture turns an iPhone capture (**photos**, **handheld video**, or **LiDAR**) into a dimensioned,
stitched whole-property floor plan. The output covers walls, ceiling heights, floor areas, openings, damage regions,
concealed-damage flags and scope line items. Every number carries a 95% confidence interval, and the JSON follows
[`schema/scan2plan_output.schema.json`](schema/scan2plan_output.schema.json).

> ### ⚠️ Important disclosure: sample data only, no real-time measurements
> **I couldn't use real-time measurements due to device constraints: I have no access to an iPhone (and no LiDAR
> iPhone), so I could not make my own captures or take live tape/laser measurements.** As agreed with the reviewer,
> everything in this repository was developed and benchmarked on the **sample data supplied with the assignment**:
> three StrayScanner LiDAR captures of one apartment (`single_room`, `single_scan_floor_only`, `single_scan_with_ceiling`).
> Consequences, stated plainly:
> - **No tape/laser ground truth exists.** Gates are scored against clearly labelled *proxies*: capture-vs-capture
>   agreement, the LiDAR-tier plan as reference for the video and photo tiers, and split-half for ceilings.
>   Fill `bench/ground_truth/<capture>.yaml` with real measurements and re-run to score against truth.
> - The **video tier** is benchmarked on the RGB video and ARKit poses inside those captures, with the LiDAR depth
>   ignored (the video tier uses the phone's motion tracking, which every iPhone has; see the protocol). A plain-RGB
>   fallback (`--plain-video`) also runs.
>   The **photo tier** is benchmarked on stills extracted from that video (`bench/make_photo_tier.py`).
>   Neither comes from a separate stock-camera capture.
> - **No staged damage** exists in the sample data. Damage detection is validated on synthetic damage composited
>   into real frames (`bench/synthetic_damage.py`) and on false-positive rate on the clean captures.
> - **Head-to-head vs a consumer app (Part 3) was not run**: it needs an iPhone to produce a Polycam or magicplan
>   export of the same rooms. The comparison code is ready for an export.

## Quick start (fresh machine)

```bash
bash scripts/setup.sh                       # Python 3.12 venv via uv, deps, pretrained weights
bash scripts/fetch_data.sh                  # unpack the three sample zips (repo root, or pass their folder) into data/
.venv/bin/scan2plan run data/c00a170fe1     # one command per capture
open out/c00a170fe1/lidar/plan.png          # rendered plan; plan.json is the full contract
```

Measured on a fresh copy of the repo (Apple M4 Pro): setup 41 s and the first LiDAR capture 9 s, with the uv package
cache and Hugging Face weights already on disk. A truly cold machine also downloads roughly 2.5 GB (torch plus model
weights), so allow 5-10 minutes depending on bandwidth.

Input type is detected automatically; override with `--tier`:

| capture | command |
|---|---|
| StrayScanner folder (LiDAR) | `scan2plan run <folder>` |
| video + phone motion: NeRFCapture export (folder or `.zip`, any iPhone) | `scan2plan run <folder or zip>` |
| video, StrayScanner folder with its LiDAR depth ignored (benchmark) | `scan2plan run <folder> --tier video` |
| plain video `.mov` / `.mp4` (RGB only, low accuracy) | `scan2plan run clip.mov` (or `--plain-video`) |
| folder of per-room photo folders | `scan2plan run <parent folder>` |

Useful flags: `--no-drift` (drift ablation), `--no-damage`, `--out DIR`.
Validate any output: `scan2plan validate out/.../plan.json`.

## Results at a glance (sample data; full tables in `reports/benchmark.md`)

| | LiDAR | Video (phone motion) | Photos |
|---|---|---|---|
| rooms found (`c00a170fe1` / `1a8384c3f6` / `c7d28f72c6`) | 3 / 4 / 4 | 3 / 5 / 6 | 3 / 4 / 4 |
| footprint vs LiDAR plan (proxy) | reference | +1.1 / +14.1 / +20.9% | +9.3 / −48.8 / −52.4% |
| median wall-length error vs LiDAR (proxy) | reference | 14.2 / 14.0 / 11.0% | – / 19.9 / – (few walls match) |
| tier gate | – | ±3%: not met | ±8%: not met (closest +9.3%) |
| runtime per capture, first run / cached (M4 Pro) | 74-95 s / 18-40 s (incl. damage) | 57-475 s / 11-31 s | 18-22 s / 14-19 s |

- **Exact ground truth (synthetic apartment, `tests/test_layout_synthetic.py`):** walls within 0.2 mm, door width
  within 1 mm, ceiling within 0.1 mm; drift correction recovers 0.25°/m yaw drift (max wall error 47 → 2.7 mm).
- **Repeatability (two LiDAR captures of the same rooms):** gate not met (0% of walls within 1 cm). One sample capture
  never observed the disputed walls (0 points), and room segmentation is unstable; see `docs/fix_postmortem.md`.
  Interval calibration improved by an order of magnitude (z rms 63 → 3.7 and 31 → 1.8).
- **Openings:** a gap must be seen through by the sensor, and mirrors are detected; phantom openings on
  `c7d28f72c6` dropped from 21 to 13 (3 doors, each seen from both rooms, agree within 5-34 mm).
- **Damage:** 1 false region over 490 m² of clean wall; recall on synthetic staged damage 9.1% (reported, not tuned away).
- **Plans:** final renders and JSON for every capture and tier, including the drift on/off ablation, are in
  `reports/plans/<capture>/<tier>/`.

## Reproduce every reported number

```bash
.venv/bin/python -m bench.run_benchmark     # all captures x all tiers (~40 min) -> reports/benchmark.md + .json
bash scripts/fixloop.sh                     # Part 4 before/after from raw data -> reports/fixloop/
.venv/bin/python -m pytest -q               # 71 unit tests (~1.5 min)
```

The benchmark builds the photo-tier stills itself on first run (`bench/make_photo_tier.py`, deterministic: a regenerated
set is bit-identical). Model outputs (monocular depth, detector boxes) are cached under `.cache/`, keyed by content
hash, and replay deterministically. Delete `.cache/` to force the live path; a live run and a replay produce
bit-identical plans (verified), and every number in `reports/` was regenerated from an empty cache.

## Repository map

| path | what |
|---|---|
| `scan2plan/io/stray.py` | LiDAR tier loader (StrayScanner export) |
| `scan2plan/tiers/posed_video.py` | video tier: RGB + ARKit poses, triangulation-scaled monocular depth |
| `scan2plan/tiers/video.py`, `video_vo.py`, `mono_depth.py` | plain-video fallback (mono depth + visual odometry), shared depth wrapper |
| `scan2plan/tiers/photo.py`, `scan2plan/stitch.py` | photo tier: per-room reconstruction + whole-property stitching |
| `scan2plan/fusion.py`, `manhattan.py`, `maps.py`, `walls.py`, `layout.py` | shared layout backend: rooms, walls, openings, ceilings |
| `scan2plan/drift.py` | drift correction (plane-anchored, pose-graph smoothed) |
| `scan2plan/damage.py`, `concealed.py`, `scope.py`, `rules/` | damage regions, concealed-damage rules, scope line items |
| `scan2plan/output.py`, `render.py`, `schema/` | JSON contract and rendered plan |
| `bench/` | benchmark harness, comparison, fix-loop scoring, test-input generators, debug views |
| `tests/` | 71 tests: synthetic exact-ground-truth geometry, input formats, tiers, stitching, damage rules |
| `docs/capture_protocol.md` | one-page stock-capture protocol (Route 2) |
| `docs/device_matrix.md` | which tier runs on which iPhone, and its accuracy |
| `docs/compliance_matrix.md` | requirement → file → artifact → status |
| `docs/technical_report.md` | technical report (≤ 6 pages) |
| `docs/fix_declaration.md`, `docs/fix_postmortem.md` | Part 4 fix loop |
| `reports/` | benchmark report, timing, fix-loop results, damage and depth-model evaluations, final plans (`reports/plans/`) |

## Third-party models and data (disclosure)

Pretrained models are downloaded once by `scripts/fetch_weights.py` from the Hugging Face hub. They run locally, and
nothing calls the author's infrastructure.

| model | used for | licence |
|---|---|---|
| `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` | monocular depth (video and photo tiers) | Apache-2.0 |
| `apple/DepthPro-hf` | focal-length estimate in the plain-video fallback only | Apple ML Research licence (non-commercial) |
| `google/owlv2-base-patch16-ensemble` | zero-shot damage box proposals | Apache-2.0 |

Capture apps (free, App Store): Stray Scanner for the LiDAR tier, NeRFCapture for the video tier. Their exports are
read directly.
