# Device matrix

| tier | runs on | capture app | inputs used | accuracy it honestly delivers (this repo, sample data) |
|---|---|---|---|---|
| **LiDAR** | iPhone 15 Pro / Pro Max, 16 Pro / Pro Max, 17 Pro / Pro Max; iPad Pro (M-series). Not on non-Pro iPhones, iPhone 16e, iPhone Air. | Stray Scanner (free) | LiDAR depth 256×192 + ARKit confidence, ARKit 6-DoF poses, per-frame intrinsics | Synthetic exact-GT: walls ≤ 0.2 mm, doors ≤ 1 mm, ceiling ≤ 0.1 mm (5 mm depth noise). Real data: where two captures observed the same wall plane, positions agree within 1-10 mm in the best-covered room (median 31 mm over all matched planes); the same door measured from both sides agrees within 5-34 mm; walls hidden behind furniture are flagged `observed: false` (σ 0.25 m). See `reports/benchmark.md`. |
| **Video** (images + phone motion) | any iPhone 15 or newer (all models: ARKit tracking needs no LiDAR) | NeRFCapture (free) | RGB frames + ARKit metric poses + intrinsics; **no depth sensor**. Depth from a monocular model (Depth Anything V2 Metric Indoor Small, Apache-2.0), rescaled per frame by triangulating features between frames with the metric poses, then multi-view consistency filtering | **Does not meet the ±3% gate.** Sample data (StrayScanner RGB + ARKit poses, LiDAR depth ignored): footprint +1.1 / +14.1 / +20.9%; median wall error 14.2 / 14.0 / 11.0%; 0-25% of walls within ±3%. |
| **Video** (plain clip fallback) | any iPhone | built-in Camera | RGB only: monocular depth + visual odometry | Footprint −46 to −76%, walls 37-49% off; intervals calibrated (scale σ 55%). Rough sketch only. |
| **Photos** | any iPhone 15 or newer (all models) | built-in Camera (Photo, 1x) | 2-8 stills per room + doorway photos; EXIF focal length when present; per-photo monocular metric depth, SIFT + depth-lifted registration, doorway photos as cross-room pose and scale anchors | **Does not meet the ±8% gate.** Sample data: footprint +9.3 / −48.8 / −52.4% (`c00a170fe1` / `1a8384c3f6` / `c7d28f72c6`); median wall error 19.9% on the one capture where walls matched; per-room scale error median 59%, bimodal (rooms anchored by a well-registered doorway photo within ~10%, others 70%+). Stitches every room folder into one plan with no overlaps. Intervals calibrated to scale σ 55%. |

Why accuracy drops tier by tier: LiDAR measures metric depth directly and ARKit fuses the IMU, so scale is fixed by
hardware. Video and photos must infer metric scale from a learned monocular depth model, which is the dominant error
term. Photos additionally lack continuity between views, so rooms are registered from a few overlapping stills.

All numbers are from the supplied sample captures (no iPhone available to the author; see README). The video and photo
rows are derived from the RGB stream of those LiDAR captures, not from separate stock-camera captures.
