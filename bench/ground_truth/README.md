# Ground truth (tape / laser)

None exists for the sample data (no iPhone or site access; see the top-level README).
To score against real measurements, copy `TEMPLATE.yaml` to `<capture_id>.yaml`, fill in laser readings, and run
`python -m bench.run_benchmark --score-only`. Room ids are matched by naming the room as it appears in
`out/<capture>/lidar/plan.png`; walls by the two plan corners they join (read off the rendered plan).
