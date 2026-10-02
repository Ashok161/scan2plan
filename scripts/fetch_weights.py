"""Download every pretrained model the pipeline uses (run once; afterwards everything runs offline)."""
import importlib

for mod, fn in (("scan2plan.tiers.mono_depth", "prefetch_models"),
                ("scan2plan.tiers.video", "prefetch_models"),
                ("scan2plan.damage", "prefetch_damage_models")):
    try:
        f = getattr(importlib.import_module(mod), fn)
    except (ImportError, AttributeError):
        continue
    print(f"prefetch: {mod}.{fn}")
    f()
print("weights ready")
