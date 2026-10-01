"""Monocular metric depth for the photo tier, with a disk cache.

Private to the photo tier. The video tier is expected to grow a sibling
`scan2plan/tiers/mono_depth.py` with the same `predict_depth(rgb_uint8,
model_id, cache_key) -> (depth_m, focal_px_or_None)` signature; the two should
be unified into one shared wrapper once that file exists (same cache format,
same model loader, same device logic) rather than kept as two copies.

Default model: "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
(transformers `depth-estimation` pipeline). It is small (~100 MB), fast on
CPU/MPS, outputs metric depth directly in metres (trained on indoor scenes,
a good match for room photos) and needs no network access once cached.
Depth Anything does not estimate focal length, so `focal_px` is None here;
callers fall back to EXIF or a field-of-view heuristic.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import numpy as np

DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

_pipe_cache: dict[tuple[str, str], object] = {}
_pipe_lock = threading.Lock()


def _pick_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _get_pipeline(model_id: str, device: str):
    key = (model_id, device)
    with _pipe_lock:
        pipe = _pipe_cache.get(key)
        if pipe is not None:
            return pipe
        from transformers import pipeline
        pipe = pipeline("depth-estimation", model=model_id, device=device)
        _pipe_cache[key] = pipe
        return pipe


def weights_available(model_id: str = DEFAULT_MODEL_ID) -> bool:
    """Best-effort check for whether model weights can be loaded (cached locally,
    or network reachable). Used by tests to skip gracefully."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(model_id, allow_patterns=["config.json"])
        return True
    except Exception:
        return False


def _cache_path(cache_dir: Path, model_id: str, cache_key: str) -> Path:
    h = hashlib.sha1(f"{model_id}:{cache_key}".encode()).hexdigest()[:20]
    return cache_dir / "photo_depth" / f"{h}.npz"


def predict_depth(rgb_uint8: np.ndarray, model_id: str = DEFAULT_MODEL_ID,
                  cache_key: str | None = None, cache_dir: str | Path = ".cache",
                  device: str | None = None):
    """RGB uint8 HxWx3 -> (depth_m HxW float32, focal_px_or_None).

    Disk-cached by (model_id, cache_key). If `cache_key` is None the result is
    not cached. Depth is resized to the input resolution if the model's
    native output differs.
    """
    cache_dir = Path(cache_dir)
    cpath = None
    if cache_key is not None:
        cpath = _cache_path(cache_dir, model_id, cache_key)
        if cpath.exists():
            z = np.load(cpath)
            focal = float(z["focal_px"]) if "focal_px" in z and np.isfinite(z["focal_px"]) else None
            return z["depth_m"].astype(np.float32), focal

    from PIL import Image
    device = _pick_device(device)
    pipe = _get_pipeline(model_id, device)
    img = Image.fromarray(rgb_uint8)
    out = pipe(img)
    depth = np.asarray(out["predicted_depth"], dtype=np.float32)
    # torch tensor path: predicted_depth may be a tensor; pipeline already
    # resizes "depth" (PIL image) to input size, mirror that for the array.
    if hasattr(out["predicted_depth"], "detach"):
        depth = out["predicted_depth"].detach().cpu().numpy().astype(np.float32)
    h, w = rgb_uint8.shape[:2]
    if depth.shape != (h, w):
        import cv2
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    focal_px = None  # Depth Anything V2 does not estimate intrinsics

    if cpath is not None:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cpath, depth_m=depth, focal_px=np.float32(focal_px if focal_px else np.nan))
    return depth, focal_px
