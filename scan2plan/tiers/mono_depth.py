"""Reusable monocular metric-depth wrapper, shared by the video and photo tiers.

Wraps HuggingFace `transformers` depth-estimation models behind a single
``predict_depth`` call with deterministic disk caching, so repeated runs over
the same video/photo replay without re-running the network, and so the photo
tier (built separately) can reuse the exact same model-loading / caching code.

Model choice is a tier decision (see tiers/video.py); this module only knows
how to run whichever model id it is given and cache the result.
"""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Known model ids and their licences (recorded honestly in FrameSet.meta).
# ---------------------------------------------------------------------------
DEPTH_ANYTHING_SMALL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DEPTH_ANYTHING_BASE = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
DEPTH_PRO = "apple/DepthPro-hf"

MODEL_LICENSES = {
    DEPTH_ANYTHING_SMALL: "Apache-2.0",
    DEPTH_ANYTHING_BASE: "CC-BY-NC-4.0 (non-commercial)",
    DEPTH_PRO: "apple-amlr (Apple ML Research License, non-commercial/research use only)",
}

_lock = threading.Lock()
_MODEL_CACHE: dict[tuple[str, str], tuple] = {}


def _device(device: str | None) -> str:
    if device:
        return device
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_model(model_id: str, device: str):
    key = (model_id, device)
    with _lock:
        if key not in _MODEL_CACHE:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            torch.manual_seed(0)
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
            _MODEL_CACHE[key] = (processor, model)
        return _MODEL_CACHE[key]


def _safe_model_dir(model_id: str) -> str:
    return model_id.replace("/", "__")


def _cache_path(cache_dir: str | Path, model_id: str, cache_key: str) -> Path:
    return Path(cache_dir) / "mono_depth" / _safe_model_dir(model_id) / f"{cache_key}.npz"


def predict_depth(
    rgb_uint8: np.ndarray,
    model_id: str,
    cache_key: str,
    cache_dir: str | Path = ".cache",
    device: str | None = None,
    resize_to_input: bool = True,
) -> tuple[np.ndarray, float | None]:
    """Metric depth (metres, float32) + focal_px estimate (None if the model
    doesn't predict one, e.g. Depth-Anything).

    If ``resize_to_input`` (default), depth is bicubic-resized to
    ``rgb_uint8``'s HxW. If False, depth is left at the model's native output
    resolution (smaller, cheaper to cache/hold in memory, and focal-length
    scale-independent) -- the caller is responsible for scaling intrinsics to
    match, exactly as the LiDAR loader scales K_rgb -> K_depth. Focal-length
    estimation (DepthPro) requires the true pixel width, so it is only
    returned when ``resize_to_input=True``.

    Cached to ``{cache_dir}/mono_depth/{model_id}/{cache_key}_{res}.npz``: the
    cache key is expected to already encode video-content-hash + frame-index
    (the model id and resize mode are folded into the path), so re-runs on the
    same video replay deterministically without touching the network or the
    GPU/MPS device.
    """
    suffix = "full" if resize_to_input else "native"
    path = _cache_path(cache_dir, model_id, f"{cache_key}_{suffix}")
    if path.exists():
        with np.load(path) as z:
            depth = z["depth"].astype(np.float32)
            focal = float(z["focal"]) if "focal" in z and np.isfinite(z["focal"]) else None
        return depth, focal

    import torch

    dev = _device(device)
    processor, model = _get_model(model_id, dev)
    h, w = rgb_uint8.shape[:2]
    inputs = processor(images=rgb_uint8, return_tensors="pt").to(dev)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = [(h, w)] if resize_to_input else None
    pp = processor.post_process_depth_estimation(outputs, target_sizes=target_sizes)[0]
    depth = pp["predicted_depth"].to(torch.float32).cpu().numpy()
    focal = pp.get("focal_length")
    focal_px = float(focal) if focal is not None else None

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, depth=depth.astype(np.float16),
                         focal=np.float32(focal_px if focal_px is not None else np.nan))
    os.replace(tmp, path)
    return depth, focal_px


def video_content_hash(path: str | Path, n_bytes: int = 1 << 20) -> str:
    """Cheap, stable content hash: file size + head/tail chunks (full-file hashing
    of multi-hundred-MB clips would dominate runtime for no benefit here)."""
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha1()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
        if size > n_bytes:
            f.seek(max(0, size - n_bytes))
            h.update(f.read(n_bytes))
    return h.hexdigest()[:16]


def prefetch_models(model_ids: list[str] | None = None) -> None:
    """Download model weights via the HF hub cache. Call once from a setup script
    so the live `load_video` path doesn't block on network access."""
    import torch
    ids = model_ids or [DEPTH_ANYTHING_SMALL, DEPTH_PRO]
    for mid in ids:
        _get_model(mid, "cpu")
    torch.manual_seed(0)
