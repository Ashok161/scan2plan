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
    focal_hint_px: float | None = None,
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

    ``focal_hint_px``: if given and the model predicts its own focal length
    (DepthPro), the returned depth is rescaled to what the model would have
    produced had it been given this TRUE focal length instead of its own
    FOV-head estimate, and the returned focal is ``focal_hint_px`` itself.
    This is exact, not an approximation: DepthPro's post-processing converts
    canonical inverse depth to metric depth via
    ``depth = focal_pred / (width * raw_inv_depth)`` (see
    ``transformers.models.depth_pro.image_processing_depth_pro``), which is
    linear in the focal length used, so
    ``depth(f_true) = depth(f_pred) * (f_true / f_pred)`` reproduces exactly
    what plugging ``f_true`` into that formula would give (the HF API has no
    argument to inject a focal length into the forward/post-process call
    itself; the FOV head only ever predicts one). Ignored (no-op) for models
    that don't predict a focal length at all, since there is nothing to
    rescale against.

    Cached to ``{cache_dir}/mono_depth/{model_id}/{cache_key}_{res}[_fh<px>].npz``:
    the cache key is expected to already encode video-content-hash +
    frame-index (the model id, resize mode and focal hint are folded into the
    path), so re-runs on the same video/photo replay deterministically
    without touching the network or the GPU/MPS device.
    """
    suffix = "full" if resize_to_input else "native"
    if focal_hint_px is not None:
        suffix += f"_fh{int(round(focal_hint_px))}"
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

    if focal_hint_px is not None and focal_px is not None and focal_px > 1e-6:
        depth = depth * (float(focal_hint_px) / focal_px)
        focal_px = float(focal_hint_px)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    focal32 = np.float32(focal_px if focal_px is not None else np.nan)
    np.savez_compressed(tmp, depth=depth.astype(np.float16), focal=focal32)
    os.replace(tmp, path)
    # Return exactly what a cache replay returns (float16-quantised depth, float32 focal): otherwise the
    # first, live run on a machine differs by rounding from every later replay, and downstream room
    # segmentation can amplify that difference (measured: 1 vs 3 rooms on c00a170fe1).
    depth = depth.astype(np.float16).astype(np.float32)
    focal_px = float(focal32) if np.isfinite(focal32) else None
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
