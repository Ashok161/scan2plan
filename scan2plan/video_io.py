"""Frame access for iPhone HEVC video.

OpenCV's CAP_PROP_POS_FRAMES seek is unreliable on iPhone HEVC files (it
returns empty frames), so access is sequential: requests are served from a
small cache and the decoder only ever moves forward, reopening on a rewind.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np


class VideoFrames:
    def __init__(self, path: str | Path, cache_size: int = 64):
        self.path = str(path)
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.cache_size = cache_size
        self._cap = None
        self._pos = 0
        cap = cv2.VideoCapture(self.path)
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()

    def _open(self):
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self.path)
        self._pos = 0

    def get(self, i: int) -> np.ndarray:
        """RGB uint8 frame i."""
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        if self._cap is None or i < self._pos:
            self._open()
        while self._pos < i:
            if not self._cap.grab():
                raise IndexError(f"frame {i} beyond end of {self.path}")
            self._pos += 1
        ok, bgr = self._cap.read()
        if not ok:
            raise IndexError(f"cannot decode frame {i} of {self.path}")
        self._pos += 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.cache[i] = rgb
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return rgb

    def iter_frames(self, indices):
        """Yield (i, rgb) for sorted indices in a single forward pass."""
        for i in sorted(set(int(k) for k in indices)):
            yield i, self.get(i)
