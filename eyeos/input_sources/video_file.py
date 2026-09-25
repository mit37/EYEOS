"""Recorded-video input, via OpenCV capture + MediaPipe FaceMesh. Untested in this
cloud container (no video/camera dependencies installed) — see docs/PLAN.md. Requires
``pip install eyeos[webcam]`` (same optional dependencies as the live webcam source).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eyeos.input_sources._capture_base import _OpenCVMediaPipeSource


class VideoFileInputSource(_OpenCVMediaPipeSource):
    def __init__(
        self,
        path: str | Path,
        min_detection_confidence: float = 0.5,
        max_num_faces: int = 2,
    ) -> None:
        super().__init__(
            min_detection_confidence=min_detection_confidence, max_num_faces=max_num_faces
        )
        self.path = Path(path)

    def _open_capture(self, cv2: Any) -> Any:
        if not self.path.exists():
            raise RuntimeError(f"video file not found: {self.path}")
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video file: {self.path}")
        return cap
