"""Live webcam input, via OpenCV capture + MediaPipe FaceMesh. Untested in this
cloud container (no camera, no display) — see docs/PLAN.md. Requires
``pip install eyeos[webcam]``.
"""

from __future__ import annotations

from typing import Any

from eyeos.input_sources._capture_base import _OpenCVMediaPipeSource


class WebcamInputSource(_OpenCVMediaPipeSource):
    def __init__(
        self,
        device_index: int = 0,
        min_detection_confidence: float = 0.5,
        max_num_faces: int = 2,
    ) -> None:
        super().__init__(
            min_detection_confidence=min_detection_confidence, max_num_faces=max_num_faces
        )
        self.device_index = device_index

    def _open_capture(self, cv2: Any) -> Any:
        cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            raise RuntimeError(f"could not open webcam device {self.device_index}")
        return cap
