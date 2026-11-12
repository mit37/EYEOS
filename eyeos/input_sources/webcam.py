"""Live webcam input, via OpenCV capture + MediaPipe FaceMesh. Frames are stamped
with the host clock (see _frame_timestamp_s). Never run against a real camera in
this cloud container; tested only with fake cv2/mediapipe (tests/fake_capture.py) —
see docs/PLAN.md. Requires ``pip install eyeos[webcam]``.
"""

from __future__ import annotations

import time
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
        self._t0 = 0.0

    def _open_capture(self, cv2: Any) -> Any:
        cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            raise RuntimeError(f"could not open webcam device {self.device_index}")
        self._t0 = time.perf_counter()
        return cap

    def _frame_timestamp_s(self) -> float:
        # CAP_PROP_POS_MSEC is meaningless for live devices (DirectShow returns -1,
        # AVFoundation 0), which would freeze every timing check. Use the host clock
        # when the frame arrives; perf_counter, since monotonic is ~15.6 ms on Windows.
        return time.perf_counter() - self._t0
