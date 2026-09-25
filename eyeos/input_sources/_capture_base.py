"""Shared OpenCV + MediaPipe frame loop for webcam.py and video_file.py.

Both real-camera input sources only differ in how they open a ``cv2.VideoCapture``
(device index vs. file path); everything else — running MediaPipe FaceMesh over each
frame and converting the result to a ``FaceLandmarks`` — is identical, so it lives
here once.

``opencv-python`` and ``mediapipe`` are optional dependencies (``pip install
eyeos[webcam]``) and are imported lazily, inside ``frames()``, specifically so that
importing this module — and therefore ``eyeos.input_sources.webcam`` /
``eyeos.input_sources.video_file`` — never fails just because those packages aren't
installed. Only actually *using* one of these sources requires them. That's why they
are not exercised in this project's CI: see docs/PLAN.md's "Cloud-instance
constraints".
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator
from typing import Any

from eyeos.input_sources import Frame, InputSource
from eyeos.input_sources._mediapipe_common import landmarks_from_mediapipe
from eyeos.landmarks import FaceLandmarks

_MISSING_DEPS_MESSAGE = (
    "{cls} needs opencv-python and mediapipe, which aren't installed. "
    "Install them with: pip install 'eyeos[webcam]'"
)


class _OpenCVMediaPipeSource(InputSource):
    def __init__(self, min_detection_confidence: float = 0.5, max_num_faces: int = 2) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.max_num_faces = max_num_faces
        self._cap: Any = None
        self._face_mesh: Any = None
        self._cv2: Any = None

    @abstractmethod
    def _open_capture(self, cv2: Any) -> Any:
        """Return an opened cv2.VideoCapture (or raise RuntimeError)."""

    def _open(self) -> None:
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError(_MISSING_DEPS_MESSAGE.format(cls=type(self).__name__)) from e
        try:
            import mediapipe as mp
        except ImportError as e:
            raise RuntimeError(_MISSING_DEPS_MESSAGE.format(cls=type(self).__name__)) from e

        self._cv2 = cv2
        self._cap = self._open_capture(cv2)
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.max_num_faces,
            refine_landmarks=True,  # required for the iris landmarks (468, 473)
            min_detection_confidence=self.min_detection_confidence,
        )

    def _to_landmarks(self, result: Any) -> FaceLandmarks | None:
        faces = result.multi_face_landmarks or []
        if not faces:
            return None
        # The legacy Solutions API doesn't expose a per-face confidence score;
        # Gate 1's face_count check (not confidence) is what catches multi-face frames.
        return landmarks_from_mediapipe(
            faces[0].landmark, face_count=len(faces), detection_confidence=1.0
        )

    def frames(self) -> Iterator[Frame]:
        self._open()
        cv2 = self._cv2
        try:
            while True:
                ok, frame_bgr = self._cap.read()
                if not ok:
                    break
                timestamp_s = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                result = self._face_mesh.process(frame_rgb)
                yield Frame(timestamp_s=timestamp_s, landmarks=self._to_landmarks(result))
        finally:
            self.close()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
