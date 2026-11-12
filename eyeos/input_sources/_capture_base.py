"""Shared OpenCV + MediaPipe frame loop for webcam.py and video_file.py.

Both real-camera input sources only differ in how they open a ``cv2.VideoCapture``
(device index vs. file path); everything else — running MediaPipe FaceMesh over each
frame and converting the result to a ``FaceLandmarks`` — is identical, so it lives
here once.

OpenCV (``opencv-contrib-python``) and ``mediapipe`` are optional dependencies (``pip
install eyeos[webcam]``) and are imported lazily, inside ``frames()``, specifically so
that importing this module — and therefore ``eyeos.input_sources.webcam`` /
``eyeos.input_sources.video_file`` — never fails just because those packages aren't
installed. Only actually *using* one of these sources requires them. CI runs these
sources only against the fake modules in tests/fake_capture.py: see docs/PLAN.md's
"Cloud-instance constraints".

FaceMesh is MediaPipe's legacy Solutions API, which mediapipe 0.10.30 and later no
longer ship, so the ``webcam`` extra pins ``mediapipe<0.10.30`` (wheels for Python
3.10-3.12 only).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator
from typing import Any

from eyeos.input_sources import Frame, InputSource
from eyeos.input_sources._mediapipe_common import landmarks_from_mediapipe
from eyeos.landmarks import FaceLandmarks

_MISSING_DEPS_MESSAGE = (
    "{cls} needs OpenCV and MediaPipe, which aren't installed. "
    "Install them with: pip install 'eyeos[webcam]' (Python 3.10-3.12)"
)
_NO_SOLUTIONS_API_MESSAGE = (
    "{cls} needs MediaPipe's legacy Solutions API (mediapipe<0.10.30); installed "
    "mediapipe {version} doesn't have it. Reinstall with: pip install 'eyeos[webcam]' "
    "(Python 3.10-3.12)"
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
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                _NO_SOLUTIONS_API_MESSAGE.format(
                    cls=type(self).__name__, version=getattr(mp, "__version__", None) or "?"
                )
            )

        self._cv2 = cv2
        # FaceMesh first: if it fails, no capture has been opened yet.
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.max_num_faces,
            refine_landmarks=True,  # required for the iris landmarks (468, 473)
            min_detection_confidence=self.min_detection_confidence,
        )
        self._cap = self._open_capture(cv2)

    def _frame_timestamp_s(self) -> float:
        """Timestamp of the frame just read. Video files: the container's position."""
        return float(self._cap.get(self._cv2.CAP_PROP_POS_MSEC)) / 1000.0

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
        try:
            self._open()
            cv2 = self._cv2
            while True:
                ok, frame_bgr = self._cap.read()
                if not ok:
                    break
                timestamp_s = self._frame_timestamp_s()
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
