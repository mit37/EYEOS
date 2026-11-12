"""Landmark types shared by every input source.

An ``InputSource`` (webcam, video file, or synthetic) is responsible for turning
whatever it reads (camera frames, a CSV row, ...) into a ``FaceLandmarks``. Everything
downstream (``features.py``, calibration, gates) only ever sees this type, so it never
needs to know MediaPipe's landmark index numbering or care whether the frame came from
a real camera or a fixture file.

Points are normalized image coordinates: x, y in [0, 1] with (0, 0) at the top-left of
the frame, matching MediaPipe's convention. z is relative depth (MediaPipe units,
smaller = closer to the camera) and is currently unused by ``features.py`` but kept for
future work.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class FaceLandmarks:
    """One frame's worth of face geometry, already reduced to the points EyeOS needs."""

    face_count: int
    detection_confidence: float

    left_iris_center: Point3D
    right_iris_center: Point3D

    left_eye_inner: Point3D
    left_eye_outer: Point3D
    left_eye_top: Point3D
    left_eye_bottom: Point3D

    right_eye_inner: Point3D
    right_eye_outer: Point3D
    right_eye_top: Point3D
    right_eye_bottom: Point3D

    nose_tip: Point3D
    chin: Point3D
    forehead: Point3D
    left_face_edge: Point3D
    right_face_edge: Point3D

    @property
    def both_irises_visible(self) -> bool:
        """True when both eyes were detected open enough to place an iris center.

        MediaPipe still emits iris landmarks on a fully closed eye (it extrapolates),
        so "visible" here means "the eye opening is wide enough to trust the iris
        position" — the eyelid gap is at least a hair's width relative to the eye's
        own width. Below that, a blink or a near-closed eye is producing noise, not a
        gaze signal.
        """
        left_gap = abs(self.left_eye_top.y - self.left_eye_bottom.y)
        left_width = abs(self.left_eye_outer.x - self.left_eye_inner.x)
        right_gap = abs(self.right_eye_top.y - self.right_eye_bottom.y)
        right_width = abs(self.right_eye_outer.x - self.right_eye_inner.x)
        min_open_ratio = 0.05
        left_open = left_width > 0 and (left_gap / left_width) >= min_open_ratio
        right_open = right_width > 0 and (right_gap / right_width) >= min_open_ratio
        return left_open and right_open


# The 15 named points, in the same order as FaceLandmarks' constructor fields (after
# face_count/detection_confidence). Used by anything that (de)serializes a
# FaceLandmarks row-wise, e.g. eyeos/synthetic_format.py.
POINT_FIELDS: tuple[str, ...] = (
    "left_iris_center",
    "right_iris_center",
    "left_eye_inner",
    "left_eye_outer",
    "left_eye_top",
    "left_eye_bottom",
    "right_eye_inner",
    "right_eye_outer",
    "right_eye_top",
    "right_eye_bottom",
    "nose_tip",
    "chin",
    "forehead",
    "left_face_edge",
    "right_face_edge",
)
