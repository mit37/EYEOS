"""Landmark types shared by every input source.

An ``InputSource`` (webcam, video file, or synthetic) is responsible for turning
whatever it reads (camera frames, a CSV row, ...) into a ``FaceLandmarks``. Everything
downstream (``features.py``, calibration, gates) only ever sees this type, so it never
needs to know MediaPipe's landmark index numbering or care whether the frame came from
a real camera or a fixture file.

Points are image coordinates in units of the frame's height, with (0, 0) at the
top-left: y runs from 0 to 1 and x from 0 to width / height. MediaPipe reports x as a
fraction of the width, so the MediaPipe sources rescale it; synthetic CSVs are taken
to be in this unit already. z is relative depth (MediaPipe units, smaller = closer to
the camera) and is currently unused by ``features.py`` but kept for future work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class FaceLandmarks:
    """One frame's worth of face geometry, already reduced to the points EyeOS needs.

    x and y share one unit (the frame's height), so distances and angles mean the
    same in both directions; see input_sources/_mediapipe_common.py.
    """

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
        min_open_ratio = 0.05
        left = _open_ratio(
            self.left_eye_top, self.left_eye_bottom, self.left_eye_outer, self.left_eye_inner
        )
        right = _open_ratio(
            self.right_eye_top, self.right_eye_bottom, self.right_eye_inner, self.right_eye_outer
        )
        return left >= min_open_ratio and right >= min_open_ratio


def _open_ratio(top: Point3D, bottom: Point3D, corner_a: Point3D, corner_b: Point3D) -> float:
    """The lid gap across the eye's corner line, divided by the eye's width along it.

    Measured in the eye's own frame, so it means the same at any head roll: a plain
    y gap over an x width only works for an upright face.
    """
    axis_x, axis_y = corner_b.x - corner_a.x, corner_b.y - corner_a.y
    width = math.hypot(axis_x, axis_y)
    if width <= 0:
        return 0.0
    gap_x, gap_y = top.x - bottom.x, top.y - bottom.y
    return abs(gap_x * axis_y - gap_y * axis_x) / (width * width)


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
