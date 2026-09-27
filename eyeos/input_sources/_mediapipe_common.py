"""Shared MediaPipe Face Landmarker glue for webcam.py and video_file.py.

MediaPipe's FaceMesh/FaceLandmarker landmark indices are a fixed, well-known numbering
(468 points, +10 iris points when ``refine_landmarks``/iris output is enabled). These
are the standard indices for the points EyeOS needs; see
https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
for the reference diagram.

This module has no import-time dependency on ``mediapipe`` itself: it only indexes
into whatever list-of-landmarks object the caller (webcam.py / video_file.py) already
obtained from the MediaPipe API, duck-typing on ``landmark[i].x/.y/.z``.
"""

from __future__ import annotations

from typing import Protocol

from eyeos.landmarks import FaceLandmarks, Point3D

LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10
LEFT_FACE_EDGE = 234
RIGHT_FACE_EDGE = 454


class _LandmarkLike(Protocol):
    x: float
    y: float
    z: float


def landmarks_from_mediapipe(
    face_landmarks: list[_LandmarkLike],
    face_count: int,
    detection_confidence: float,
) -> FaceLandmarks:
    """Build a FaceLandmarks from one face's worth of raw MediaPipe landmarks."""

    def pt(index: int) -> Point3D:
        lm = face_landmarks[index]
        return Point3D(x=lm.x, y=lm.y, z=lm.z)

    return FaceLandmarks(
        face_count=face_count,
        detection_confidence=detection_confidence,
        left_iris_center=pt(LEFT_IRIS_CENTER),
        right_iris_center=pt(RIGHT_IRIS_CENTER),
        left_eye_inner=pt(LEFT_EYE_INNER),
        left_eye_outer=pt(LEFT_EYE_OUTER),
        left_eye_top=pt(LEFT_EYE_TOP),
        left_eye_bottom=pt(LEFT_EYE_BOTTOM),
        right_eye_inner=pt(RIGHT_EYE_INNER),
        right_eye_outer=pt(RIGHT_EYE_OUTER),
        right_eye_top=pt(RIGHT_EYE_TOP),
        right_eye_bottom=pt(RIGHT_EYE_BOTTOM),
        nose_tip=pt(NOSE_TIP),
        chin=pt(CHIN),
        forehead=pt(FOREHEAD),
        left_face_edge=pt(LEFT_FACE_EDGE),
        right_face_edge=pt(RIGHT_FACE_EDGE),
    )
