"""Landmarks -> features: iris offsets (both eyes) and an approximate head pose.

The head pose here is a **deliberately simplified** geometric estimate (ratios of
landmark positions scaled to degrees), not a full ``cv2.solvePnP`` 3D fit. That keeps
this module pure-numpy and unit-testable on fixture landmarks with no OpenCV/MediaPipe
dependency. It is accurate enough for Gate 2's job — deciding whether the current head
pose has drifted too far from the calibration pose — because that only needs consistent
relative angles, not camera-calibrated absolute ones. A future pass could swap in
``solvePnP`` behind the same ``extract_features`` signature without touching callers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from eyeos.landmarks import FaceLandmarks, Point3D

# Scales the nose-offset ratio (-1..1 across the visible face) to degrees. These are
# rough, hand-picked constants for a roughly frontal, roughly centered face — good
# enough for relative "has the pose drifted" comparisons, not for absolute pose truth.
_YAW_SCALE_DEG = 60.0
_PITCH_SCALE_DEG = 50.0


@dataclass(frozen=True)
class FeatureVector:
    left_iris_x: float
    left_iris_y: float
    right_iris_x: float
    right_iris_y: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float

    def iris_vector(self) -> tuple[float, float, float, float]:
        """The 4 iris-offset numbers fed into the calibration regression."""
        return (self.left_iris_x, self.left_iris_y, self.right_iris_x, self.right_iris_y)

    def head_pose(self) -> tuple[float, float, float]:
        return (self.yaw_deg, self.pitch_deg, self.roll_deg)


def _iris_offset(iris: Point3D, corner_a: Point3D, corner_b: Point3D) -> tuple[float, float]:
    """The iris centre relative to the midpoint of the eye's two corners, in units of
    the eye's half-width. x runs along the corner line (toward +x in the image), y
    perpendicular to it (toward +y, i.e. down).

    The eyelids are deliberately not used. They move during every blink while the
    corners don't, and a lid-relative offset made a half-closed lid look like a gaze
    jump of hundreds of pixels.
    """
    left, right = (corner_a, corner_b) if corner_a.x <= corner_b.x else (corner_b, corner_a)
    axis_x, axis_y = right.x - left.x, right.y - left.y
    width = math.hypot(axis_x, axis_y)
    if width <= 1e-9:
        return 0.0, 0.0
    ux, uy = axis_x / width, axis_y / width
    dx = iris.x - (left.x + right.x) / 2.0
    dy = iris.y - (left.y + right.y) / 2.0
    half_width = width / 2.0
    return (dx * ux + dy * uy) / half_width, (-dx * uy + dy * ux) / half_width


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def extract_features(landmarks: FaceLandmarks) -> FeatureVector:
    left_x, left_y = _iris_offset(
        landmarks.left_iris_center, landmarks.left_eye_inner, landmarks.left_eye_outer
    )
    right_x, right_y = _iris_offset(
        landmarks.right_iris_center, landmarks.right_eye_inner, landmarks.right_eye_outer
    )

    # Roll from the eye corners too, so a blink (or a wink) can't change the head pose.
    left_eye_center = Point3D(
        (landmarks.left_eye_inner.x + landmarks.left_eye_outer.x) / 2.0,
        (landmarks.left_eye_inner.y + landmarks.left_eye_outer.y) / 2.0,
    )
    right_eye_center = Point3D(
        (landmarks.right_eye_inner.x + landmarks.right_eye_outer.x) / 2.0,
        (landmarks.right_eye_inner.y + landmarks.right_eye_outer.y) / 2.0,
    )
    roll_deg = math.degrees(
        math.atan2(
            right_eye_center.y - left_eye_center.y,
            right_eye_center.x - left_eye_center.x,
        )
    )

    face_mid_x = (landmarks.left_face_edge.x + landmarks.right_face_edge.x) / 2.0
    face_half_width = abs(landmarks.right_face_edge.x - landmarks.left_face_edge.x) / 2.0
    yaw_ratio = (
        (landmarks.nose_tip.x - face_mid_x) / face_half_width if face_half_width > 1e-9 else 0.0
    )
    yaw_deg = _clamp(yaw_ratio, -1.0, 1.0) * _YAW_SCALE_DEG

    face_mid_y = (landmarks.forehead.y + landmarks.chin.y) / 2.0
    face_half_height = abs(landmarks.chin.y - landmarks.forehead.y) / 2.0
    pitch_ratio = (
        (landmarks.nose_tip.y - face_mid_y) / face_half_height if face_half_height > 1e-9 else 0.0
    )
    pitch_deg = _clamp(pitch_ratio, -1.0, 1.0) * _PITCH_SCALE_DEG

    return FeatureVector(
        left_iris_x=left_x,
        left_iris_y=left_y,
        right_iris_x=right_x,
        right_iris_y=right_y,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
    )


def head_pose_delta_deg(
    pose_a: tuple[float, float, float], pose_b: tuple[float, float, float]
) -> float:
    """Largest per-axis absolute difference between two (yaw, pitch, roll) poses."""
    return max(abs(a - b) for a, b in zip(pose_a, pose_b, strict=True))
