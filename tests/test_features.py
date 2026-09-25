"""Feature extraction tested on hand-built fixture landmarks (no camera needed)."""

from __future__ import annotations

import pytest

from eyeos.features import extract_features, head_pose_delta_deg
from eyeos.landmarks import FaceLandmarks, Point3D


def make_landmarks(**overrides) -> FaceLandmarks:
    """A centered, frontal, symmetric face. Iris offsets and head pose are all ~0."""
    defaults = dict(
        face_count=1,
        detection_confidence=0.95,
        left_iris_center=Point3D(0.38, 0.40),
        right_iris_center=Point3D(0.62, 0.40),
        left_eye_inner=Point3D(0.42, 0.40),
        left_eye_outer=Point3D(0.34, 0.40),
        left_eye_top=Point3D(0.38, 0.38),
        left_eye_bottom=Point3D(0.38, 0.42),
        right_eye_inner=Point3D(0.58, 0.40),
        right_eye_outer=Point3D(0.66, 0.40),
        right_eye_top=Point3D(0.62, 0.38),
        right_eye_bottom=Point3D(0.62, 0.42),
        nose_tip=Point3D(0.50, 0.50),
        chin=Point3D(0.50, 0.70),
        forehead=Point3D(0.50, 0.30),
        left_face_edge=Point3D(0.30, 0.50),
        right_face_edge=Point3D(0.70, 0.50),
    )
    defaults.update(overrides)
    return FaceLandmarks(**defaults)


def test_centered_frontal_face_has_near_zero_features():
    f = extract_features(make_landmarks())
    assert f.left_iris_x == pytest.approx(0.0, abs=1e-9)
    assert f.left_iris_y == pytest.approx(0.0, abs=1e-9)
    assert f.right_iris_x == pytest.approx(0.0, abs=1e-9)
    assert f.right_iris_y == pytest.approx(0.0, abs=1e-9)
    assert f.yaw_deg == pytest.approx(0.0, abs=1e-9)
    assert f.pitch_deg == pytest.approx(0.0, abs=1e-9)
    assert f.roll_deg == pytest.approx(0.0, abs=1e-9)


def test_iris_offset_toward_inner_corner_is_positive_x():
    # Left eye inner corner is at x=0.42, the eye center is at x=0.38: an iris at the
    # inner corner is fully offset toward +x.
    f = extract_features(make_landmarks(left_iris_center=Point3D(0.42, 0.40)))
    assert f.left_iris_x == pytest.approx(1.0)
    assert f.left_iris_y == pytest.approx(0.0, abs=1e-9)


def test_iris_offset_toward_outer_corner_is_negative_x():
    f = extract_features(make_landmarks(left_iris_center=Point3D(0.34, 0.40)))
    assert f.left_iris_x == pytest.approx(-1.0)


def test_iris_offset_toward_top_is_negative_y():
    f = extract_features(make_landmarks(left_iris_center=Point3D(0.38, 0.38)))
    assert f.left_iris_y == pytest.approx(-1.0)


def test_iris_offset_toward_bottom_is_positive_y():
    f = extract_features(make_landmarks(left_iris_center=Point3D(0.38, 0.42)))
    assert f.left_iris_y == pytest.approx(1.0)


def test_right_eye_offset_independent_of_left():
    f = extract_features(
        make_landmarks(
            left_iris_center=Point3D(0.42, 0.40),
            right_iris_center=Point3D(0.58, 0.40),
        )
    )
    assert f.left_iris_x == pytest.approx(1.0)
    assert f.right_iris_x == pytest.approx(-1.0)


def test_zero_width_eye_box_does_not_divide_by_zero():
    f = extract_features(
        make_landmarks(left_eye_inner=Point3D(0.38, 0.40), left_eye_outer=Point3D(0.38, 0.40))
    )
    assert f.left_iris_x == 0.0


def test_head_turned_right_gives_positive_yaw():
    f = extract_features(make_landmarks(nose_tip=Point3D(0.60, 0.50)))
    assert f.yaw_deg > 0


def test_head_turned_left_gives_negative_yaw():
    f = extract_features(make_landmarks(nose_tip=Point3D(0.40, 0.50)))
    assert f.yaw_deg < 0


def test_head_tilted_down_gives_positive_pitch():
    f = extract_features(make_landmarks(nose_tip=Point3D(0.50, 0.60)))
    assert f.pitch_deg > 0


def test_head_tilted_up_gives_negative_pitch():
    f = extract_features(make_landmarks(nose_tip=Point3D(0.50, 0.40)))
    assert f.pitch_deg < 0


def test_head_rolled_gives_nonzero_roll():
    f = extract_features(
        make_landmarks(
            left_eye_top=Point3D(0.38, 0.34),
            left_eye_bottom=Point3D(0.38, 0.38),
            right_eye_top=Point3D(0.62, 0.42),
            right_eye_bottom=Point3D(0.62, 0.46),
        )
    )
    assert f.roll_deg != pytest.approx(0.0, abs=1e-6)


def test_yaw_is_clamped_at_extremes():
    f = extract_features(make_landmarks(nose_tip=Point3D(5.0, 0.50)))
    assert f.yaw_deg == pytest.approx(60.0)


def test_iris_vector_matches_fields():
    f = extract_features(make_landmarks())
    assert f.iris_vector() == (f.left_iris_x, f.left_iris_y, f.right_iris_x, f.right_iris_y)


def test_head_pose_tuple_matches_fields():
    f = extract_features(make_landmarks())
    assert f.head_pose() == (f.yaw_deg, f.pitch_deg, f.roll_deg)


def test_both_irises_visible_true_for_open_eyes():
    lm = make_landmarks()
    assert lm.both_irises_visible


def test_both_irises_visible_false_when_one_eye_closed():
    lm = make_landmarks(left_eye_top=Point3D(0.38, 0.400), left_eye_bottom=Point3D(0.38, 0.401))
    assert not lm.both_irises_visible


def test_both_irises_visible_false_when_both_eyes_closed():
    lm = make_landmarks(
        left_eye_top=Point3D(0.38, 0.400),
        left_eye_bottom=Point3D(0.38, 0.401),
        right_eye_top=Point3D(0.62, 0.400),
        right_eye_bottom=Point3D(0.62, 0.401),
    )
    assert not lm.both_irises_visible


def test_head_pose_delta_deg_is_max_abs_per_axis():
    a = (10.0, -5.0, 0.0)
    b = (12.0, 0.0, 20.0)
    assert head_pose_delta_deg(a, b) == pytest.approx(20.0)


def test_head_pose_delta_deg_zero_for_identical_poses():
    a = (3.0, -2.0, 1.0)
    assert head_pose_delta_deg(a, a) == 0.0
