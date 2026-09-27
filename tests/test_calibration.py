"""Calibration tested against a known synthetic ground-truth linear mapping."""

from __future__ import annotations

import math

import pytest

from eyeos.calibration import Calibration, CalibrationStatus, target_points
from eyeos.features import FeatureVector

# A fixed "true" affine mapping from [1, x1, x2, x3, x4] (basis) -> screen px.
TRUE_WX = (960.0, 500.0, -50.0, 300.0, 80.0)
TRUE_WY = (540.0, -50.0, 400.0, 60.0, 250.0)

# 9 iris-vector points spread through [-0.4, 0.4]^4, generic enough to make the
# 9x5 design matrix (with the constant column) full rank.
IRIS_VECTORS = [
    (-0.4, -0.4, -0.4, -0.4),
    (-0.4, -0.4, 0.4, 0.4),
    (-0.4, 0.4, -0.4, 0.4),
    (-0.4, 0.4, 0.4, -0.4),
    (0.0, 0.0, 0.0, 0.0),
    (0.4, -0.4, -0.4, 0.4),
    (0.4, -0.4, 0.4, -0.4),
    (0.4, 0.4, -0.4, -0.4),
    (0.4, 0.4, 0.4, 0.4),
]


def _true_point(iris_vector: tuple[float, float, float, float]) -> tuple[float, float]:
    basis = (1.0, *iris_vector)
    x = sum(w * b for w, b in zip(TRUE_WX, basis, strict=True))
    y = sum(w * b for w, b in zip(TRUE_WY, basis, strict=True))
    return x, y


def _feature_vector(iris_vector: tuple[float, float, float, float]) -> FeatureVector:
    x1, x2, x3, x4 = iris_vector
    return FeatureVector(
        left_iris_x=x1,
        left_iris_y=x2,
        right_iris_x=x3,
        right_iris_y=x4,
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
    )


def _fit_calibration(degree: int = 1) -> Calibration:
    cal = Calibration(screen_width_px=1920, screen_height_px=1080, degree=degree)
    for iv in IRIS_VECTORS:
        cal.add_sample(_true_point(iv), _feature_vector(iv))
    cal.fit()
    return cal


def test_target_points_is_a_3x3_grid_at_10_50_90_percent():
    pts = target_points(1920, 1080)
    assert len(pts) == 9
    assert (192.0, 108.0) in pts  # 10%, 10%
    assert (960.0, 540.0) in pts  # 50%, 50%
    assert (1728.0, 972.0) in pts  # 90%, 90%


def test_fit_raises_below_minimum_samples():
    cal = Calibration(1920, 1080)
    for iv in IRIS_VECTORS[:4]:
        cal.add_sample(_true_point(iv), _feature_vector(iv))
    with pytest.raises(ValueError):
        cal.fit()


def test_predict_raises_before_fit():
    cal = Calibration(1920, 1080)
    with pytest.raises(RuntimeError):
        cal.predict(_feature_vector((0.0, 0.0, 0.0, 0.0)))


def test_status_before_fit_reports_no_calibration():
    cal = Calibration(1920, 1080)
    status = cal.status()
    assert status == CalibrationStatus.none()
    assert status.exists is False
    assert math.isinf(status.validation_error_px)


def test_affine_fit_recovers_clean_linear_mapping():
    cal = _fit_calibration(degree=1)
    assert cal.is_fitted
    # Noiseless, well-determined linear system: ridge (tiny lambda) should recover
    # it almost exactly, both on held-out cross-validation and on a fresh point.
    assert cal.validation_error_px < 5.0

    test_iv = (0.2, -0.1, 0.1, 0.3)
    expected = _true_point(test_iv)
    predicted = cal.predict(_feature_vector(test_iv))
    assert predicted[0] == pytest.approx(expected[0], abs=5.0)
    assert predicted[1] == pytest.approx(expected[1], abs=5.0)


def test_status_after_fit_reports_calibration():
    cal = _fit_calibration()
    status = cal.status()
    assert status.exists is True
    assert status.validation_error_px < 5.0
    assert status.calibration_head_pose == (0.0, 0.0, 0.0)


def test_calibration_head_pose_is_mean_of_sample_poses():
    cal = Calibration(1920, 1080)
    poses = [(-10.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    for i, iv in enumerate(IRIS_VECTORS):
        fv = FeatureVector(*iv, *poses[i % 2])
        cal.add_sample(_true_point(iv), fv)
    cal.fit()
    expected_mean = sum(poses[i % 2][0] for i in range(len(IRIS_VECTORS))) / len(IRIS_VECTORS)
    assert cal.calibration_head_pose[0] == pytest.approx(expected_mean)


def test_degree_2_fits_without_crashing():
    cal = _fit_calibration(degree=2)
    assert cal.is_fitted
    predicted = cal.predict(_feature_vector((0.1, 0.1, 0.1, 0.1)))
    assert isinstance(predicted[0], float)
    assert isinstance(predicted[1], float)


def test_noisy_targets_produce_nonzero_validation_error():
    cal = Calibration(1920, 1080)
    for i, iv in enumerate(IRIS_VECTORS):
        tx, ty = _true_point(iv)
        noise = 200.0 if i % 2 == 0 else -200.0
        cal.add_sample((tx + noise, ty + noise), _feature_vector(iv))
    cal.fit()
    assert cal.validation_error_px > 50.0
