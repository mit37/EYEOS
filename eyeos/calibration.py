"""9-point calibration: collect (iris features -> screen point) samples, fit a
per-user ridge regression, and report a validation error Gate 2 can check.

Ridge regression is used over plain least squares because a live calibration session
gives you exactly 9 points — with a higher-order polynomial basis that is an
underdetermined or barely-determined system, and ridge keeps the fit well-posed. The
default basis is affine (degree=1: a constant plus the 4 raw iris-offset numbers),
which is well-determined by 9 points and — for the kind of roughly-linear
iris-offset-to-screen relationship a calibration session produces — recovers the
mapping cleanly. ``degree=2`` (adding squares and pairwise products) is available for
a closer fit at the cost of needing more, cleaner calibration points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np

from eyeos.features import FeatureVector

IrisVector = tuple[float, float, float, float]
HeadPose = tuple[float, float, float]
ScreenPoint = tuple[float, float]

MIN_SAMPLES_TO_FIT = 5


@dataclass(frozen=True)
class CalibrationStatus:
    """What Gate 2 needs: does a calibration exist, how good is it, and at what pose."""

    exists: bool
    validation_error_px: float
    calibration_head_pose: HeadPose

    @staticmethod
    def none() -> CalibrationStatus:
        return CalibrationStatus(
            exists=False, validation_error_px=math.inf, calibration_head_pose=(0.0, 0.0, 0.0)
        )


@dataclass(frozen=True)
class CalibrationSample:
    target_px: ScreenPoint
    iris_vector: IrisVector
    head_pose: HeadPose


def target_points(screen_width_px: float, screen_height_px: float) -> list[ScreenPoint]:
    """The standard 9-point grid (3x3, at 10/50/90% of each screen axis)."""
    fracs = (0.1, 0.5, 0.9)
    return [(fx * screen_width_px, fy * screen_height_px) for fy in fracs for fx in fracs]


def _poly_features(iris_vector: IrisVector, degree: int) -> np.ndarray:
    terms = [1.0, *iris_vector]
    if degree >= 2:
        terms += [a * b for a, b in combinations_with_replacement(iris_vector, 2)]
    if degree > 2:
        raise ValueError(f"unsupported degree: {degree}")
    return np.array(terms, dtype=float)


def _ridge_fit(X: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    n_features = X.shape[1]
    reg = ridge_lambda * np.eye(n_features)
    reg[0, 0] = 0.0  # never regularize the intercept
    return np.linalg.solve(X.T @ X + reg, X.T @ y)


class Calibration:
    """Collects calibration samples, fits a ridge regression, reports validation error."""

    def __init__(
        self,
        screen_width_px: float,
        screen_height_px: float,
        degree: int = 1,
        ridge_lambda: float = 1e-3,
    ) -> None:
        self.screen_width_px = screen_width_px
        self.screen_height_px = screen_height_px
        self.degree = degree
        self.ridge_lambda = ridge_lambda
        self._samples: list[CalibrationSample] = []
        self._weights_x: np.ndarray | None = None
        self._weights_y: np.ndarray | None = None
        self._validation_error_px: float | None = None
        self._calibration_pose: HeadPose | None = None

    @property
    def samples(self) -> list[CalibrationSample]:
        return list(self._samples)

    @property
    def is_fitted(self) -> bool:
        return self._weights_x is not None and self._weights_y is not None

    @property
    def validation_error_px(self) -> float:
        return self._validation_error_px if self._validation_error_px is not None else math.inf

    @property
    def calibration_head_pose(self) -> HeadPose:
        return self._calibration_pose if self._calibration_pose is not None else (0.0, 0.0, 0.0)

    def add_sample(self, target_px: ScreenPoint, features: FeatureVector) -> None:
        self._samples.append(
            CalibrationSample(
                target_px=target_px,
                iris_vector=features.iris_vector(),
                head_pose=features.head_pose(),
            )
        )

    def fit(self) -> None:
        if len(self._samples) < MIN_SAMPLES_TO_FIT:
            raise ValueError(
                f"need at least {MIN_SAMPLES_TO_FIT} calibration samples, got {len(self._samples)}"
            )
        X = np.array([_poly_features(s.iris_vector, self.degree) for s in self._samples])
        yx = np.array([s.target_px[0] for s in self._samples])
        yy = np.array([s.target_px[1] for s in self._samples])
        self._weights_x = _ridge_fit(X, yx, self.ridge_lambda)
        self._weights_y = _ridge_fit(X, yy, self.ridge_lambda)
        mean_pose = np.mean([s.head_pose for s in self._samples], axis=0)
        self._calibration_pose = (float(mean_pose[0]), float(mean_pose[1]), float(mean_pose[2]))
        self._validation_error_px = self._leave_one_out_rms_error()

    def _predict_weights(
        self, iris_vector: IrisVector, weights_x: np.ndarray, weights_y: np.ndarray
    ) -> ScreenPoint:
        phi = _poly_features(iris_vector, self.degree)
        return float(phi @ weights_x), float(phi @ weights_y)

    def _leave_one_out_rms_error(self) -> float:
        n = len(self._samples)
        if n < MIN_SAMPLES_TO_FIT + 1:
            return 0.0
        squared_errors = []
        for held_out in range(n):
            train = [s for i, s in enumerate(self._samples) if i != held_out]
            X = np.array([_poly_features(s.iris_vector, self.degree) for s in train])
            wx = _ridge_fit(X, np.array([s.target_px[0] for s in train]), self.ridge_lambda)
            wy = _ridge_fit(X, np.array([s.target_px[1] for s in train]), self.ridge_lambda)
            held = self._samples[held_out]
            px, py = self._predict_weights(held.iris_vector, wx, wy)
            squared_errors.append((px - held.target_px[0]) ** 2 + (py - held.target_px[1]) ** 2)
        return math.sqrt(sum(squared_errors) / len(squared_errors))

    def predict(self, features: FeatureVector) -> ScreenPoint:
        if not self.is_fitted:
            raise RuntimeError("Calibration.fit() has not been called (or has no data)")
        assert self._weights_x is not None and self._weights_y is not None
        return self._predict_weights(features.iris_vector(), self._weights_x, self._weights_y)

    def status(self) -> CalibrationStatus:
        if not self.is_fitted:
            return CalibrationStatus.none()
        return CalibrationStatus(
            exists=True,
            validation_error_px=self.validation_error_px,
            calibration_head_pose=self.calibration_head_pose,
        )
