"""The gaze models eval/public compares: EyeOS's shipping calibration, and experiments.

Experiments live here, not in eyeos/: nothing in this file changes what EyeOS ships.
Every model is deterministic (closed-form least squares; no random initialisation, no
shuffling), so rerunning a comparison on the same cache gives the same numbers.

The variants (``--variants``, default all):

- ``eyeos``: the shipping calibration, eyeos.calibration.Calibration with its defaults.
- ``eyeos-poly2``: the same with ``degree=2`` (squares and pairwise products).
- ``ridge-{0.001,0.1,1,10}``: the shipping affine model with another ridge strength
  (``ridge-0.001`` restates the shipping default).
- ``per-dot-{1,5,10,20}``: the shipping model fitted on K images per calibration dot
  (``per-dot-5`` restates the default K).
- ``with-head-pose``: a ridge regression on the 4 iris offsets plus EyeOS's yaw, pitch
  and roll, standardised; an experiment, not in eyeos/.
- ``global-lopo``: one ridge regression trained on every OTHER participant's valid
  frames (iris offsets plus head pose -> screen fractions), then a per-user affine
  correction fitted on this participant's own 9-dot calibration set.
- ``eyeos-live9``: the shipping calibration built the way a live session builds it
  (eyeos/calibration_session.py): each dot's K images are averaged into one sample,
  exactly as collect_dot_features averages a dot's frames (the 4 iris offsets and the
  3 head-pose angles), and added at the GRID POINT, not at the images' true targets.
  Calibration is fitted on those 9 samples, so its validation error is the statistic
  live Gate 2 judges. It differs from live in what is averaged: K photos of separate
  moments (often separate days), not ~10 consecutive frames of one.
- ``centre``: baseline; always predicts the screen centre (W/2, H/2) and ignores the
  image. Its validation error is the RMS distance of the calibration targets to the
  centre: the leave-one-out of a constant that ignores the data.
- ``calibration-mean``: baseline; always predicts the mean of the calibration targets.
  Its validation error is the true leave-one-out RMS of that mean predictor.

The baselines run through the same protocol as every model (same calibration and test
frames, same rounding and clamping), so their errors are directly comparable: they say
how much of a model's error is left after it has used the image at all.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eval.public.protocol import FitContext, FittedModel, PreparedPerson, Variant
from eyeos.calibration import MIN_SAMPLES_TO_FIT, Calibration
from eyeos.features import FeatureVector

# The ridge strength of the experimental regressions below, on standardised features.
# Chosen before any results were seen and not tuned on them.
EXPERIMENT_RIDGE_LAMBDA = 1e-3


def leave_one_out_rms_px(
    targets_px: np.ndarray, fit_predict: Callable[[np.ndarray, int], np.ndarray]
) -> float:
    """Calibration's validation error, for any model: refit without each sample in turn,
    predict it, and take the RMS of the Euclidean errors (unclamped). Like Calibration,
    fewer than MIN_SAMPLES_TO_FIT + 1 samples is "not validated" (inf), never 0."""
    targets = np.asarray(targets_px, dtype=float).reshape(-1, 2)
    n = len(targets)
    if n < MIN_SAMPLES_TO_FIT + 1:
        return math.inf
    squared = []
    for held_out in range(n):
        train = np.delete(np.arange(n), held_out)
        predicted = np.asarray(fit_predict(train, held_out), dtype=float).reshape(2)
        squared.append(float(np.sum((predicted - targets[held_out]) ** 2)))
    return math.sqrt(sum(squared) / n)


def feature_matrix(features: Sequence[FeatureVector]) -> np.ndarray:
    """(n, 7): the 4 iris offsets, then yaw, pitch, roll in degrees."""
    rows = [(*f.iris_vector(), *f.head_pose()) for f in features]
    return np.array(rows, dtype=float).reshape(-1, 7)


@dataclass(frozen=True)
class StandardizedRidge:
    """Ridge regression on z-scored features, with an unregularised intercept.

    Each column is centred and scaled by its standard deviation over the training rows;
    a constant column (std 0) is only centred, so it drops out instead of dividing by 0.
    """

    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray  # (1 + n_features, n_outputs)

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, ridge_lambda: float) -> StandardizedRidge:
        x = np.asarray(x, dtype=float)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-12] = 1.0
        design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
        regularizer = ridge_lambda * np.eye(design.shape[1])
        regularizer[0, 0] = 0.0
        weights = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
        return cls(mean, scale, weights)

    def predict(self, x: np.ndarray) -> np.ndarray:
        design = np.column_stack([np.ones(len(x)), (np.asarray(x) - self.mean) / self.scale])
        result: np.ndarray = design @ self.weights
        return result


# ---------------------------------------------------------------------------
# the shipping calibration
# ---------------------------------------------------------------------------


class CalibrationModel:
    """eyeos.calibration.Calibration, fitted; predictions go through Calibration.predict."""

    def __init__(self, calibration: Calibration) -> None:
        self.calibration = calibration

    @property
    def validation_error_px(self) -> float:
        return self.calibration.validation_error_px

    def predict(self, features: Sequence[FeatureVector]) -> np.ndarray:
        return np.array([self.calibration.predict(f) for f in features], dtype=float).reshape(-1, 2)


@dataclass(frozen=True)
class CalibrationVariant:
    """eyeos.calibration.Calibration with the given constructor arguments (none = the
    shipping defaults), fitted on K images per calibration dot."""

    name: str
    description: str
    calibration_kwargs: dict[str, Any] = field(default_factory=dict)
    k_per_dot: int | None = None
    needs_other_persons: bool = False

    def fit(
        self, features: Sequence[FeatureVector], targets_px: np.ndarray, context: FitContext
    ) -> FittedModel:
        width, height = context.screen_px
        calibration = Calibration(width, height, **self.calibration_kwargs)
        for f, (x, y) in zip(features, np.asarray(targets_px, dtype=float), strict=True):
            calibration.add_sample((float(x), float(y)), f)
        calibration.fit()
        return CalibrationModel(calibration)


def average_dot_features(features: Sequence[FeatureVector]) -> FeatureVector:
    """One calibration sample from a dot's frames, computed exactly as
    eyeos.calibration_session.collect_dot_features computes it: the mean of the 4 iris
    offsets and the 3 head-pose angles."""
    mean = np.mean([[*f.iris_vector(), *f.head_pose()] for f in features], axis=0)
    return FeatureVector(*(float(v) for v in mean))


@dataclass(frozen=True)
class LiveStyleCalibrationVariant:
    """The shipping calibration fitted on 9 samples, one per dot, each the average of
    that dot's images and placed at the dot: what calibrate_interactively fits live."""

    name: str = "eyeos-live9"
    description: str = (
        "shipping eyeos.calibration.Calibration fitted the way a live session fits it: each "
        "dot's K images averaged into one sample (as calibration_session.collect_dot_features "
        "averages frames) at the grid point, not the images' true targets; validation error "
        "= leave-one-out over those 9 samples, the statistic live Gate 2 judges"
    )
    k_per_dot: int | None = None
    needs_other_persons: bool = False

    def fit(
        self, features: Sequence[FeatureVector], targets_px: np.ndarray, context: FitContext
    ) -> FittedModel:
        # targets_px is deliberately unused: a live session only knows where the dot was.
        if not context.dot_rows or len(context.dot_rows) != len(context.grid_px):
            raise ValueError(
                f"{self.name} needs the calibration grid and each dot's images "
                "(FitContext.grid_px and dot_rows)"
            )
        width, height = context.screen_px
        calibration = Calibration(width, height)
        for (x, y), rows in zip(context.grid_px, context.dot_rows, strict=True):
            calibration.add_sample((x, y), average_dot_features([features[i] for i in rows]))
        calibration.fit()
        return CalibrationModel(calibration)


# ---------------------------------------------------------------------------
# with-head-pose
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RidgeModel:
    regressor: StandardizedRidge
    validation_error_px: float

    def predict(self, features: Sequence[FeatureVector]) -> np.ndarray:
        return self.regressor.predict(feature_matrix(features))


@dataclass(frozen=True)
class HeadPoseRidgeVariant:
    name: str = "with-head-pose"
    description: str = (
        "experiment: ridge regression (lambda 1e-3, standardised features) on the 4 iris "
        "offsets plus EyeOS's yaw, pitch and roll; not part of eyeos/"
    )
    ridge_lambda: float = EXPERIMENT_RIDGE_LAMBDA
    k_per_dot: int | None = None
    needs_other_persons: bool = False

    def fit(
        self, features: Sequence[FeatureVector], targets_px: np.ndarray, context: FitContext
    ) -> FittedModel:
        x = feature_matrix(features)
        y = np.asarray(targets_px, dtype=float)

        def fit_predict(train: np.ndarray, held_out: int) -> np.ndarray:
            fold = StandardizedRidge.fit(x[train], y[train], self.ridge_lambda)
            return fold.predict(x[held_out : held_out + 1])[0]

        return RidgeModel(
            StandardizedRidge.fit(x, y, self.ridge_lambda), leave_one_out_rms_px(y, fit_predict)
        )


# ---------------------------------------------------------------------------
# global-lopo
# ---------------------------------------------------------------------------


def lopo_training_set(others: Sequence[PreparedPerson]) -> tuple[np.ndarray, np.ndarray]:
    """Every valid frame of the given (other) persons: (n, 7) features and (n, 2) targets
    as fractions of each person's own screen, so different screens share one scale."""
    features = [feature_matrix(p.features) for p in others]
    fractions = [
        p.data.targets_px[p.valid_rows] / np.asarray(p.data.screen_px, dtype=float) for p in others
    ]
    return np.vstack(features), np.vstack(fractions)


def _fit_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares affine map, (n, 2) -> (n, 2), as a (3, 2) matrix on [1, u, v]."""
    design = np.column_stack([np.ones(len(source)), source])
    solution: np.ndarray = np.linalg.lstsq(design, target, rcond=None)[0]
    return solution


def _apply_affine(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.column_stack([np.ones(len(points)), points]) @ affine
    return result


@dataclass(frozen=True)
class GlobalLopoModel:
    regressor: StandardizedRidge
    affine: np.ndarray
    screen_px: tuple[float, float]
    validation_error_px: float

    def predict(self, features: Sequence[FeatureVector]) -> np.ndarray:
        fractions = _apply_affine(self.affine, self.regressor.predict(feature_matrix(features)))
        result: np.ndarray = fractions * np.asarray(self.screen_px, dtype=float)
        return result


@dataclass(frozen=True)
class GlobalLopoVariant:
    name: str = "global-lopo"
    description: str = (
        "experiment: leave-one-person-out ridge (lambda 1e-3, standardised iris offsets plus "
        "head pose -> screen fractions) trained on every other participant, then a "
        "per-user affine correction fitted on this participant's 9-dot calibration set"
    )
    ridge_lambda: float = EXPERIMENT_RIDGE_LAMBDA
    k_per_dot: int | None = None
    needs_other_persons: bool = True

    def fit(
        self, features: Sequence[FeatureVector], targets_px: np.ndarray, context: FitContext
    ) -> FittedModel:
        if not context.other_persons:
            raise ValueError("global-lopo needs at least one other person to train on")
        train_x, train_y = lopo_training_set(context.other_persons)
        regressor = StandardizedRidge.fit(train_x, train_y, self.ridge_lambda)
        screen = np.asarray(context.screen_px, dtype=float)
        targets = np.asarray(targets_px, dtype=float)
        global_fractions = regressor.predict(feature_matrix(features))
        true_fractions = targets / screen

        def fit_predict(train: np.ndarray, held_out: int) -> np.ndarray:
            fold = _fit_affine(global_fractions[train], true_fractions[train])
            return _apply_affine(fold, global_fractions[held_out : held_out + 1])[0] * screen

        return GlobalLopoModel(
            regressor=regressor,
            affine=_fit_affine(global_fractions, true_fractions),
            screen_px=context.screen_px,
            validation_error_px=leave_one_out_rms_px(targets, fit_predict),
        )


# ---------------------------------------------------------------------------
# baselines: constant predictors that ignore the image
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstantModel:
    point_px: tuple[float, float]
    validation_error_px: float

    def predict(self, features: Sequence[FeatureVector]) -> np.ndarray:
        return np.tile(np.asarray(self.point_px, dtype=float), (len(features), 1))


@dataclass(frozen=True)
class CentreBaseline:
    name: str = "centre"
    description: str = (
        "baseline: always predicts the screen centre and ignores the image; validation "
        "error = RMS distance of the calibration targets to the centre"
    )
    k_per_dot: int | None = None
    needs_other_persons: bool = False

    def fit(
        self, features: Sequence[FeatureVector], targets_px: np.ndarray, context: FitContext
    ) -> FittedModel:
        width, height = context.screen_px
        centre = (width / 2.0, height / 2.0)
        point = np.array(centre, dtype=float)
        # A constant that ignores the data predicts the same whatever is held out.
        validation = leave_one_out_rms_px(targets_px, lambda train, held_out: point)
        return ConstantModel(centre, validation)


@dataclass(frozen=True)
class CalibrationMeanBaseline:
    name: str = "calibration-mean"
    description: str = (
        "baseline: always predicts the mean of the calibration targets and ignores the "
        "image; validation error = leave-one-out RMS of that mean"
    )
    k_per_dot: int | None = None
    needs_other_persons: bool = False

    def fit(
        self, features: Sequence[FeatureVector], targets_px: np.ndarray, context: FitContext
    ) -> FittedModel:
        targets = np.asarray(targets_px, dtype=float).reshape(-1, 2)

        def fit_predict(train: np.ndarray, held_out: int) -> np.ndarray:
            result: np.ndarray = targets[train].mean(axis=0)
            return result

        mean = targets.mean(axis=0)
        return ConstantModel(
            (float(mean[0]), float(mean[1])), leave_one_out_rms_px(targets, fit_predict)
        )


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------


def build_variants() -> dict[str, Variant]:
    """Every variant, in report order."""
    variants: list[Variant] = [
        CalibrationVariant(
            "eyeos",
            "shipping EyeOS: eyeos.calibration.Calibration with its defaults (affine in the "
            "4 iris offsets, ridge 1e-3)",
        ),
        CalibrationVariant(
            "eyeos-poly2",
            "eyeos.calibration.Calibration(degree=2): adds squares and pairwise products",
            {"degree": 2},
        ),
    ]
    for ridge_lambda in (0.001, 0.1, 1.0, 10.0):
        variants.append(
            CalibrationVariant(
                f"ridge-{ridge_lambda:g}",
                f"eyeos.calibration.Calibration(ridge_lambda={ridge_lambda:g})"
                + (" (the shipping default)" if ridge_lambda == 0.001 else ""),
                {"ridge_lambda": ridge_lambda},
            )
        )
    for k in (1, 5, 10, 20):
        variants.append(
            CalibrationVariant(
                f"per-dot-{k}",
                f"shipping calibration fitted on {k} image(s) per calibration dot",
                k_per_dot=k,
            )
        )
    variants += [HeadPoseRidgeVariant(), GlobalLopoVariant()]
    variants += [CentreBaseline(), CalibrationMeanBaseline(), LiveStyleCalibrationVariant()]
    return {v.name: v for v in variants}


VARIANT_NAMES: tuple[str, ...] = tuple(build_variants())
