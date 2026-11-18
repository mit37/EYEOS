"""Where the public-dataset error comes from: diagnostics on top of the EyeOS protocol.

``python -m eval.public.mpiifacegaze evaluate`` measures how far EyeOS's shipping
calibration misses on real photos. These diagnostics take that number apart. Each
section answers one question (``SECTIONS``), every metric has a name and a one-line
description (``METRICS``), and the results JSON carries both next to the numbers.

Like protocol.py, nothing here knows a dataset: a loader hands over one
``DiagnosticPerson`` per participant (for MPIIFaceGaze,
``eval.public.mpiifacegaze.load_diagnostic_person``). Everything is deterministic:
closed-form least squares, no randomness.

The in-sample sections fit on every valid frame and score on those same frames. They
bound what a calibration of these features could reach; they are not results.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from eval.public import protocol
from eval.public.models import (
    EXPERIMENT_RIDGE_LAMBDA,
    StandardizedRidge,
    build_variants,
    feature_matrix,
)
from eyeos.calibration import Calibration, target_points
from eyeos.features import head_pose_delta_deg

# The shipping calibration's ridge strength, read from it rather than restated.
SHIPPING_RIDGE_LAMBDA: float = Calibration(1.0, 1.0).ridge_lambda
# An affine fit of the 4 iris offsets has 5 parameters per axis; fewer valid frames than
# this and a person's in-sample sections are skipped.
MIN_FRAMES = 6
# A day folder needs at least this many valid images for the within-day protocol (and
# always at least K per dot x 9 dots + 1 test image).
DEFAULT_MIN_DAY_IMAGES = 60
# The protocol variant every protocol section runs: the shipping calibration.
PROTOCOL_VARIANT = "eyeos"
# Head-pose delta bins in degrees, closed on the right like Gate 2's "<= limit":
# [0, 5], (5, 10], (10, 15], (15, 20], (20, inf).
POSE_BIN_EDGES_DEG: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0)

SECTIONS: dict[str, str] = {
    "feature_sanity": (
        "a. Are the iris offsets sane? Ranges per eye, whether both eyes move together, "
        "whether both eyes' x offsets have the same sign against the target's x, and the "
        "EyeOS roll spread. A broken or swapped eye shows here."
    ),
    "in_sample_ceiling": (
        "b. How well could ANY calibration of these features do? Each model is fitted on "
        "all of a person's valid frames and scored on the same frames, so it is an "
        "optimistic bound, not a result. Far above Gate 2's limit means the features, not "
        "the calibration set, are the limit."
    ),
    "constant_baselines": (
        "c. What does a model that ignores the eyes score? A yardstick for every other "
        "pixel number."
    ),
    "cross_day": (
        "d. The standard protocol, exactly as `evaluate` runs it for the shipping `eyeos` "
        "variant: K images per dot chosen from all of a person's frames (so from many "
        "days), tested on every other frame. The reference for within_day."
    ),
    "within_day": (
        "d. The same protocol inside one day folder at a time: K images per dot chosen "
        "only from that day, tested on that day's other images. Within-day far below "
        "cross-day means drift between sittings is a large share of the error, which a "
        "live session (calibrated in the same sitting) would not have; both high means "
        "the features themselves are the bottleneck. A day can hold several sittings, "
        "and a day's calibration images lie further from the dots (see "
        "calibration_mean_distance_px), so this is not a live session either."
    ),
    "error_by_head_pose": (
        "e. Under the cross-day protocol, does the error grow as the head pose moves away "
        "from the calibration pose, and what is it on the frames Gate 2 would let through "
        "(within its head-pose limit)? The delta is Gate 2's own: the largest per-axis "
        "difference of EyeOS's yaw, pitch and roll from the calibration frames' mean, in "
        "EyeOS's degrees, whose scale head_pose_agreement checks."
    ),
}

METRICS: dict[str, str] = {
    "n_images": "images of this person in the dataset",
    "n_valid": "images that Gate 1 would pass (exactly one face, both irises visible)",
    "skipped": "why a section could not be computed for this person",
    "feature_sanity.left_iris_x": (
        "EyeOS left-eye iris offset along the eye's corner line, in eye half-widths: "
        "median, p5, p95 over the valid frames"
    ),
    "feature_sanity.left_iris_y": (
        "EyeOS left-eye iris offset across the corner line (down = +), in eye half-widths: "
        "median, p5, p95"
    ),
    "feature_sanity.right_iris_x": (
        "as left_iris_x, for the right eye (the eye on the image's right)"
    ),
    "feature_sanity.right_iris_y": "as left_iris_y, for the right eye",
    "feature_sanity.corr_left_right_x": (
        "Pearson correlation of the two eyes' x offsets; both eyes turn together, so near +1 "
        "is healthy and noise pulls it toward 0"
    ),
    "feature_sanity.corr_left_right_y": "Pearson correlation of the two eyes' y offsets",
    "feature_sanity.corr_left_x_target_x": (
        "Pearson correlation of the left eye's x offset with the target's x"
    ),
    "feature_sanity.corr_right_x_target_x": (
        "Pearson correlation of the right eye's x offset with the target's x"
    ),
    "feature_sanity.target_x_signs_match": (
        "whether both eyes' x offsets correlate with the target's x with the same sign "
        "(false = one eye's x runs the wrong way)"
    ),
    "feature_sanity.corr_left_y_target_y": (
        "Pearson correlation of the left eye's y offset with the target's y"
    ),
    "feature_sanity.corr_right_y_target_y": (
        "Pearson correlation of the right eye's y offset with the target's y"
    ),
    "feature_sanity.eyeos_roll_deg": (
        "EyeOS head roll (from the eye corners), degrees: median, p5, p95; a few degrees is "
        "expected for someone sitting at a laptop"
    ),
    "in_sample_ceiling.both_eyes": (
        "the shipping calibration's model (affine in the 4 iris offsets, its ridge strength) "
        "fitted on all valid frames: mean_px and rms_px of the rounded, clamped prediction's "
        "error, and r2_x, r2_y of the raw fit per screen axis"
    ),
    "in_sample_ceiling.left_eye_only": "as both_eyes, on the left eye's 2 offsets alone",
    "in_sample_ceiling.right_eye_only": "as both_eyes, on the right eye's 2 offsets alone",
    "in_sample_ceiling.with_head_pose": (
        "as both_eyes, with EyeOS's yaw, pitch and roll added, fitted like the with-head-pose "
        "variant (standardised ridge)"
    ),
    "constant_baselines.screen_centre_mean_px": (
        "mean px error of always predicting the screen centre, over all valid frames"
    ),
    "constant_baselines.own_mean_target_mean_px": (
        "mean px error of always predicting this person's mean target over all valid frames "
        "(an in-sample constant)"
    ),
    "cross_day.n_calibration": "calibration images (K per dot x 9 dots)",
    "cross_day.n_test": "test images: every other valid image",
    "cross_day.mean_px": "mean px error on the test images (as evaluate's pixel_error_px.mean)",
    "cross_day.mean_deg": "mean angular error in degrees (null without a trusted screen model)",
    "cross_day.validation_error_px": (
        "the calibration's own leave-one-out RMS error, what Gate 2 checks"
    ),
    "cross_day.passes_gate2": "whether validation_error_px is below Gate 2's limit",
    "cross_day.calibration_mean_distance_px": (
        "mean distance from the calibration images' true targets to their dots"
    ),
    "cross_day.calibration_days": (
        "distinct day folders among the calibration images (null without day folders)"
    ),
    "within_day.days_total": "day folders with at least one valid image",
    "within_day.days_used": (
        "days with at least the minimum number of valid images (config.min_day_images, and "
        "always K x 9 + 1), each calibrated and tested on its own"
    ),
    "within_day.days_skipped": "days with too few valid images (listed in skipped_days)",
    "within_day.mean_px": "mean over the days used of each day's mean px test error",
    "within_day.mean_deg": "mean over the days used of each day's mean angular error",
    "within_day.validation_error_px_mean": (
        "mean over the days used of each day's leave-one-out validation error (Gate 2's input)"
    ),
    "within_day.gate2_passed_days": "days whose validation error is below Gate 2's limit",
    "within_day.calibration_mean_distance_px_mean": (
        "mean over the days used of the calibration images' mean distance to their dots"
    ),
    "within_day.cross_day_mean_px": (
        "cross_day.mean_px again, null when no day was used, so the aggregate compares the "
        "two protocols on the same persons"
    ),
    "within_day.cross_day_mean_deg": "cross_day.mean_deg, on the same terms",
    "within_day.cross_day_validation_error_px": "cross_day.validation_error_px, on the same terms",
    "within_day.within_over_cross_mean_px": (
        "within_day.mean_px / cross_day.mean_px: well below 1 = drift between days matters"
    ),
    "within_day.per_day": (
        "one entry per day used: day, n_valid, and the cross_day metrics for that day alone"
    ),
    "within_day.skipped_days": "the days not used, with their number of valid images",
    "aggregate.within_day.days_used_total": "days used, summed over persons",
    "aggregate.within_day.gate2_passed_days_total": (
        "days whose calibration passes Gate 2, summed over persons"
    ),
    "error_by_head_pose.bins": (
        "per head-pose delta bin ([0, 5], (5, 10], (10, 15], (15, 20], (20, inf) degrees): "
        "lo_deg, hi_deg (null = open), n test frames and their mean_px"
    ),
    "error_by_head_pose.within_limit_share": (
        "share of test frames within Gate 2's head-pose limit (evaluate's "
        "head_pose_within_limit_share)"
    ),
    "error_by_head_pose.within_limit_mean_px": (
        "mean px error over the test frames within the limit only: what Gate 2 would act on"
    ),
    "error_by_head_pose.beyond_limit_mean_px": (
        "mean px error over the test frames beyond the limit (null if there are none)"
    ),
    "aggregate.error_by_head_pose.bins": (
        "the bins over all persons: n (frames), pooled_mean_px (over those frames), "
        "mean_of_person_means_px and persons (with at least one frame in the bin)"
    ),
}

AGGREGATE_NOTE = (
    "aggregate.<section>.<metric> summarises the per-person metric of the same name: "
    "mean, median, min and max over the persons where it is not null (persons counts "
    "them); a true/false metric counts the persons where it is true; a text metric counts "
    "persons per value. Each person weighs the same."
)


@dataclass
class DiagnosticPerson:
    """One participant: the protocol's PersonData, plus what only the diagnostics need.
    Each list or array is aligned with ``data.landmarks`` (one entry per image)."""

    data: protocol.PersonData
    days: list[str | None] | None = None  # the day (session) folder of each image
    head_rotation: np.ndarray | None = None  # (n, 3, 3): dataset head pose, head -> camera
    head_translation_mm: np.ndarray | None = None  # (n, 3): dataset head position

    @property
    def name(self) -> str:
        return self.data.person


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _spread(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "p5": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson correlation; None when either side is constant (it is undefined)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _r2(truth: np.ndarray, predicted: np.ndarray) -> float | None:
    """1 - SS_res / SS_tot over every column together; None when truth is constant."""
    truth = np.asarray(truth, dtype=float)
    total = float(np.sum((truth - truth.mean(axis=0)) ** 2))
    if total <= 1e-12:
        return None
    return 1.0 - float(np.sum((truth - predicted) ** 2)) / total


def _errors_px(
    predicted: np.ndarray, targets: np.ndarray, screen: tuple[float, float]
) -> np.ndarray:
    """Pixel errors exactly as the protocol scores them: rounded and clamped first."""
    clamped = protocol.clamp_to_screen(predicted, *screen)
    errors: np.ndarray = np.hypot(*(clamped - targets).T)
    return errors


def _skipped(reason: str) -> dict[str, Any]:
    return {"skipped": reason}


# ---------------------------------------------------------------------------
# the shipping model's fit, without Calibration's O(n^2) leave-one-out
# ---------------------------------------------------------------------------


def shipping_affine_fit(
    x: np.ndarray, y: np.ndarray, ridge_lambda: float = SHIPPING_RIDGE_LAMBDA
) -> np.ndarray:
    """eyeos.calibration.Calibration's fit (degree 1): ridge regression on [1, x] with the
    intercept unregularised, one column of weights per output column. (Calibration also
    computes a leave-one-out validation, which costs O(n^2) refits and means nothing for
    an in-sample fit; a test checks this matches Calibration.predict.)"""
    x = np.asarray(x, dtype=float).reshape(len(x), -1)
    design = np.column_stack([np.ones(len(x)), x])
    regularizer = ridge_lambda * np.eye(design.shape[1])
    regularizer[0, 0] = 0.0
    weights: np.ndarray = np.linalg.solve(
        design.T @ design + regularizer, design.T @ np.asarray(y, dtype=float)
    )
    return weights


def affine_predict(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(len(x), -1)
    result: np.ndarray = np.column_stack([np.ones(len(x)), x]) @ weights
    return result


def _fit_quality(
    predicted: np.ndarray, targets: np.ndarray, screen: tuple[float, float]
) -> dict[str, Any]:
    errors = _errors_px(predicted, targets, screen)
    return {
        "mean_px": float(np.mean(errors)),
        "rms_px": float(np.sqrt(np.mean(errors**2))),
        "r2_x": _r2(targets[:, 0], predicted[:, 0]),
        "r2_y": _r2(targets[:, 1], predicted[:, 1]),
    }


# ---------------------------------------------------------------------------
# the sections
# ---------------------------------------------------------------------------


@dataclass
class _Frames:
    """A person's valid frames, as the sections use them."""

    person: DiagnosticPerson
    prepared: protocol.PreparedPerson
    targets: np.ndarray  # (n_valid, 2)
    features: np.ndarray  # (n_valid, 7): 4 iris offsets, then EyeOS yaw, pitch, roll
    days: list[str | None] | None  # per valid frame; None when no image has a day

    @property
    def screen(self) -> tuple[float, float]:
        return self.person.data.screen_px

    @property
    def iris(self) -> np.ndarray:
        return self.features[:, :4]


@dataclass
class _Settings:
    k_per_dot: int
    min_day_images: int
    limits: protocol.Limits


def _feature_sanity(frames: _Frames) -> dict[str, Any]:
    iris, targets = frames.iris, frames.targets
    left_x, right_x = (_corr(iris[:, i], targets[:, 0]) for i in (0, 2))
    signs_match = None
    if left_x is not None and right_x is not None:
        signs_match = bool(left_x * right_x > 0)
    return {
        "left_iris_x": _spread(iris[:, 0]),
        "left_iris_y": _spread(iris[:, 1]),
        "right_iris_x": _spread(iris[:, 2]),
        "right_iris_y": _spread(iris[:, 3]),
        "corr_left_right_x": _corr(iris[:, 0], iris[:, 2]),
        "corr_left_right_y": _corr(iris[:, 1], iris[:, 3]),
        "corr_left_x_target_x": left_x,
        "corr_right_x_target_x": right_x,
        "target_x_signs_match": signs_match,
        "corr_left_y_target_y": _corr(iris[:, 1], targets[:, 1]),
        "corr_right_y_target_y": _corr(iris[:, 3], targets[:, 1]),
        "eyeos_roll_deg": _spread(frames.features[:, 6]),
    }


def _shipping_in_sample(frames: _Frames) -> np.ndarray:
    """The shipping model's in-sample predictions (raw, before rounding) on all frames."""
    return affine_predict(shipping_affine_fit(frames.iris, frames.targets), frames.iris)


def _in_sample_ceiling(frames: _Frames) -> dict[str, Any]:
    iris, targets, screen = frames.iris, frames.targets, frames.screen
    with_pose = StandardizedRidge.fit(frames.features, targets, EXPERIMENT_RIDGE_LAMBDA)
    return {
        "both_eyes": _fit_quality(_shipping_in_sample(frames), targets, screen),
        "left_eye_only": _fit_quality(
            affine_predict(shipping_affine_fit(iris[:, :2], targets), iris[:, :2]),
            targets,
            screen,
        ),
        "right_eye_only": _fit_quality(
            affine_predict(shipping_affine_fit(iris[:, 2:], targets), iris[:, 2:]),
            targets,
            screen,
        ),
        "with_head_pose": _fit_quality(with_pose.predict(frames.features), targets, screen),
    }


def _constant_baselines(frames: _Frames) -> dict[str, Any]:
    targets, (width, height) = frames.targets, frames.screen
    n = len(targets)
    centre = np.tile([width / 2.0, height / 2.0], (n, 1))
    own = np.tile(targets.mean(axis=0), (n, 1))
    return {
        "screen_centre_mean_px": float(np.mean(_errors_px(centre, targets, frames.screen))),
        "own_mean_target_mean_px": float(np.mean(_errors_px(own, targets, frames.screen))),
    }


def _protocol_metrics(result: protocol.PersonResult) -> dict[str, Any]:
    """The protocol's per-person metrics that the protocol sections report."""
    metrics = result.metrics
    angular = metrics["angular_error_deg"]
    return {
        "n_calibration": metrics["n_calibration"],
        "n_test": metrics["n_test"],
        "mean_px": metrics["pixel_error_px"]["mean"],
        "mean_deg": angular["mean"] if angular else None,
        "validation_error_px": metrics["validation_error_px"],
        "passes_gate2": metrics["passes_gate2"],
        "calibration_mean_distance_px": metrics["calibration_mean_distance_px"],
    }


def calibration_split(
    person: protocol.PreparedPerson, k_per_dot: int
) -> tuple[list[int], np.ndarray]:
    """The calibration and test frames (indices into the valid frames) exactly as
    protocol.evaluate_person picks them: the K valid images nearest each of the 9 dots,
    and every other valid image."""
    width, height = person.data.screen_px
    targets = np.asarray(person.data.targets_px, dtype=float)[person.valid_rows]
    selected, _ = protocol.select_calibration_set(targets, target_points(width, height), k_per_dot)
    calibration = [i for per_dot in selected for i in per_dot]
    is_calibration = np.zeros(len(targets), dtype=bool)
    is_calibration[calibration] = True
    return calibration, np.flatnonzero(~is_calibration)


@dataclass
class _CrossDay:
    """The standard protocol's run on one person, kept for the sections that read it."""

    metrics: dict[str, Any]
    pixel_errors: np.ndarray  # per test frame, in the order of ``test``
    calibration: list[int]  # indices into the valid frames
    test: np.ndarray


def _cross_day(frames: _Frames, settings: _Settings) -> tuple[dict[str, Any], _CrossDay | None]:
    variant = build_variants()[PROTOCOL_VARIANT]
    result = protocol.evaluate_person(
        variant, frames.prepared, [], settings.k_per_dot, settings.limits
    )
    if "skipped" in result.metrics or result.pixel_errors is None:
        return _skipped(result.metrics.get("skipped", "not evaluated")), None
    metrics = _protocol_metrics(result)
    calibration, test = calibration_split(frames.prepared, settings.k_per_dot)
    if len(test) != len(result.pixel_errors):  # pragma: no cover - a protocol change
        raise RuntimeError("the cross-day split no longer matches protocol.evaluate_person")
    metrics["calibration_days"] = (
        None
        if frames.days is None
        else len({frames.days[i] for i in calibration if frames.days[i] is not None})
    )
    return metrics, _CrossDay(metrics, result.pixel_errors, calibration, test)


def _subset(data: protocol.PersonData, rows: np.ndarray, name: str) -> protocol.PersonData:
    """The PersonData of some of a person's images only (``rows`` index its arrays)."""

    def take(array: np.ndarray | None) -> np.ndarray | None:
        return None if array is None else np.asarray(array)[rows]

    return dataclasses.replace(
        data,
        person=name,
        targets_px=np.asarray(data.targets_px)[rows],
        landmarks=[data.landmarks[i] for i in rows],
        face_centers_mm=take(data.face_centers_mm),
        gaze_targets_mm=take(data.gaze_targets_mm),
    )


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present and len(present) == len(values) else None


def _within_day(frames: _Frames, cross: _CrossDay | None, settings: _Settings) -> dict[str, Any]:
    if frames.days is None:
        return _skipped("no image is in a day folder, so there is no day to calibrate within")
    variant = build_variants()[PROTOCOL_VARIANT]
    grid_size = len(target_points(*frames.screen))
    needed = max(settings.min_day_images, settings.k_per_dot * grid_size + 1)
    by_day: dict[str, list[int]] = {}
    for index, day in enumerate(frames.days):
        if day is not None:
            by_day.setdefault(day, []).append(index)
    per_day: list[dict[str, Any]] = []
    skipped_days: list[dict[str, Any]] = []
    for day in sorted(by_day):
        indices = by_day[day]
        if len(indices) < needed:
            skipped_days.append({"day": day, "n_valid": len(indices)})
            continue
        rows = frames.prepared.valid_rows[indices]
        subset = _subset(frames.person.data, rows, f"{frames.person.name} {day}")
        result = protocol.evaluate_person(
            variant, protocol.prepare(subset), [], settings.k_per_dot, settings.limits
        )
        per_day.append({"day": day, "n_valid": len(indices), **_protocol_metrics(result)})

    used = bool(per_day)
    cross_px = cross.metrics["mean_px"] if (cross and used) else None
    mean_px = _mean_or_none([d["mean_px"] for d in per_day]) if used else None
    return {
        "days_total": len(by_day),
        "days_used": len(per_day),
        "days_skipped": len(skipped_days),
        "mean_px": mean_px,
        "mean_deg": _mean_or_none([d["mean_deg"] for d in per_day]) if used else None,
        "validation_error_px_mean": (
            _mean_or_none([d["validation_error_px"] for d in per_day]) if used else None
        ),
        "gate2_passed_days": sum(1 for d in per_day if d["passes_gate2"]),
        "calibration_mean_distance_px_mean": (
            _mean_or_none([d["calibration_mean_distance_px"] for d in per_day]) if used else None
        ),
        "cross_day_mean_px": cross_px,
        "cross_day_mean_deg": cross.metrics["mean_deg"] if (cross and used) else None,
        "cross_day_validation_error_px": (
            cross.metrics["validation_error_px"] if (cross and used) else None
        ),
        "within_over_cross_mean_px": (
            mean_px / cross_px if (mean_px is not None and cross_px) else None
        ),
        "per_day": per_day,
        "skipped_days": skipped_days,
    }


def bin_errors(
    deltas_deg: np.ndarray, errors_px: np.ndarray, edges: Sequence[float] = POSE_BIN_EDGES_DEG
) -> list[dict[str, Any]]:
    """Mean error per delta bin: [e0, e1], (e1, e2], ..., (e_last, inf)."""
    deltas = np.asarray(deltas_deg, dtype=float)
    errors = np.asarray(errors_px, dtype=float)
    bounds = [*edges, math.inf]
    bins = []
    for i, (low, high) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        chosen = (deltas <= high) & ((deltas >= low) if i == 0 else (deltas > low))
        bins.append(
            {
                "lo_deg": float(low),
                "hi_deg": None if math.isinf(high) else float(high),
                "n": int(np.sum(chosen)),
                "mean_px": float(np.mean(errors[chosen])) if chosen.any() else None,
            }
        )
    return bins


def _error_by_head_pose(
    frames: _Frames, cross: _CrossDay | None, settings: _Settings
) -> dict[str, Any]:
    if cross is None:
        return _skipped("the cross-day protocol did not run (see cross_day)")
    poses = frames.features[:, 4:7]
    mean = poses[cross.calibration].mean(axis=0)
    calibration_pose = (float(mean[0]), float(mean[1]), float(mean[2]))
    deltas = np.array(
        [
            head_pose_delta_deg((float(p[0]), float(p[1]), float(p[2])), calibration_pose)
            for p in poses[cross.test]
        ]
    )
    within = deltas <= settings.limits.max_head_pose_delta_deg
    errors = cross.pixel_errors
    return {
        "bins": bin_errors(deltas, errors),
        "within_limit_share": float(np.mean(within)),
        "within_limit_mean_px": float(np.mean(errors[within])) if within.any() else None,
        "beyond_limit_mean_px": float(np.mean(errors[~within])) if (~within).any() else None,
    }


def _valid_days(person: DiagnosticPerson, valid_rows: np.ndarray) -> list[str | None] | None:
    if person.days is None:
        return None
    days = [person.days[i] for i in valid_rows]
    return days if any(d is not None for d in days) else None


def _check_aligned(person: DiagnosticPerson) -> None:
    n = len(person.data.landmarks)
    for label, value in (
        ("days", person.days),
        ("head_rotation", person.head_rotation),
        ("head_translation_mm", person.head_translation_mm),
    ):
        if value is not None and len(value) != n:
            raise ValueError(f"{person.name}: {label} has {len(value)} entries for {n} images")


def diagnose_person(person: DiagnosticPerson, settings: _Settings) -> dict[str, Any]:
    """Every section for one person; a section that cannot run says why."""
    _check_aligned(person)
    prepared = protocol.prepare(person.data)
    n_valid = len(prepared.valid_rows)
    result: dict[str, Any] = {"n_images": len(person.data.landmarks), "n_valid": n_valid}
    if n_valid < MIN_FRAMES:
        reason = f"{n_valid} valid frames; at least {MIN_FRAMES} are needed"
        for section in SECTIONS:
            result[section] = _skipped(reason)
        return result
    frames = _Frames(
        person=person,
        prepared=prepared,
        targets=np.asarray(person.data.targets_px, dtype=float)[prepared.valid_rows],
        features=feature_matrix(prepared.features),
        days=_valid_days(person, prepared.valid_rows),
    )
    result["feature_sanity"] = _feature_sanity(frames)
    result["in_sample_ceiling"] = _in_sample_ceiling(frames)
    result["constant_baselines"] = _constant_baselines(frames)
    result["cross_day"], cross = _cross_day(frames, settings)
    result["within_day"] = _within_day(frames, cross, settings)
    result["error_by_head_pose"] = _error_by_head_pose(frames, cross, settings)
    return result


# ---------------------------------------------------------------------------
# across persons
# ---------------------------------------------------------------------------


def _summarise(values: list[Any]) -> Any:
    """One per-person metric across persons (see AGGREGATE_NOTE); None if nobody has it."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    first = present[0]
    if isinstance(first, bool):
        return {"true": sum(1 for v in present if v), "persons": len(present)}
    if isinstance(first, (int, float)):
        array = np.array(present, dtype=float)
        return {
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
            "persons": len(present),
        }
    if isinstance(first, str):
        counts: dict[str, int] = {}
        for value in present:
            counts[value] = counts.get(value, 0) + 1
        return {"counts": dict(sorted(counts.items())), "persons": len(present)}
    if isinstance(first, dict):
        keys = list(dict.fromkeys(k for v in present for k in v))
        summary = {k: _summarise([v.get(k) for v in present]) for k in keys}
        return {k: s for k, s in summary.items() if s is not None}
    return None  # lists (per-day and per-bin details) are summarised by their section


def aggregate(persons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "persons": len(persons),
        "n_images": _summarise([p["n_images"] for p in persons.values()]),
        "n_valid": _summarise([p["n_valid"] for p in persons.values()]),
    }
    for section in SECTIONS:
        done = [p[section] for p in persons.values() if "skipped" not in p.get(section, {})]
        result[section] = _summarise(done) or {}
    within = [p["within_day"] for p in persons.values() if "days_used" in p.get("within_day", {})]
    result["within_day"]["days_used_total"] = sum(w["days_used"] for w in within)
    result["within_day"]["gate2_passed_days_total"] = sum(w["gate2_passed_days"] for w in within)
    result["error_by_head_pose"]["bins"] = _pool_bins(
        [
            p["error_by_head_pose"]["bins"]
            for p in persons.values()
            if "bins" in p.get("error_by_head_pose", {})
        ]
    )
    return result


def _pool_bins(per_person: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not per_person:
        return []
    pooled = []
    for i, first in enumerate(per_person[0]):
        parts = [bins[i] for bins in per_person if bins[i]["n"] > 0]
        n = sum(b["n"] for b in parts)
        pooled.append(
            {
                "lo_deg": first["lo_deg"],
                "hi_deg": first["hi_deg"],
                "n": n,
                "pooled_mean_px": sum(b["n"] * b["mean_px"] for b in parts) / n if n else None,
                "mean_of_person_means_px": (
                    float(np.mean([b["mean_px"] for b in parts])) if parts else None
                ),
                "persons": len(parts),
            }
        )
    return pooled


def _clean(value: Any) -> Any:
    """Plain JSON types; NaN and infinities become null."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def diagnose(
    persons: Sequence[DiagnosticPerson],
    k_per_dot: int = protocol.DEFAULT_K_PER_DOT,
    min_day_images: int = DEFAULT_MIN_DAY_IMAGES,
    limits: protocol.Limits | None = None,
) -> dict[str, Any]:
    """Every section for every person, the aggregate across persons, and the name and
    description of every metric. Deterministic."""
    settings = _Settings(k_per_dot, min_day_images, limits or protocol.Limits())
    per_person = {p.name: diagnose_person(p, settings) for p in persons}
    return _clean(
        {
            "config": {
                "protocol_variant": PROTOCOL_VARIANT,
                "k_per_dot": settings.k_per_dot,
                "min_day_images": settings.min_day_images,
                "gate2_max_validation_error_px": settings.limits.gate2_max_error_px,
                "max_head_pose_delta_deg": settings.limits.max_head_pose_delta_deg,
                "in_sample_ridge_lambda": SHIPPING_RIDGE_LAMBDA,
                "with_head_pose_ridge_lambda": EXPERIMENT_RIDGE_LAMBDA,
            },
            "sections": SECTIONS,
            "metrics": METRICS,
            "aggregate_note": AGGREGATE_NOTE,
            "persons": per_person,
            "aggregate": aggregate(per_person),
        }
    )


# ---------------------------------------------------------------------------
# the terminal summary
# ---------------------------------------------------------------------------


def _cell(value: Any, spec: str = ".1f") -> str:
    return "-" if value is None else format(value, spec)


def _get(tree: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(tree, dict) or key not in tree:
            return None
        tree = tree[key]
    return tree


# (title, path into a person's result, format); the "all" row shows the mean across
# persons of the same path.
_COLUMNS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("valid", ("n_valid",), "d"),
    ("cross px", ("cross_day", "mean_px"), ".1f"),
    ("cross LOO", ("cross_day", "validation_error_px"), ".1f"),
    ("in-pose px", ("error_by_head_pose", "within_limit_mean_px"), ".1f"),
    ("within px", ("within_day", "mean_px"), ".1f"),
    ("days", ("within_day", "days_used"), "d"),
    ("ceiling px", ("in_sample_ceiling", "both_eyes", "mean_px"), ".1f"),
    ("+pose px", ("in_sample_ceiling", "with_head_pose", "mean_px"), ".1f"),
    ("centre px", ("constant_baselines", "screen_centre_mean_px"), ".1f"),
    ("corr L/R x", ("feature_sanity", "corr_left_right_x"), ".2f"),
)


def summary_table(result: dict[str, Any]) -> str:
    """A short text table for the terminal: one line per person and one across persons
    (the mean of the per-person values)."""
    lines = [f"{'person':<8}" + "".join(f"{title:>11}" for title, _, _ in _COLUMNS)]
    for name, person in result["persons"].items():
        lines.append(
            f"{name:<8}"
            + "".join(f"{_cell(_get(person, *path), spec):>11}" for _, path, spec in _COLUMNS)
        )
    means = [
        _cell(_get(result["aggregate"], *path, "mean"), ".1f" if spec == "d" else spec)
        for _, path, spec in _COLUMNS
    ]
    lines.append(f"{'all':<8}" + "".join(f"{cell:>11}" for cell in means))
    return "\n".join(lines)
