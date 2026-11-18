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

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from eval.public import protocol
from eval.public.models import EXPERIMENT_RIDGE_LAMBDA, StandardizedRidge, feature_matrix
from eyeos.calibration import Calibration

# The shipping calibration's ridge strength, read from it rather than restated.
SHIPPING_RIDGE_LAMBDA: float = Calibration(1.0, 1.0).ridge_lambda
# An affine fit of the 4 iris offsets has 5 parameters per axis; fewer valid frames than
# this and a person's in-sample sections are skipped.
MIN_FRAMES = 6

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

    @property
    def screen(self) -> tuple[float, float]:
        return self.person.data.screen_px

    @property
    def iris(self) -> np.ndarray:
        return self.features[:, :4]


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


def diagnose_person(person: DiagnosticPerson) -> dict[str, Any]:
    """Every section for one person; a section that cannot run says why."""
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
    )
    result["feature_sanity"] = _feature_sanity(frames)
    result["in_sample_ceiling"] = _in_sample_ceiling(frames)
    result["constant_baselines"] = _constant_baselines(frames)
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
    return result


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


def diagnose(persons: Sequence[DiagnosticPerson]) -> dict[str, Any]:
    """Every section for every person, the aggregate across persons, and the name and
    description of every metric. Deterministic."""
    per_person = {p.name: diagnose_person(p) for p in persons}
    return _clean(
        {
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


_COLUMNS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("valid", ("n_valid",), "d"),
    ("ceiling px", ("in_sample_ceiling", "both_eyes", "mean_px"), ".1f"),
    ("+pose px", ("in_sample_ceiling", "with_head_pose", "mean_px"), ".1f"),
    ("centre px", ("constant_baselines", "screen_centre_mean_px"), ".1f"),
    ("corr L/R x", ("feature_sanity", "corr_left_right_x"), ".2f"),
)


def summary_table(result: dict[str, Any]) -> str:
    """A short text table for the terminal: one line per person and one across persons
    (the mean of the per-person values)."""
    lines = [f"{'person':<8}" + "".join(f"{title:>12}" for title, _, _ in _COLUMNS)]
    for name, person in result["persons"].items():
        lines.append(
            f"{name:<8}"
            + "".join(f"{_cell(_get(person, *path), spec):>12}" for _, path, spec in _COLUMNS)
        )
    aggregate_row = []
    for _, path, _spec in _COLUMNS:
        mean = _get(result["aggregate"], *path, "mean")
        aggregate_row.append(f"{_cell(mean, '.2f' if path[-1].startswith('corr') else '.1f'):>12}")
    lines.append(f"{'all':<8}" + "".join(aggregate_row))
    return "\n".join(lines)
