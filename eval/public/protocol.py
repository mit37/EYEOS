"""The EyeOS-faithful evaluation protocol, independent of any one dataset.

A dataset loader (e.g. eval/public/mpiifacegaze.py) turns each participant into a
``PersonData``: their screen size, one detected ``FaceLandmarks`` (or None) and one true
on-screen target per image, and, if the dataset has 3D labels, the face centre, the 3D
gaze target and a function mapping screen pixels to camera millimetres. Everything below
treats those the way a live EyeOS session would:

1. A frame is usable only if Gate 1 could pass it: exactly one face, both irises
   visible (``FaceLandmarks.both_irises_visible``).
2. Calibration: for each point of EyeOS's own 9-point grid (eyeos.calibration.
   target_points on this person's screen), the K usable images whose true target is
   nearest the dot stand in for "looking at the dot". They are added with their TRUE
   targets, and the model is fitted on them.
3. Every other usable image is a test frame. Predictions are rounded and clamped to the
   screen exactly as Pipeline._to_screen_pixel does for the cursor.

Static photos cannot exercise the One Euro filter or dwell, so neither is used.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from eyeos.calibration import target_points
from eyeos.config import Config
from eyeos.features import FeatureVector, extract_features, head_pose_delta_deg
from eyeos.landmarks import FaceLandmarks

ScreenPoint = tuple[float, float]
DEFAULT_K_PER_DOT = 5


@dataclass
class PersonData:
    """One participant, as a dataset loader hands them to the protocol."""

    person: str
    screen_px: tuple[float, float]  # (width, height) of the screen the targets are on
    targets_px: np.ndarray  # (n, 2) true on-screen targets
    landmarks: list[FaceLandmarks | None]  # n detections, None = no face
    face_centers_mm: np.ndarray | None = None  # (n, 3), camera coordinates
    gaze_targets_mm: np.ndarray | None = None  # (n, 3), camera coordinates
    to_camera_mm: Callable[[np.ndarray], np.ndarray] | None = None  # px (n, 2) -> mm (n, 3)
    angular_note: str | None = None  # why angular error is not available, if it is not


@dataclass
class PreparedPerson:
    """A PersonData with the usable frames picked out and their features computed."""

    data: PersonData
    valid_rows: np.ndarray  # indices of usable frames into the PersonData arrays
    features: list[FeatureVector]  # one per valid row
    detection: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.data.person


@dataclass(frozen=True)
class Limits:
    """The shipping Gate 2 thresholds the results are judged against."""

    gate2_max_error_px: float = Config().gates.calibration_max_error_px
    max_head_pose_delta_deg: float = Config().gates.calibration_max_head_pose_delta_deg


@dataclass(frozen=True)
class FitContext:
    screen_px: tuple[float, float]
    # The other participants, for variants that learn across people (global-lopo). The
    # protocol never puts the person being evaluated in here.
    other_persons: tuple[PreparedPerson, ...] = ()


class FittedModel(Protocol):
    @property
    def validation_error_px(self) -> float:
        """Leave-one-out RMS error over the calibration samples (inf = not validated)."""

    def predict(self, features: Sequence[FeatureVector]) -> np.ndarray:
        """(n, 2) screen points, before rounding and clamping."""


class Variant(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def k_per_dot(self) -> int | None:
        """Images per calibration dot; None means the run's --k-per-dot."""

    @property
    def needs_other_persons(self) -> bool: ...

    def fit(
        self, features: Sequence[FeatureVector], targets_px: np.ndarray, context: FitContext
    ) -> FittedModel: ...


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------


def is_valid(landmarks: FaceLandmarks | None) -> bool:
    """Would Gate 1 let this frame through? Exactly one face, both irises visible."""
    return landmarks is not None and landmarks.face_count == 1 and landmarks.both_irises_visible


def detection_summary(landmarks: Sequence[FaceLandmarks | None]) -> dict[str, Any]:
    no_face = sum(1 for lm in landmarks if lm is None)
    multiple = sum(1 for lm in landmarks if lm is not None and lm.face_count != 1)
    closed = sum(
        1
        for lm in landmarks
        if lm is not None and lm.face_count == 1 and not lm.both_irises_visible
    )
    valid = len(landmarks) - no_face - multiple - closed
    return {
        "images": len(landmarks),
        "valid": valid,
        "no_face": no_face,
        "multiple_faces": multiple,
        "irises_not_visible": closed,
        "detection_rate": valid / len(landmarks) if landmarks else 0.0,
    }


def prepare(data: PersonData) -> PreparedPerson:
    rows = [i for i, lm in enumerate(data.landmarks) if is_valid(lm)]
    features = [extract_features(data.landmarks[i]) for i in rows]  # type: ignore[arg-type]
    return PreparedPerson(
        data=data,
        valid_rows=np.array(rows, dtype=int),
        features=features,
        detection=detection_summary(data.landmarks),
    )


# ---------------------------------------------------------------------------
# calibration-set selection, prediction, error measures
# ---------------------------------------------------------------------------


def select_calibration_set(
    targets_px: np.ndarray, grid: Sequence[ScreenPoint], k: int
) -> tuple[list[list[int]], list[float]]:
    """For each grid point in order, the ``k`` images whose true target is nearest it.

    No image is used twice: a later grid point takes the next-nearest unused images.
    Equal distances go to the earlier image. Returns the chosen row indices per grid
    point and, per grid point, their mean distance to it (how far "looking at the dot"
    is approximated).
    """
    targets = np.asarray(targets_px, dtype=float).reshape(-1, 2)
    needed = k * len(grid)
    if len(targets) < needed:
        raise ValueError(
            f"{k} images per dot x {len(grid)} dots needs {needed} images, got {len(targets)}"
        )
    used = np.zeros(len(targets), dtype=bool)
    selected: list[list[int]] = []
    mean_distance: list[float] = []
    for gx, gy in grid:
        distance = np.hypot(targets[:, 0] - gx, targets[:, 1] - gy)
        order = np.argsort(distance, kind="stable")
        chosen = [int(i) for i in order if not used[i]][:k]
        used[chosen] = True
        selected.append(chosen)
        mean_distance.append(float(np.mean(distance[chosen])))
    return selected, mean_distance


def clamp_to_screen(points_px: np.ndarray, width: float, height: float) -> np.ndarray:
    """Round to whole pixels and clamp to the screen, as Pipeline._to_screen_pixel does
    (both round halves to even)."""
    points = np.rint(np.asarray(points_px, dtype=float).reshape(-1, 2))
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return points


def angular_errors_deg(
    predicted_mm: np.ndarray, face_centers_mm: np.ndarray, gaze_targets_mm: np.ndarray
) -> np.ndarray:
    """Angle between the predicted ray (P - fc) and the true ray (gt - fc), in degrees."""
    predicted = np.asarray(predicted_mm, dtype=float) - face_centers_mm
    true = np.asarray(gaze_targets_mm, dtype=float) - face_centers_mm
    cross = np.linalg.norm(np.cross(predicted, true), axis=1)
    dot = np.sum(predicted * true, axis=1)
    result: np.ndarray = np.degrees(np.arctan2(cross, dot))
    return result


def summary_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


# ---------------------------------------------------------------------------
# one variant on one person
# ---------------------------------------------------------------------------


@dataclass
class PersonResult:
    metrics: dict[str, Any]
    pixel_errors: np.ndarray | None = None
    angular_errors: np.ndarray | None = None


def evaluate_person(
    variant: Variant,
    person: PreparedPerson,
    others: Sequence[PreparedPerson],
    k_per_dot: int,
    limits: Limits,
) -> PersonResult:
    k = variant.k_per_dot or k_per_dot
    data = person.data
    width, height = data.screen_px
    grid = target_points(width, height)
    n_valid = len(person.valid_rows)
    if n_valid < k * len(grid) + 1:
        return PersonResult(
            {
                "skipped": (
                    f"{n_valid} valid frames; {k} per dot x {len(grid)} dots plus at least "
                    "one test frame are needed"
                )
            }
        )
    if variant.needs_other_persons and not others:
        return PersonResult({"skipped": "needs at least 2 persons"})

    valid_targets = data.targets_px[person.valid_rows]
    selected, mean_distance = select_calibration_set(valid_targets, grid, k)
    calibration = [i for per_dot in selected for i in per_dot]
    is_calibration = np.zeros(n_valid, dtype=bool)
    is_calibration[calibration] = True
    test = np.flatnonzero(~is_calibration)

    model = variant.fit(
        [person.features[i] for i in calibration],
        valid_targets[calibration],
        FitContext(screen_px=(width, height), other_persons=tuple(others)),
    )
    test_features = [person.features[i] for i in test]
    predicted = clamp_to_screen(model.predict(test_features), width, height)
    pixel_errors = np.hypot(*(predicted - valid_targets[test]).T)
    diagonal = math.hypot(width, height)

    calibration_pose = np.mean([person.features[i].head_pose() for i in calibration], axis=0)
    pose = (float(calibration_pose[0]), float(calibration_pose[1]), float(calibration_pose[2]))
    within = [
        head_pose_delta_deg(f.head_pose(), pose) <= limits.max_head_pose_delta_deg
        for f in test_features
    ]

    validation = model.validation_error_px
    metrics: dict[str, Any] = {
        "n_calibration": len(calibration),
        "n_test": len(test),
        "calibration_mean_distance_px": float(np.mean(mean_distance)),
        "calibration_mean_distance_per_dot_px": mean_distance,
        "validation_error_px": _finite_or_none(validation),
        "passes_gate2": bool(math.isfinite(validation) and validation < limits.gate2_max_error_px),
        "pixel_error_px": summary_stats(pixel_errors),
        "pixel_error_pct_diagonal": summary_stats(100.0 * pixel_errors / diagonal),
        "angular_error_deg": None,
        "head_pose_within_limit_share": float(np.mean(within)),
    }
    angular_errors = None
    if (
        data.to_camera_mm is not None
        and data.face_centers_mm is not None
        and data.gaze_targets_mm is not None
    ):
        rows = person.valid_rows[test]
        angular_errors = angular_errors_deg(
            data.to_camera_mm(predicted), data.face_centers_mm[rows], data.gaze_targets_mm[rows]
        )
        metrics["angular_error_deg"] = summary_stats(angular_errors)
    else:
        metrics["angular_error_note"] = data.angular_note or "no 3D labels in this dataset"
    return PersonResult(metrics, pixel_errors, angular_errors)


# ---------------------------------------------------------------------------
# across persons
# ---------------------------------------------------------------------------


def _aggregate_errors(
    per_person: list[dict[str, float]], pooled: list[np.ndarray]
) -> dict[str, float] | None:
    if not per_person:
        return None
    means = np.array([stats["mean"] for stats in per_person])
    frames = np.concatenate(pooled)
    return {
        "mean_of_person_means": float(np.mean(means)),
        "median_of_person_means": float(np.median(means)),
        "p90_of_person_means": float(np.percentile(means, 90)),
        "pooled_median": float(np.median(frames)),
        "pooled_p90": float(np.percentile(frames, 90)),
    }


def aggregate(results: dict[str, PersonResult]) -> dict[str, Any]:
    done = {name: r for name, r in results.items() if "skipped" not in r.metrics}
    angular = [r for r in done.values() if r.angular_errors is not None]
    validation = [
        r.metrics["validation_error_px"]
        for r in done.values()
        if r.metrics["validation_error_px"] is not None
    ]
    return {
        "persons_evaluated": len(done),
        "persons_skipped": sorted(set(results) - set(done)),
        "pixel_error_px": _aggregate_errors(
            [r.metrics["pixel_error_px"] for r in done.values()],
            [r.pixel_errors for r in done.values() if r.pixel_errors is not None],
        ),
        "pixel_error_pct_diagonal_mean_of_person_means": (
            float(np.mean([r.metrics["pixel_error_pct_diagonal"]["mean"] for r in done.values()]))
            if done
            else None
        ),
        # Only over persons with a trusted screen model; see each person's note otherwise.
        "angular_error_deg": _aggregate_errors(
            [r.metrics["angular_error_deg"] for r in angular],
            [r.angular_errors for r in angular if r.angular_errors is not None],
        ),
        "angular_error_persons": len(angular),
        "validation_error_px_mean": float(np.mean(validation)) if validation else None,
        "gate2_passed": sum(1 for r in done.values() if r.metrics["passes_gate2"]),
        "head_pose_within_limit_share_mean": (
            float(np.mean([r.metrics["head_pose_within_limit_share"] for r in done.values()]))
            if done
            else None
        ),
    }


def evaluate(
    persons: Sequence[PersonData],
    variants: Sequence[Variant],
    k_per_dot: int = DEFAULT_K_PER_DOT,
    limits: Limits | None = None,
) -> dict[str, Any]:
    """Every variant on every person. Deterministic: no randomness anywhere."""
    limits = limits or Limits()
    prepared = [prepare(p) for p in persons]
    detection = {p.name: p.detection for p in prepared}
    output: dict[str, Any] = {
        "detection": {
            "persons": detection,
            "aggregate": {
                "detection_rate_mean": (
                    float(np.mean([d["detection_rate"] for d in detection.values()]))
                    if detection
                    else None
                ),
                "images": sum(d["images"] for d in detection.values()),
                "valid": sum(d["valid"] for d in detection.values()),
            },
        },
        "variants": {},
    }
    for variant in variants:
        results = {
            p.name: evaluate_person(
                variant,
                p,
                [o for o in prepared if o.name != p.name] if variant.needs_other_persons else [],
                k_per_dot,
                limits,
            )
            for p in prepared
        }
        output["variants"][variant.name] = {
            "description": variant.description,
            "k_per_dot": variant.k_per_dot or k_per_dot,
            "aggregate": aggregate(results),
            "persons": {name: r.metrics for name, r in results.items()},
        }
    return output
