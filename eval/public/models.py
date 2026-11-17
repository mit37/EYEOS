"""The gaze models eval/public compares: EyeOS's shipping calibration, and experiments.

Experiments live here, not in eyeos/: nothing in this file changes what EyeOS ships.
Every model is deterministic (no random initialisation, no shuffling).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eval.public.protocol import FitContext, FittedModel, Variant
from eyeos.calibration import Calibration
from eyeos.features import FeatureVector


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


def build_variants() -> dict[str, Variant]:
    """Every variant, in report order."""
    variants: list[Variant] = [
        CalibrationVariant(
            "eyeos",
            "shipping EyeOS: eyeos.calibration.Calibration with its defaults (affine in the "
            "4 iris offsets, ridge 1e-3)",
        ),
    ]
    return {v.name: v for v in variants}


VARIANT_NAMES: tuple[str, ...] = tuple(build_variants())
