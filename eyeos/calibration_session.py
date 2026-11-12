"""In-process calibration for a live session: show the 9 dots, and on each keypress
average a short burst of open-eye frames from the SAME frame stream the pipeline will
then run on (PRD §2: "a calibration exists for this session")."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from eyeos.calibration import Calibration
from eyeos.config import Config
from eyeos.features import FeatureVector, extract_features
from eyeos.input_sources import Frame

# Frames dropped after each keypress before sampling: camera drivers buffer a few
# frames, which may predate the user settling on the dot.
SETTLE_FRAMES = 5
SAMPLE_FRAMES = 10
MAX_FRAMES_PER_DOT = 60


def collect_dot_features(
    frames: Iterator[Frame],
    settle: int = SETTLE_FRAMES,
    want: int = SAMPLE_FRAMES,
    max_frames: int = MAX_FRAMES_PER_DOT,
) -> FeatureVector | None:
    """Average ``want`` usable frames (one face, both eyes open) into one sample.

    Returns None if fewer than ``want // 2`` usable frames turn up within
    ``max_frames``, so the caller can ask for the dot again.
    """
    usable: list[FeatureVector] = []
    for i, frame in enumerate(frames):
        if i >= max_frames:
            break
        if i < settle:
            continue
        lm = frame.landmarks
        if lm is None or lm.face_count != 1 or not lm.both_irises_visible:
            continue
        usable.append(extract_features(lm))
        if len(usable) >= want:
            break
    if len(usable) < max(1, want // 2):
        return None
    mean = np.mean([[*f.iris_vector(), *f.head_pose()] for f in usable], axis=0)
    return FeatureVector(*(float(v) for v in mean))


def calibrate_interactively(
    config: Config, frames: Iterator[Frame], ui_factory: Callable[..., Any]
) -> Calibration | None:
    """Run the dot screen built by ``ui_factory`` (CalibrationUI in real use) and fit.

    Returns None if the user cancels.
    """
    calibration = Calibration(config.screen.width_px, config.screen.height_px)

    def capture(target_px: tuple[float, float]) -> bool:
        features = collect_dot_features(frames)
        if features is None:
            return False
        calibration.add_sample(target_px, features)
        return True

    ui = ui_factory(config.screen.width_px, config.screen.height_px, capture_sample=capture)
    if not ui.run():
        return None
    calibration.fit()
    return calibration
