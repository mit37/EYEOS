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


class InputEndedError(RuntimeError):
    """The frame stream ended while a dot was being sampled (camera stopped)."""


def collect_dot_features(
    frames: Iterator[Frame],
    settle: int = SETTLE_FRAMES,
    want: int = SAMPLE_FRAMES,
    max_frames: int = MAX_FRAMES_PER_DOT,
) -> FeatureVector | None:
    """Average ``want`` usable frames (one face, both eyes open) into one sample.

    Returns None if fewer than ``want // 2`` usable frames turn up within
    ``max_frames``, so the caller can ask for the dot again. Raises InputEndedError if
    the stream ends first: asking again could never succeed.
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
    else:
        raise InputEndedError("the input source ran out of frames")
    if len(usable) < max(1, want // 2):
        return None
    mean = np.mean([[*f.iris_vector(), *f.head_pose()] for f in usable], axis=0)
    return FeatureVector(*(float(v) for v in mean))


def calibrate_interactively(
    config: Config, frames: Iterator[Frame], ui_factory: Callable[..., Any]
) -> Calibration | None:
    """Run the dot screen built by ``ui_factory`` (CalibrationUI in real use) and fit.

    Returns None if the user cancels, or if any of the ``config.calibration.points``
    dots is left without a sample. If reading the frames fails, or the stream ends,
    during a dot, the dot screen is closed and that error is raised once it has gone
    (an end of stream as "camera stopped during calibration: ...").

    ``ui_factory(width, height, capture_sample=...)`` must return an object with
    ``run() -> bool`` and ``cancel()``.
    """
    calibration = Calibration(config.screen.width_px, config.screen.height_px)
    failures: list[Exception] = []

    def dots_with_a_sample() -> int:
        return len({sample.target_px for sample in calibration.samples})

    def capture(target_px: tuple[float, float]) -> bool:
        # This runs inside the dot screen's key handler, where Tk would print an
        # exception behind the fullscreen window and carry on. Keep it, close the
        # screen, and raise it once run() has returned.
        try:
            features = collect_dot_features(frames)
        except InputEndedError as e:
            stopped = RuntimeError(
                f"camera stopped during calibration: {e} at point "
                f"{dots_with_a_sample() + 1}/{config.calibration.points}"
            )
            stopped.__cause__ = e
            failures.append(stopped)
        except Exception as e:
            failures.append(e)
        else:
            if features is None:
                return False
            calibration.add_sample(target_px, features)
            return True
        ui.cancel()
        return False

    ui = ui_factory(config.screen.width_px, config.screen.height_px, capture_sample=capture)
    completed = ui.run()
    if failures:
        raise failures[0]
    if not completed:
        return None
    # Each captured dot adds exactly one sample. Whatever the dot screen reported, never
    # fit on fewer dots than the calibration asks for.
    if dots_with_a_sample() < config.calibration.points:
        return None
    calibration.fit()
    return calibration
