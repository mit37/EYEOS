"""The pipeline end-to-end: input source -> features -> calibration -> filter ->
gates -> output, dry-run. This is the PRD Definition-of-Done item "dry-run mode runs
the whole pipeline on a video file" — exercised here against a SyntheticInputSource
CSV fixture instead of a real video, since this container has no camera/video deps
(see docs/PLAN.md)."""

from __future__ import annotations

import csv
from collections.abc import Iterator

from eyeos.calibration import Calibration
from eyeos.config import Config
from eyeos.input_sources import Frame, InputSource
from eyeos.input_sources.synthetic import SyntheticInputSource
from eyeos.landmarks import Point3D
from eyeos.output.dry_run import DryRunOutputAdapter
from eyeos.pipeline import Pipeline
from eyeos.synthetic_format import CSV_FIELDNAMES, landmarks_to_row
from tests.test_calibration import IRIS_VECTORS, _feature_vector, _true_point
from tests.test_features import make_landmarks


class ListInputSource(InputSource):
    """A trivial in-memory InputSource for tests that don't need a real CSV file."""

    def __init__(self, frames: list[Frame]) -> None:
        self._frames = frames

    def frames(self) -> Iterator[Frame]:
        yield from self._frames


def _fitted_calibration(screen_w=1920, screen_h=1080) -> Calibration:
    cal = Calibration(screen_w, screen_h)
    for iv in IRIS_VECTORS:
        cal.add_sample(_true_point(iv), _feature_vector(iv))
    cal.fit()
    return cal


# ---------------------------------------------------------------------------
# Without a calibration: gates correctly refuse to move (Gate 2 fails cleanly)
# ---------------------------------------------------------------------------


def test_uncalibrated_pipeline_never_moves_or_clicks():
    frames = [
        Frame(timestamp_s=t, landmarks=make_landmarks()) for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    ]
    output = DryRunOutputAdapter(echo=None)
    pipeline = Pipeline(Config(), ListInputSource(frames), output, calibration=None)

    result = pipeline.run()

    assert result.frame_count == 6
    assert result.total_moves == 0
    assert result.total_clicks == 0
    assert output.actions == []
    assert all(not s.gate2_calibration for s in result.statuses)
    assert all("recalibrate" in s.reasons["gate2"] for s in result.statuses)


# ---------------------------------------------------------------------------
# With a calibration: the whole pipeline moves the cursor and fires a click
# ---------------------------------------------------------------------------


def test_calibrated_pipeline_moves_and_clicks_end_to_end():
    calibration = _fitted_calibration()
    # make_landmarks() is centered/frontal => iris_vector (0,0,0,0) => predicted
    # point is _true_point((0,0,0,0)) == (960.0, 540.0), constant across frames.
    frames = [
        Frame(timestamp_s=t, landmarks=make_landmarks()) for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    ]
    output = DryRunOutputAdapter(echo=None)

    def external_signals(timestamp_s: float) -> tuple[bool, bool]:
        return False, timestamp_s >= 1.0  # confirm_keypress on the last frame

    pipeline = Pipeline(
        Config(),
        ListInputSource(frames),
        output,
        calibration=calibration,
        external_signals=external_signals,
    )
    result = pipeline.run()

    assert result.frame_count == 6
    assert result.total_moves == 6
    assert result.total_clicks == 1
    assert ("click",) in output.actions
    moves = [a for a in output.actions if a[0] == "move"]
    assert all(a == ("move", 960, 540) for a in moves)


def test_on_status_hook_is_called_once_per_frame():
    frames = [Frame(timestamp_s=t, landmarks=make_landmarks()) for t in (0.0, 0.2, 0.4)]
    output = DryRunOutputAdapter(echo=None)
    seen = []
    pipeline = Pipeline(Config(), ListInputSource(frames), output, on_status=seen.append)
    result = pipeline.run()
    assert seen == result.statuses
    assert len(seen) == 3


def test_max_frames_stops_the_run_early():
    frames = [Frame(timestamp_s=t, landmarks=make_landmarks()) for t in (0.0, 0.2, 0.4, 0.6)]
    output = DryRunOutputAdapter(echo=None)
    pipeline = Pipeline(Config(), ListInputSource(frames), output)
    result = pipeline.run(max_frames=2)
    assert result.frame_count == 2


def test_dropout_frame_freezes_and_does_not_crash():
    frames = [
        Frame(timestamp_s=0.0, landmarks=make_landmarks()),
        Frame(timestamp_s=0.2, landmarks=None),
        Frame(timestamp_s=0.4, landmarks=make_landmarks()),
    ]
    output = DryRunOutputAdapter(echo=None)
    pipeline = Pipeline(
        Config(), ListInputSource(frames), output, calibration=_fitted_calibration()
    )
    result = pipeline.run()
    assert result.statuses[1].move_allowed is False
    assert result.statuses[1].gate1_presence is False


# ---------------------------------------------------------------------------
# Blink-hold confirmation gesture resolution
# ---------------------------------------------------------------------------


def test_blink_hold_within_grace_period_still_reports_visible_and_accumulates_hold():
    frames = [
        Frame(timestamp_s=0.0, landmarks=make_landmarks()),
        Frame(
            timestamp_s=0.3,
            landmarks=make_landmarks(
                left_eye_top=Point3D(0.38, 0.400),
                left_eye_bottom=Point3D(0.38, 0.401),
                right_eye_top=Point3D(0.62, 0.400),
                right_eye_bottom=Point3D(0.62, 0.401),
            ),
        ),
    ]
    output = DryRunOutputAdapter(echo=None)
    pipeline = Pipeline(
        Config(), ListInputSource(frames), output, calibration=_fitted_calibration()
    )
    result = pipeline.run()
    # A 0.3s closed-eye stretch is within confirmation_hold_s(0.4) + grace(0.5): Gate 1
    # should still see "visible", so movement/dwell survive the blink.
    assert result.statuses[1].gate1_presence is True
    assert result.statuses[1].move_allowed is True


def test_prolonged_eye_closure_past_grace_is_reported_as_not_visible():
    closed_eyes = dict(
        left_eye_top=Point3D(0.38, 0.400),
        left_eye_bottom=Point3D(0.38, 0.401),
        right_eye_top=Point3D(0.62, 0.400),
        right_eye_bottom=Point3D(0.62, 0.401),
    )
    # The hold duration accumulates across *consecutive* closed-eye frames, so this
    # needs a run of them (not a single frame with a big timestamp jump) to actually
    # pass confirmation_hold_s(0.4) + grace(0.5).
    frames = [Frame(timestamp_s=0.0, landmarks=make_landmarks())] + [
        Frame(timestamp_s=t, landmarks=make_landmarks(**closed_eyes))
        for t in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
    ]
    output = DryRunOutputAdapter(echo=None)
    pipeline = Pipeline(
        Config(), ListInputSource(frames), output, calibration=_fitted_calibration()
    )
    result = pipeline.run()
    # Early closed-eye frames (t=0.2, 0.4) are still within the grace window.
    assert result.statuses[1].gate1_presence is True
    # By t=1.2 the hold (1.2s) has exceeded confirmation_hold_s(0.4) + grace(0.5).
    assert result.statuses[-1].gate1_presence is False


# ---------------------------------------------------------------------------
# Reading frames from an actual CSV file via SyntheticInputSource
# ---------------------------------------------------------------------------


def test_pipeline_runs_over_a_real_synthetic_csv_file(tmp_path):
    p = tmp_path / "session.csv"
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            writer.writerow(landmarks_to_row(t, make_landmarks()))

    output = DryRunOutputAdapter(echo=None)
    pipeline = Pipeline(
        Config(),
        SyntheticInputSource(p),
        output,
        calibration=_fitted_calibration(),
    )
    result = pipeline.run()
    assert result.frame_count == 6
    assert result.total_moves == 6
