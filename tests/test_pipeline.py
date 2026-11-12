"""The pipeline end-to-end: input source -> features -> calibration -> filter ->
gates -> output, dry-run. This is the PRD Definition-of-Done item "dry-run mode runs
the whole pipeline on a video file" — exercised here against a SyntheticInputSource
CSV fixture instead of a real video, since this container has no camera/video deps
(see docs/PLAN.md)."""

from __future__ import annotations

import csv
from collections.abc import Iterator

import pytest

from eyeos.calibration import Calibration
from eyeos.config import Config
from eyeos.hotkeys import FakeKeySource, KeySignals
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
    assert result.statuses[1].cursor_target_px == pytest.approx((960.0, 540.0))
    assert "blink-hold grace" in result.statuses[1].reasons["gate1"]


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


# ---------------------------------------------------------------------------
# Kill switch (PRD §1: "one hotkey kills it instantly")
# ---------------------------------------------------------------------------


class _KillAt(InputSource):
    """Yields frames and taps the kill key while frame ``n`` is being delivered."""

    def __init__(self, frames, source, n):
        self._frames, self._source, self._n = frames, source, n

    def frames(self):
        for i, frame in enumerate(self._frames):
            if i == self._n:
                self._source.press("esc")
            if i == self._n + 2:
                self._source.release("esc")
            yield frame


def test_kill_hotkey_ends_the_session_and_nothing_is_emitted_after_it():
    source = FakeKeySource()
    keys = KeySignals("esc", source)
    frames = [Frame(timestamp_s=i / 30, landmarks=make_landmarks()) for i in range(30)]
    output = DryRunOutputAdapter(echo=None)
    n = 10
    pipeline = Pipeline(
        Config(),
        _KillAt(frames, source, n),
        output,
        calibration=_fitted_calibration(),
        external_signals=keys,
    )

    result = pipeline.run()

    assert result.killed
    assert len(result.statuses) == n + 1
    assert result.statuses[-1].reasons["gate4"] == "kill switch pressed"
    assert result.statuses[-1].move_allowed is False
    assert len([a for a in output.actions if a[0] == "move"]) == n
    assert ("click",) not in output.actions


def test_pipeline_refuses_a_real_cursor_backend_without_a_kill_switch():
    class RealCursor(DryRunOutputAdapter):
        moves_real_cursor = True

    with pytest.raises(ValueError, match="kill switch"):
        Pipeline(Config(), ListInputSource([]), RealCursor(echo=None))


# ---------------------------------------------------------------------------
# Blink-hold click end to end (30 fps): gaze must hold while the eyes are closed
# ---------------------------------------------------------------------------

_CLOSED = dict(
    left_eye_top=Point3D(0.38, 0.400),
    left_eye_bottom=Point3D(0.38, 0.401),
    right_eye_top=Point3D(0.62, 0.400),
    right_eye_bottom=Point3D(0.62, 0.401),
)


def _blink_session(open_s, closed_s, closed_overrides=None, fps=30, after_open_s=0.2):
    closed_overrides = closed_overrides or _CLOSED
    frames, t, dt = [], 0.0, 1.0 / fps
    for _ in range(round(open_s * fps)):
        frames.append(Frame(timestamp_s=t, landmarks=make_landmarks()))
        t += dt
    for _ in range(round(closed_s * fps)):
        frames.append(Frame(timestamp_s=t, landmarks=make_landmarks(**closed_overrides)))
        t += dt
    for _ in range(round(after_open_s * fps)):
        frames.append(Frame(timestamp_s=t, landmarks=make_landmarks()))
        t += dt
    return frames


def _run_frames(frames, config=None):
    output = DryRunOutputAdapter(echo=None)
    pipeline = Pipeline(
        config or Config(), ListInputSource(frames), output, calibration=_fitted_calibration()
    )
    return pipeline.run(), output


def test_deliberate_blink_hold_after_dwell_clicks_once_at_the_dwelled_point():
    result, output = _run_frames(_blink_session(open_s=1.0, closed_s=0.5))

    assert result.total_clicks == 1
    clicked = next(s for s in result.statuses if s.click_fired)
    assert clicked.cursor_target_px == pytest.approx((960.0, 540.0))
    assert all(a == ("move", 960, 540) for a in output.actions if a[0] == "move")


def test_cursor_never_moves_on_closed_eye_landmarks():
    # Upper lid drops onto a still-centred iris: the closed-eye offset is garbage.
    lid_down = dict(
        left_eye_top=Point3D(0.38, 0.418),
        left_eye_bottom=Point3D(0.38, 0.420),
        right_eye_top=Point3D(0.62, 0.418),
        right_eye_bottom=Point3D(0.62, 0.420),
    )
    result, output = _run_frames(_blink_session(1.0, 0.9, closed_overrides=lid_down))

    moves = [a for a in output.actions if a[0] == "move"]
    assert moves and all(a == ("move", 960, 540) for a in moves)


def test_short_natural_blinks_do_not_break_the_dwell():
    frames, t, dt = [], 0.0, 1.0 / 30
    blink_starts = (0.5, 1.2, 1.9, 2.6)
    while t < 3.0:
        closed = any(b <= t < b + 0.1 for b in blink_starts)
        lm = make_landmarks(**_CLOSED) if closed else make_landmarks()
        frames.append(Frame(timestamp_s=t, landmarks=lm))
        t += dt

    result, output = _run_frames(frames)

    assert result.total_clicks == 0
    assert all(a == ("move", 960, 540) for a in output.actions if a[0] == "move")
    assert result.statuses[-1].gate3_stability is True


def test_blink_hold_cannot_click_inside_a_no_click_zone():
    from dataclasses import replace

    from eyeos.config import NoClickZone

    config = Config()
    config = replace(
        config, gates=replace(config.gates, no_click_zones=(NoClickZone(0, 0, 1920, 40),))
    )
    lid_down = dict(
        left_eye_top=Point3D(0.38, 0.418),
        left_eye_bottom=Point3D(0.38, 0.420),
        right_eye_top=Point3D(0.62, 0.418),
        right_eye_bottom=Point3D(0.62, 0.420),
    )
    result, output = _run_frames(_blink_session(1.0, 0.9, closed_overrides=lid_down), config)

    for status in result.statuses:
        if status.click_fired:
            x, y = status.cursor_target_px
            assert not (0 <= x <= 1920 and 0 <= y <= 40)


# ---------------------------------------------------------------------------
# The gates must judge the exact pixel that is emitted (rounded, on-screen, finite)
# ---------------------------------------------------------------------------


class _FixedGaze:
    """A stand-in calibration that maps every frame to one fixed screen point."""

    is_fitted = True

    def __init__(self, point):
        self.point = point

    def predict(self, _features):
        return self.point

    def status(self):
        from eyeos.calibration import CalibrationStatus

        return CalibrationStatus(
            exists=True, validation_error_px=1.0, calibration_head_pose=(0.0, 0.0, 0.0)
        )


def _zone_config():
    from dataclasses import replace

    from eyeos.config import NoClickZone

    base = Config()
    zones = (NoClickZone(1860, 0, 1920, 40),)
    return replace(base, gates=replace(base.gates, no_click_zones=zones))


def _dwell_then_keypress(point):
    frames = [Frame(timestamp_s=i / 30, landmarks=make_landmarks()) for i in range(45)]
    output = DryRunOutputAdapter(echo=None)
    pipeline = Pipeline(
        _zone_config(),
        ListInputSource(frames),
        output,
        calibration=_FixedGaze(point),
        external_signals=lambda t: (False, t >= 1.2),
    )
    return pipeline.run(), output


@pytest.mark.parametrize("point", [(1859.6, 20.0), (1925.0, 20.0), (1900.0, -3.0)])
def test_no_click_lands_in_a_zone_after_rounding_or_clamping(point):
    result, output = _dwell_then_keypress(point)

    assert result.total_clicks == 0
    for action in output.actions:
        if action[0] == "move":
            assert 0 <= action[1] <= 1919 and 0 <= action[2] <= 1079


def test_zone_edge_is_exact_not_padded():
    result, output = _dwell_then_keypress((1859.4, 20.0))

    assert result.total_clicks == 1
    clicked = next(s for s in result.statuses if s.click_fired)
    assert clicked.cursor_target_px == (1859, 20)


def test_a_nan_landmark_does_not_poison_the_filter_or_crash_the_run():
    import math

    frames = [Frame(timestamp_s=i / 30, landmarks=make_landmarks()) for i in range(30)]
    bad = make_landmarks(left_iris_center=Point3D(float("nan"), 0.40))
    frames[15] = Frame(timestamp_s=15 / 30, landmarks=bad)
    output = DryRunOutputAdapter(echo=None)
    pipeline = Pipeline(
        Config(), ListInputSource(frames), output, calibration=_fitted_calibration()
    )

    result = pipeline.run()

    assert result.frame_count == 30
    assert result.statuses[15].cursor_target_px is None
    later = [s.cursor_target_px for s in result.statuses[16:]]
    assert later and all(p is not None and all(math.isfinite(c) for c in p) for p in later)


def test_keypress_mode_has_no_blink_grace_and_only_keypresses_click():
    from dataclasses import replace

    base = Config()
    config = replace(base, gates=replace(base.gates, confirmation_mode="keypress"))
    frames = _blink_session(open_s=1.0, closed_s=0.6)
    output = DryRunOutputAdapter(echo=None)
    result = Pipeline(
        config, ListInputSource(frames), output, calibration=_fitted_calibration()
    ).run()

    first_closed = 30
    assert result.statuses[first_closed].gate1_presence is False
    assert result.statuses[first_closed].reasons["gate1"] == "both irises not visible"
    assert result.total_clicks == 0

    frames = [Frame(timestamp_s=i / 30, landmarks=make_landmarks()) for i in range(40)]
    result = Pipeline(
        config,
        ListInputSource(frames),
        DryRunOutputAdapter(echo=None),
        calibration=_fitted_calibration(),
        external_signals=lambda t: (False, t >= 1.0),
    ).run()
    assert result.total_clicks == 1


def test_a_frozen_clock_never_moves_the_cursor():
    frames = [
        Frame(
            timestamp_s=0.0,
            landmarks=make_landmarks(left_iris_center=Point3D(0.36 + i * 0.001, 0.40)),
        )
        for i in range(60)
    ]
    result, _ = _run_frames(frames)

    assert result.total_moves <= 1
    assert all("timestamp not advancing" in s.reasons["gate1"] for s in result.statuses[1:])


# ---------------------------------------------------------------------------
# Partly closed lids: the frames on the way into and out of a blink
# ---------------------------------------------------------------------------

_TRANSITION_TOPS = (0.39, 0.40, 0.41)  # upper lid closing over three frames


def _lids(top_y, bottom_y=0.42, iris_dy=0.0):
    return dict(
        left_eye_top=Point3D(0.38, top_y),
        left_eye_bottom=Point3D(0.38, bottom_y),
        right_eye_top=Point3D(0.62, top_y),
        right_eye_bottom=Point3D(0.62, bottom_y),
        left_iris_center=Point3D(0.38, 0.40 + iris_dy),
        right_iris_center=Point3D(0.62, 0.40 + iris_dy),
    )


def _blink_with_transitions(closed_s, open_s=1.0, after_s=0.5, fps=30, drift=0.003):
    """Open, three closing frames, a closure, three reopening frames, open again. The
    iris drifts 0.003 down with the lid, as MediaPipe's iris estimate tends to."""
    frames, t, dt = [], 0.0, 1.0 / fps

    def add(**overrides):
        nonlocal t
        frames.append(Frame(timestamp_s=t, landmarks=make_landmarks(**overrides)))
        t += dt

    for _ in range(round(open_s * fps)):
        add()
    for top in _TRANSITION_TOPS:
        add(**_lids(top, iris_dy=drift))
    for _ in range(round(closed_s * fps)):
        add(**_lids(0.418, bottom_y=0.420, iris_dy=drift))
    for top in reversed(_TRANSITION_TOPS):
        add(**_lids(top, iris_dy=drift))
    for _ in range(round(after_s * fps)):
        add()
    return frames


def _max_move_distance(output, point=(960, 540)):
    moves = [a for a in output.actions if a[0] == "move"]
    assert moves
    return max(((x - point[0]) ** 2 + (y - point[1]) ** 2) ** 0.5 for _, x, y in moves)


def test_blink_hold_with_lid_transitions_clicks_once_on_target():
    result, output = _run_frames(_blink_with_transitions(closed_s=0.5))

    assert result.total_clicks == 1
    clicked = next(s for s in result.statuses if s.click_fired)
    assert clicked.cursor_target_px == pytest.approx((960.0, 540.0))
    assert _max_move_distance(output) <= 20.0


def test_long_closure_with_lid_transitions_never_moves_off_target():
    _result, output = _run_frames(_blink_with_transitions(closed_s=0.9))
    assert _max_move_distance(output) <= 20.0


def test_short_blinks_with_lid_transitions_keep_the_dwell():
    frames = []
    for i in range(4):
        part = _blink_with_transitions(closed_s=0.1, open_s=0.4, after_s=0.0)
        offset = i * (part[-1].timestamp_s + 1.0 / 30)
        frames += [Frame(f.timestamp_s + offset, f.landmarks) for f in part]
    frames += [Frame(frames[-1].timestamp_s + (k + 1) / 30, make_landmarks()) for k in range(6)]

    result, output = _run_frames(frames)

    assert result.total_clicks == 0
    assert _max_move_distance(output) <= 20.0
    assert result.statuses[-1].gate3_stability is True


def test_a_sustained_squint_is_trusted_again_after_the_hold():
    # Narrowed lids for a whole second (looking down narrows them too): after a short
    # hold the narrower opening becomes the new normal and the cursor follows again.
    fps, t, frames = 30, 0.0, []
    for _ in range(30):
        frames.append(Frame(t, make_landmarks()))
        t += 1.0 / fps
    squint_start = t
    for _ in range(30):
        frames.append(Frame(t, make_landmarks(**_lids(0.395))))
        t += 1.0 / fps

    result, _output = _run_frames(frames)

    moved = [
        f.timestamp_s
        for f, s in zip(frames, result.statuses, strict=True)
        if s.move_allowed and s.cursor_target_px != pytest.approx((960.0, 540.0))
    ]
    assert moved, "the cursor never followed the squinting gaze"
    assert moved[0] >= squint_start + 0.3 - 1e-9
