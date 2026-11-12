"""The pipeline end-to-end: input source -> features -> calibration -> filter ->
gates -> output, dry-run. This is the PRD Definition-of-Done item "dry-run mode runs
the whole pipeline on a video file" — exercised here against a SyntheticInputSource
CSV fixture instead of a real video, since this container has no camera/video deps
(see docs/PLAN.md)."""

from __future__ import annotations

import csv
import math
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
    frames = [Frame(timestamp_s=k / 30, landmarks=make_landmarks()) for k in range(31)]
    output = DryRunOutputAdapter(echo=None)
    pipeline = Pipeline(Config(), ListInputSource(frames), output, calibration=None)

    result = pipeline.run()

    assert result.frame_count == 31
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
    frames = [Frame(timestamp_s=k / 30, landmarks=make_landmarks()) for k in range(31)]
    output = DryRunOutputAdapter(echo=None)

    def external_signals(timestamp_s: float) -> tuple[bool, bool]:
        return False, timestamp_s >= 1.0 - 1e-9  # confirm_keypress on the last frame

    pipeline = Pipeline(
        Config(),
        ListInputSource(frames),
        output,
        calibration=calibration,
        external_signals=external_signals,
    )
    result = pipeline.run()

    assert result.frame_count == 31
    assert result.total_moves == 31
    assert result.total_clicks == 1
    clicked = next(s for s in result.statuses if s.click_fired)
    assert clicked is result.statuses[-1]
    assert clicked.cursor_target_px == pytest.approx((960.0, 540.0))
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
    closed = make_landmarks(
        left_eye_top=Point3D(0.38, 0.400),
        left_eye_bottom=Point3D(0.38, 0.401),
        right_eye_top=Point3D(0.62, 0.400),
        right_eye_bottom=Point3D(0.62, 0.401),
    )
    # Eyes open for 0.1 s, then closed for 0.3 s, at 30 fps.
    frames = [Frame(timestamp_s=k / 30, landmarks=make_landmarks()) for k in range(3)]
    frames += [Frame(timestamp_s=(3 + k) / 30, landmarks=closed) for k in range(10)]
    output = DryRunOutputAdapter(echo=None)
    pipeline = Pipeline(
        Config(), ListInputSource(frames), output, calibration=_fitted_calibration()
    )
    last = pipeline.run().statuses[-1]
    # A 0.3s closed-eye stretch is within confirmation_hold_s(0.4) + grace(0.5): Gate 1
    # should still see "visible", so movement/dwell survive the blink.
    assert last.gate1_presence is True
    assert last.move_allowed is True
    assert last.cursor_target_px == pytest.approx((960.0, 540.0))
    assert "eyes closed 0.30s: blink-hold grace" in last.reasons["gate1"]


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

    def run_with(zones):
        config = Config()
        config = replace(config, gates=replace(config.gates, no_click_zones=zones))
        return _run_frames(_blink_session(open_s=1.0, closed_s=0.5), config)

    # Without a zone this dwell + blink-hold clicks once at (960, 540)...
    assert run_with(())[0].total_clicks == 1
    # ...and with a zone around that point it never clicks, nor moves there.
    result, output = run_with((NoClickZone(900, 500, 1000, 600),))
    assert result.total_clicks == 0
    assert not [a for a in output.actions if a[0] == "move"]


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
    # A closure shorter than the 0.1 s debounce is held as a detection dropout; past it,
    # closed eyes fail Gate 1 (no blink-hold grace in keypress mode).
    assert result.statuses[first_closed].gate1_presence is True
    assert result.statuses[first_closed + 4].gate1_presence is False
    assert result.statuses[first_closed + 4].reasons["gate1"] == "both irises not visible"
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
# Lids moving during a blink don't move the gaze (offsets are measured from the corners)
# ---------------------------------------------------------------------------

_APERTURE_RAMP = (0.875, 0.75, 0.5, 0.25)  # fractions of the open lid gap (0.04)


def _lids(fraction_open, iris_dy=0.0):
    top_y = 0.42 - 0.04 * fraction_open  # the upper lid moves; the lower lid doesn't
    return dict(
        left_eye_top=Point3D(0.38, top_y),
        right_eye_top=Point3D(0.62, top_y),
        left_iris_center=Point3D(0.38, 0.40 + iris_dy),
        right_iris_center=Point3D(0.62, 0.40 + iris_dy),
    )


def _blink_with_transitions(closed_s, open_s=1.0, after_s=0.5, fps=30, drift=0.0, ramp=None):
    """Open, a closing ramp of partly closed frames, a closure, the ramp reversed, open
    again. ``drift`` moves the iris estimate down with the lid on every non-open frame."""
    ramp = _APERTURE_RAMP if ramp is None else ramp
    frames, t, dt = [], 0.0, 1.0 / fps

    def add(**overrides):
        nonlocal t
        frames.append(Frame(timestamp_s=t, landmarks=make_landmarks(**overrides)))
        t += dt

    for _ in range(round(open_s * fps)):
        add()
    for f in ramp:
        add(**_lids(f, iris_dy=drift))
    for _ in range(round(closed_s * fps)):
        add(**_lids(0.05 / 2, iris_dy=drift))  # gap 0.001: below both_irises_visible
    for f in reversed(ramp):
        add(**_lids(f, iris_dy=drift))
    for _ in range(round(after_s * fps)):
        add()
    return frames


def _max_move_distance(output, point=(960, 540)):
    moves = [a for a in output.actions if a[0] == "move"]
    assert moves
    return max(((x - point[0]) ** 2 + (y - point[1]) ** 2) ** 0.5 for _, x, y in moves)


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_blink_hold_with_lid_transitions_clicks_once_on_target(fps):
    # Includes a frame with the lid only 12.5% closed, at the start and the end.
    result, output = _run_frames(_blink_with_transitions(closed_s=0.5, fps=fps))

    assert result.total_clicks == 1
    clicked = next(s for s in result.statuses if s.click_fired)
    assert clicked.cursor_target_px == pytest.approx((960.0, 540.0))
    assert _max_move_distance(output) == 0.0


def test_long_closure_with_lid_transitions_never_moves_off_target():
    _result, output = _run_frames(_blink_with_transitions(closed_s=0.9))
    assert _max_move_distance(output) == 0.0


def test_short_blinks_with_lid_transitions_keep_the_dwell():
    frames = []
    for i in range(4):
        part = _blink_with_transitions(closed_s=0.1, open_s=0.4, after_s=0.0)
        offset = i * (part[-1].timestamp_s + 1.0 / 30)
        frames += [Frame(f.timestamp_s + offset, f.landmarks) for f in part]
    frames += [Frame(frames[-1].timestamp_s + (k + 1) / 30, make_landmarks()) for k in range(6)]

    result, output = _run_frames(frames)

    assert result.total_clicks == 0
    assert _max_move_distance(output) == 0.0
    assert result.statuses[-1].gate3_stability is True


def test_an_iris_estimate_that_drifts_with_the_lid_fails_closed():
    # Documented limitation: if the iris estimate itself moves as the lid closes, the
    # partly closed frames are a real gaze change. The dwell restarts and no click fires.
    result, _output = _run_frames(_blink_with_transitions(closed_s=0.5, drift=0.003))
    assert result.total_clicks == 0


def test_looking_down_with_narrowed_lids_then_confirming_never_clicks_the_old_target():
    # Dwell on A, then look down toward B: the lids narrow to 70% as the gaze drops. A
    # confirm key 0.13 s later must not click A (nor B, which was never dwelled on).
    from dataclasses import replace

    config = Config()
    config = replace(config, gates=replace(config.gates, confirmation_mode="keypress"))
    narrowed = {k: v for k, v in _lids(0.7).items() if k.endswith("_eye_top")}
    frames = _seq(
        [
            (1.0, lambda: _gaze_landmarks(960, 300)),
            (0.5, lambda: _gaze_landmarks(960, 950, **narrowed)),
        ]
    )
    pressed_from = frames[30].timestamp_s + 0.13

    def signals(t):
        return False, t >= pressed_from - 1e-9

    output = DryRunOutputAdapter(echo=None)
    keys_pipeline = Pipeline(
        config,
        ListInputSource(frames),
        output,
        calibration=_fitted_calibration(),
        external_signals=signals,
    )
    assert keys_pipeline.run().total_clicks == 0


def _flutter(pattern, fps=30):
    """1 s dwelling on A, then one frame per character: 'c' closed, 'o' a one-frame
    glimpse with the lids 30% open (Gate 1 counts that as open)."""
    frames = _seq([(1.0, lambda: _gaze_landmarks(960, 300))], fps=fps)
    t0 = frames[-1].timestamp_s
    for k, ch in enumerate(pattern, start=1):
        lids = _lids(0.3) if ch == "o" else _lids(0.025)
        lm = _gaze_landmarks(960, 300, **{n: v for n, v in lids.items() if n.endswith("_eye_top")})
        frames.append(Frame(t0 + k / fps, lm))
    return frames


@pytest.mark.parametrize(
    "pattern",
    [
        ("c" * 12 + "o") * 70,  # closed stretches just under confirmation_hold_s
        "c" * 14 + ("c" * 9 + "o") * 30 + "c" * 14,  # click, flutter, then a long closure
        ("c" * 4 + "o") * 150 + "c" * 14,
        ("c" * 14 + "o") * 60,
    ],
    ids=["under-hold", "click-flutter-click", "dense", "every-15"],
)
def test_eyelid_flutter_in_a_long_closure_clicks_at_most_once(pattern):
    # The eyes are never really reopened, so one blink-hold click at most.
    result, _output = _run_frames(_flutter(pattern))
    assert result.total_clicks <= 1


@pytest.mark.parametrize("seed", range(6))
def test_random_eyelid_flutter_clicks_at_most_once(seed):
    import random

    rng = random.Random(seed)
    pattern = "".join("o" if rng.random() < 0.25 else "c" for _ in range(900))
    result, _output = _run_frames(_flutter(pattern))
    assert result.total_clicks <= 1


# ---------------------------------------------------------------------------
# Held gaze (eyes closed in the blink-hold grace window) is not an observation
# ---------------------------------------------------------------------------


def _gaze_landmarks(px, py, closed=False, **overrides):
    """Landmarks whose iris offset _fitted_calibration() maps to (px, py)."""
    dx, dy = px - 960, py - 540
    det = 800 * 650 - 30 * 10
    a = (dx * 650 - 30 * dy) / det
    b = (800 * dy - 10 * dx) / det
    lm = dict(
        left_iris_center=Point3D(0.38 + 0.04 * a, 0.40 + 0.04 * b),
        right_iris_center=Point3D(0.62 + 0.04 * a, 0.40 + 0.04 * b),
    )
    if closed:
        lm.update(_CLOSED)
    lm.update(overrides)
    return make_landmarks(**lm)


def _seq(spec, fps=30):
    """(duration_s, landmark factory or None for a lost face) segments at 30 fps."""
    frames, t = [], 0.0
    for duration_s, factory in spec:
        for _ in range(round(duration_s * fps)):
            frames.append(Frame(timestamp_s=t, landmarks=factory() if factory else None))
            t += 1.0 / fps
    return frames


def _targets_while_eyes_closed(frames, result):
    return {
        s.cursor_target_px
        for f, s in zip(frames, result.statuses, strict=True)
        if s.cursor_target_px is not None and not (f.landmarks and f.landmarks.both_irises_visible)
    }


@pytest.mark.parametrize("open_frames", [1, 2, 3])
def test_eyes_closed_time_never_counts_toward_the_dwell(open_frames):
    # A glance of a few frames at B, then the eyes close for 0.87 s (inside the grace
    # window). The 0.8 s dwell was never observed with open eyes, so nothing may click.
    b = (1500.0, 300.0)
    frames = _seq(
        [
            (1.0, lambda: _gaze_landmarks(960, 540)),
            (2.0, None),  # face lost long enough for Gate 4's speed average to allow B
            (open_frames / 30, lambda: _gaze_landmarks(*b)),
            (0.87, lambda: _gaze_landmarks(*b, closed=True)),
            (0.3, lambda: _gaze_landmarks(*b)),
        ]
    )
    result, _output = _run_frames(frames)
    assert result.total_clicks == 0


def test_blink_hold_holds_nothing_after_a_frame_gate1_rejected():
    # Dwelling at A, one frame with a second face in view (MediaPipe's first face may
    # not be the user's), then the user's eyes close.
    frames = _seq(
        [
            (1.0, lambda: _gaze_landmarks(960, 540)),
            (1 / 30, lambda: _gaze_landmarks(1060, 600, face_count=2)),
            (0.6, lambda: _gaze_landmarks(1060, 600, closed=True)),
        ]
    )
    result, output = _run_frames(frames)

    assert result.total_clicks == 0
    assert {a for a in output.actions if a[0] == "move"} == {("move", 960, 540)}
    assert _targets_while_eyes_closed(frames, result) <= {(960.0, 540.0)}


def test_blink_hold_never_relocates_to_a_point_gate4_blocked():
    # A 300 px jump that Gate 4's speed limit blocks, then the eyes close: the cursor
    # must not move there (or click there) with the eyes shut.
    frames = _seq(
        [
            (1.0, lambda: _gaze_landmarks(960, 540)),
            (2 / 30, lambda: _gaze_landmarks(1260, 540)),
            (0.85, lambda: _gaze_landmarks(1260, 540, closed=True)),
        ]
    )
    result, _output = _run_frames(frames)

    assert result.total_clicks == 0
    assert _targets_while_eyes_closed(frames, result) <= {(960.0, 540.0)}


def test_a_head_turn_during_a_blink_hold_fails_gate2():
    turned = dict(nose_tip=Point3D(0.60, 0.50))  # about 25 deg of yaw
    frames = _seq(
        [
            (1.0, lambda: _gaze_landmarks(960, 540)),
            (0.5, lambda: _gaze_landmarks(960, 540, closed=True, **turned)),
        ]
    )
    result, _output = _run_frames(frames)

    assert result.total_clicks == 0
    assert not result.statuses[-1].gate2_calibration


def test_a_camera_stall_mid_blink_hold_does_not_click():
    # A dwell met with open eyes, the eyes close, the camera stalls 0.3 s, the eyes stay
    # closed. The stall breaks the observation, so the blink-hold can't confirm.
    frames = _seq(
        [
            (1.0, lambda: _gaze_landmarks(960, 540)),
            (0.1, lambda: _gaze_landmarks(960, 540, closed=True)),
        ]
    )
    t = frames[-1].timestamp_s + 0.3
    frames += [Frame(t + k / 30, _gaze_landmarks(960, 540, closed=True)) for k in range(15)]

    result, _output = _run_frames(frames)

    assert result.total_clicks == 0
    assert any("frame gap" in s.reasons["gate1"] for s in result.statuses)


def test_a_second_face_mid_blink_hold_does_not_click():
    frames = _seq(
        [
            (1.0, lambda: _gaze_landmarks(960, 540)),
            (0.1, lambda: _gaze_landmarks(960, 540, closed=True)),
            (1 / 30, lambda: _gaze_landmarks(960, 540, closed=True, face_count=2)),
            (0.5, lambda: _gaze_landmarks(960, 540, closed=True)),
        ]
    )
    result, _output = _run_frames(frames)
    assert result.total_clicks == 0


@pytest.mark.parametrize("bad_t", [math.nan, math.inf])
@pytest.mark.parametrize("position", ["first", "middle"])
def test_a_non_finite_timestamp_fails_closed_without_crashing(bad_t, position):
    frames = _seq([(1.0, lambda: _gaze_landmarks(960, 540))])
    index = 0 if position == "first" else 15
    frames[index] = Frame(bad_t, frames[index].landmarks)

    result, output = _run_frames(frames)

    assert not result.statuses[index].gate1_presence
    assert "not finite" in result.statuses[index].reasons["gate1"]
    assert result.statuses[-1].move_allowed  # later valid frames still pass
    assert {a for a in output.actions if a[0] == "move"} == {("move", 960, 540)}


def test_closed_eyes_past_the_grace_window_carry_no_gaze_and_never_reach_the_filter():
    lid_down = dict(
        left_eye_top=Point3D(0.38, 0.418),
        left_eye_bottom=Point3D(0.38, 0.420),
        right_eye_top=Point3D(0.62, 0.418),
        right_eye_bottom=Point3D(0.62, 0.420),
    )
    frames = _blink_session(1.0, 1.5, closed_overrides=lid_down, after_open_s=0.5)
    result, output = _run_frames(frames)

    past_grace = [s for s in result.statuses if s.reasons["gate1"] == "both irises not visible"]
    assert len(past_grace) >= 10
    assert all(s.smoothed_gaze_px is None for s in past_grace)
    # Had closed-eye offsets reached the filter, reopening would drift in from elsewhere.
    assert {a for a in output.actions if a[0] == "move"} == {("move", 960, 540)}


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_one_to_one_lid_flutter_clicks_at_most_once(fps):
    hold = int(0.4 * fps) + 2
    pattern = "c" * hold + "co" * 3 * fps + "c" * hold
    result, _output = _run_frames(_flutter(pattern, fps=fps))
    assert result.total_clicks <= 1


@pytest.mark.parametrize("open_fraction", [0.3, 0.35, 0.4, 0.45])
@pytest.mark.parametrize("seed", range(8))
def test_random_lid_flutter_at_15_fps_clicks_at_most_once(open_fraction, seed):
    # Random glimpses, but never open for more than 2 frames in a row (0.13 s at 15 fps):
    # the eyes are never really reopened. A longer open run is a real reopening, and a
    # dwell met then may click again; that is by design.
    import random

    rng = random.Random(seed)
    pattern = ""
    for _ in range(450):
        glimpse = rng.random() < open_fraction and not pattern.endswith("oo")
        pattern += "o" if glimpse else "c"
    result, _output = _run_frames(_flutter(pattern, fps=15))
    assert result.total_clicks <= 1


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_blinks_at_an_earlier_target_do_not_spoil_the_next_click(fps):
    # Reading at A with natural blinks, then a button at B: one natural blink there, a
    # dwell, and a deliberate blink-hold must click B once.
    at_a = lambda: _gaze_landmarks(960, 540)  # noqa: E731
    at_b = lambda: _gaze_landmarks(1060, 540)  # noqa: E731
    closed_b = lambda: _gaze_landmarks(1060, 540, closed=True)  # noqa: E731
    closed_a = lambda: _gaze_landmarks(960, 540, closed=True)  # noqa: E731
    spec = []
    for _ in range(4):
        spec += [(0.6, at_a), (0.1, closed_a)]
    spec += [(0.4, at_b), (0.1, closed_b), (1.2, at_b), (0.5, closed_b), (0.3, at_b)]
    frames = _seq(spec, fps=fps)

    result, _output = _run_frames(frames)

    assert result.total_clicks == 1
    clicked = next(s for s in result.statuses if s.click_fired)
    assert clicked.cursor_target_px == pytest.approx((1060.0, 540.0))


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_a_closure_just_short_of_the_hold_never_clicks(fps):
    # 1/3 s of closed eyes after a dwell, then open again: below confirmation_hold_s.
    # The reopening's first frames must not push the hold over 0.4 s.
    frames = _seq(
        [
            (1.0, lambda: _gaze_landmarks(960, 540)),
            (1 / 3, lambda: _gaze_landmarks(960, 540, closed=True)),
            (0.5, lambda: _gaze_landmarks(960, 540)),
        ],
        fps=fps,
    )
    result, _output = _run_frames(frames)
    assert result.total_clicks == 0


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_a_blink_hold_click_only_fires_while_the_eyes_are_closed(fps):
    frames = _seq(
        [
            (1.0, lambda: _gaze_landmarks(960, 540)),
            (0.5, lambda: _gaze_landmarks(960, 540, closed=True)),
            (0.5, lambda: _gaze_landmarks(960, 540)),
        ],
        fps=fps,
    )
    result, _output = _run_frames(frames)
    clicked = [f for f, s in zip(frames, result.statuses, strict=True) if s.click_fired]
    assert len(clicked) == 1
    assert not clicked[0].landmarks.both_irises_visible


@pytest.mark.parametrize("fps", [30, 60])
def test_a_key_pressed_just_after_reopening_on_another_point_never_clicks_the_old_one(fps):
    # Default "either" mode: dwell on A, blink while the gaze moves to B, press the
    # confirm key 0.05 s after the eyes reopen on B. A must not be clicked.
    frames = _seq(
        [
            (1.0, lambda: _gaze_landmarks(960, 540)),
            (2 / 30, lambda: _gaze_landmarks(1060, 540, closed=True)),
            (0.5, lambda: _gaze_landmarks(1060, 540)),
        ],
        fps=fps,
    )
    reopened = next(
        f.timestamp_s for f in frames if f.timestamp_s > 1.0 and f.landmarks.both_irises_visible
    )

    def signals(t):
        return False, reopened + 0.05 - 1e-9 <= t <= reopened + 0.1

    output = DryRunOutputAdapter(echo=None)
    result = Pipeline(
        Config(),
        ListInputSource(frames),
        output,
        calibration=_fitted_calibration(),
        external_signals=signals,
    ).run()
    assert result.total_clicks == 0


@pytest.mark.parametrize("every", [6, 7])
def test_frequent_one_frame_iris_dropouts_still_allow_a_dwell_and_a_click(every):
    # MediaPipe losing the iris for single frames is not a blink and must not cost the
    # reopening debounce each time.
    frames = _seq([(3.0, lambda: _gaze_landmarks(1060, 540))])
    frames = [
        (
            Frame(f.timestamp_s, _gaze_landmarks(1060, 540, closed=True))
            if k % every == every - 1
            else f
        )
        for k, f in enumerate(frames)
    ]
    t = frames[-1].timestamp_s
    frames += [Frame(t + (k + 1) / 30, _gaze_landmarks(1060, 540, closed=True)) for k in range(15)]

    result, _output = _run_frames(frames)

    assert any(s.gate3_stability for s in result.statuses[:90])
    assert result.total_clicks == 1


@pytest.mark.parametrize("fps, blink, every", [(15, 2, 3), (30, 3, 4), (60, 6, 7)])
def test_a_blink_followed_by_iris_dropouts_never_adds_up_to_a_click(fps, blink, every):
    # The blink-hold is one unbroken run of closed frames: a 0.1 s blink plus one-frame
    # dropouts must not add up to 0.4 s and click.
    frames = _seq([(1.0, lambda: _gaze_landmarks(960, 540))], fps=fps)
    t0 = frames[-1].timestamp_s
    pattern = "c" * blink + ("o" * (every - 1) + "c") * (3 * fps // every)
    for k, ch in enumerate(pattern, start=1):
        frames.append(Frame(t0 + k / fps, _gaze_landmarks(960, 540, closed=(ch == "c"))))

    result, _output = _run_frames(frames)
    assert result.total_clicks == 0


@pytest.mark.parametrize("seed", range(8))
def test_noisy_lid_flutter_at_15_fps_clicks_at_most_once(seed):
    # As the random flutter test, with +-20 px of gaze noise on every glimpse.
    import random

    rng = random.Random(seed)
    frames = _seq([(1.0, lambda: _gaze_landmarks(960, 300))], fps=15)
    t0, pattern = frames[-1].timestamp_s, ""
    for _ in range(450):
        pattern += "o" if rng.random() < 0.45 and not pattern.endswith("oo") else "c"
    for k, ch in enumerate(pattern, start=1):
        lids = _lids(0.3) if ch == "o" else _lids(0.025)
        dx, dy = rng.uniform(-20, 20), rng.uniform(-20, 20)
        tops = {n: v for n, v in lids.items() if n.endswith("_eye_top")}
        frames.append(Frame(t0 + k / 15, _gaze_landmarks(960 + dx, 300 + dy, **tops)))

    result, _output = _run_frames(frames)
    assert result.total_clicks <= 1


def test_keypress_mode_tolerates_one_frame_iris_dropouts():
    from dataclasses import replace

    base = Config()
    config = replace(base, gates=replace(base.gates, confirmation_mode="keypress"))
    frames = _seq([(2.0, lambda: _gaze_landmarks(960, 540))])
    frames = [
        Frame(f.timestamp_s, _gaze_landmarks(960, 540, closed=True)) if k % 12 == 11 else f
        for k, f in enumerate(frames)
    ]
    result = Pipeline(
        config,
        ListInputSource(frames),
        DryRunOutputAdapter(echo=None),
        calibration=_fitted_calibration(),
        external_signals=lambda t: (False, t >= 1.85),
    ).run()
    assert result.total_clicks == 1


def _run_with_key(frames, key_down, mode="either"):
    from dataclasses import replace

    base = Config()
    config = replace(base, gates=replace(base.gates, confirmation_mode=mode))
    output = DryRunOutputAdapter(echo=None)
    result = Pipeline(
        config,
        ListInputSource(frames),
        output,
        calibration=_fitted_calibration(),
        external_signals=lambda t: (False, key_down(t)),
    ).run()
    return result


@pytest.mark.parametrize("mode", ["either", "keypress"])
@pytest.mark.parametrize("fps", [15, 30, 60])
def test_a_confirm_key_held_through_a_blink_clicks_once(mode, fps):
    frames = _seq(
        [
            (1.0, lambda: _gaze_landmarks(960, 540)),
            (0.2, lambda: _gaze_landmarks(960, 540, closed=True)),
            (1.5, lambda: _gaze_landmarks(960, 540)),
        ],
        fps=fps,
    )
    result = _run_with_key(frames, lambda t: t >= 0.9, mode)
    assert result.total_clicks == 1


@pytest.mark.parametrize("mode", ["either", "keypress"])
@pytest.mark.parametrize("fps", [30, 60])
def test_a_key_pressed_on_a_closed_frame_never_clicks_the_point_held_from_before(mode, fps):
    # Dwell on A, blink while the gaze moves to B, key down from the last closed frame.
    frames = _seq(
        [
            (1.0, lambda: _gaze_landmarks(960, 540)),
            (2 / 30, lambda: _gaze_landmarks(1060, 540, closed=True)),
            (0.5, lambda: _gaze_landmarks(1060, 540)),
        ],
        fps=fps,
    )
    reopened = next(
        f.timestamp_s for f in frames if f.timestamp_s > 1.0 and f.landmarks.both_irises_visible
    )
    result = _run_with_key(frames, lambda t: reopened - 1 / fps - 1e-9 <= t <= reopened + 0.1, mode)
    assert result.total_clicks == 0


@pytest.mark.parametrize("fps, unit", [(15, "co"), (30, "coo"), (60, "ccoooo")])
def test_a_lost_face_frame_does_not_turn_lid_flutter_into_looking(fps, unit):
    hold = "c" * (int(0.4 * fps) + 2)
    flutter = unit * (fps // len(unit))
    frames = _flutter(hold + flutter, fps=fps)
    t = frames[-1].timestamp_s
    frames.append(Frame(t + 1 / fps, None))
    tail = _flutter(unit * (2 * fps // len(unit)) + hold, fps=fps)[round(fps) :]
    offset = t + 2 / fps - tail[0].timestamp_s
    frames += [Frame(f.timestamp_s + offset, f.landmarks) for f in tail]

    result, _output = _run_frames(frames)
    assert result.total_clicks == 1


@pytest.mark.parametrize("fps, unit", [(15, "coo"), (30, "coo"), (60, "ccoooo")])
@pytest.mark.parametrize("how", ["lost_face", "dropped_frame", "stale_timestamp"])
def test_unobserved_or_stale_frames_never_stretch_a_glimpse_into_looking(fps, unit, how):
    # Click, then lid flutter whose open runs never reach 0.1 s. One of its closed
    # frames is replaced by a lost face, dropped entirely, or re-sent with an old
    # timestamp. None of that may end the closure: one click only.
    hold = "c" * (int(0.4 * fps) + 2)
    frames = _flutter(hold + unit * (4 * fps // len(unit)) + hold, fps=fps)
    first_closed = round(fps)  # _flutter dwells 1 s, then the pattern starts
    target = first_closed + len(hold) + len(unit) * (fps // len(unit))
    if how == "lost_face":
        frames[target] = Frame(frames[target].timestamp_s, None)
    elif how == "dropped_frame":
        del frames[target]
    else:
        stale = frames[first_closed].timestamp_s + 0.05
        frames[target + 1] = Frame(stale, frames[target + 1].landmarks)

    result, _output = _run_frames(frames)
    assert result.total_clicks == 1
