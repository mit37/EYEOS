"""In-process calibration (`eyeos run --calibrate`), tested with a fake dot screen."""

from __future__ import annotations

import csv
import sys

import pytest

from eval.generate_synthetic import make_landmarks, offset_for_target
from eyeos.calibration import target_points
from eyeos.calibration_session import (
    SAMPLE_FRAMES,
    SETTLE_FRAMES,
    calibrate_interactively,
    collect_dot_features,
)
from eyeos.config import Config
from eyeos.input_sources import Frame
from eyeos.synthetic_format import CSV_FIELDNAMES, landmarks_to_row
from tests.fake_tk import make_fake_tk

PER_DOT = SETTLE_FRAMES + SAMPLE_FRAMES


class FakeDotScreen:
    """Stands in for CalibrationUI: 'presses space' on each dot, retrying a failed one.

    Like Tk, it doesn't raise an exception from ``capture_sample``: it records it in
    ``callback_errors`` and counts that press as a failed capture.
    """

    def __init__(self, width, height, capture_sample, cancel=False):
        self.targets = target_points(width, height)
        self.capture_sample = capture_sample
        self.cancel_at_start = cancel
        self.cancelled = False
        self.presses = 0
        self.callback_errors = []

    def cancel(self):
        self.cancelled = True

    def run(self):
        if self.cancel_at_start:
            return False
        for target in self.targets:
            for _attempt in range(3):
                if self.cancelled:
                    return False
                self.presses += 1
                try:
                    captured = self.capture_sample(target)
                except SystemExit:
                    raise
                except BaseException as e:  # like Tk's CallWrapper: even KeyboardInterrupt
                    self.callback_errors.append(e)
                    captured = False
                if captured is not False:
                    break
        return not self.cancelled


def _calibration_frames(t0=0.0, fps=30):
    frames, t = [], t0
    for target in target_points(1920, 1080):
        ox, oy = offset_for_target(*target)
        for _ in range(PER_DOT):
            frames.append(Frame(timestamp_s=t, landmarks=make_landmarks(ox, oy)))
            t += 1 / fps
    return frames, t


def test_collect_dot_features_skips_settle_frames_and_unusable_frames():
    good = make_landmarks(0.2, -0.1)
    stale = make_landmarks(-0.9, 0.9)
    frames = [Frame(i / 30, stale) for i in range(SETTLE_FRAMES)]
    frames += [Frame(1.0, None), Frame(1.1, make_landmarks(0.2, -0.1, face_count=2))]
    frames += [Frame(1.2 + i / 30, good) for i in range(SAMPLE_FRAMES)]

    features = collect_dot_features(iter(frames))

    assert features is not None
    assert features.left_iris_x == pytest.approx(0.2)
    assert features.left_iris_y == pytest.approx(-0.05)  # offset_y in eye half-widths


def test_collect_dot_features_gives_up_when_too_few_usable_frames():
    frames = [Frame(i / 30, None) for i in range(100)]
    assert collect_dot_features(iter(frames)) is None


def test_collect_dot_features_reports_the_end_of_the_input():
    # Not "too few usable frames": the stream itself stopped (camera unplugged).
    frames = [Frame(i / 30, make_landmarks(0.2, -0.1)) for i in range(SETTLE_FRAMES + 3)]
    with pytest.raises(RuntimeError, match="ran out of frames"):
        collect_dot_features(iter(frames))


def test_calibrate_interactively_fits_on_the_stream():
    frames, _ = _calibration_frames()

    calibration = calibrate_interactively(Config(), iter(frames), FakeDotScreen)

    assert calibration is not None and calibration.is_fitted
    assert len(calibration.samples) == 9
    assert calibration.validation_error_px < 5.0


def test_calibrate_interactively_returns_none_when_cancelled():
    frames, _ = _calibration_frames()

    def cancelled(w, h, capture_sample):
        return FakeDotScreen(w, h, capture_sample, cancel=True)

    assert calibrate_interactively(Config(), iter(frames), cancelled) is None


@pytest.mark.parametrize("dots", [3, 6, 8])
def test_calibrate_interactively_needs_a_sample_for_every_dot(dots):
    # A dot screen that reports success early must not yield a fit on fewer dots
    # (with fewer than 5, fitting would even crash).
    frames, _ = _calibration_frames()

    class EndsEarly(FakeDotScreen):
        def run(self):
            for target in self.targets[:dots]:
                assert self.capture_sample(target) is True
            return True

    assert calibrate_interactively(Config(), iter(frames), EndsEarly) is None


@pytest.mark.parametrize(
    "ending, message",
    [
        ("raises", "camera unplugged"),
        ("ends", r"camera stopped during calibration: .* at point 4/9"),
    ],
)
def test_calibrate_interactively_reports_a_stream_that_fails_mid_calibration(ending, message):
    frames, _ = _calibration_frames()

    def stream():
        yield from frames[: 3 * PER_DOT]  # three dots' worth, then the camera fails
        if ending == "raises":
            raise RuntimeError("camera unplugged")

    screens = []

    def dot_screen(width, height, capture_sample):
        screens.append(FakeDotScreen(width, height, capture_sample))
        return screens[-1]

    with pytest.raises(RuntimeError, match=message):
        calibrate_interactively(Config(), stream(), dot_screen)
    assert screens[0].cancelled
    assert screens[0].presses == 4  # the dot screen closed at the dot that failed


def _write_session_csv(tmp_path, frames):
    csv_path = tmp_path / "session.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(landmarks_to_row(fr.timestamp_s, fr.landmarks) for fr in frames)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"input:\n  source: synthetic\n  synthetic_path: {csv_path}\n"
        "output:\n  backend: dry_run\n"
    )
    return config_path


def _calibration_then_dwell_frames():
    frames, t = _calibration_frames()
    for i in range(30):  # then dwell on the screen centre
        frames.append(Frame(t + i / 30, make_landmarks(0.0, 0.0)))
    return frames


def test_eyeos_run_calibrate_moves_the_cursor_end_to_end(tmp_path, monkeypatch, capsys):
    import eyeos.cli as cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "CalibrationUI", FakeDotScreen)
    monkeypatch.setitem(sys.modules, "tkinter", make_fake_tk())
    config_path = _write_session_csv(tmp_path, _calibration_then_dwell_frames())

    exit_code = cli.main(
        ["run", "--config", str(config_path), "--calibrate", "--overlay", "--no-log"]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "calibration validation error" in out
    assert "processed 30 frames: 30 moves" in out
    assert "move_cursor(960, 540)" in out


def test_eyeos_run_calibrate_closing_the_window_emits_nothing(tmp_path, monkeypatch, capsys):
    # The real dot screen on a fake tkinter: 6 dots captured, then Alt+F4.
    import eyeos.cli as cli
    from eyeos.calibration_ui import CalibrationUI

    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "tkinter", make_fake_tk())

    def closed_after_six_dots(width, height, capture_sample):
        ui = CalibrationUI(width, height, capture_sample=capture_sample)

        def user(root):
            for _ in range(6):
                root.fire("<space>")
            root.close_window()

        ui.root.on_mainloop = user
        return ui

    monkeypatch.setattr(cli, "CalibrationUI", closed_after_six_dots)
    config_path = _write_session_csv(tmp_path, _calibration_then_dwell_frames())

    exit_code = cli.main(["run", "--config", str(config_path), "--calibrate", "--no-log"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "calibration cancelled" in captured.err
    assert captured.out == ""  # no validation error, no frames processed, no moves


def _scripted_dot_screen(monkeypatch, presses, shown):
    """Replace eyeos run's dot screen with the real one on a fake tkinter, where the
    user presses space ``presses`` times."""
    import eyeos.cli as cli
    from eyeos.calibration_ui import CalibrationUI

    monkeypatch.setitem(sys.modules, "tkinter", make_fake_tk())

    def dot_screen(width, height, capture_sample):
        ui = CalibrationUI(width, height, capture_sample=capture_sample)
        ui.root.on_mainloop = lambda root: [root.fire("<space>") for _ in range(presses)]
        shown.append(ui)
        return ui

    monkeypatch.setattr(cli, "CalibrationUI", dot_screen)


def _webcam_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("input:\n  source: webcam\noutput:\n  backend: dry_run\n")
    return config_path


@pytest.mark.parametrize(
    "cv2_kwargs, mp_kwargs, message",
    [
        ({"opened": False}, {}, "could not open webcam device 0"),
        ({}, {"solutions": False}, "legacy Solutions API"),
    ],
)
def test_eyeos_run_calibrate_reports_a_camera_that_cannot_open_before_any_dot(
    tmp_path, monkeypatch, capsys, cv2_kwargs, mp_kwargs, message
):
    from eyeos.cli import main
    from tests.fake_capture import FakeCv2, FakeMediapipe, install_fakes

    monkeypatch.chdir(tmp_path)
    install_fakes(
        monkeypatch,
        cv2=FakeCv2(n_frames=1000, **cv2_kwargs),
        mediapipe=FakeMediapipe(**mp_kwargs),
    )
    shown = []
    _scripted_dot_screen(monkeypatch, presses=9, shown=shown)

    with pytest.raises(RuntimeError, match=message):
        main(["run", "--config", str(_webcam_config(tmp_path)), "--calibrate", "--no-log"])

    assert shown == []  # failed before the fullscreen window could hide the error
    assert "calibration cancelled" not in capsys.readouterr().err


def test_eyeos_run_calibrate_refuses_a_source_with_no_frames(tmp_path, monkeypatch, capsys):
    from eyeos.cli import main
    from tests.fake_capture import FakeCv2, FakeMediapipe, install_fakes

    monkeypatch.chdir(tmp_path)
    install_fakes(monkeypatch, cv2=FakeCv2(n_frames=0), mediapipe=FakeMediapipe())
    shown = []
    _scripted_dot_screen(monkeypatch, presses=9, shown=shown)

    exit_code = main(["run", "--config", str(_webcam_config(tmp_path)), "--calibrate", "--no-log"])

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "gave no frames" in err
    assert "calibration cancelled" not in err
    assert shown == []


def test_eyeos_run_calibrate_reports_a_camera_that_stops_mid_calibration(
    tmp_path, monkeypatch, capsys
):
    from eyeos.cli import main

    monkeypatch.chdir(tmp_path)
    frames, _ = _calibration_frames()
    config_path = _write_session_csv(tmp_path, frames[: 3 * PER_DOT])  # ends at dot 4
    shown = []
    _scripted_dot_screen(monkeypatch, presses=9, shown=shown)

    with pytest.raises(RuntimeError, match="camera stopped during calibration"):
        main(["run", "--config", str(config_path), "--calibrate", "--no-log"])

    (ui,) = shown
    assert ui.root.destroyed
    assert ui.root.callback_errors == []  # nothing left for Tk to swallow
    captured = capsys.readouterr()
    assert "calibration cancelled" not in captured.err
    assert captured.out == ""


def test_ctrl_c_while_a_dot_is_sampled_is_reported_as_an_interrupt():
    # Real Tk swallows a KeyboardInterrupt raised in a key handler. It must still end
    # calibration as an interrupt, not as "camera stopped" on the next press.
    def frames():
        for i in range(SETTLE_FRAMES + SAMPLE_FRAMES):
            yield Frame(i / 30, make_landmarks(0.2, -0.1))
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        calibrate_interactively(Config(), frames(), FakeDotScreen)


def test_the_last_dot_keeps_a_sample_when_the_input_ends_with_enough_frames():
    # A recording that ends right after the last dot, with one blink among its frames.
    frames = [Frame(i / 30, make_landmarks(0.2, -0.1)) for i in range(SETTLE_FRAMES)]
    frames += [Frame(1.0, None)]
    frames += [Frame(1.1 + i / 30, make_landmarks(0.2, -0.1)) for i in range(SAMPLE_FRAMES - 1)]
    features = collect_dot_features(iter(frames))
    assert features is not None
    assert features.left_iris_x == pytest.approx(0.2)
