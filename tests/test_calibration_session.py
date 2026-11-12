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
    """Stands in for CalibrationUI: 'presses space' on each dot, retrying a failed one."""

    def __init__(self, width, height, capture_sample, cancel=False):
        self.targets = target_points(width, height)
        self.capture_sample = capture_sample
        self.cancel = cancel

    def run(self):
        if self.cancel:
            return False
        for target in self.targets:
            for _attempt in range(3):
                if self.capture_sample(target) is not False:
                    break
        return True


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
    assert features.left_iris_y == pytest.approx(-0.1)


def test_collect_dot_features_gives_up_when_too_few_usable_frames():
    frames = [Frame(i / 30, None) for i in range(100)]
    assert collect_dot_features(iter(frames)) is None


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


def test_eyeos_run_calibrate_moves_the_cursor_end_to_end(tmp_path, monkeypatch, capsys):
    import eyeos.cli as cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "CalibrationUI", FakeDotScreen)
    monkeypatch.setitem(sys.modules, "tkinter", make_fake_tk())

    frames, t = _calibration_frames()
    for i in range(30):  # then dwell on the screen centre
        frames.append(Frame(t + i / 30, make_landmarks(0.0, 0.0)))
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

    exit_code = cli.main(
        ["run", "--config", str(config_path), "--calibrate", "--overlay", "--no-log"]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "calibration validation error" in out
    assert "processed 30 frames: 30 moves" in out
    assert "move_cursor(960, 540)" in out
