"""Smoke test for the `eyeos run` CLI against the synthetic fixture path."""

from __future__ import annotations

import csv

import pytest

from eyeos.cli import main
from eyeos.synthetic_format import CSV_FIELDNAMES, landmarks_to_row
from tests.test_features import make_landmarks


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    # Session logs default to ./logs; keep test runs from writing into the repo.
    monkeypatch.chdir(tmp_path)


def _write_synthetic_csv(path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for t in (0.0, 0.2, 0.4):
            writer.writerow(landmarks_to_row(t, make_landmarks()))


def test_cli_run_against_synthetic_source(tmp_path, capsys):
    csv_path = tmp_path / "session.csv"
    _write_synthetic_csv(csv_path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"input:\n  source: synthetic\n  synthetic_path: {csv_path}\noutput:\n  backend: dry_run\n"
    )

    exit_code = main(["run", "--config", str(config_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "processed 3 frames" in out


def test_cli_run_respects_max_frames(tmp_path, capsys):
    csv_path = tmp_path / "session.csv"
    _write_synthetic_csv(csv_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"input:\n  source: synthetic\n  synthetic_path: {csv_path}\noutput:\n  backend: dry_run\n"
    )

    exit_code = main(["run", "--config", str(config_path), "--max-frames", "1"])
    assert exit_code == 0
    assert "processed 1 frames" in capsys.readouterr().out


def test_cli_run_writes_a_session_log(tmp_path):
    csv_path = tmp_path / "session.csv"
    _write_synthetic_csv(csv_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"input:\n  source: synthetic\n  synthetic_path: {csv_path}\noutput:\n  backend: dry_run\n"
    )
    log_path = tmp_path / "session.jsonl"

    main(["run", "--config", str(config_path), "--log", str(log_path)])

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 3


def test_cli_unknown_output_backend_raises(tmp_path):
    csv_path = tmp_path / "session.csv"
    _write_synthetic_csv(csv_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "input:\n"
        "  source: synthetic\n"
        f"  synthetic_path: {csv_path}\n"
        "output:\n"
        "  backend: not_a_backend\n"
    )
    with pytest.raises(ValueError, match="not_a_backend"):
        main(["run", "--config", str(config_path)])


class _RecordingRealCursor:
    """A fake real-cursor backend: Pipeline must treat it as able to move the mouse."""

    moves_real_cursor = True

    def __init__(self):
        self.actions = []
        self.closed = False

    def move_cursor(self, x, y):
        self.actions.append(("move", x, y))

    def click(self):
        self.actions.append(("click",))

    def close(self):
        self.closed = True


def _real_backend_config(tmp_path):
    csv_path = tmp_path / "session.csv"
    _write_synthetic_csv(csv_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "input:\n"
        "  source: synthetic\n"
        f"  synthetic_path: {csv_path}\n"
        "output:\n"
        "  backend: pynput\n"
    )
    return config_path


def test_cli_refuses_real_cursor_backend_without_a_kill_switch(tmp_path, monkeypatch, capsys):
    import eyeos.cli as cli

    built = []
    monkeypatch.setattr(
        cli, "_build_output", lambda config: built.append(1) or _RecordingRealCursor()
    )

    def no_keyboard(config):
        raise RuntimeError("no keyboard backend")

    monkeypatch.setattr(cli, "_build_key_signals", no_keyboard)

    exit_code = main(["run", "--config", str(_real_backend_config(tmp_path))])

    assert exit_code != 0
    assert built == []
    assert "kill switch" in capsys.readouterr().err


def test_cli_dry_run_never_needs_a_keyboard(tmp_path, monkeypatch):
    import eyeos.cli as cli

    def no_keyboard(config):
        raise AssertionError("dry_run must not build a key listener")

    monkeypatch.setattr(cli, "_build_key_signals", no_keyboard)
    csv_path = tmp_path / "session.csv"
    _write_synthetic_csv(csv_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"input:\n  source: synthetic\n  synthetic_path: {csv_path}\noutput:\n  backend: dry_run\n"
    )
    assert main(["run", "--config", str(config_path)]) == 0


def test_cli_wires_and_closes_the_kill_switch_for_real_backends(tmp_path, monkeypatch):
    import eyeos.cli as cli
    from eyeos.hotkeys import FakeKeySource, KeySignals

    source = FakeKeySource()
    seen = {}
    original_pipeline = cli.Pipeline

    def spy_pipeline(*args, **kwargs):
        seen["external_signals"] = kwargs.get("external_signals")
        return original_pipeline(*args, **kwargs)

    monkeypatch.setattr(cli, "Pipeline", spy_pipeline)
    monkeypatch.setattr(cli, "_build_output", lambda config: _RecordingRealCursor())
    monkeypatch.setattr(cli, "_build_key_signals", lambda config: KeySignals("esc", source))

    assert main(["run", "--config", str(_real_backend_config(tmp_path))]) == 0
    assert isinstance(seen["external_signals"], KeySignals)
    assert not source.running


def _synthetic_config(tmp_path, extra=""):
    csv_path = tmp_path / "session.csv"
    _write_synthetic_csv(csv_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"input:\n  source: synthetic\n  synthetic_path: {csv_path}\n"
        f"output:\n  backend: dry_run\n{extra}"
    )
    return config_path


@pytest.mark.parametrize(
    "mode, backend, warned",
    [("keypress", "dry_run", True), ("either", "dry_run", False), ("keypress", "pynput", False)],
)
def test_cli_warns_that_a_keypress_dry_run_cannot_click(
    tmp_path, monkeypatch, capsys, mode, backend, warned
):
    import eyeos.cli as cli
    from eyeos.hotkeys import FakeKeySource, KeySignals

    if backend != "dry_run":
        monkeypatch.setattr(cli, "_build_output", lambda config: _RecordingRealCursor())
        monkeypatch.setattr(
            cli, "_build_key_signals", lambda config: KeySignals("esc", FakeKeySource(), "f9")
        )
    config_path = _write_n_frames_config(tmp_path, 3, backend=backend)
    with config_path.open("a") as f:
        f.write(f"  confirm_key: f9\ngates:\n  confirmation_mode: {mode}\n")

    assert main(["run", "--config", str(config_path), "--no-log"]) == 0

    err = capsys.readouterr().err
    assert ("dry runs don't listen to the keyboard" in err) is warned
    if warned:
        assert len(err.strip().splitlines()) == 1


def test_cli_writes_a_session_log_by_default(tmp_path):
    log_dir = tmp_path / "my-logs"
    config_path = _synthetic_config(tmp_path, f"logging:\n  session_log_dir: {log_dir}\n")

    assert main(["run", "--config", str(config_path)]) == 0

    logs = list(log_dir.glob("session-*.jsonl"))
    assert len(logs) == 1
    assert len(logs[0].read_text().strip().splitlines()) == 3


def test_cli_no_log_writes_nothing(tmp_path):
    log_dir = tmp_path / "my-logs"
    config_path = _synthetic_config(tmp_path, f"logging:\n  session_log_dir: {log_dir}\n")

    assert main(["run", "--config", str(config_path), "--no-log"]) == 0
    assert not log_dir.exists()


@pytest.mark.parametrize("source", ["webcam", "video_file"])
def test_cli_forwards_gate1_threshold_to_live_sources(tmp_path, source):
    from eyeos.cli import _build_input_source
    from eyeos.config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"input:\n  source: {source}\n  video_path: {tmp_path / 'clip.mp4'}\n"
        "gates:\n  presence_confidence_threshold: 0.9\n"
    )
    built = _build_input_source(load_config(config_path))

    assert built.min_detection_confidence == 0.9
    assert built.min_tracking_confidence == 0.9


# ---------------------------------------------------------------------------
# `eyeos run` on a video file, end to end through fake cv2/mediapipe
# ---------------------------------------------------------------------------


def _video_config(tmp_path, video_line=None):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")  # only has to exist: the fake cv2 does the "decoding"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "input:\n  source: video_file\n"
        + (video_line if video_line is not None else f"  video_path: {video}\n")
        + f"output:\n  backend: dry_run\nlogging:\n  session_log_dir: {tmp_path / 'logs'}\n"
    )
    return config_path


@pytest.mark.parametrize("max_frames", [None, 5])
def test_cli_runs_a_video_file_end_to_end(tmp_path, capsys, monkeypatch, max_frames):
    import json

    from tests.fake_capture import FakeCv2, FakeMediapipe, install_fakes

    fake_cv2 = FakeCv2(n_frames=12)
    fake_mp = FakeMediapipe(faces_per_frame=[1, 0, 2] * 4)
    install_fakes(monkeypatch, cv2=fake_cv2, mediapipe=fake_mp)
    argv = ["run", "--config", str(_video_config(tmp_path))]
    if max_frames is not None:
        argv += ["--max-frames", str(max_frames)]

    assert main(argv) == 0

    expected = max_frames or 12
    assert f"processed {expected} frames" in capsys.readouterr().out
    (capture,) = fake_cv2.captures
    (mesh,) = fake_mp.face_meshes
    assert capture.release_calls == 1  # released once, on a full run and an early stop
    assert mesh.close_calls == 1
    assert mesh.kwargs["refine_landmarks"] is True

    (log,) = (tmp_path / "logs").glob("session-*.jsonl")
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == expected
    # Container timestamps (index / fps), and every face count reaches Gate 1.
    assert [r["timestamp_s"] for r in records] == pytest.approx([i / 30 for i in range(expected)])
    reasons = [r["reasons"]["gate1"] for r in records[:3]]
    assert reasons[1] == "expected exactly one face, saw 0"
    assert reasons[2] == "expected exactly one face, saw 2"
    assert not any(r["move_allowed"] for r in records)  # no calibration: Gate 2 holds


def test_cli_reports_a_missing_video_file(tmp_path, capsys, monkeypatch):
    from tests.fake_capture import FakeCv2, FakeMediapipe, install_fakes

    install_fakes(monkeypatch, cv2=FakeCv2(), mediapipe=FakeMediapipe())
    config_path = _video_config(tmp_path, f"  video_path: {tmp_path / 'missing.mp4'}\n")

    with pytest.raises(RuntimeError, match="video file not found"):
        main(["run", "--config", str(config_path), "--no-log"])


def test_cli_requires_video_path_for_a_video_source(tmp_path):
    config_path = _video_config(tmp_path, video_line="")

    with pytest.raises(ValueError, match="video_path is not set"):
        main(["run", "--config", str(config_path), "--no-log"])


def test_two_runs_in_the_same_second_keep_both_session_logs(tmp_path, monkeypatch):
    import eyeos.cli as cli

    class FrozenDatetime:
        @staticmethod
        def now(tz=None):
            from datetime import datetime

            return datetime(2026, 11, 12, 14, 0, 0, tzinfo=tz)

    monkeypatch.setattr(cli, "datetime", FrozenDatetime)
    log_dir = tmp_path / "logs"
    config_path = _synthetic_config(tmp_path, f"logging:\n  session_log_dir: {log_dir}\n")

    assert main(["run", "--config", str(config_path), "--max-frames", "3"]) == 0
    assert main(["run", "--config", str(config_path), "--max-frames", "1"]) == 0

    logs = sorted(log_dir.glob("session-*.jsonl"))
    assert [p.name for p in logs] == [
        "session-20261112T140000Z-2.jsonl",
        "session-20261112T140000Z.jsonl",
    ]
    assert [len(p.read_text().splitlines()) for p in logs] == [1, 3]


# ---------------------------------------------------------------------------
# Closing the overlay window, and cleanup when a close() fails
# ---------------------------------------------------------------------------


def _write_n_frames_config(tmp_path, n_frames, backend="pynput"):
    csv_path = tmp_path / "session.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for i in range(n_frames):
            writer.writerow(landmarks_to_row(i / 30, make_landmarks()))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"input:\n  source: synthetic\n  synthetic_path: {csv_path}\n"
        f"output:\n  backend: {backend}\n"
    )
    return config_path


def test_closing_the_overlay_mid_run_keeps_the_session_and_closes_everything(
    tmp_path, monkeypatch, capsys
):
    import sys

    import eyeos.cli as cli
    from eyeos.hotkeys import FakeKeySource, KeySignals
    from eyeos.logging_utils import SessionLogger
    from eyeos.overlay import GateOverlay
    from tests.fake_tk import make_fake_tk

    monkeypatch.setitem(sys.modules, "tkinter", make_fake_tk())
    keyboard = FakeKeySource()
    output = _RecordingRealCursor()
    overlays, loggers = [], []

    class UserClosesTheOverlay(GateOverlay):
        def __init__(self):
            super().__init__()
            self.frames_seen = 0
            # The user closes the window while Tk is drawing frame 2.
            self.root.on_update = lambda root: root.close_window() if root.updates == 2 else None
            overlays.append(self)

        def update(self, status):
            self.frames_seen += 1
            if self.frames_seen == 5:
                keyboard.press("esc")  # then ends the session with the kill hotkey
            super().update(status)

    class SpyLogger(SessionLogger):
        def __init__(self, path):
            super().__init__(path)
            loggers.append(self)

    monkeypatch.setattr(cli, "GateOverlay", UserClosesTheOverlay)
    monkeypatch.setattr(cli, "SessionLogger", SpyLogger)
    monkeypatch.setattr(cli, "_build_output", lambda config: output)
    monkeypatch.setattr(cli, "_build_key_signals", lambda config: KeySignals("esc", keyboard))
    config_path = _write_n_frames_config(tmp_path, 10)

    exit_code = main(["run", "--config", str(config_path), "--overlay"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "processed 6 frames" in out  # frames kept coming after the window closed
    assert "kill switch (esc) pressed: session ended" in out
    (overlay,) = overlays
    assert overlay.root.destroy_calls == 1
    assert overlay.root.updates == 2  # nothing drawn after the close
    assert output.closed
    assert loggers[0]._fh.closed
    assert not keyboard.running


@pytest.mark.parametrize("failing", ["input", "overlay", "output", "logger", "keys"])
def test_every_resource_is_closed_even_if_one_close_raises(tmp_path, monkeypatch, failing):
    import sys

    import eyeos.cli as cli
    from eyeos.hotkeys import FakeKeySource, KeySignals
    from eyeos.logging_utils import SessionLogger
    from eyeos.overlay import GateOverlay
    from tests.fake_tk import make_fake_tk

    monkeypatch.setitem(sys.modules, "tkinter", make_fake_tk())
    closed = []

    def recording(name, resource):
        original = resource.close

        def close():
            closed.append(name)
            original()
            if name == failing:
                raise RuntimeError(f"{name} close failed")

        resource.close = close
        return resource

    build_input = cli._build_input_source
    monkeypatch.setattr(cli, "_build_input_source", lambda c: recording("input", build_input(c)))
    monkeypatch.setattr(cli, "GateOverlay", lambda: recording("overlay", GateOverlay()))
    monkeypatch.setattr(cli, "_build_output", lambda c: recording("output", _RecordingRealCursor()))
    monkeypatch.setattr(cli, "SessionLogger", lambda p: recording("logger", SessionLogger(p)))
    keyboard = FakeKeySource()
    monkeypatch.setattr(
        cli, "_build_key_signals", lambda c: recording("keys", KeySignals("esc", keyboard))
    )
    config_path = _write_n_frames_config(tmp_path, 3)

    with pytest.raises(RuntimeError, match=f"{failing} close failed"):
        main(["run", "--config", str(config_path), "--overlay"])

    assert set(closed) == {"input", "overlay", "output", "logger", "keys"}
    assert not keyboard.running


def test_the_eyeos_command_prints_errors_as_one_line(tmp_path, capsys, monkeypatch):
    from eyeos.cli import console_main
    from tests.fake_capture import FakeCv2, FakeMediapipe, install_fakes

    install_fakes(monkeypatch, cv2=FakeCv2(), mediapipe=FakeMediapipe())
    config_path = _video_config(tmp_path, f"  video_path: {tmp_path / 'missing.mp4'}\n")

    assert console_main(["run", "--config", str(config_path), "--no-log"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("eyeos: error: video file not found")
    assert "Traceback" not in err

    assert console_main(["run", "--config", str(tmp_path / "nope.yaml")]) == 2
    assert "eyeos: error:" in capsys.readouterr().err

    for name, text in [
        ("bad.yaml", "output: {kill_hotkey: [esc\n"),
        ("int.yaml", "output: {kill_hotkey: 5}\n"),
    ]:
        (tmp_path / name).write_text(text)
        assert console_main(["run", "--config", str(tmp_path / name)]) == 2
        err = capsys.readouterr().err
        assert err.startswith("eyeos: error:") and "Traceback" not in err


def test_the_eyeos_command_reports_ctrl_c_cleanly(tmp_path, capsys, monkeypatch):
    import eyeos.cli as cli

    def interrupted(_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_run", interrupted)
    assert cli.console_main(["run", "--config", str(_synthetic_config(tmp_path))]) == 130
    assert capsys.readouterr().err == "eyeos: interrupted\n"


def test_the_installed_command_is_console_main():
    from pathlib import Path

    tomllib = pytest.importorskip("tomllib")  # Python 3.11+

    pyproject = tomllib.loads(
        (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    )
    assert pyproject["project"]["scripts"]["eyeos"] == "eyeos.cli:console_main"
