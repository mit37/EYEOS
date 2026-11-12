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
