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
