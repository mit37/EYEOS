"""Smoke test for the `eyeos run` CLI against the synthetic fixture path."""

from __future__ import annotations

import csv

import pytest

from eyeos.cli import main
from eyeos.synthetic_format import CSV_FIELDNAMES, landmarks_to_row
from tests.test_features import make_landmarks


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
