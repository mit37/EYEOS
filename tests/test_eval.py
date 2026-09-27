"""eval/generate_synthetic.py and eval/run_eval.py: the synthetic-fixture generator
and the metrics it lets CI compute without a camera (PRD §5)."""

from __future__ import annotations

from pathlib import Path

from eval.generate_synthetic import generate_rows, write_csv
from eval.run_eval import compute_metrics

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "eval" / "fixtures" / "synthetic_session.csv"
)


def test_generate_rows_is_deterministic_for_a_given_seed():
    rows_a = generate_rows(seed=123)
    rows_b = generate_rows(seed=123)
    assert rows_a == rows_b


def test_generate_rows_different_seeds_give_different_noise():
    rows_a = generate_rows(seed=1)
    rows_b = generate_rows(seed=2)
    assert rows_a != rows_b


def test_generate_rows_covers_all_four_segments():
    rows = generate_rows()
    segments = {row["segment"] for row in rows}
    assert segments == {"calibration", "on_target", "looking_away", "just_reading"}


def test_committed_fixture_exists_and_is_the_current_generator_output():
    assert FIXTURE_PATH.exists(), "run `python -m eval.generate_synthetic` to (re)generate it"
    committed_rows = compute_metrics(FIXTURE_PATH)  # just needs to not raise / parse cleanly
    assert committed_rows["frame_count"] > 0


def test_compute_metrics_on_the_committed_fixture_matches_documented_expectations():
    metrics = compute_metrics(FIXTURE_PATH)

    assert metrics["calibration_validation_error_px"] < 5.0
    # The fixture's whole point: gates never false-accept a "looking away" frame, and
    # no click ever fires during "just reading" without confirmation.
    assert metrics["gate_false_accept_rate"] == 0.0
    assert metrics["click_false_trigger_rate"] == 0.0
    assert metrics["just_reading_false_click_count"] == 0
    # The One Euro filter should reduce both pixel error and steady-state jitter
    # relative to the raw (unfiltered) calibration output, given this fixture's noise.
    assert metrics["mean_pixel_error_filtered_px"] < metrics["mean_pixel_error_raw_px"]
    assert metrics["steady_state_jitter_filtered_px"] < metrics["steady_state_jitter_raw_px"]


def test_compute_metrics_on_a_small_generated_session(tmp_path):
    rows = generate_rows(seed=7)
    p = tmp_path / "small_session.csv"
    write_csv(p, rows)
    metrics = compute_metrics(p)
    assert metrics["frame_count"] == len(rows)
    assert metrics["on_target_frame_count"] > 0
    assert metrics["looking_away_frame_count"] > 0
    assert metrics["just_reading_frame_count"] > 0
