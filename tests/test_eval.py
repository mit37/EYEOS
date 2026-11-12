"""eval/generate_synthetic.py and eval/run_eval.py: the synthetic-fixture generator
and the metrics it lets CI compute without a camera (PRD §5)."""

from __future__ import annotations

from pathlib import Path

from eval.generate_synthetic import (
    DT,
    generate_rows,
    make_landmarks,
    make_row,
    offset_for_target,
    write_csv,
)
from eval.run_eval import build_calibration, compute_metrics, load_raw_rows
from eyeos.config import Config
from eyeos.input_sources.synthetic import SyntheticInputSource
from eyeos.output.dry_run import DryRunOutputAdapter
from eyeos.pipeline import Pipeline

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "eval" / "fixtures" / "synthetic_session.csv"
)


def _on_target_move_allowed(path: Path) -> list[bool]:
    rows = load_raw_rows(path)
    config = Config()
    calibration = build_calibration(rows, config.screen.width_px, config.screen.height_px)
    pipeline = Pipeline(
        config, SyntheticInputSource(path), DryRunOutputAdapter(echo=None), calibration=calibration
    )
    statuses = pipeline.run().statuses
    return [
        s.move_allowed
        for row, s in zip(rows, statuses, strict=True)
        if row["segment"] == "on_target"
    ]


def _two_noise_free_blocks_session(path: Path) -> None:
    """Calibration rows, then two adjacent noise-free on_target blocks >400px apart, so
    Gate 4's speed limit suppresses each block's switch frame and true jitter is ~0."""
    rows = [row for row in generate_rows() if row["segment"] == "calibration"]
    t = float(rows[-1]["timestamp_s"]) + DT
    for target in [(300.0, 200.0), (1600.0, 900.0)]:
        landmarks = make_landmarks(*offset_for_target(*target))
        for _ in range(15):
            rows.append(make_row(t, landmarks, target, "on_target"))
            t += DT
    write_csv(path, rows)


def test_generate_rows_is_deterministic_for_a_given_seed():
    rows_a = generate_rows(seed=123)
    rows_b = generate_rows(seed=123)
    assert rows_a == rows_b


def test_generate_rows_different_seeds_give_different_noise():
    rows_a = generate_rows(seed=1)
    rows_b = generate_rows(seed=2)
    assert rows_a != rows_b


def test_calibration_frames_carry_noise_within_each_dot():
    rows = [row for row in generate_rows() if row["segment"] == "calibration"]
    by_dot: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        by_dot.setdefault((row["target_x"], row["target_y"]), set()).add(row["left_iris_center_x"])
    assert len(by_dot) == 9
    assert all(len(xs) > 1 for xs in by_dot.values())


def test_generate_rows_covers_all_four_segments():
    rows = generate_rows()
    segments = {row["segment"] for row in rows}
    assert segments == {"calibration", "on_target", "looking_away", "just_reading"}


def test_committed_fixture_is_the_current_generator_output():
    from eval.run_eval import load_raw_rows

    assert FIXTURE_PATH.exists(), "run `python -m eval.generate_synthetic` to (re)generate it"
    assert (
        load_raw_rows(FIXTURE_PATH) == generate_rows()
    ), "fixture is stale: run python -m eval.generate_synthetic"


def test_compute_metrics_on_the_committed_fixture_matches_documented_expectations():
    metrics = compute_metrics(FIXTURE_PATH)

    # The calibration frames are noisy too, so the error measures something, and it stays
    # under Gate 2's limit (at or above it the cursor would freeze and every zero below
    # would be vacuous).
    max_error = Config().gates.calibration_max_error_px
    assert 1.0 < metrics["calibration_validation_error_px"] < max_error
    # The fixture's whole point: gates never false-accept a "looking away" frame, and
    # no click ever fires during "just reading" without confirmation.
    assert metrics["gate_false_accept_rate"] == 0.0
    assert metrics["click_false_trigger_rate"] == 0.0
    assert metrics["just_reading_false_click_count"] == 0
    # The One Euro filter should reduce both pixel error and steady-state jitter
    # relative to the raw (unfiltered) calibration output, given this fixture's noise.
    assert metrics["mean_pixel_error_filtered_px"] < metrics["mean_pixel_error_raw_px"]
    assert metrics["steady_state_jitter_filtered_px"] < metrics["steady_state_jitter_raw_px"]


def test_filtered_pixel_error_uses_the_same_frames_as_raw():
    # Only meaningful if the gates suppress some on-target frames in the fixture.
    assert not all(_on_target_move_allowed(FIXTURE_PATH))
    metrics = compute_metrics(FIXTURE_PATH)
    assert metrics["on_target_frame_count_filtered"] == metrics["on_target_frame_count"]


def test_jitter_never_pairs_frames_across_a_target_switch(tmp_path):
    p = tmp_path / "two_blocks.csv"
    _two_noise_free_blocks_session(p)
    assert not all(_on_target_move_allowed(p))

    metrics = compute_metrics(p, jitter_warmup_frames=1)

    assert metrics["on_target_frame_count_filtered"] == metrics["on_target_frame_count"]
    assert metrics["steady_state_jitter_raw_px"] < 1.0
    assert metrics["steady_state_jitter_filtered_px"] < 1.0

    # With no warm-up only the filter's step response is left; a single cross-block
    # pair (>400px over 28 pairs) would push the mean well past this bound.
    no_warmup = compute_metrics(p, jitter_warmup_frames=0)
    assert no_warmup["steady_state_jitter_raw_px"] < 1.0
    assert no_warmup["steady_state_jitter_filtered_px"] < 5.0


def test_compute_metrics_on_a_small_generated_session(tmp_path):
    rows = generate_rows(seed=7)
    p = tmp_path / "small_session.csv"
    write_csv(p, rows)
    metrics = compute_metrics(p)
    assert metrics["frame_count"] == len(rows)
    assert metrics["on_target_frame_count"] > 0
    assert metrics["looking_away_frame_count"] > 0
    assert metrics["just_reading_frame_count"] > 0


def test_dwell_reachability_is_reported_next_to_the_false_click_rate():
    metrics = compute_metrics(FIXTURE_PATH)

    # Documented negative result: at the fixture's noise level dwell never holds on target,
    # while the static just-reading hold satisfies it almost all the time.
    assert metrics["on_target_dwell_satisfied_frames"] == 0
    assert (
        metrics["just_reading_dwell_satisfied_frames"] > 0.9 * metrics["just_reading_frame_count"]
    )


def test_looking_away_covers_gate2_and_off_screen_modes_with_per_mode_rates():
    metrics = compute_metrics(FIXTURE_PATH)
    by_mode = metrics["looking_away_by_mode"]

    assert set(by_mode) == {
        "two_faces",
        "low_confidence",
        "no_face",
        "head_turned",
        "eyes_off_screen",
    }
    for mode in ("two_faces", "low_confidence", "no_face", "head_turned"):
        assert by_mode[mode]["false_accepts"] == 0, mode
    # Documented negative result: an eyes-only glance past the screen edge passes every gate.
    assert by_mode["eyes_off_screen"] == {"frames": 10, "false_accepts": 10}
    # The aggregate stays over the Gate-1 modes only, as the README row says.
    assert metrics["looking_away_frame_count"] == 30
