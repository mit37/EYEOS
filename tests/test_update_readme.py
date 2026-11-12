"""scripts/update_readme.py: the guard that keeps README numbers generated."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from eval.generate_synthetic import generate_rows

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "update_readme.py"
_spec = importlib.util.spec_from_file_location("update_readme", _SCRIPT)
assert _spec is not None and _spec.loader is not None
update_readme = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(update_readme)


def test_inline_marker_must_occur_exactly_once():
    text = "**<!-- test-count -->1<!-- /test-count --> tests**"
    assert update_readme.inline("test-count", 7, text) == (
        "**<!-- test-count -->7<!-- /test-count --> tests**"
    )
    with pytest.raises(RuntimeError, match="test-count marker exactly once"):
        update_readme.inline("test-count", 7, "**230 tests**")  # marker deleted in an edit
    with pytest.raises(RuntimeError, match="found 2"):
        update_readme.inline("test-count", 7, text + text)


def test_region_markers_must_exist():
    text = "a\n<!-- results:start -->\nold\n<!-- results:end -->\nb"
    assert update_readme.region("results", "new", text) == (
        "a\n<!-- results:start -->\nnew\n<!-- results:end -->\nb"
    )
    with pytest.raises(RuntimeError, match="missing the results markers"):
        update_readme.region("results", "new", "a\nb")


def test_results_labels_are_read_from_the_fixture():
    facts = update_readme.fixture_facts(generate_rows())
    assert facts["calibration_frames"] == 45
    assert facts["calibration_dots"] == 9
    assert facts["fps"] == pytest.approx(10.0)
    assert facts["just_reading_s"] == pytest.approx(30.0)
    assert 15.0 < facts["head_turned_yaw_deg"] < 30.0

    # A different fixture changes the labels, not just the values.
    rows = generate_rows()
    shorter = [r for i, r in enumerate(rows) if not (r["segment"] == "just_reading" and i % 2)]
    assert update_readme.fixture_facts(shorter)["just_reading_s"] < facts["just_reading_s"]


def test_negative_result_tags_follow_the_numbers():
    metrics = {
        "calibration_validation_error_px": 30.0,
        "on_target_frame_count": 90,
        "on_target_frame_count_filtered": 90,
        "mean_pixel_error_raw_px": 20.0,
        "mean_pixel_error_filtered_px": 15.0,
        "steady_state_jitter_raw_px": 12.0,
        "steady_state_jitter_filtered_px": 8.0,
        "on_target_dwell_satisfied_frames": 60,
        "looking_away_by_mode": {"eyes_off_screen": {"frames": 10, "false_accepts": 0}},
        "just_reading_dwell_satisfied_frames": 290,
        "just_reading_frame_count": 300,
        "just_reading_false_click_count": 0,
    }
    facts = update_readme.fixture_facts(generate_rows())

    assert "negative result" not in update_readme.results_table(metrics, facts)
    assert "Negative result" not in update_readme.results_notes(metrics)
    assert "held on 60 of 90" in update_readme.results_notes(metrics)

    # Holding on 1 frame in 90 is still not usable clicking: still a negative result.
    rare = dict(metrics, on_target_dwell_satisfied_frames=1)
    assert "negative result" in update_readme.results_table(rare, facts)
    assert "Negative result: the dwell rarely holds" in update_readme.results_notes(rare)
    assert "(1 of 90 frames)" in update_readme.results_notes(rare)


def test_the_labels_change_with_the_fixture():
    metrics = update_readme.compute_metrics(update_readme.FIXTURE)
    facts = dict(
        update_readme.fixture_facts(generate_rows()),
        calibration_frames=36,
        fps=15.0,
        just_reading_s=20.0,
        head_turned_yaw_deg=33.0,
    )
    table = update_readme.results_table(metrics, facts)
    for label in ("36 calibration frames, 4 per dot", "at 15 fps", "20 s synthetic", "~33°"):
        assert label in table


def test_config_values_quoted_in_prose_are_filled_from_the_defaults():
    from eyeos.config import Config

    stale = "a gap over <!-- cfg:gates.max_frame_gap_s -->9.9<!-- /cfg --> s"
    want = Config().gates.max_frame_gap_s
    assert update_readme.config_values(stale) == (
        f"a gap over <!-- cfg:gates.max_frame_gap_s -->{want:g}<!-- /cfg --> s"
    )
    with pytest.raises(RuntimeError, match="unknown config value: gates.no_such_setting"):
        update_readme.config_values("<!-- cfg:gates.no_such_setting -->1<!-- /cfg -->")
