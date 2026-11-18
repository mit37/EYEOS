"""scripts/update_readme.py: the guard that keeps README numbers generated."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import shlex
import statistics
from pathlib import Path

import numpy as np
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


def _public_results(
    variants: tuple[str, ...] = ("eyeos", "global-lopo"),
    noise: float = 0.0,
    limit: float = 60.0,
    persons: tuple[str, ...] = ("p00", "p01"),
) -> dict:
    """What `eval.public.mpiifacegaze evaluate` writes, from the real protocol on
    noise-free (or, with ``noise``, noisy) fake persons (no scipy or dataset needed)."""
    import random

    import numpy as np

    from eval.generate_synthetic import make_landmarks
    from eval.public import models, protocol
    from tests.fake_mpiifacegaze import offsets_for_target

    def person(name: str) -> protocol.PersonData:
        rng = random.Random(name)
        width, height = 1280.0, 800.0
        targets = [(rng.uniform(0, width - 1), rng.uniform(0, height - 1)) for _ in range(400)]
        landmarks: list = []
        for x, y in targets:
            ox, oy = offsets_for_target(x, y, width, height)
            if noise:
                ox, oy = ox + rng.gauss(0.0, noise), oy + rng.gauss(0.0, noise)
            landmarks.append(make_landmarks(ox, oy))
        landmarks[0] = None  # one image without a face
        return protocol.PersonData(name, (width, height), np.array(targets), landmarks)

    registry = models.build_variants()
    out = protocol.evaluate(
        [person(name) for name in persons],
        [registry[name] for name in variants],
        limits=protocol.Limits(gate2_max_error_px=limit),
    )
    return {
        "generated": "2026-11-18T02:00:00+00:00",
        "git_commit": "0123456789abcdef0123456789abcdef01234567",
        "git_dirty": False,
        "config": {
            "root": "datasets/MPIIFaceGaze",
            "cache": "datasets/cache/mpiifacegaze",
            "persons": list(persons),
            "variants": list(variants),
            "k_per_dot": protocol.DEFAULT_K_PER_DOT,
            "gate2_max_validation_error_px": limit,
        },
        **out,
    }


def test_public_results_without_a_results_file_say_not_measured():
    text = update_readme.public_results(None)
    assert "not measured" in text
    assert "|" not in text


def test_public_results_table_has_a_row_per_variant_with_its_numbers():
    results = _public_results()
    text = update_readme.public_results(results)
    detection = results["detection"]["aggregate"]
    assert f"{detection['valid']} of {detection['images']}" in text
    assert "photos of 2 people" in text
    assert "0123456" in text and "2026-11-18" in text
    for name, variant in results["variants"].items():
        agg = variant["aggregate"]
        row = next(line for line in text.splitlines() if line.startswith(f"| `{name}`"))
        assert f"{agg['pixel_error_px']['mean_of_person_means']:.1f}" in row
        assert f"{agg['gate2_passed']} / {agg['persons_evaluated']}" in row
    # The shipping calibration is the row to read first.
    assert "| `eyeos` (shipping)" in text


def test_public_results_mark_missing_angles_and_skipped_persons():
    results = _public_results()
    agg = results["variants"]["eyeos"]["aggregate"]
    agg["angular_error_deg"] = None
    agg["persons_skipped"] = ["p07"]
    results["git_dirty"] = True
    text = update_readme.public_results(results)
    row = next(line for line in text.splitlines() if line.startswith("| `eyeos`"))
    assert "not measured" in row
    assert "p07" in text
    assert "uncommitted changes" in text


# ---------------------------------------------------------------------------
# the MPIIFaceGaze section: negative result, honest check, baselines, restatements
# ---------------------------------------------------------------------------


def _table(text: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    """The MPIIFaceGaze table: its header, and each row's cells by column, keyed by the
    variant's name, in the table's order."""
    lines = [line for line in text.splitlines() if line.startswith("| ")]
    header = [cell.strip() for cell in lines[0].strip(" |").split("|")]
    rows: dict[str, dict[str, str]] = {}
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip(" |").split("|")]
        name = re.match(r"`([^`]+)`", cells[0])
        assert name is not None, line
        rows[name.group(1)] = dict(zip(header, cells, strict=True))
    return header, rows


def _column(header: list[str], prefix: str) -> str:
    [column] = [name for name in header if name.startswith(prefix)]
    return column


def _paragraph(text: str, needle: str) -> str:
    [paragraph] = [p for p in text.split("\n\n") if needle in p]
    return paragraph


def _footnote(text: str) -> str:
    [footnote] = [p for p in text.split("\n\n") if p.startswith("† ")]
    return footnote


def _validation(results: dict, variant: str) -> list[float]:
    return [p["validation_error_px"] for p in results["variants"][variant]["persons"].values()]


def _person_means(results: dict, variant: str) -> dict[str, float]:
    persons = results["variants"][variant]["persons"]
    return {name: p["pixel_error_px"]["mean"] for name, p in persons.items()}


def test_public_results_state_the_negative_result_from_the_validation_error():
    results = _public_results(noise=0.3, limit=75.0)
    errors = _validation(results, "eyeos")
    assert min(errors) > 75.0  # the noise sinks every calibration

    text = update_readme.public_results(results)

    paragraph = _paragraph(text, "Negative result")
    assert paragraph.startswith("**Negative result")
    assert "over its 45 calibration photos" in paragraph
    assert f"averages {np.mean(errors):.1f} px per person" in paragraph
    assert f"({min(errors):.1f} to {max(errors):.1f} px)" in paragraph
    assert "Gate 2's 75 px limit" in paragraph  # the limit is the JSON's, not a default
    assert "0 of 2 persons pass" in paragraph
    # This file has no live-style figure, and the text says that the 45-photo one is not it.
    assert "Live Gate 2 judges a different statistic (†)" in paragraph
    assert "which this file does not report" in paragraph


def test_public_results_have_no_negative_result_when_every_calibration_passes():
    results = _public_results()  # noise-free: every calibration recovers its targets

    text = update_readme.public_results(results)

    assert "Negative result" not in text
    assert "2 of 2 persons pass" in _paragraph(text, "leave-one-out validation error")


def test_public_results_give_the_live_style_statistic_and_name_it_live_gate_2s():
    results = _public_results(("eyeos", "eyeos-live9"), noise=0.3)
    live = _validation(results, "eyeos-live9")
    passed = sum(p["passes_gate2"] for p in results["variants"]["eyeos-live9"]["persons"].values())

    text = update_readme.public_results(results)

    paragraph = _paragraph(text, "Negative result")
    assert "(`eyeos-live9`)" in paragraph
    assert "the statistic live Gate 2 judges" in paragraph
    assert f"{np.mean(live):.1f} px per person ({min(live):.1f} to {max(live):.1f} px)" in (
        paragraph
    )
    assert f"{passed} of 2 pass" in paragraph
    assert "does not report" not in paragraph


def test_a_failing_live_style_check_is_a_negative_result_even_when_the_photo_check_passes():
    # Noise-free, the 45-photo leave-one-out is ~0, but eyeos-live9 labels each averaged
    # sample with the dot, not the images' true targets, so its error is not.
    first = _public_results(("eyeos", "eyeos-live9"))
    limit = (max(_validation(first, "eyeos")) + min(_validation(first, "eyeos-live9"))) / 2
    assert max(_validation(first, "eyeos")) < limit < min(_validation(first, "eyeos-live9"))
    results = _public_results(("eyeos", "eyeos-live9"), limit=limit)

    paragraph = _paragraph(update_readme.public_results(results), "Negative result")

    assert "2 of 2 persons pass" in paragraph  # the photo check
    assert "and 0 of 2 pass" in paragraph  # the live-style check
    assert f"Gate 2's {limit:g} px limit" in paragraph


def test_public_results_name_the_calibration_check_for_what_it_is():
    results = _public_results(limit=75.0)
    results["config"]["k_per_dot"] = 7  # the photo count is read from the config

    text = update_readme.public_results(results)

    header, rows = _table(text)
    assert "Gate 2 would accept" not in text
    check = _column(header, "Calibration check passed")
    assert check == "Calibration check passed (leave-one-out < 75 px)†"
    assert rows["eyeos"][check] == "2 / 2"
    loo = _column(header, "Mean leave-one-out error")
    agg = results["variants"]["eyeos"]["aggregate"]
    assert rows["eyeos"][loo] == f"{agg['validation_error_px_mean']:.1f} px"
    footnote = _footnote(text)
    assert "63 calibration photos" in footnote and "live Gate 2" in footnote
    assert "For every variant it is" in footnote and "eyeos-live9" not in footnote

    with_live = update_readme.public_results(_public_results(("eyeos", "eyeos-live9")))
    footnote = _footnote(with_live)
    assert "For every variant except `eyeos-live9` it is" in footnote
    assert "45 calibration photos" in footnote
    assert "`eyeos-live9` computes that one" in footnote


def test_public_results_count_the_persons_each_variant_beats_eyeos_for():
    persons = ("p00", "p01", "p02")
    results = _public_results(("eyeos", "global-lopo", "centre"), noise=0.1, persons=persons)
    eyeos = _person_means(results, "eyeos")

    header, rows = _table(update_readme.public_results(results))

    better = _column(header, "Better than `eyeos` for")
    assert rows["eyeos"][better] == "-"
    for name in ("global-lopo", "centre"):
        means = _person_means(results, name)
        count = sum(means[p] < eyeos[p] for p in persons)
        assert rows[name][better] == f"{count} / 3", name
    assert rows["centre"][better] == "0 / 3"

    # By hand: lower for p01 only (p02 ties), then p02 skipped.
    lopo = results["variants"]["global-lopo"]["persons"]
    lopo["p00"]["pixel_error_px"]["mean"] = eyeos["p00"] + 1.0
    lopo["p01"]["pixel_error_px"]["mean"] = eyeos["p01"] - 1.0
    lopo["p02"]["pixel_error_px"]["mean"] = eyeos["p02"]
    _, rows = _table(update_readme.public_results(results))
    assert rows["global-lopo"][better] == "1 / 3"
    lopo["p02"] = {"skipped": "needs at least one other person with valid frames"}
    _, rows = _table(update_readme.public_results(results))
    assert rows["global-lopo"][better] == "1 / 2"


def test_public_results_mark_rows_whose_results_are_eyeoss():
    results = _public_results(("eyeos", "ridge-0.001", "per-dot-5", "ridge-0.1"), noise=0.1)

    text = update_readme.public_results(results)

    _, rows = _table(text)
    assert rows["eyeos"]["Variant"] == "`eyeos` (shipping)"
    assert rows["ridge-0.001"]["Variant"] == "`ridge-0.001` (= eyeos)"
    assert rows["per-dot-5"]["Variant"] == "`per-dot-5` (= eyeos)"
    assert rows["ridge-0.1"]["Variant"] == "`ridge-0.1`"
    note = _paragraph(text, "(= eyeos):")
    assert "not an independent" in note

    # One number apart is a different result.
    results["variants"]["per-dot-5"]["persons"]["p01"]["validation_error_px"] += 1e-9
    results["variants"]["ridge-0.001"]["persons"]["p00"]["n_test"] -= 1
    text = update_readme.public_results(results)
    _, rows = _table(text)
    assert rows["per-dot-5"]["Variant"] == "`per-dot-5`"
    assert rows["ridge-0.001"]["Variant"] == "`ridge-0.001`"
    assert "(= eyeos)" not in text


def test_public_results_put_the_baselines_first_and_scale_eyeos_against_the_centre():
    results = _public_results(
        ("eyeos", "global-lopo", "centre", "calibration-mean", "eyeos-live9"), noise=0.1
    )

    text = update_readme.public_results(results)

    _, rows = _table(text)
    assert list(rows) == ["centre", "calibration-mean", "eyeos", "global-lopo", "eyeos-live9"]
    assert rows["centre"]["Variant"] == "`centre` (baseline)"
    assert rows["calibration-mean"]["Variant"] == "`calibration-mean` (baseline)"
    assert rows["eyeos-live9"]["Variant"] == "`eyeos-live9` (live-style)"
    mean = {
        name: results["variants"][name]["aggregate"]["pixel_error_px"]["mean_of_person_means"]
        for name in ("eyeos", "centre")
    }
    below = 100.0 * (mean["centre"] - mean["eyeos"]) / mean["centre"]
    assert below > 50.0
    sentence = _paragraph(text, "`centre` baseline")
    assert f"{mean['eyeos']:.1f} px, is {below:.1f}% below" in sentence
    assert f"`centre` baseline's {mean['centre']:.1f} px" in sentence
    assert "Negative result" not in sentence


def test_a_shipping_calibration_no_better_than_the_centre_is_a_negative_result():
    results = _public_results(("eyeos", "centre"), noise=0.1)
    centre = results["variants"]["centre"]["aggregate"]["pixel_error_px"]["mean_of_person_means"]
    results["variants"]["eyeos"]["aggregate"]["pixel_error_px"]["mean_of_person_means"] = (
        1.1 * centre
    )

    sentence = _paragraph(update_readme.public_results(results), "`centre` baseline")

    assert sentence.startswith("**Negative result.**")
    assert "10.0% above" in sentence


def test_public_results_summarise_person_means_by_median_and_max():
    results = _public_results(("eyeos",), noise=0.1, persons=("p00", "p01", "p02", "p03"))
    means = sorted(_person_means(results, "eyeos").values())

    text = update_readme.public_results(results)

    header, rows = _table(text)
    column = _column(header, "Median / max of person means")
    median = (means[1] + means[2]) / 2  # four persons: the middle two's mean
    assert median == pytest.approx(statistics.median(means))
    assert rows["eyeos"][column] == f"{median:.1f} / {means[3]:.1f} px"
    assert "p90" not in text


def test_public_results_give_the_commands_that_reproduce_them():
    from eval.public import mpiifacegaze as mpii

    results = _public_results()
    results["config"].update(root="data/MPIIFaceGaze", cache="data/cache/mpii")

    text = update_readme.public_results(results)

    block = text.split("```bash\n", 1)[1].split("```", 1)[0]
    download, extract, evaluate = (shlex.split(line) for line in block.splitlines())
    assert download[:3] == ["python", "scripts/download_mpiifacegaze.py", "--accept-license"]
    assert download[download.index("--dest") + 1] == "data"
    assert [extract[3], evaluate[3]] == ["extract", "evaluate"]
    for command in (extract, evaluate):
        assert command[:3] == ["python", "-m", "eval.public.mpiifacegaze"]
        args = mpii.build_parser().parse_args(command[3:])
        assert args.root == Path("data/MPIIFaceGaze")
        assert args.cache == Path("data/cache/mpii")
    args = mpii.build_parser().parse_args(evaluate[3:])
    assert args.out == Path("eval/results/mpiifacegaze.json")
    assert "--k-per-dot" not in evaluate  # the default

    results["config"]["k_per_dot"] = 10
    block = update_readme.public_results(results).split("```bash\n", 1)[1].split("```", 1)[0]
    evaluate = shlex.split(block.splitlines()[2])
    assert mpii.build_parser().parse_args(evaluate[3:]).k_per_dot == 10


def test_public_results_render_a_results_file_from_before_the_baselines():
    # The committed file, without whatever a later run added: no environment block, no
    # baselines and no eyeos-live9.
    old = json.loads(update_readme.PUBLIC_RESULTS.read_text())
    old.pop("environment", None)
    for name in ("centre", "calibration-mean", "eyeos-live9"):
        old["variants"].pop(name, None)
    eyeos = old["variants"]["eyeos"]
    errors = _validation(old, "eyeos")
    passed = sum(p["passes_gate2"] for p in eyeos["persons"].values())
    evaluated = eyeos["aggregate"]["persons_evaluated"]

    text = update_readme.public_results(old)

    assert ("Negative result" in text) == (passed < evaluated)
    assert f"{passed} of {evaluated} persons pass" in text
    assert f"({min(errors):.1f} to {max(errors):.1f} px)" in text
    assert "baseline" not in text and "eyeos-live9" not in text
    header, rows = _table(text)
    assert list(rows) == list(old["variants"])
    for name, variant in old["variants"].items():
        repeats = name != "eyeos" and variant["persons"] == eyeos["persons"]
        assert ("(= eyeos)" in rows[name]["Variant"]) is repeats, name
    limit = old["config"]["gate2_max_validation_error_px"]
    assert f"Calibration check passed (leave-one-out < {limit:g} px)†" in header


def test_public_results_without_an_eyeos_row_compare_nothing():
    results = _public_results(("global-lopo",))

    text = update_readme.public_results(results)

    header, rows = _table(text)
    assert rows["global-lopo"][_column(header, "Better than")] == "-"
    assert "shipping calibration" not in text


def test_public_results_count_a_calibration_that_could_not_be_validated_as_failed():
    results = _public_results(noise=0.3)
    persons = results["variants"]["eyeos"]["persons"]
    persons["p01"] = dict(persons["p01"], validation_error_px=None, passes_gate2=False)
    kept = persons["p00"]["validation_error_px"]
    before = copy.deepcopy(results)

    paragraph = _paragraph(update_readme.public_results(results), "Negative result")

    assert f"averages {kept:.1f} px per person ({kept:.1f} to {kept:.1f} px)" in paragraph
    assert "1 could not be validated" in paragraph
    assert "0 of 2 persons pass" in paragraph
    assert results == before  # the generator never edits the results it reads
