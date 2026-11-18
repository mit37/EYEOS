"""Regenerate README.md's numbers from this repo's own code (STANDARDS §1.3: every
metric "produced by a script in this repo"; §3: test count "generated, not typed").

It fills the Results table (values *and* the fixture facts in its labels), the
negative-result note under it, the MPIIFaceGaze section (its table, negative result,
baseline comparison and reproducing commands, all read from the committed
``eval/results/mpiifacegaze.json``, so the check needs no dataset), the CLI transcript
and the test counts.

    python scripts/update_readme.py           # rewrite the generated regions
    python scripts/update_readme.py --check   # exit 1 with a diff if README is stale (CI)

Generated regions are delimited by HTML comments in README.md; text outside them is
hand-written and never touched.
"""

from __future__ import annotations

import argparse
import difflib
import io
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.generate_synthetic import NOISE_SIGMA  # noqa: E402
from eval.public.models import VARIANT_NAMES  # noqa: E402
from eval.public.protocol import DEFAULT_K_PER_DOT  # noqa: E402
from eval.run_eval import compute_metrics, load_raw_rows  # noqa: E402
from eyeos.calibration_session import SAMPLE_FRAMES  # noqa: E402
from eyeos.cli import main as eyeos_main  # noqa: E402
from eyeos.config import Config  # noqa: E402
from eyeos.features import extract_features  # noqa: E402
from eyeos.synthetic_format import row_to_frame  # noqa: E402

README = ROOT / "README.md"
FIXTURE = ROOT / "eval" / "fixtures" / "synthetic_session.csv"
CLI_ARGS = ["run", "--config", "config.example.yaml", "--no-log"]


def collected_tests(*paths: str) -> int:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    match = re.search(r"(\d+) tests? collected", out)
    if match is None:
        raise RuntimeError(f"could not parse pytest's collected count from:\n{out}")
    return int(match.group(1))


def cli_transcript() -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = eyeos_main(CLI_ARGS)
    if code != 0:
        raise RuntimeError(f"eyeos {' '.join(CLI_ARGS)} exited {code}")
    lines = [line for line in buf.getvalue().splitlines() if not line.startswith("[dry-run]")]
    return "```\n$ eyeos " + " ".join(CLI_ARGS) + "\n" + "\n".join(lines) + "\n```"


def _rate(count: int, total: int) -> str:
    return f"**{count / total:.1f}** ({count} / {total})"


def fixture_facts(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Facts about the fixture that the Results labels quote, read from the fixture
    itself so a generator change can't leave a stale label behind."""
    calibration = [r for r in rows if r["segment"] == "calibration"]
    dots = {(r["target_x"], r["target_y"]) for r in calibration}
    times = [float(r["timestamp_s"]) for r in rows]
    steps = sorted(b - a for a, b in zip(times, times[1:], strict=False))
    dt = steps[len(steps) // 2]
    head_turned = next(r for r in rows if r.get("away_mode") == "head_turned")
    _, landmarks = row_to_frame(head_turned)
    assert landmarks is not None
    return {
        "calibration_frames": len(calibration),
        "calibration_dots": len(dots),
        "fps": 1.0 / dt,
        "just_reading_s": sum(r["segment"] == "just_reading" for r in rows) * dt,
        "head_turned_yaw_deg": abs(extract_features(landmarks).yaw_deg),
    }


def _away_mode_label(mode: str, stats: dict[str, int], facts: dict[str, Any]) -> str:
    labels = {
        "two_faces": "two faces (Gate 1)",
        "low_confidence": "low detection confidence (Gate 1; only possible on synthetic input)",
        "no_face": "no face (Gate 1)",
        "head_turned": (
            f"head turned ~{facts['head_turned_yaw_deg']:.0f}° away (Gate 2's pose check)"
        ),
    }
    if mode == "eyes_off_screen":
        tag = " — **negative result**" if stats["false_accepts"] else ""
        return (
            f"eyes glancing past the screen edge, head still{tag}: no gate covers this by"
            " design, the cursor pins to the screen edge"
        )
    return labels.get(mode, mode)


def _looking_away_rows(
    by_mode: dict[str, dict[str, int]], facts: dict[str, Any]
) -> list[tuple[str, str]]:
    return [
        (
            f"Gate false-accept rate, looking away: {_away_mode_label(mode, stats, facts)}",
            _rate(stats["false_accepts"], stats["frames"]),
        )
        for mode, stats in by_mode.items()
    ]


def results_table(m: dict[str, Any], facts: dict[str, Any]) -> str:
    n_target = m["on_target_frame_count"]
    per_dot = facts["calibration_frames"] / facts["calibration_dots"]
    dwell_tag = " (**negative result**, see below)" if _dwell_rarely_holds(m) else ""
    rows = [
        ("Metric", "Value"),
        ("---", "---"),
        (
            "Calibration validation error (leave-one-out RMS over the"
            f" {facts['calibration_frames']} calibration frames, {per_dot:g} per dot, carrying"
            " the same injected noise as the on-target frames; Gate 2 refuses a calibration at"
            f" or above {Config().gates.calibration_max_error_px:g} px)",
            f"{m['calibration_validation_error_px']:.2f} px",
        ),
        (
            f"Mean pixel error, **raw** calibration output, {n_target} noisy on-target frames",
            f"{m['mean_pixel_error_raw_px']:.1f} px",
        ),
        (
            "Mean pixel error, **filtered** (One Euro output the gates see), same"
            f" {m['on_target_frame_count_filtered']} frames",
            f"{m['mean_pixel_error_filtered_px']:.1f} px",
        ),
        (
            "Mean angular error",
            "**not measured**: the fixture has no real screen size or viewing distance",
        ),
        (
            f"Steady-state jitter, **raw** (px per frame at {facts['fps']:.0f} fps, same frame"
            " pairs)",
            f"{m['steady_state_jitter_raw_px']:.1f} px",
        ),
        (
            "Steady-state jitter, **filtered**",
            f"{m['steady_state_jitter_filtered_px']:.1f} px",
        ),
        (
            f"Gate 3 dwell satisfied on noisy on-target frames{dwell_tag}",
            f"{m['on_target_dwell_satisfied_frames']} / {n_target}",
        ),
        *_looking_away_rows(m["looking_away_by_mode"], facts),
        (
            f"Click false-trigger rate, {facts['just_reading_s']:.0f} s synthetic steady-gaze hold"
            " with no blinks, saccades"
            " or confirmation input — 0 by construction; checks only that an unconfirmed dwell"
            f" never clicks (dwell held on {m['just_reading_dwell_satisfied_frames']} /"
            f" {m['just_reading_frame_count']} frames)",
            _rate(m["just_reading_false_click_count"], m["just_reading_frame_count"]),
        ),
        (
            "Click false-trigger rate on a real 5-minute reading segment (PRD §5)",
            "**not measured**: needs a recorded session",
        ),
    ]
    return "\n".join(f"| {a} | {b} |" for a, b in rows)


def _dwell_rarely_holds(m: dict[str, Any]) -> bool:
    """Clicking isn't usable if the dwell holds on fewer than half the on-target frames."""
    return bool(m["on_target_dwell_satisfied_frames"] < m["on_target_frame_count"] / 2)


def results_notes(m: dict[str, Any]) -> str:
    """The note under the Results table, built from the same numbers it describes."""
    gates = Config().gates
    jitter = m["steady_state_jitter_filtered_px"]
    compared = "larger than" if jitter > gates.dwell_jitter_px else "within"
    held, total = m["on_target_dwell_satisfied_frames"], m["on_target_frame_count"]
    sigma = f"σ = {NOISE_SIGMA:g} in the generator's offset units"
    if 0 < held and _dwell_rarely_holds(m):
        return (
            f"**Negative result: the dwell rarely holds on the noisy on-target gaze** ({held} of"
            f" {total} frames). At the fixture's injected noise ({sigma}), the filtered"
            f" frame-to-frame jitter ({jitter:.1f} px) is"
            f" {compared} Gate 3's {gates.dwell_jitter_px:g} px dwell radius, so an intended"
            " click is unreliable. Whether real webcam noise is above or below this level is"
            " **not measured**."
        )
    if held == 0:
        return (
            "**Negative result: no intended click could fire on the noisy on-target gaze.** At"
            f" the fixture's injected noise ({sigma}), the"
            f" filtered frame-to-frame jitter ({jitter:.1f} px) is {compared} Gate 3's"
            f" {gates.dwell_jitter_px:g} px dwell radius. So the"
            f" {gates.dwell_time_s:g} s dwell never holds on target, and no click of any kind"
            ' could fire there. The "0 false clicks" row therefore has no matching hit rate.'
            " Whether real webcam noise is above or below this level is **not measured**."
            " Either the noise would need to be lower, or `dwell_jitter_px` raised, before"
            " clicking is usable. That trade-off needs a real recording to settle."
        )
    return (
        f"At the fixture's injected noise ({sigma}), Gate 3's"
        f" {gates.dwell_time_s:g} s dwell within {gates.dwell_jitter_px:g} px held on"
        f" {m['on_target_dwell_satisfied_frames']} of {m['on_target_frame_count']} noisy"
        " on-target frames."
    )


PUBLIC_RESULTS_PATH = "eval/results/mpiifacegaze.json"  # as README text and commands name it
PUBLIC_RESULTS = ROOT / PUBLIC_RESULTS_PATH

PUBLIC_NOT_MEASURED = (
    "**Public dataset: not measured yet.** `eval/public/` scores the same calibration on "
    "MPIIFaceGaze, 15 people's real laptop-webcam photos with known on-screen targets "
    f"(`docs/PUBLIC_DATASETS.md`). No results file (`{PUBLIC_RESULTS_PATH}`) has "
    "been committed yet, so no MPIIFaceGaze number exists."
)

SHIPPING = "eyeos"
LIVE_STYLE = "eyeos-live9"  # the shipping calibration fitted as a live session fits it
BASELINES = ("centre", "calibration-mean")  # constant predictors that ignore the image
CALIBRATION_DOTS = 9  # eyeos.calibration.target_points: the 3 x 3 grid
MPII_PERSONS = tuple(f"p{i:02d}" for i in range(15))  # MPIIFaceGaze's participants
DATASET_ROOT = "datasets/MPIIFaceGaze"  # where docs/PUBLIC_DATASETS.md puts the dataset
DATASET_CACHE = "datasets/cache/mpiifacegaze"


def _num(value: float | None, spec: str, unit: str = "") -> str:
    return "-" if value is None else f"{value:{spec}}{unit}"


def _evaluated(variant: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The per-person records of the persons a variant evaluated (a skipped person's
    record has no errors)."""
    return [(p, r) for p, r in variant["persons"].items() if "pixel_error_px" in r]


def _person_means(variant: dict[str, Any]) -> dict[str, float]:
    return {p: r["pixel_error_px"]["mean"] for p, r in _evaluated(variant)}


def _median_and_max(variant: dict[str, Any]) -> str:
    """Median / max of the per-person mean errors: with 15 persons a p90 is the 2nd-3rd
    worst person interpolated, so the worst person is reported as such."""
    means = list(_person_means(variant).values())
    if not means:
        return "-"
    return f"{statistics.median(means):.1f} / {max(means):.1f} px"


def _better_than(variant: dict[str, Any], reference: dict[str, Any]) -> str:
    """For how many persons (of those both evaluated) the variant's mean error is lower."""
    ours, theirs = _person_means(variant), _person_means(reference)
    common = [p for p in ours if p in theirs]
    if not common:
        return "-"
    return f"{sum(ours[p] < theirs[p] for p in common)} / {len(common)}"


def _repeats(variant: dict[str, Any], reference: dict[str, Any]) -> bool:
    """Are the per-person results identical, number for number?"""
    return bool(reference["persons"]) and variant["persons"] == reference["persons"]


def _validation(variant: dict[str, Any]) -> tuple[str, int, int, int]:
    """How a variant's per-person validation errors read in a sentence, and (passed,
    evaluated, not validated) counts."""
    records = [r for _, r in _evaluated(variant)]
    errors = [r["validation_error_px"] for r in records if r.get("validation_error_px") is not None]
    passed = sum(bool(r.get("passes_gate2")) for r in records)
    if errors:
        phrase = (
            f"averages {statistics.fmean(errors):.1f} px per person "
            f"({min(errors):.1f} to {max(errors):.1f} px)"
        )
    else:
        phrase = "could not be computed for anyone"
    return phrase, passed, len(records), len(records) - len(errors)


def _pass_count(passed: int, evaluated: int, unvalidated: int, persons: str = "") -> str:
    text = f"{passed} of {evaluated}{persons} pass"
    return text + (f" ({unvalidated} could not be validated)" if unvalidated else "")


def _photos(results: dict[str, Any]) -> int | None:
    k = results["config"].get("k_per_dot")
    return None if k is None else CALIBRATION_DOTS * int(k)


def _validation_result(results: dict[str, Any]) -> str | None:
    """The shipping calibration's validation error against Gate 2's limit, and, when the
    file has it, the live-style statistic: a negative result unless everyone passes."""
    variants = results["variants"]
    if SHIPPING not in variants:
        return None
    limit = results["config"]["gate2_max_validation_error_px"]
    photos = _photos(results)
    over = f"its {photos} calibration photos" if photos else "its calibration photos"
    phrase, passed, evaluated, unvalidated = _validation(variants[SHIPPING])
    text = (
        f"The shipping calibration's leave-one-out validation error over {over} {phrase}, "
        f"against Gate 2's {limit:g} px limit: "
        f"{_pass_count(passed, evaluated, unvalidated, ' persons')}."
    )
    negative = passed < evaluated
    live = variants.get(LIVE_STYLE)
    if live is None:
        text += " Live Gate 2 judges a different statistic (†), which this file does not report."
    else:
        phrase, passed, evaluated, unvalidated = _validation(live)
        text += (
            f" Fitted the way a live session fits it (`{LIVE_STYLE}`), its leave-one-out over "
            f"{CALIBRATION_DOTS} samples, one per dot, is the same statistic as live Gate 2's, "
            f"but not a live result: {_not_live(results, live)}. On these photos it "
            f"{phrase}, and {_pass_count(passed, evaluated, unvalidated)}."
        )
        negative = negative or passed < evaluated
    return ("**Negative result.** " if negative else "") + text


def _not_live(results: dict[str, Any], live: dict[str, Any]) -> str:
    """How eyeos-live9's samples differ from a live session's (eyeos.calibration_session):
    a dot's photos come from separate moments, and their true targets are not the dot."""
    k = results["config"].get("k_per_dot")
    photos = f"a dot's {k} photos" if k is not None else "a dot's K photos"
    text = (
        f"each sample averages {photos}, taken at different moments (often on different "
        f"days) rather than up to {SAMPLE_FRAMES} frames of one look, and is placed at the dot"
    )
    distances = [
        r["calibration_mean_distance_px"]
        for _, r in _evaluated(live)
        if r.get("calibration_mean_distance_px") is not None
    ]
    if distances:
        text += (
            " although their true targets lie on average "
            f"{statistics.fmean(distances):.1f} px from it"
        )
    return text


def _centre_comparison(results: dict[str, Any]) -> str | None:
    """How far the shipping calibration's mean error is below the centre baseline's."""
    variants = results["variants"]
    if SHIPPING not in variants or "centre" not in variants:
        return None
    ours, centre = (
        (variants[name]["aggregate"].get("pixel_error_px") or {}).get("mean_of_person_means")
        for name in (SHIPPING, "centre")
    )
    if ours is None or not centre:
        return None
    change = 100.0 * (centre - ours) / centre
    direction = f"{change:.1f}% below" if ours < centre else f"{-change:.1f}% above"
    text = (
        f"The shipping calibration's mean error, {ours:.1f} px, is {direction} the `centre` "
        f"baseline's {centre:.1f} px (a constant guess at the screen centre that ignores the "
        "image)."
    )
    return text if ours < centre else "**Negative result.** " + text


def _check_footnote(results: dict[str, Any]) -> str:
    limit = results["config"]["gate2_max_validation_error_px"]
    photos = _photos(results)
    k = results["config"].get("k_per_dot")
    over = (
        f"its {photos} calibration photos ({CALIBRATION_DOTS} dots × K = {k}; "
        f"{CALIBRATION_DOTS} × K for `per-dot-K`)"
        if photos
        else f"its {CALIBRATION_DOTS} × K calibration photos"
    )
    live = LIVE_STYLE in results["variants"]
    which = f"For every variant except `{LIVE_STYLE}` it is" if live else "For every variant it is"
    return (
        "† Each variant's leave-one-out error, and the persons for whom it is below Gate 2's "
        f"{limit:g} px limit. {which} over {over}, each at its own true target: not the "
        f"statistic live Gate 2 judges, which is over {CALIBRATION_DOTS} samples, one per dot, "
        "each that dot's frames averaged and placed at the dot. "
        + (
            f"`{LIVE_STYLE}` computes that statistic on these photos (see above)."
            if live
            else "This file does not report that one."
        )
    )


def _repo_relative(path: str) -> bool:
    """Is this a relative POSIX path inside the repo, safe to print in the README (no
    absolute, home, Windows or ``..`` path that names one machine's folders)?"""
    parts = PurePosixPath(path).parts
    return (
        bool(parts)
        and not PurePosixPath(path).is_absolute()
        and "\\" not in path
        and ":" not in path
        and not path.startswith("~")
        and ".." not in parts
    )


def _paths(config: dict[str, Any]) -> tuple[str, str, bool]:
    """(root, cache, the run's own?) for the reproducing commands: the run's paths when
    they are repo-relative and root is the downloader's ``<dest>/MPIIFaceGaze``, else the
    documented ones."""
    root, cache = str(config.get("root", DATASET_ROOT)), str(config.get("cache", DATASET_CACHE))
    if (
        _repo_relative(root)
        and _repo_relative(cache)
        and PurePosixPath(root).name == "MPIIFaceGaze"
    ):
        return str(PurePosixPath(root)), str(PurePosixPath(cache)), True
    return DATASET_ROOT, DATASET_CACHE, False


def _subset(config: dict[str, Any], key: str, everything: Sequence[str]) -> str:
    """``--persons``/``--variants`` with the run's list, when it is not the default."""
    chosen = config.get(key)
    if chosen is None or list(chosen) == list(everything):
        return ""
    return f" --{key} {shlex.quote(','.join(chosen))}"


def _reproduce(results: dict[str, Any]) -> str:
    """The commands that regenerate the results file, with its persons, variants and K,
    and its root and cache when they are repo-relative (else the documented ones)."""
    config = results["config"]
    root, cache, own = _paths(config)
    where = f"--root {shlex.quote(root)} --cache {shlex.quote(cache)}"
    persons = _subset(config, "persons", MPII_PERSONS)
    variants = _subset(config, "variants", VARIANT_NAMES)
    k = config.get("k_per_dot")
    k_flag = "" if k is None or k == DEFAULT_K_PER_DOT else f" --k-per-dot {k}"
    note = (
        ""
        if own
        else "; the run itself used other local paths, so these commands use the documented ones"
    )
    return "\n".join(
        [
            f"Reproduce it with the dataset (`docs/PUBLIC_DATASETS.md`{note}):",
            "",
            "```bash",
            "python scripts/download_mpiifacegaze.py --accept-license "
            f"--dest {shlex.quote(str(PurePosixPath(root).parent))}",
            f"python -m eval.public.mpiifacegaze extract {where}{persons}",
            f"python -m eval.public.mpiifacegaze evaluate {where}{persons}{variants} "
            f"--out {PUBLIC_RESULTS_PATH}{k_flag}",
            "```",
        ]
    )


def public_results(results: dict[str, Any] | None) -> str:
    """The MPIIFaceGaze section, from the committed output of
    ``python -m eval.public.mpiifacegaze evaluate``; "not measured" without one. Works
    with a file from before the baselines, eyeos-live9 and the environment record."""
    if results is None:
        return PUBLIC_NOT_MEASURED
    detection = results["detection"]["aggregate"]
    commit = (results.get("git_commit") or "unknown")[:7]
    dirty = " (with uncommitted changes)" if results.get("git_dirty") else ""
    date = str(results.get("generated", ""))[:10] or "an unknown date"
    rate = detection.get("detection_rate_mean")
    people = len(results["detection"]["persons"])
    limit = results["config"]["gate2_max_validation_error_px"]
    variants = results["variants"]
    shipping = variants.get(SHIPPING)
    paragraphs = [
        f"**Public dataset: MPIIFaceGaze**, laptop-webcam photos of {people} people with "
        "known on-screen targets (`docs/PUBLIC_DATASETS.md`). Generated from the committed "
        f"`{PUBLIC_RESULTS_PATH}`, written by the evaluation at commit `{commit}`"
        f"{dirty} on {date}. MediaPipe found one usable face in {detection['valid']} of "
        f"{detection['images']} images (mean detection rate per person "
        f"{_num(None if rate is None else rate * 100, '.1f', '%')}). Each person is "
        "calibrated on the images nearest EyeOS's 9 calibration points and tested on all "
        "their other images. Errors are per person, then summarised across persons."
    ]
    paragraphs += [p for p in (_validation_result(results), _centre_comparison(results)) if p]

    lines = [
        "| Variant | Persons | Mean error | Median / max of person means | Share of screen "
        "diagonal | Mean angular error | Better than `eyeos` for | Mean leave-one-out error† "
        f"| Calibration check passed (leave-one-out < {limit:g} px)† |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    order = [n for n in BASELINES if n in variants] + [n for n in variants if n not in BASELINES]
    skipped: list[str] = []
    repeated = False
    for name in order:
        variant = variants[name]
        agg = variant["aggregate"]
        px = agg.get("pixel_error_px") or {}
        deg = agg.get("angular_error_deg")
        evaluated = agg["persons_evaluated"]
        if deg is None:
            angle = "not measured (no trusted screen model)"
        else:
            angle = _num(deg.get("mean_of_person_means"), ".2f", "°")
            if agg.get("angular_error_persons", evaluated) < evaluated:
                angle += f" ({agg['angular_error_persons']} persons)"
        label = f"`{name}`"
        if name == SHIPPING:
            label += " (shipping)"
        elif name in BASELINES:
            label += " (baseline)"
        elif name == LIVE_STYLE:
            label += " (live-style)"
        if shipping is not None and name != SHIPPING and _repeats(variant, shipping):
            label += " (= eyeos)"
            repeated = True
        better = "-" if shipping is None or name == SHIPPING else _better_than(variant, shipping)
        lines.append(
            f"| {label} | {evaluated} | {_num(px.get('mean_of_person_means'), '.1f', ' px')} "
            f"| {_median_and_max(variant)} "
            f"| {_num(agg.get('pixel_error_pct_diagonal_mean_of_person_means'), '.2f', '%')} "
            f"| {angle} | {better} | {_num(agg.get('validation_error_px_mean'), '.1f', ' px')} "
            f"| {agg['gate2_passed']} / {evaluated} |"
        )
        if agg.get("persons_skipped"):
            skipped.append(f"`{name}`: {', '.join(agg['persons_skipped'])}")
    paragraphs.append("\n".join(lines))
    if skipped:
        paragraphs.append("Persons skipped (too few usable images): " + "; ".join(skipped) + ".")
    paragraphs.append(_check_footnote(results))
    if repeated:
        paragraphs.append(
            "(= eyeos): per-person results identical to `eyeos`'s, number for number, so the "
            "same result listed again, not an independent confirmation."
        )
    paragraphs.append(_reproduce(results))
    return "\n\n".join(paragraphs)


def load_public_results(path: Path = PUBLIC_RESULTS) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def region(name: str, body: str, src: str) -> str:
    """Replace the text between ``<!-- name:start -->`` and ``<!-- name:end -->``."""
    pattern = re.compile(
        rf"(<!-- {name}:start -->\n)(?:.*?\n)?(<!-- {name}:end -->)", flags=re.DOTALL
    )
    if not pattern.search(src):
        raise RuntimeError(f"README.md is missing the {name} markers")
    return pattern.sub(lambda mt: mt.group(1) + body + "\n" + mt.group(2), src)


def inline(name: str, value: int, src: str) -> str:
    """Replace the number in ``<!-- name -->N<!-- /name -->``, which must occur once."""
    new, n = re.subn(
        rf"<!-- {name} -->\d*<!-- /{name} -->", f"<!-- {name} -->{value}<!-- /{name} -->", src
    )
    if n != 1:
        raise RuntimeError(f"README.md must contain the {name} marker exactly once (found {n})")
    return new


def config_value(name: str) -> str:
    """A configuration default quoted in README prose, by dotted name."""
    from eyeos.pipeline import _BLINK_GESTURE_GRACE_S, _MIN_REOPEN_S

    if name == "pipeline.blink_gesture_grace_s":
        return f"{_BLINK_GESTURE_GRACE_S:g}"
    if name == "pipeline.min_reopen_s":
        return f"{_MIN_REOPEN_S:g}"
    section, _, field = name.partition(".")
    value = getattr(getattr(Config(), section, None), field, None)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"README.md quotes an unknown config value: {name}")
    return f"{value:g}"


def config_values(src: str) -> str:
    """Fill every ``<!-- cfg:section.field -->N<!-- /cfg -->`` from the defaults."""
    return re.sub(
        r"<!-- cfg:([\w.]+) -->[^<]*<!-- /cfg -->",
        lambda m: f"<!-- cfg:{m.group(1)} -->{config_value(m.group(1))}<!-- /cfg -->",
        src,
    )


def render(text: str) -> str:
    metrics = compute_metrics(FIXTURE)
    facts = fixture_facts(load_raw_rows(FIXTURE))
    total = collected_tests()
    gates = collected_tests("tests/test_gates.py")

    text = region("cli-demo", cli_transcript(), text)
    text = region("results", results_table(metrics, facts), text)
    text = region("results-notes", results_notes(metrics), text)
    text = region("public-results", public_results(load_public_results()), text)
    text = inline("test-count", total, text)
    text = inline("gate-test-count", gates, text)
    text = config_values(text)
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if README.md is stale")
    args = parser.parse_args(argv)

    os.chdir(ROOT)
    current = README.read_text()
    updated = render(current)
    if args.check:
        if updated != current:
            sys.stdout.writelines(
                difflib.unified_diff(
                    current.splitlines(keepends=True),
                    updated.splitlines(keepends=True),
                    "README.md",
                    "README.md (regenerated)",
                )
            )
            print("\nREADME.md is stale: run `python scripts/update_readme.py`", file=sys.stderr)
            return 1
        print("README.md is up to date")
        return 0
    README.write_text(updated)
    print("README.md regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
