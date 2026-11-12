"""Regenerate README.md's numbers from this repo's own code (STANDARDS §1.3: every
metric "produced by a script in this repo"; §3: test count "generated, not typed").

It fills the Results table (values *and* the fixture facts in its labels), the
negative-result note under it, the CLI transcript and the test counts.

    python scripts/update_readme.py           # rewrite the generated regions
    python scripts/update_readme.py --check   # exit 1 with a diff if README is stale (CI)

Generated regions are delimited by HTML comments in README.md; text outside them is
hand-written and never touched.
"""

from __future__ import annotations

import argparse
import difflib
import io
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.generate_synthetic import NOISE_SIGMA  # noqa: E402
from eval.run_eval import compute_metrics, load_raw_rows  # noqa: E402
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
    from eyeos.pipeline import _BLINK_GESTURE_GRACE_S

    if name == "pipeline.blink_gesture_grace_s":
        return f"{_BLINK_GESTURE_GRACE_S:g}"
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
