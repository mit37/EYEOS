"""Regenerate every number in README.md from this repo's own code (STANDARDS §1.3: every
metric "produced by a script in this repo"; §3: test count "generated, not typed").

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

from eval.run_eval import compute_metrics  # noqa: E402
from eyeos.cli import main as eyeos_main  # noqa: E402

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


_AWAY_MODE_LABELS = {
    "two_faces": "two faces (Gate 1)",
    "low_confidence": "low detection confidence (Gate 1; only possible on synthetic input)",
    "no_face": "no face (Gate 1)",
    "head_turned": "head turned ~21° away (Gate 2's pose check)",
    "eyes_off_screen": (
        "eyes glancing past the screen edge, head still — **negative result**: no gate"
        " covers this by design, the cursor pins to the screen edge"
    ),
}


def _looking_away_rows(by_mode: dict[str, dict[str, int]]) -> list[tuple[str, str]]:
    return [
        (
            f"Gate false-accept rate, looking away: {_AWAY_MODE_LABELS.get(mode, mode)}",
            _rate(stats["false_accepts"], stats["frames"]),
        )
        for mode, stats in by_mode.items()
    ]


def results_table(m: dict[str, Any]) -> str:
    n_target = m["on_target_frame_count"]
    rows = [
        ("Metric", "Value"),
        ("---", "---"),
        (
            "Calibration validation error (leave-one-out RMS over the 45 calibration frames,"
            " 5 per dot, carrying the same injected noise as the on-target frames; Gate 2"
            " refuses a calibration at or above 60 px)",
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
            "Steady-state jitter, **raw** (px per frame at 10 fps, same frame pairs)",
            f"{m['steady_state_jitter_raw_px']:.1f} px",
        ),
        (
            "Steady-state jitter, **filtered**",
            f"{m['steady_state_jitter_filtered_px']:.1f} px",
        ),
        (
            "Gate 3 dwell satisfied on noisy on-target frames (**negative result**, see below)",
            f"{m['on_target_dwell_satisfied_frames']} / {n_target}",
        ),
        *_looking_away_rows(m["looking_away_by_mode"]),
        (
            "Click false-trigger rate, 30 s synthetic steady-gaze hold with no blinks, saccades"
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


def render(text: str) -> str:
    metrics = compute_metrics(FIXTURE)
    total = collected_tests()
    gates = collected_tests("tests/test_gates.py")

    def region(name: str, body: str, src: str) -> str:
        pattern = re.compile(
            rf"(<!-- {name}:start -->\n)(?:.*?\n)?(<!-- {name}:end -->)", flags=re.DOTALL
        )
        if not pattern.search(src):
            raise RuntimeError(f"README.md is missing the {name} markers")
        return pattern.sub(lambda mt: mt.group(1) + body + "\n" + mt.group(2), src)

    def inline(name: str, value: int, src: str) -> str:
        return re.sub(
            rf"<!-- {name} -->\d*<!-- /{name} -->", f"<!-- {name} -->{value}<!-- /{name} -->", src
        )

    text = region("cli-demo", cli_transcript(), text)
    text = region("results", results_table(metrics), text)
    text = inline("test-count", total, text)
    text = inline("gate-test-count", gates, text)
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
