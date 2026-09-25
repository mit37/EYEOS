"""Computes the eval metrics from PRD §5, against the committed synthetic session
(``eval/fixtures/synthetic_session.csv``, produced by ``eval/generate_synthetic.py``)
or any other session CSV in the same format. Reuses the real Calibration/Pipeline/
gates code — nothing here re-implements the pipeline's math, so these numbers reflect
exactly what the shipped code does, not a separate "eval model" of it.

    python -m eval.run_eval --fixtures eval/fixtures --out eval/results.json

Per STANDARDS.md §1.3: every number below comes from this script, running against
this repo's code, and is reproducible with the command above. Numbers this synthetic
fixture cannot support (see "notes" in the output) are reported as not measured, not
guessed or omitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from eyeos.calibration import Calibration
from eyeos.config import Config
from eyeos.features import extract_features
from eyeos.input_sources.synthetic import SyntheticInputSource
from eyeos.output.dry_run import DryRunOutputAdapter
from eyeos.pipeline import Pipeline
from eyeos.synthetic_format import row_to_frame

ScreenPoint = tuple[float, float]

# Frames to skip at the start of each on-target dwell block before counting jitter,
# so the filter's settling lag after a target switch isn't counted as jitter.
_JITTER_WARMUP_FRAMES = 5


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _dist(a: ScreenPoint, b: ScreenPoint) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def load_raw_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def build_calibration(rows: list[dict[str, str]], screen_w: float, screen_h: float) -> Calibration:
    cal = Calibration(screen_w, screen_h)
    for row in rows:
        if row.get("segment") != "calibration":
            continue
        _, landmarks = row_to_frame(row)
        if landmarks is None:
            continue
        features = extract_features(landmarks)
        target = (float(row["target_x"]), float(row["target_y"]))
        cal.add_sample(target, features)
    cal.fit()
    return cal


def raw_predictions(
    rows: list[dict[str, str]], calibration: Calibration
) -> list[ScreenPoint | None]:
    preds: list[ScreenPoint | None] = []
    for row in rows:
        _, landmarks = row_to_frame(row)
        if landmarks is None:
            preds.append(None)
            continue
        preds.append(calibration.predict(extract_features(landmarks)))
    return preds


def compute_metrics(path: Path) -> dict[str, Any]:
    rows = load_raw_rows(path)
    config = Config()  # screen defaults (1920x1080) match the fixture's ground truth
    calibration = build_calibration(rows, config.screen.width_px, config.screen.height_px)
    raw_preds = raw_predictions(rows, calibration)

    output = DryRunOutputAdapter(echo=None)
    pipeline = Pipeline(config, SyntheticInputSource(path), output, calibration=calibration)
    result = pipeline.run()
    if len(result.statuses) != len(rows):
        raise RuntimeError("pipeline frame count does not match the session CSV row count")

    on_target_errors_raw: list[float] = []
    on_target_errors_filtered: list[float] = []
    jitter_raw: list[float] = []
    jitter_filtered: list[float] = []
    prev_raw: ScreenPoint | None = None
    prev_filtered: ScreenPoint | None = None
    prev_target_key: tuple[str, str] | None = None
    block_frame_index = 0

    looking_away_total = 0
    looking_away_false_accepts = 0
    just_reading_total = 0
    just_reading_false_clicks = 0

    for row, raw_pred, status in zip(rows, raw_preds, result.statuses, strict=True):
        segment = row.get("segment", "")
        target_key = (row["target_x"], row["target_y"])

        if segment == "on_target" and raw_pred is not None:
            target = (float(row["target_x"]), float(row["target_y"]))
            on_target_errors_raw.append(_dist(raw_pred, target))
            if status.cursor_target_px is not None:
                on_target_errors_filtered.append(_dist(status.cursor_target_px, target))

            same_block = target_key == prev_target_key
            block_frame_index = block_frame_index + 1 if same_block else 0
            # Skip each dwell block's first few frames for the jitter metric: right
            # after a target switch the filter is still lagging toward the new point
            # (settling, not jitter), which would otherwise swamp the steady-state
            # jitter number — see the README's Results section for what this looks
            # like if you don't skip it.
            past_warmup = same_block and block_frame_index >= _JITTER_WARMUP_FRAMES
            if past_warmup and prev_raw is not None:
                jitter_raw.append(_dist(raw_pred, prev_raw))
            if past_warmup and prev_filtered is not None and status.cursor_target_px is not None:
                jitter_filtered.append(_dist(status.cursor_target_px, prev_filtered))
            prev_raw = raw_pred
            if status.cursor_target_px is not None:
                prev_filtered = status.cursor_target_px
            prev_target_key = target_key
        else:
            prev_raw = prev_filtered = None
            prev_target_key = None
            block_frame_index = 0

        if segment == "looking_away":
            looking_away_total += 1
            if status.move_allowed:
                looking_away_false_accepts += 1

        if segment == "just_reading":
            just_reading_total += 1
            if status.click_fired:
                just_reading_false_clicks += 1

    return {
        "frame_count": len(rows),
        "calibration_validation_error_px": calibration.validation_error_px,
        "on_target_frame_count": len(on_target_errors_raw),
        "mean_pixel_error_raw_px": _mean(on_target_errors_raw),
        "mean_pixel_error_filtered_px": _mean(on_target_errors_filtered),
        "mean_angular_error_deg": None,
        "steady_state_jitter_raw_px": _mean(jitter_raw),
        "steady_state_jitter_filtered_px": _mean(jitter_filtered),
        "looking_away_frame_count": looking_away_total,
        "gate_false_accept_rate": (
            looking_away_false_accepts / looking_away_total if looking_away_total else None
        ),
        "just_reading_frame_count": just_reading_total,
        "just_reading_false_click_count": just_reading_false_clicks,
        "click_false_trigger_rate": (
            just_reading_false_clicks / just_reading_total if just_reading_total else None
        ),
        "notes": {
            "mean_angular_error_deg": (
                "not measured: this synthetic fixture only has normalized image-plane "
                "landmark offsets, no real screen size or viewing distance, so there is "
                "nothing to convert pixel error into a visual angle with. A real "
                "recording (with a measured screen + distance) would give this."
            ),
            "just_reading_duration": (
                "PRD asks for a 5-minute just-reading segment; this synthetic fixture "
                "uses a 30s stand-in (300 frames at 10fps) to keep CI fast — scaled "
                "down, not the real target length."
            ),
            "steady_state_jitter": (
                f"the first {_JITTER_WARMUP_FRAMES} frames of every on-target dwell "
                "block are excluded from the jitter numbers: right after a target "
                "switch the One Euro filter is still settling toward the new point, "
                "and that lag isn't jitter. Including those frames makes the filtered "
                "number look *worse* than the raw one (a real, reported effect — see "
                "the README) even though the filter clearly reduces steady-state "
                "jitter once it has settled."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures", default="eval/fixtures", help="directory holding synthetic_session.csv"
    )
    parser.add_argument(
        "--session", default=None, help="explicit session CSV path (overrides --fixtures)"
    )
    parser.add_argument("--out", default="eval/results.json")
    args = parser.parse_args(argv)

    session_path = (
        Path(args.session) if args.session else Path(args.fixtures) / "synthetic_session.csv"
    )
    metrics = compute_metrics(session_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
