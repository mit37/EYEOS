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

# Leading frames of each on-target block kept out of jitter pairs: frame 0 is the target
# switch, where the filter output is still mid-step toward the new point.
_JITTER_WARMUP_FRAMES = 1


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


def compute_metrics(
    path: Path, jitter_warmup_frames: int = _JITTER_WARMUP_FRAMES
) -> dict[str, Any]:
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
    on_target_dwell_frames = 0
    just_reading_dwell_frames = 0

    for row, raw_pred, status in zip(rows, raw_preds, result.statuses, strict=True):
        segment = row.get("segment", "")
        target_key = (row["target_x"], row["target_y"]) if segment == "on_target" else None
        filtered = status.smoothed_gaze_px

        if segment == "on_target" and raw_pred is not None:
            target = (float(row["target_x"]), float(row["target_y"]))
            on_target_errors_raw.append(_dist(raw_pred, target))
            on_target_dwell_frames += status.gate3_stability
            if filtered is not None:
                on_target_errors_filtered.append(_dist(filtered, target))

        if target_key is not None and target_key == prev_target_key:
            block_frame_index += 1
        else:
            block_frame_index = 0
            prev_raw = prev_filtered = None
        # The earlier frame of the pair (block_frame_index - 1) must be past the warm-up.
        if (
            block_frame_index > jitter_warmup_frames
            and raw_pred is not None
            and prev_raw is not None
            and filtered is not None
            and prev_filtered is not None
        ):
            jitter_raw.append(_dist(raw_pred, prev_raw))
            jitter_filtered.append(_dist(filtered, prev_filtered))
        prev_raw, prev_filtered = raw_pred, filtered
        prev_target_key = target_key

        if segment == "looking_away":
            looking_away_total += 1
            if status.move_allowed:
                looking_away_false_accepts += 1

        if segment == "just_reading":
            just_reading_total += 1
            just_reading_dwell_frames += status.gate3_stability
            if status.click_fired:
                just_reading_false_clicks += 1

    return {
        "frame_count": len(rows),
        "calibration_validation_error_px": calibration.validation_error_px,
        "on_target_frame_count": len(on_target_errors_raw),
        "on_target_frame_count_filtered": len(on_target_errors_filtered),
        "mean_pixel_error_raw_px": _mean(on_target_errors_raw),
        "mean_pixel_error_filtered_px": _mean(on_target_errors_filtered),
        "mean_angular_error_deg": None,
        "steady_state_jitter_raw_px": _mean(jitter_raw),
        "steady_state_jitter_filtered_px": _mean(jitter_filtered),
        "looking_away_frame_count": looking_away_total,
        "gate_false_accept_rate": (
            looking_away_false_accepts / looking_away_total if looking_away_total else None
        ),
        "on_target_dwell_satisfied_frames": on_target_dwell_frames,
        "just_reading_dwell_satisfied_frames": just_reading_dwell_frames,
        "just_reading_frame_count": just_reading_total,
        "just_reading_false_click_count": just_reading_false_clicks,
        "click_false_trigger_rate": (
            just_reading_false_clicks / just_reading_total if just_reading_total else None
        ),
        "notes": {
            "dwell_reachability": (
                "how often Gate 3 (dwell) was satisfied. A false-click rate means little "
                "without this: if dwell never holds on the noisy on-target gaze, no "
                "intended click could fire either. At this fixture's injected noise the "
                "filtered frame-to-frame jitter exceeds gates.dwell_jitter_px, so dwell "
                "never holds on target; the noise level itself is an assumption, not a "
                "measurement of real webcam noise."
            ),
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
            "mean_pixel_error": (
                "raw is the calibration output; filtered is the One Euro output the gates "
                "see (GateStatus.smoothed_gaze_px). Both are averaged over the same "
                "on-target frames, including frames where the gates held the cursor still "
                "(e.g. Gate 4's speed limit on each target switch)."
            ),
            "steady_state_jitter": (
                "mean frame-to-frame displacement in px per frame (the committed fixture "
                "runs at 10 fps), raw vs One Euro output, over the same pairs of "
                "consecutive frames within one on-target block. Pairs that touch the "
                f"first {jitter_warmup_frames} frame(s) of a block are left out: frame 0 "
                "is the target switch, where the filter is still mid-step toward the new "
                "point. An earlier version of this eval reported 88.6px filtered vs 42.5px "
                "raw jitter and blamed filter settling. That was an eval bookkeeping bug, "
                "now fixed, not a property of the filter: Gate 4 suppressed each block's "
                "switch frame, so the block's second frame was compared with the previous "
                "block's last cursor position."
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
