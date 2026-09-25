"""Generates the committed synthetic landmark sequence CI evaluates against (PRD §5:
"The cloud CI runs the eval on a small committed synthetic landmark sequence
(generated), so the pipeline is tested without video.").

Fully synthetic and deterministic (seeded RNG) — no recorded video, no third-party
data, reproducible with one command:

    python -m eval.generate_synthetic

The session simulates, in order:

1. ``calibration``  -- the standard 9-point grid, clean (no noise): the calibration
   a real session would collect.
2. ``on_target``    -- 6 held gaze targets, WITH per-frame noise on the (synthetic)
   iris offset, so run_eval.py has something for the One Euro filter to smooth and a
   noisy-vs-calibrated pixel error to report.
3. ``looking_away`` -- three failure modes in turn (two faces, low confidence, no
   face at all) that Gate 1 must reject, for the gate false-accept-rate metric.
4. ``just_reading`` -- a long steady, in-calibration, never-confirmed gaze hold, for
   the click false-trigger-rate metric (target: 0 clicks).

The (target_x, target_y, segment) columns beyond the normal synthetic_format schema
are eval-only ground truth; eyeos.input_sources.synthetic.SyntheticInputSource (and
therefore the CLI/pipeline) ignores unknown columns, so this same file also works as
an ordinary session recording — it's what config.example.yaml points to by default.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from eyeos.calibration import target_points
from eyeos.landmarks import FaceLandmarks, Point3D
from eyeos.synthetic_format import CSV_FIELDNAMES, landmarks_to_row

SCREEN_W, SCREEN_H = 1920.0, 1080.0
DT = 0.1  # 10 fps

FIELDNAMES = [*CSV_FIELDNAMES, "target_x", "target_y", "segment"]

# A frontal, centered eye/face geometry, held fixed for every synthetic frame — only
# the iris offset (within each eye box) and face_count/detection_confidence vary.
_LEFT_EYE_CENTER = (0.38, 0.40)
_RIGHT_EYE_CENTER = (0.62, 0.40)
_HALF_W, _HALF_H = 0.04, 0.02


def make_landmarks(
    offset_x: float,
    offset_y: float,
    face_count: int = 1,
    detection_confidence: float = 0.9,
) -> FaceLandmarks:
    """A frontal face with both irises offset by (offset_x, offset_y) — the same
    offset units features.py's extract_features() reads back out (roughly -1..1 =
    fully toward one eye corner)."""
    return FaceLandmarks(
        face_count=face_count,
        detection_confidence=detection_confidence,
        left_iris_center=Point3D(
            _LEFT_EYE_CENTER[0] + offset_x * _HALF_W, _LEFT_EYE_CENTER[1] + offset_y * _HALF_H
        ),
        right_iris_center=Point3D(
            _RIGHT_EYE_CENTER[0] + offset_x * _HALF_W, _RIGHT_EYE_CENTER[1] + offset_y * _HALF_H
        ),
        left_eye_inner=Point3D(0.42, 0.40),
        left_eye_outer=Point3D(0.34, 0.40),
        left_eye_top=Point3D(0.38, 0.38),
        left_eye_bottom=Point3D(0.38, 0.42),
        right_eye_inner=Point3D(0.58, 0.40),
        right_eye_outer=Point3D(0.66, 0.40),
        right_eye_top=Point3D(0.62, 0.38),
        right_eye_bottom=Point3D(0.62, 0.42),
        nose_tip=Point3D(0.50, 0.50),
        chin=Point3D(0.50, 0.70),
        forehead=Point3D(0.50, 0.30),
        left_face_edge=Point3D(0.30, 0.50),
        right_face_edge=Point3D(0.70, 0.50),
    )


def offset_for_target(target_x: float, target_y: float) -> tuple[float, float]:
    """The ground-truth (noiseless) inverse of the calibration mapping this fixture
    was built with: a simple, symmetric, affine iris-offset-to-screen model."""
    return (target_x - SCREEN_W / 2) / (SCREEN_W / 2), (target_y - SCREEN_H / 2) / (SCREEN_H / 2)


def make_row(
    t: float, landmarks: FaceLandmarks | None, target: tuple[float, float] | None, segment: str
) -> dict[str, str]:
    row = landmarks_to_row(t, landmarks)
    row["target_x"] = repr(target[0]) if target else ""
    row["target_y"] = repr(target[1]) if target else ""
    row["segment"] = segment
    return row


def generate_rows(seed: int = 20260925) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    t = 0.0

    # 1. Calibration: the standard 9-point grid, clean.
    for target in target_points(SCREEN_W, SCREEN_H):
        ox, oy = offset_for_target(*target)
        for _ in range(5):
            rows.append(make_row(t, make_landmarks(ox, oy), target, "calibration"))
            t += DT

    # 2. On-target dwell, with noise: what the filter and pixel-error metrics use.
    on_target_points = [
        (300.0, 200.0),
        (1600.0, 200.0),
        (960.0, 540.0),
        (300.0, 900.0),
        (1600.0, 900.0),
        (960.0, 150.0),
    ]
    noise_sigma = 0.03  # in iris-offset units
    for target in on_target_points:
        ox, oy = offset_for_target(*target)
        for _ in range(15):
            noisy = (ox + rng.gauss(0, noise_sigma), oy + rng.gauss(0, noise_sigma))
            rows.append(make_row(t, make_landmarks(*noisy), target, "on_target"))
            t += DT

    # 3. Looking away: three failure modes Gate 1 must catch.
    for _ in range(10):
        rows.append(make_row(t, make_landmarks(0.0, 0.0, face_count=2), None, "looking_away"))
        t += DT
    for _ in range(10):
        rows.append(
            make_row(t, make_landmarks(0.0, 0.0, detection_confidence=0.2), None, "looking_away")
        )
        t += DT
    for _ in range(10):
        rows.append(make_row(t, None, None, "looking_away"))
        t += DT

    # 4. Just reading: long steady in-calibration hold, never confirmed.
    for _ in range(300):
        rows.append(make_row(t, make_landmarks(0.0, 0.0), (960.0, 540.0), "just_reading"))
        t += DT

    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="eval/fixtures/synthetic_session.csv", help="output CSV path"
    )
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args(argv)

    rows = generate_rows(seed=args.seed)
    write_csv(Path(args.out), rows)
    print(f"wrote {len(rows)} frames to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
