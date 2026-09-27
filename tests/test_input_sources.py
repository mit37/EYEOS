"""Input sources: the synthetic CSV reader (fully exercised) and the camera/video
sources (only their guarded-missing-dependency path — this container has neither
opencv-python nor mediapipe installed, matching the "no webcam in the cloud"
constraint in docs/PLAN.md).
"""

from __future__ import annotations

import csv

import pytest

from eyeos.input_sources import Frame
from eyeos.input_sources._mediapipe_common import landmarks_from_mediapipe
from eyeos.input_sources.synthetic import SyntheticInputSource
from eyeos.input_sources.video_file import VideoFileInputSource
from eyeos.input_sources.webcam import WebcamInputSource
from eyeos.landmarks import FaceLandmarks, Point3D
from eyeos.synthetic_format import CSV_FIELDNAMES, landmarks_to_row, row_to_frame

SAMPLE_LANDMARKS = FaceLandmarks(
    face_count=1,
    detection_confidence=0.9,
    left_iris_center=Point3D(0.38, 0.40),
    right_iris_center=Point3D(0.62, 0.40),
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


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# synthetic_format round-trip
# ---------------------------------------------------------------------------


def test_landmarks_round_trip_through_a_row():
    row = landmarks_to_row(1.5, SAMPLE_LANDMARKS)
    ts, landmarks = row_to_frame(row)
    assert ts == pytest.approx(1.5)
    assert landmarks == SAMPLE_LANDMARKS


def test_no_face_round_trips_to_none_landmarks():
    row = landmarks_to_row(2.0, None)
    ts, landmarks = row_to_frame(row)
    assert ts == pytest.approx(2.0)
    assert landmarks is None
    assert row["face_count"] == "0"


def test_row_fieldnames_match_csv_fieldnames():
    row = landmarks_to_row(0.0, SAMPLE_LANDMARKS)
    assert set(row.keys()) == set(CSV_FIELDNAMES)


# ---------------------------------------------------------------------------
# SyntheticInputSource
# ---------------------------------------------------------------------------


def test_synthetic_source_yields_frames_in_order(tmp_path):
    p = tmp_path / "session.csv"
    rows = [
        landmarks_to_row(0.0, SAMPLE_LANDMARKS),
        landmarks_to_row(0.1, None),
        landmarks_to_row(0.2, SAMPLE_LANDMARKS),
    ]
    _write_csv(p, rows)

    frames = list(SyntheticInputSource(p).frames())
    assert [f.timestamp_s for f in frames] == pytest.approx([0.0, 0.1, 0.2])
    assert frames[0].landmarks == SAMPLE_LANDMARKS
    assert frames[1].landmarks is None
    assert frames[2].landmarks == SAMPLE_LANDMARKS
    assert all(isinstance(f, Frame) for f in frames)


def test_synthetic_source_context_manager_closes_cleanly(tmp_path):
    p = tmp_path / "session.csv"
    _write_csv(p, [landmarks_to_row(0.0, SAMPLE_LANDMARKS)])
    with SyntheticInputSource(p) as src:
        frames = list(src.frames())
    assert len(frames) == 1


def test_synthetic_source_empty_file_yields_no_frames(tmp_path):
    p = tmp_path / "empty.csv"
    _write_csv(p, [])
    assert list(SyntheticInputSource(p).frames()) == []


# ---------------------------------------------------------------------------
# _mediapipe_common index mapping (pure function, no mediapipe import needed)
# ---------------------------------------------------------------------------


class _FakeLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x, self.y, self.z = x, y, z


def test_landmarks_from_mediapipe_indexes_the_right_points():
    raw = [_FakeLandmark(0.0, 0.0, 0.0) for _ in range(478)]
    raw[468] = _FakeLandmark(0.11, 0.22, 0.33)  # left iris center
    raw[473] = _FakeLandmark(0.44, 0.55, 0.66)  # right iris center
    raw[1] = _FakeLandmark(0.5, 0.5, 0.0)  # nose tip

    landmarks = landmarks_from_mediapipe(raw, face_count=1, detection_confidence=0.8)

    assert landmarks.left_iris_center == Point3D(0.11, 0.22, 0.33)
    assert landmarks.right_iris_center == Point3D(0.44, 0.55, 0.66)
    assert landmarks.nose_tip == Point3D(0.5, 0.5, 0.0)
    assert landmarks.face_count == 1
    assert landmarks.detection_confidence == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Webcam / video-file sources: guarded missing-dependency path
# ---------------------------------------------------------------------------


def test_webcam_source_raises_clear_error_without_opencv_mediapipe():
    with pytest.raises(RuntimeError, match="opencv-python and mediapipe"):
        next(iter(WebcamInputSource(device_index=0).frames()))


def test_video_file_source_raises_clear_error_without_opencv_mediapipe(tmp_path):
    missing = tmp_path / "does_not_matter.mp4"
    with pytest.raises(RuntimeError, match="opencv-python and mediapipe"):
        next(iter(VideoFileInputSource(missing).frames()))
