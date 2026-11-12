"""Input sources: the synthetic CSV reader (fully exercised) and the camera/video
sources. This container has neither OpenCV nor MediaPipe installed, matching the "no
webcam in the cloud" constraint in docs/PLAN.md, so the camera/video sources are tested
on their missing-dependency path and against the fakes in tests/fake_capture.py.
"""

from __future__ import annotations

import csv
import types
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from eyeos.input_sources import Frame
from eyeos.input_sources._mediapipe_common import landmarks_from_mediapipe
from eyeos.input_sources.synthetic import SyntheticInputSource
from eyeos.input_sources.video_file import VideoFileInputSource
from eyeos.input_sources.webcam import WebcamInputSource
from eyeos.landmarks import FaceLandmarks, Point3D
from eyeos.synthetic_format import CSV_FIELDNAMES, landmarks_to_row, row_to_frame
from tests.fake_capture import FakeCv2, FakeMediapipe, install_fakes

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
    with pytest.raises(RuntimeError, match="needs OpenCV and MediaPipe"):
        next(iter(WebcamInputSource(device_index=0).frames()))


def test_video_file_source_raises_clear_error_without_opencv_mediapipe(tmp_path):
    missing = tmp_path / "does_not_matter.mp4"
    with pytest.raises(RuntimeError, match="needs OpenCV and MediaPipe"):
        next(iter(VideoFileInputSource(missing).frames()))


# ---------------------------------------------------------------------------
# Webcam / video-file sources: MediaPipe without the Solutions API, and cleanup
# ---------------------------------------------------------------------------


def _video_file_source(tmp_path):
    video = tmp_path / "clip.mp4"
    video.touch()
    return VideoFileInputSource(video)


def _webcam_source(tmp_path):
    return WebcamInputSource(device_index=0)


CAPTURE_SOURCES = pytest.mark.parametrize(
    "make_source", [_video_file_source, _webcam_source], ids=["video_file", "webcam"]
)


@CAPTURE_SOURCES
@pytest.mark.parametrize(
    "mediapipe, shown_version",
    [
        (types.ModuleType("mediapipe"), "?"),
        (FakeMediapipe(solutions=False, version="1.2.3"), "1.2.3"),
    ],
    ids=["bare", "1.2.3"],
)
def test_source_raises_clear_error_when_mediapipe_lacks_solutions_api(
    make_source, mediapipe, shown_version, tmp_path, monkeypatch
):
    fake_cv2 = FakeCv2(n_frames=3)
    install_fakes(monkeypatch, cv2=fake_cv2, mediapipe=mediapipe)

    with pytest.raises(RuntimeError, match=r"Solutions API|mediapipe<0\.10\.30") as excinfo:
        next(iter(make_source(tmp_path).frames()))

    message = str(excinfo.value)
    assert f"installed mediapipe {shown_version} " in message
    assert "aren't installed" not in message
    assert all(cap.release_calls == 1 for cap in fake_cv2.captures)


@CAPTURE_SOURCES
def test_face_mesh_failure_leaves_no_capture_open(make_source, tmp_path, monkeypatch):
    fake_cv2 = FakeCv2(n_frames=3)
    fake_mp = FakeMediapipe(face_mesh_error=RuntimeError("face mesh model failed to load"))
    install_fakes(monkeypatch, cv2=fake_cv2, mediapipe=fake_mp)

    with pytest.raises(RuntimeError, match="face mesh model failed to load"):
        next(iter(make_source(tmp_path).frames()))

    assert all(cap.release_calls == 1 for cap in fake_cv2.captures)


@CAPTURE_SOURCES
def test_capture_open_failure_closes_face_mesh(make_source, tmp_path, monkeypatch):
    fake_mp = FakeMediapipe()
    install_fakes(monkeypatch, cv2=FakeCv2(opened=False), mediapipe=fake_mp)

    with pytest.raises(RuntimeError, match="could not open"):
        next(iter(make_source(tmp_path).frames()))

    assert all(mesh.close_calls == 1 for mesh in fake_mp.face_meshes)


@CAPTURE_SOURCES
def test_full_read_releases_capture_and_closes_face_mesh_once(make_source, tmp_path, monkeypatch):
    fake_cv2 = FakeCv2(n_frames=3)
    fake_mp = FakeMediapipe()
    install_fakes(monkeypatch, cv2=fake_cv2, mediapipe=fake_mp)

    frames = list(make_source(tmp_path).frames())

    assert len(frames) == 3
    assert all(f.landmarks is not None for f in frames)
    [cap] = fake_cv2.captures
    [mesh] = fake_mp.face_meshes
    assert cap.release_calls == 1
    assert mesh.close_calls == 1
    assert mesh.kwargs["refine_landmarks"] is True


def test_webcam_extra_pins_a_mediapipe_with_the_solutions_api():
    tomllib = pytest.importorskip("tomllib")
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with open(pyproject, "rb") as f:
        extra = tomllib.load(f)["project"]["optional-dependencies"]["webcam"]
    requirements = {r.name: r for r in map(Requirement, extra)}

    # mediapipe itself depends on opencv-contrib-python; opencv-python would share its cv2/.
    assert "opencv-python" not in requirements
    assert "opencv-contrib-python" in requirements
    mediapipe = requirements["mediapipe"].specifier
    assert mediapipe.contains("0.10.21")
    assert not mediapipe.contains("0.10.30")
    assert not mediapipe.contains("1.2.3")


@pytest.mark.parametrize("pos_msec", [0.0, -1.0], ids=["avfoundation-0", "dshow-minus-1"])
def test_webcam_stamps_frames_with_the_host_clock(pos_msec, monkeypatch):
    import itertools
    import time

    ticks = itertools.count(start=100.0, step=1 / 30)
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))
    install_fakes(
        monkeypatch, cv2=FakeCv2(n_frames=5, pos_msec=pos_msec), mediapipe=FakeMediapipe()
    )

    stamps = [f.timestamp_s for f in WebcamInputSource(device_index=0).frames()]

    assert len(stamps) == 5
    assert all(b > a for a, b in zip(stamps, stamps[1:], strict=False))


def test_video_file_keeps_the_container_timestamps(tmp_path, monkeypatch):
    install_fakes(monkeypatch, cv2=FakeCv2(n_frames=4, fps=25.0), mediapipe=FakeMediapipe())

    stamps = [f.timestamp_s for f in _video_file_source(tmp_path).frames()]

    assert stamps == pytest.approx([0.0, 0.04, 0.08, 0.12])
