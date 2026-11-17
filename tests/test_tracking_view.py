"""The live tracking view (``eyeos run --view``): the pupil markers, their shift from
the eye's corner midpoint (the number EyeOS turns into the cursor), a mini screen map
and the gate lights, drawn over the camera image.

The geometry is pure and tested directly. The OpenCV layer is tested against the fake
``cv2`` in tests/fake_capture.py, which records every drawing and window call; only the
optional smoke tests at the bottom use a real OpenCV.
"""

from __future__ import annotations

import pytest

from eyeos.input_sources import Frame
from eyeos.input_sources.video_file import VideoFileInputSource
from eyeos.input_sources.webcam import WebcamInputSource
from tests.fake_capture import FakeCv2, FakeImage, FakeMediapipe, install_fakes
from tests.test_features import make_landmarks

# ---------------------------------------------------------------------------
# Plumbing: frames carry the camera image only when asked
# ---------------------------------------------------------------------------


def test_frame_equality_ignores_the_image():
    landmarks = make_landmarks()
    with_image = Frame(timestamp_s=0.5, landmarks=landmarks, image=FakeImage("bgr", 0))
    without = Frame(timestamp_s=0.5, landmarks=landmarks)

    assert with_image == without
    assert hash(with_image) == hash(without)
    assert without.image is None
    assert "image" not in repr(with_image)


def _video_file_source(tmp_path, **kwargs):
    video = tmp_path / "clip.mp4"
    video.touch()
    return VideoFileInputSource(video, **kwargs)


def _webcam_source(tmp_path, **kwargs):
    return WebcamInputSource(device_index=0, **kwargs)


CAPTURE_SOURCES = pytest.mark.parametrize(
    "make_source", [_video_file_source, _webcam_source], ids=["video_file", "webcam"]
)


@CAPTURE_SOURCES
def test_capture_sources_drop_the_camera_image_by_default(make_source, tmp_path, monkeypatch):
    install_fakes(monkeypatch, cv2=FakeCv2(n_frames=3), mediapipe=FakeMediapipe())

    frames = list(make_source(tmp_path).frames())

    assert len(frames) == 3
    assert [f.image for f in frames] == [None, None, None]


@CAPTURE_SOURCES
def test_capture_sources_attach_the_bgr_frame_when_asked(make_source, tmp_path, monkeypatch):
    install_fakes(
        monkeypatch,
        cv2=FakeCv2(n_frames=3, frame_shape=(480, 640, 3)),
        mediapipe=FakeMediapipe(),
    )

    frames = list(make_source(tmp_path, keep_images=True).frames())

    # The frame as read() returned it (BGR), not the RGB copy MediaPipe was given.
    assert [f.image for f in frames] == [FakeImage("bgr", i, (480, 640, 3)) for i in range(3)]
    assert all(f.landmarks is not None for f in frames)


# ---------------------------------------------------------------------------
# Pipeline.on_frame: one call per frame, after on_status
# ---------------------------------------------------------------------------


def test_on_frame_gets_each_frame_its_gate_input_and_status_after_on_status():
    from eyeos.config import Config
    from eyeos.gates import FrameInput
    from eyeos.output.dry_run import DryRunOutputAdapter
    from eyeos.pipeline import Pipeline
    from tests.test_pipeline import ListInputSource

    frames = [
        Frame(timestamp_s=t, landmarks=make_landmarks(), image=FakeImage("bgr", i))
        for i, t in enumerate((0.0, 0.2, 0.4))
    ]
    events = []
    pipeline = Pipeline(
        Config(),
        ListInputSource(frames),
        DryRunOutputAdapter(echo=None),
        on_status=lambda status: events.append(("status", status)),
        on_frame=lambda *args: events.append(("frame", *args)),
    )

    result = pipeline.run()

    assert [e[0] for e in events] == ["status", "frame"] * 3
    seen = [e for e in events if e[0] == "frame"]
    assert [e[1] for e in seen] == frames
    assert [e[1].image for e in seen] == [FakeImage("bgr", i) for i in range(3)]
    assert all(isinstance(e[2], FrameInput) for e in seen)
    assert [e[2].timestamp_s for e in seen] == [0.0, 0.2, 0.4]
    assert [e[3] for e in seen] == result.statuses
