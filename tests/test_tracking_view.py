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


# ---------------------------------------------------------------------------
# compute_view: the geometry, with no OpenCV at all
# ---------------------------------------------------------------------------

SCREEN = (1920, 1080)


def _status(
    *,
    move_allowed=True,
    target=(960.0, 540.0),
    smoothed=(960.0, 540.0),
    gates=(True, True, True, True),
    reasons=None,
):
    from eyeos.gates import GateStatus

    return GateStatus(
        gate1_presence=bool(gates[0]),
        gate2_calibration=bool(gates[1]),
        gate3_stability=bool(gates[2]),
        gate4_kill_and_limits=bool(gates[3]),
        move_allowed=move_allowed,
        cursor_target_px=target,
        click_fired=False,
        reasons=reasons if reasons is not None else {f"gate{i}": "ok" for i in range(1, 5)},
        smoothed_gaze_px=smoothed,
    )


def test_landmarks_are_frame_heights_so_pixels_scale_both_axes_by_the_height():
    from eyeos.landmarks import Point3D
    from eyeos.tracking_view import compute_view

    # On a 16:9 frame x runs 0..16/9: the image centre is x = 8/9, not x = 0.5.
    centre = Point3D(0.5 * 1280 / 720, 0.5)
    view = compute_view(make_landmarks(left_iris_center=centre), (1280, 720), None, SCREEN)
    assert view.left.iris_px == pytest.approx((640.0, 360.0))

    view = compute_view(make_landmarks(), (1280, 720), None, SCREEN)
    left, right = view.left, view.right
    assert left.outer_px == pytest.approx((0.34 * 720, 0.40 * 720))
    assert left.inner_px == pytest.approx((0.42 * 720, 0.40 * 720))
    assert left.top_px == pytest.approx((0.38 * 720, 0.38 * 720))
    assert left.bottom_px == pytest.approx((0.38 * 720, 0.42 * 720))
    assert left.iris_px == pytest.approx((0.38 * 720, 0.40 * 720))
    assert right.inner_px == pytest.approx((0.58 * 720, 0.40 * 720))
    assert right.outer_px == pytest.approx((0.66 * 720, 0.40 * 720))
    assert right.iris_px == pytest.approx((0.62 * 720, 0.40 * 720))
    assert view.image_size == (1280, 720)


def test_the_shift_runs_from_the_corner_midpoint_to_the_iris_and_the_arrow_magnifies_it():
    from eyeos.landmarks import Point3D
    from eyeos.tracking_view import compute_view

    landmarks = make_landmarks(
        left_iris_center=Point3D(0.39, 0.41), right_iris_center=Point3D(0.60, 0.40)
    )
    view = compute_view(landmarks, (1280, 720), None, SCREEN)

    left = view.left
    assert left.mid_px == pytest.approx((0.38 * 720, 0.40 * 720))
    assert left.shift_px == pytest.approx((0.01 * 720, 0.01 * 720))
    assert left.arrow_end_px == pytest.approx((0.38 * 720 + 4 * 7.2, 0.40 * 720 + 4 * 7.2))
    assert view.right.shift_px == pytest.approx((-0.02 * 720, 0.0), abs=1e-9)

    magnified = compute_view(landmarks, (1280, 720), None, SCREEN, gain=10.0)
    assert magnified.left.arrow_end_px == pytest.approx((273.6 + 72.0, 288.0 + 72.0))
    assert magnified.left.shift_px == pytest.approx(left.shift_px)


def test_the_offsets_are_the_numbers_eyeos_feeds_the_calibration():
    from eyeos.features import extract_features
    from eyeos.landmarks import Point3D
    from eyeos.tracking_view import compute_view

    landmarks = make_landmarks(
        left_iris_center=Point3D(0.39, 0.41), right_iris_center=Point3D(0.60, 0.40)
    )
    view = compute_view(landmarks, (640, 480), None, SCREEN)

    features = extract_features(landmarks)
    assert view.iris_offsets == pytest.approx(features.iris_vector())
    # In eye half-widths: 0.01 / 0.04 along and across the left eye's corner line.
    assert view.left.offset == pytest.approx((0.25, 0.25))
    assert view.right.offset == pytest.approx((features.right_iris_x, features.right_iris_y))
    assert view.right.offset == pytest.approx((-0.5, 0.0), abs=1e-9)


def test_no_face_means_no_eye_markers_but_still_a_map_and_lights():
    from eyeos.tracking_view import compute_view

    view = compute_view(None, (640, 480), _status(move_allowed=False, target=None), SCREEN)

    assert view.left is None and view.right is None
    assert view.iris_offsets is None
    assert view.minimap is not None
    assert len(view.gates) == 4


def test_an_eye_with_non_finite_landmarks_is_not_drawn():
    from eyeos.landmarks import Point3D
    from eyeos.tracking_view import compute_view

    landmarks = make_landmarks(left_iris_center=Point3D(float("nan"), 0.40))
    view = compute_view(landmarks, (640, 480), None, SCREEN)

    assert view.left is None
    assert view.right is not None
    assert view.iris_offsets is None


@pytest.mark.parametrize(
    "image_size, screen",
    [((640, 480), (1920, 1080)), ((1280, 720), (1920, 1080)), ((640, 480), (1080, 1920))],
    ids=["640x480-landscape", "1280x720-landscape", "640x480-portrait"],
)
def test_the_mini_map_sits_in_the_top_right_corner_with_the_screen_aspect(image_size, screen):
    from eyeos.tracking_view import compute_view

    view = compute_view(make_landmarks(), image_size, None, screen)

    x0, y0, x1, y1 = view.minimap.rect
    w, h = image_size
    assert 0 < x0 < x1 < w and 0 < y0 < y1 < h
    assert x0 > w / 2 and y1 < h / 2  # top-right quadrant
    assert w - x1 == y0  # the same margin from both edges
    assert (x1 - x0) / (y1 - y0) == pytest.approx(screen[0] / screen[1], rel=0.02)
    assert view.minimap.screen == screen


def test_an_allowed_move_puts_the_cursor_marker_at_the_target():
    from eyeos.tracking_view import compute_view

    status = _status(target=(960.0, 540.0), smoothed=(100.0, 100.0))
    minimap = compute_view(make_landmarks(), (640, 480), status, SCREEN).minimap

    x0, y0, x1, y1 = minimap.rect
    assert minimap.cursor_state == "allowed"
    assert minimap.screen_point == (960.0, 540.0)
    assert minimap.cursor_px == pytest.approx(((x0 + x1) / 2, (y0 + y1) / 2))


def test_a_blocked_move_shows_the_smoothed_gaze_marked_blocked():
    from eyeos.tracking_view import compute_view

    status = _status(move_allowed=False, target=None, smoothed=(0.0, 0.0), gates=(1, 0, 0, 1))
    minimap = compute_view(make_landmarks(), (640, 480), status, SCREEN).minimap

    x0, y0, _, _ = minimap.rect
    assert minimap.cursor_state == "blocked"
    assert minimap.screen_point == (0.0, 0.0)
    assert minimap.cursor_px == pytest.approx((x0, y0))


@pytest.mark.parametrize(
    "status",
    [
        None,
        _status(move_allowed=False, target=None, smoothed=None),
        _status(move_allowed=False, target=None, smoothed=(float("nan"), 5.0)),
    ],
    ids=["no-status", "no-gaze", "nan-gaze"],
)
def test_without_a_usable_gaze_there_is_no_cursor_marker(status):
    from eyeos.tracking_view import compute_view

    minimap = compute_view(make_landmarks(), (640, 480), status, SCREEN).minimap

    assert minimap.cursor_state == "none"
    assert minimap.cursor_px is None


def test_the_gate_lights_follow_the_status():
    from eyeos.tracking_view import compute_view

    reasons = {
        "gate1": "ok",
        "gate2": "no calibration for this session — recalibrate",
        "gate3": "dwell 0.10s < 0.80s",
        "gate4": "ok",
    }
    status = _status(move_allowed=False, target=None, gates=(1, 0, 0, 1), reasons=reasons)

    lights = compute_view(make_landmarks(), (640, 480), status, SCREEN).gates

    assert [light.label for light in lights] == [
        "1 Presence",
        "2 Calibration",
        "3 Stability",
        "4 Kill/limits",
    ]
    assert [light.ok for light in lights] == [True, False, False, True]
    assert [light.reason for light in lights] == [reasons[f"gate{i}"] for i in range(1, 5)]
    assert compute_view(make_landmarks(), (640, 480), None, SCREEN).gates == ()


@pytest.mark.parametrize(
    "image_size, screen", [((0, 480), SCREEN), ((640, 480), (1920, 0)), ((-1, 5), SCREEN)]
)
def test_compute_view_rejects_empty_sizes(image_size, screen):
    from eyeos.tracking_view import compute_view

    with pytest.raises(ValueError):
        compute_view(make_landmarks(), image_size, None, screen)
