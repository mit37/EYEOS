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
from tests.fake_capture import FakeCv2, FakeCv2Error, FakeImage, FakeMediapipe, install_fakes
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


# ---------------------------------------------------------------------------
# TrackingView: the OpenCV window, against the fake cv2
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_cv2(monkeypatch):
    """A fake cv2 with drawing/window recording, and a display to draw on."""
    cv2 = FakeCv2()
    install_fakes(monkeypatch, cv2=cv2)
    monkeypatch.setenv("DISPLAY", ":99")
    return cv2


def _install_cv2(monkeypatch, **kwargs):
    cv2 = FakeCv2(**kwargs)
    install_fakes(monkeypatch, cv2=cv2)
    monkeypatch.setenv("DISPLAY", ":99")
    return cv2


def _frame_input(t=0.0):
    from eyeos.gates import FrameInput

    return FrameInput(
        timestamp_s=t,
        face_count=1,
        detection_confidence=0.95,
        both_irises_visible=True,
        gaze_point_px=(960.0, 540.0),
        head_pose_deg=(0.0, 0.0, 0.0),
    )


def _drawn(cv2, name):
    return [call for call in cv2.drawn if call.name == name]


def _texts(cv2):
    return [call.params["text"] for call in _drawn(cv2, "putText")]


def _centers(cv2):
    return [call.params["center"] for call in _drawn(cv2, "circle")]


def _round(point):
    return (round(point[0]), round(point[1]))


def _update(view, n_frames, landmarks="default", status="default"):
    for i in range(n_frames):
        t = i / 30
        face = make_landmarks() if landmarks == "default" else landmarks
        view.update(Frame(t, face), _frame_input(t), _status() if status == "default" else status)


def test_the_view_needs_opencv(monkeypatch):
    import sys

    from eyeos.tracking_view import TrackingView, annotate

    monkeypatch.setitem(sys.modules, "cv2", None)  # makes `import cv2` raise ImportError
    monkeypatch.setenv("DISPLAY", ":99")

    with pytest.raises(RuntimeError, match="needs OpenCV"):
        TrackingView(SCREEN)
    with pytest.raises(RuntimeError, match="needs OpenCV"):
        annotate(FakeImage("bgr", 0, (480, 640, 3)), make_landmarks())


@pytest.mark.parametrize(
    "platform, environ, refused",
    [
        ("linux", {}, True),
        ("linux", {"DISPLAY": ""}, True),
        ("linux", {"DISPLAY": ":0"}, False),
        ("linux", {"WAYLAND_DISPLAY": "wayland-0"}, False),
        ("win32", {}, False),
        ("darwin", {}, False),
    ],
)
def test_a_linux_session_without_a_display_is_refused_up_front(platform, environ, refused):
    # OpenCV's Qt window aborts the whole process (no exception to catch) when there is
    # no X or Wayland display, so the view must refuse before its first imshow().
    from eyeos.tracking_view import _display_problem

    assert (_display_problem(platform, environ) is not None) is refused


def test_the_view_refuses_to_start_without_a_display(monkeypatch):
    from eyeos.tracking_view import TrackingView

    install_fakes(monkeypatch, cv2=FakeCv2())
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    with pytest.raises(RuntimeError, match="needs a display"):
        TrackingView(SCREEN)


def test_update_draws_the_pupils_their_shift_the_map_and_the_lights(fake_cv2):
    from eyeos.landmarks import Point3D
    from eyeos.tracking_view import TrackingView, compute_view

    landmarks = make_landmarks(
        left_iris_center=Point3D(0.39, 0.41), right_iris_center=Point3D(0.60, 0.40)
    )
    camera = FakeImage("bgr", 7, (720, 1280, 3))
    frame = Frame(timestamp_s=1.0, landmarks=landmarks, image=camera)
    status = _status(target=(480.0, 270.0))
    view = TrackingView(SCREEN)

    view.update(frame, _frame_input(1.0), status)

    expected = compute_view(landmarks, (1280, 720), status, SCREEN)
    # One image shown in the "EyeOS tracking" window: a copy of the camera frame, so the
    # frame itself is never drawn on.
    (shown,) = fake_cv2.shown
    assert shown == camera and shown is not camera
    assert fake_cv2.window_calls[:2] == [("imshow", "EyeOS tracking", shown), ("waitKey", 1)]
    assert all(call.image is shown for call in fake_cv2.drawn)

    circles = _centers(fake_cv2)
    lines = [(c.params["pt1"], c.params["pt2"]) for c in _drawn(fake_cv2, "line")]
    arrows = [(c.params["pt1"], c.params["pt2"]) for c in _drawn(fake_cv2, "arrowedLine")]
    for eye in (expected.left, expected.right):
        # The pupil marker, the corner midpoint, both corners and both lids.
        for point in (eye.iris_px, eye.mid_px, eye.outer_px, eye.inner_px):
            assert _round(point) in circles
        for point in (eye.top_px, eye.bottom_px):
            assert _round(point) in circles
        assert (_round(eye.outer_px), _round(eye.inner_px)) in lines  # the corner line
        assert (_round(eye.mid_px), _round(eye.arrow_end_px)) in arrows  # the shift x gain
    assert len(arrows) == 2

    # The mini map, with the cursor at the target the gates allowed.
    x0, y0, x1, y1 = expected.minimap.rect
    rectangles = [(c.params["pt1"], c.params["pt2"]) for c in _drawn(fake_cv2, "rectangle")]
    assert rectangles == [((x0, y0), (x1, y1))]
    assert expected.minimap.cursor_px == pytest.approx(
        (x0 + 0.25 * (x1 - x0), y0 + 0.25 * (y1 - y0))
    )
    assert _round(expected.minimap.cursor_px) in circles

    texts = _texts(fake_cv2)
    for label in ("1 Presence", "2 Calibration", "3 Stability", "4 Kill/limits"):
        assert any(text.startswith(label) for text in texts), label
    # The numbers the calibration is fed, in eye half-widths, and the arrow's gain.
    assert any("L +0.25 +0.25" in t and "R -0.50 +0.00" in t for t in texts)
    assert any("x4" in t for t in texts)
    assert "blocked" not in texts


def test_gate_lights_are_green_or_red_with_their_reasons(fake_cv2):
    from eyeos.tracking_view import GREEN, RED, TrackingView, gate_light_centers

    reasons = {"gate1": "ok", "gate2": "no calibration", "gate3": "dwell", "gate4": "ok"}
    status = _status(move_allowed=False, target=None, gates=(1, 0, 0, 1), reasons=reasons)
    view = TrackingView(SCREEN)

    view.update(Frame(0.0, make_landmarks()), _frame_input(), status)

    texts = _texts(fake_cv2)
    assert "1 Presence: ok" in texts
    assert "2 Calibration: no calibration" in texts
    centers = gate_light_centers()
    lights = {
        c.params["center"]: c.params["color"]
        for c in _drawn(fake_cv2, "circle")
        if c.params["center"] in centers and c.params["thickness"] == fake_cv2.FILLED
    }
    assert [lights[center] for center in centers] == [GREEN, RED, RED, GREEN]


def test_a_blocked_cursor_is_drawn_at_the_smoothed_gaze_and_labelled(fake_cv2):
    from eyeos.tracking_view import RED, TrackingView, compute_view

    status = _status(move_allowed=False, target=None, smoothed=(1920.0, 1080.0), gates=(1, 0, 0, 1))
    view = TrackingView(SCREEN)

    view.update(Frame(0.0, make_landmarks()), _frame_input(), status)

    minimap = compute_view(make_landmarks(), (640, 480), status, SCREEN).minimap
    assert _round(minimap.cursor_px) == minimap.rect[2:]  # bottom-right corner
    marker = [c for c in _drawn(fake_cv2, "circle") if c.params["center"] == minimap.rect[2:]]
    assert marker and all(c.params["color"] == RED for c in marker)
    assert "blocked" in _texts(fake_cv2)


def test_an_allowed_cursor_is_green(fake_cv2):
    from eyeos.tracking_view import GREEN, TrackingView

    view = TrackingView(SCREEN)
    view.update(Frame(0.0, make_landmarks()), _frame_input(), _status(target=(0.0, 0.0)))

    x0, y0 = _drawn(fake_cv2, "rectangle")[0].params["pt1"]
    marker = [c for c in _drawn(fake_cv2, "circle") if c.params["center"] == (x0, y0)]
    assert marker and all(c.params["color"] == GREEN for c in marker)


def test_reasons_are_drawn_in_plain_ascii(fake_cv2):
    # OpenCV's Hershey fonts only have ASCII: an em dash would come out as "??".
    from eyeos.tracking_view import TrackingView

    reasons = {
        "gate1": "ok",
        "gate2": "no calibration for this session — recalibrate",
        "gate3": "head pose 20.0° ≥ limit",
        "gate4": "ok",
    }
    status = _status(move_allowed=False, target=None, gates=(1, 0, 0, 1), reasons=reasons)
    view = TrackingView(SCREEN)

    view.update(Frame(0.0, make_landmarks()), _frame_input(), status)

    texts = _texts(fake_cv2)
    assert "2 Calibration: no calibration for this session - recalibrate" in texts
    assert "3 Stability: head pose 20.0deg >= limit" in texts
    assert all(text.isascii() for text in texts)


def test_without_a_camera_image_the_view_draws_on_a_black_640x480_canvas(fake_cv2):
    import numpy as np

    from eyeos.tracking_view import TrackingView, compute_view

    view = TrackingView(SCREEN)
    view.update(Frame(0.0, make_landmarks()), _frame_input(), _status())

    (shown,) = fake_cv2.shown
    assert isinstance(shown, np.ndarray)
    assert shown.shape == (480, 640, 3) and shown.dtype == np.uint8
    assert not shown.any()  # the fake cv2 draws nothing on it
    expected = compute_view(make_landmarks(), (640, 480), _status(), SCREEN)
    arrows = [c.params["pt1"] for c in _drawn(fake_cv2, "arrowedLine")]
    assert arrows == [_round(expected.left.mid_px), _round(expected.right.mid_px)]


def test_no_face_is_written_on_the_image(fake_cv2):
    from eyeos.tracking_view import TrackingView

    status = _status(move_allowed=False, target=None, smoothed=None, gates=(0, 0, 0, 1))
    view = TrackingView(SCREEN)

    view.update(Frame(0.0, None), _frame_input(), status)

    assert "no face" in _texts(fake_cv2)
    assert _drawn(fake_cv2, "arrowedLine") == []
    assert len(fake_cv2.shown) == 1


@pytest.mark.parametrize("key", [ord("q"), ord("Q")])
def test_pressing_q_closes_the_window_and_stops_drawing(monkeypatch, key):
    from eyeos.tracking_view import TrackingView

    cv2 = _install_cv2(monkeypatch, keys=[-1, key])
    view = TrackingView(SCREEN)

    _update(view, 4)

    assert view.closed
    assert len(cv2.shown) == 2  # nothing drawn after 'q'
    destroys = [c for c in cv2.window_calls if c[0] == "destroyWindow"]
    assert destroys == [("destroyWindow", "EyeOS tracking")]


def test_other_keys_leave_the_window_open(monkeypatch):
    # Esc is the kill switch's job (eyeos.hotkeys), not the view's.
    from eyeos.tracking_view import TrackingView

    cv2 = _install_cv2(monkeypatch, keys=[27, ord("a"), ord(" ")])
    view = TrackingView(SCREEN)

    _update(view, 3)

    assert not view.closed
    assert len(cv2.shown) == 3


def test_a_window_closed_by_the_user_stops_drawing(monkeypatch):
    from eyeos.tracking_view import TrackingView

    # The user closes the window while the second image is up.
    cv2 = _install_cv2(monkeypatch, window_visible=lambda shown: 1.0 if shown < 2 else 0.0)
    view = TrackingView(SCREEN)

    _update(view, 5)

    assert view.closed
    assert len(cv2.shown) == 2
    props = [c for c in cv2.window_calls if c[0] == "getWindowProperty"]
    assert props == [("getWindowProperty", "EyeOS tracking", cv2.WND_PROP_VISIBLE)] * 2


def test_a_window_property_error_counts_as_a_closed_window(monkeypatch):
    from eyeos.tracking_view import TrackingView

    cv2 = _install_cv2(monkeypatch, window_property_error=FakeCv2Error("NULL window"))
    view = TrackingView(SCREEN)

    _update(view, 2)

    assert view.closed
    assert len(cv2.shown) == 1


def test_close_is_idempotent_and_tolerates_an_already_destroyed_window(monkeypatch):
    from eyeos.tracking_view import TrackingView

    cv2 = _install_cv2(monkeypatch, destroy_error=FakeCv2Error("NULL window"))
    view = TrackingView(SCREEN)
    _update(view, 1)

    view.close()
    view.close()

    assert view.closed
    destroys = [c for c in cv2.window_calls if c[0] == "destroyWindow"]
    assert destroys == [("destroyWindow", "EyeOS tracking")]
    # HighGUI only takes a destroyed window off the screen while it handles events.
    assert cv2.window_calls[-1] == ("waitKey", 1)


def test_closing_a_view_that_never_showed_a_frame_touches_no_window(fake_cv2):
    from eyeos.tracking_view import TrackingView

    view = TrackingView(SCREEN)
    view.close()

    assert view.closed
    assert fake_cv2.window_calls == []


def test_an_opencv_without_window_support_fails_with_a_clear_error(monkeypatch):
    # opencv-python-headless raises cv2.error from imshow ("The function is not
    # implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support").
    from eyeos.tracking_view import TrackingView

    _install_cv2(monkeypatch, imshow_error=FakeCv2Error("The function is not implemented."))
    view = TrackingView(SCREEN)

    with pytest.raises(RuntimeError, match="could not open its window"):
        _update(view, 1)


# ---------------------------------------------------------------------------
# annotate(): the same drawing on a still image
# ---------------------------------------------------------------------------


def test_annotate_draws_on_a_copy_of_a_still_image(fake_cv2):
    from eyeos.tracking_view import annotate, compute_view

    image = FakeImage("bgr", 3, (480, 640, 3))

    out = annotate(image, make_landmarks())

    assert out == image and out is not image
    assert all(call.image is out for call in fake_cv2.drawn)
    expected = compute_view(make_landmarks(), (640, 480), None, None)
    circles = _centers(fake_cv2)
    assert _round(expected.left.iris_px) in circles
    assert _round(expected.right.iris_px) in circles
    assert len(_drawn(fake_cv2, "arrowedLine")) == 2
    assert _drawn(fake_cv2, "rectangle") == []  # no screen, no map
    assert not any(t.startswith("1 Presence") for t in _texts(fake_cv2))  # no gates
    assert fake_cv2.window_calls == []  # and no window


def test_annotate_puts_a_given_cursor_on_the_mini_map(fake_cv2):
    from eyeos.tracking_view import CYAN, annotate

    annotate(
        FakeImage("bgr", 3, (480, 640, 3)),
        make_landmarks(),
        cursor_px=(1920.0, 0.0),
        screen=SCREEN,
    )

    (rectangle,) = _drawn(fake_cv2, "rectangle")
    (_, y0), (x1, _) = rectangle.params["pt1"], rectangle.params["pt2"]
    marker = [c for c in _drawn(fake_cv2, "circle") if c.params["center"] == (x1, y0)]
    assert marker and all(c.params["color"] == CYAN for c in marker)
    assert "blocked" not in _texts(fake_cv2)  # no gate decided anything here


def test_compute_view_takes_a_plain_cursor_point_for_still_images():
    from eyeos.tracking_view import compute_view

    minimap = compute_view(
        make_landmarks(), (640, 480), None, SCREEN, cursor_px=(960.0, 540.0)
    ).minimap

    x0, y0, x1, y1 = minimap.rect
    assert minimap.cursor_state == "point"
    assert minimap.cursor_px == pytest.approx(((x0 + x1) / 2, (y0 + y1) / 2))
    with pytest.raises(ValueError, match="not both"):
        compute_view(make_landmarks(), (640, 480), _status(), SCREEN, cursor_px=(1.0, 2.0))


def test_annotate_needs_the_screen_to_place_a_cursor(fake_cv2):
    from eyeos.tracking_view import annotate

    with pytest.raises(ValueError, match="screen"):
        annotate(FakeImage("bgr", 3, (480, 640, 3)), make_landmarks(), cursor_px=(1.0, 2.0))


def test_annotate_with_no_face(fake_cv2):
    from eyeos.tracking_view import annotate

    annotate(FakeImage("bgr", 3, (480, 640, 3)), None, screen=SCREEN)

    assert "no face" in _texts(fake_cv2)
    assert len(_drawn(fake_cv2, "rectangle")) == 1


def test_annotate_with_real_opencv_draws_on_a_numpy_image():
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from eyeos.tracking_view import annotate, compute_view

    assert not isinstance(cv2, FakeCv2)  # a real OpenCV
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cursor = (960.0, 540.0)

    out = annotate(image, make_landmarks(), cursor_px=cursor, screen=SCREEN)

    assert isinstance(out, np.ndarray) and out.shape == image.shape and out.dtype == np.uint8
    assert not image.any()  # the input is untouched
    geometry = compute_view(make_landmarks(), (640, 480), None, SCREEN, cursor_px=cursor)
    for point in (geometry.left.iris_px, geometry.right.iris_px, geometry.minimap.cursor_px):
        x, y = _round(point)
        assert out[y, x].any()  # the pupil markers and the cursor on the mini map
