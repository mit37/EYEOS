"""A live view of what EyeOS tracks (``eyeos run --view``): a marker on each pupil, its
shift from the eye's corner midpoint (the number EyeOS turns into the cursor), a mini map
of the screen with the cursor on it, and the four gate lights, drawn over the camera
image. docs/DEMO.md says what each drawn element means.

``compute_view`` is the pure geometry: it turns one frame's landmarks and gate status
into pixel positions and needs no OpenCV, so it is tested directly. ``TrackingView`` is
a thin OpenCV layer over it that shows each frame in a HighGUI window, and ``annotate``
draws the same thing on a still image. Both import OpenCV lazily, so importing this
module never needs it.

The window has only been driven against the fake ``cv2`` in tests/fake_capture.py and,
during development, a real OpenCV 4.11 (Qt) on a virtual X display (Xvfb); never on a
real desktop, webcam or Windows. See docs/PLAN.md's "Cloud-instance constraints".
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from eyeos.features import extract_features
from eyeos.gates import FrameInput, GateStatus, ScreenPoint
from eyeos.input_sources import Frame
from eyeos.landmarks import FaceLandmarks, Point3D

Pixel = tuple[float, float]
CursorState = Literal["allowed", "blocked", "point", "none"]

GATE_LABELS = ("1 Presence", "2 Calibration", "3 Stability", "4 Kill/limits")

# The mini map is a quarter of the image wide (less for a tall screen, so it never
# takes more than this share of the image's height), this far from the top-right corner.
_MAP_WIDTH_FRACTION = 0.25
_MAP_MAX_HEIGHT_FRACTION = 0.4
_MAP_MARGIN_PX = 10


@dataclass(frozen=True)
class EyeView:
    """One eye, in image pixels. ``offset`` is what EyeOS measures from it."""

    outer_px: Pixel  # eye corners
    inner_px: Pixel
    top_px: Pixel  # eyelids (drawn only: EyeOS's gaze ignores them)
    bottom_px: Pixel
    iris_px: Pixel  # the pupil marker
    mid_px: Pixel  # midpoint of the two corners
    shift_px: Pixel  # iris_px - mid_px
    arrow_end_px: Pixel  # mid_px + shift_px * gain
    # The iris offset from features.extract_features, in eye half-widths: x along the
    # corner line, y across it. These two numbers per eye feed the calibration.
    offset: tuple[float, float]


@dataclass(frozen=True)
class MiniMap:
    """The screen, scaled into a rectangle of the image, with the cursor on it."""

    rect: tuple[int, int, int, int]  # x0, y0, x1, y1 in image pixels
    screen: tuple[int, int]  # width, height in screen pixels
    # Where the marker goes, in image pixels; None when there's no gaze to show.
    cursor_px: Pixel | None
    # "allowed": the cursor was sent there this frame (status.cursor_target_px).
    # "blocked": the gates froze the cursor; the marker shows the smoothed gaze instead.
    # "point": a point given to annotate(), with no gate decision behind it.
    cursor_state: CursorState
    screen_point: ScreenPoint | None  # the marker's point, in screen pixels


@dataclass(frozen=True)
class GateLight:
    label: str
    ok: bool
    reason: str


@dataclass(frozen=True)
class ViewGeometry:
    image_size: tuple[int, int]
    face_found: bool  # False when the frame had no landmarks at all
    left: EyeView | None  # the eye on the left of the image; None if not measured
    right: EyeView | None
    # extract_features(landmarks).iris_vector(): (left x, left y, right x, right y).
    iris_offsets: tuple[float, float, float, float] | None
    minimap: MiniMap | None
    gates: tuple[GateLight, ...]  # empty without a status


def _to_px(point: Point3D, height: int) -> Pixel:
    # FaceLandmarks x and y are both in units of the frame's height (landmarks.py).
    return point.x * height, point.y * height


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def _eye_view(
    outer: Point3D,
    inner: Point3D,
    top: Point3D,
    bottom: Point3D,
    iris: Point3D,
    offset: tuple[float, float],
    height: int,
    gain: float,
) -> EyeView | None:
    points = (outer, inner, top, bottom, iris)
    if not all(_finite(p.x, p.y) for p in points) or not _finite(*offset):
        return None
    outer_px, inner_px = _to_px(outer, height), _to_px(inner, height)
    iris_px = _to_px(iris, height)
    mid_px = ((outer_px[0] + inner_px[0]) / 2.0, (outer_px[1] + inner_px[1]) / 2.0)
    shift_px = (iris_px[0] - mid_px[0], iris_px[1] - mid_px[1])
    eye = EyeView(
        outer_px=outer_px,
        inner_px=inner_px,
        top_px=_to_px(top, height),
        bottom_px=_to_px(bottom, height),
        iris_px=iris_px,
        mid_px=mid_px,
        shift_px=shift_px,
        arrow_end_px=(mid_px[0] + shift_px[0] * gain, mid_px[1] + shift_px[1] * gain),
        offset=offset,
    )
    # Finite landmarks can still overflow once scaled (to inf, or inf - inf = NaN).
    pixels = (
        eye.outer_px,
        eye.inner_px,
        eye.top_px,
        eye.bottom_px,
        eye.iris_px,
        eye.mid_px,
        eye.shift_px,
        eye.arrow_end_px,
    )
    if not all(_finite(*pixel) for pixel in pixels):
        return None
    return eye


def _minimap(
    image_size: tuple[int, int],
    screen: tuple[int, int],
    point: ScreenPoint | None,
    state: CursorState,
) -> MiniMap:
    width, height = image_size
    screen_w, screen_h = screen
    map_w = width * _MAP_WIDTH_FRACTION
    map_h = map_w * screen_h / screen_w
    if map_h > height * _MAP_MAX_HEIGHT_FRACTION:
        map_h = height * _MAP_MAX_HEIGHT_FRACTION
        map_w = map_h * screen_w / screen_h
    x1, y0 = width - _MAP_MARGIN_PX, _MAP_MARGIN_PX
    x0, y1 = x1 - round(map_w), y0 + round(map_h)
    rect = (x0, y0, x1, y1)

    if point is None or not _finite(*point):
        return MiniMap(rect, screen, None, "none", None)
    # Clamped onto the map: the pipeline already clamps to the screen, but a marker must
    # never be drawn outside the rectangle it stands for.
    fx = min(max(point[0] / screen_w, 0.0), 1.0)
    fy = min(max(point[1] / screen_h, 0.0), 1.0)
    cursor_px = (x0 + fx * (x1 - x0), y0 + fy * (y1 - y0))
    return MiniMap(rect, screen, cursor_px, state, point)


def _cursor(
    status: GateStatus | None, cursor_px: ScreenPoint | None
) -> tuple[ScreenPoint | None, CursorState]:
    if cursor_px is not None:
        return cursor_px, "point"
    if status is None:
        return None, "none"
    if status.move_allowed and status.cursor_target_px is not None:
        return status.cursor_target_px, "allowed"
    if status.smoothed_gaze_px is not None:
        return status.smoothed_gaze_px, "blocked"
    return None, "none"


def _gate_lights(status: GateStatus | None) -> tuple[GateLight, ...]:
    if status is None:
        return ()
    passes = (
        status.gate1_presence,
        status.gate2_calibration,
        status.gate3_stability,
        status.gate4_kill_and_limits,
    )
    return tuple(
        GateLight(label, ok, status.reasons.get(f"gate{i}", ""))
        for i, (label, ok) in enumerate(zip(GATE_LABELS, passes, strict=True), start=1)
    )


def compute_view(
    landmarks: FaceLandmarks | None,
    image_size: tuple[int, int],
    status: GateStatus | None,
    screen: tuple[int, int] | None,
    gain: float = 4.0,
    *,
    cursor_px: ScreenPoint | None = None,
) -> ViewGeometry:
    """Everything the tracking view draws for one frame, in image pixels.

    ``image_size`` and ``screen`` are (width, height). ``gain`` stretches each eye's
    shift arrow, since a real iris moves only a few pixels. Without ``screen`` there is
    no mini map; without ``status`` there are no gate lights and no cursor marker.
    ``cursor_px`` (screen pixels) puts a plain marker on the map instead of a gated
    cursor, for still images; it can't be combined with a ``status``.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {image_size}")
    if screen is not None and (screen[0] <= 0 or screen[1] <= 0):
        raise ValueError(f"screen size must be positive, got {screen}")
    if cursor_px is not None and status is not None:
        raise ValueError("pass a status or a cursor_px, not both")
    if cursor_px is not None and screen is None:
        raise ValueError("a cursor_px needs the screen size, to place it on the mini map")

    left = right = None
    iris_offsets = None
    if landmarks is not None:
        features = extract_features(landmarks)
        left = _eye_view(
            landmarks.left_eye_outer,
            landmarks.left_eye_inner,
            landmarks.left_eye_top,
            landmarks.left_eye_bottom,
            landmarks.left_iris_center,
            (features.left_iris_x, features.left_iris_y),
            height,
            gain,
        )
        right = _eye_view(
            landmarks.right_eye_outer,
            landmarks.right_eye_inner,
            landmarks.right_eye_top,
            landmarks.right_eye_bottom,
            landmarks.right_iris_center,
            (features.right_iris_x, features.right_iris_y),
            height,
            gain,
        )
        if _finite(*features.iris_vector()):
            iris_offsets = features.iris_vector()

    minimap = None
    if screen is not None:
        point, state = _cursor(status, cursor_px)
        minimap = _minimap((width, height), screen, point, state)

    return ViewGeometry(
        image_size=(width, height),
        face_found=landmarks is not None,
        left=left,
        right=right,
        iris_offsets=iris_offsets,
        minimap=minimap,
        gates=_gate_lights(status),
    )


# ---------------------------------------------------------------------------
# Drawing (OpenCV)
# ---------------------------------------------------------------------------

WINDOW_NAME = "EyeOS tracking"
# The canvas when the source has no camera image (a synthetic landmark CSV).
BLANK_SIZE = (640, 480)

# BGR, as OpenCV wants them.
GREEN = (60, 200, 60)
RED = (40, 40, 230)
CYAN = (230, 210, 0)
YELLOW = (0, 210, 255)
MAGENTA = (230, 0, 230)
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
BLACK = (0, 0, 0)

_FONT_SCALE = 0.4
_OUTLINE_THICKNESS = 3  # the dark outline under every text: its widest stroke
_LINE_H = 18
_LIGHT_X, _LIGHT_Y0, _LIGHT_STEP, _LIGHT_RADIUS = 14, 16, 20, 6
_PUPIL_RING_RADIUS = 6
_CURSOR_RADIUS = 5
# OpenCV parses drawing coordinates as 32-bit ints; anything this far off the image is
# clipped anyway.
_MAX_COORD = 100_000

# The keys that close the view (with Shift or Caps Lock, some backends report Q).
_QUIT_KEYS = (ord("q"), ord("Q"))

# OpenCV's Hershey fonts only cover ASCII.
_ASCII = {"—": "-", "–": "-", "°": "deg", "≥": ">=", "≤": "<="}


def gate_light_centers() -> tuple[tuple[int, int], ...]:
    """Where the four gate lights are drawn (top-left of the image), in image pixels."""
    return tuple((_LIGHT_X, _LIGHT_Y0 + i * _LIGHT_STEP) for i in range(len(GATE_LABELS)))


def _import_cv2(who: str) -> Any:
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            f"{who} needs OpenCV, which isn't installed. Install it with: "
            "pip install 'eyeos[webcam]' (Python 3.10-3.12)"
        ) from e
    return cv2


def _display_problem(platform: str, environ: Mapping[str, str]) -> str | None:
    """Why no window can open here, or None.

    Off Windows and macOS, the opencv-python wheels draw their window with Qt, and the
    only Qt platform plugin they bundle is xcb (X11). Without an X display Qt aborts
    the whole process instead of raising, so this has to be checked before the first
    window opens. A Wayland desktop offers an X display through XWayland ($DISPLAY);
    $WAYLAND_DISPLAY alone is not enough. (A GTK build of OpenCV could draw on Wayland
    directly, but this check doesn't tell the builds apart.)"""
    if platform in ("win32", "darwin"):
        return None
    if environ.get("DISPLAY"):
        return None
    return (
        "The tracking view needs an X display ($DISPLAY is not set; on a Wayland desktop "
        "XWayland provides one). Run it in a desktop session, or drop --view."
    )


def _qt_platform_override(platform: str, environ: Mapping[str, str]) -> str | None:
    """The QT_QPA_PLATFORM to set before the first window opens, or None to leave it.

    A QT_QPA_PLATFORM that names only plugins the opencv wheels don't ship (wayland,
    offscreen, ...) makes Qt abort the process even with $DISPLAY set, so it is pointed
    at xcb. A list that includes xcb ("wayland;xcb") is left alone: Qt falls back to
    xcb by itself. Only Qt reads the variable, so a GTK build is unaffected."""
    if platform in ("win32", "darwin"):
        return None
    value = environ.get("QT_QPA_PLATFORM", "").strip()
    if not value:
        return None
    # "name[:option,...]", several separated by ";" (tried in order).
    plugins = {entry.split(":", 1)[0].strip() for entry in value.split(";")}
    return None if "xcb" in plugins else "xcb"


def _window_error(error: Exception) -> RuntimeError:
    return RuntimeError(
        f"the tracking view could not open its window ({str(error).strip()}). It needs "
        "a desktop session and an OpenCV build with window support "
        "(opencv-contrib-python, not a -headless build)."
    )


def _ascii(text: str) -> str:
    for char, replacement in _ASCII.items():
        text = text.replace(char, replacement)
    return text.encode("ascii", errors="replace").decode("ascii")


def _coord(value: float) -> int:
    # Clamped before rounding: round() raises on inf and NaN. compute_view never gives
    # a NaN to draw, but if one came, it would land off the image too.
    if math.isnan(value):
        return -_MAX_COORD
    return round(min(max(value, -_MAX_COORD), _MAX_COORD))


def _pt(point: Pixel) -> tuple[int, int]:
    return _coord(point[0]), _coord(point[1])


def _image_size(image: Any) -> tuple[int, int]:
    """(width, height) of an OpenCV image, an array of shape (height, width, ...)."""
    shape = getattr(image, "shape", None)
    if not shape or len(shape) < 2:
        raise ValueError("expected an OpenCV image: an array of shape (height, width, 3)")
    return int(shape[1]), int(shape[0])


def _blank_canvas() -> Any:
    import numpy as np

    width, height = BLANK_SIZE
    return np.zeros((height, width, 3), dtype=np.uint8)


def _text(
    cv2: Any, image: Any, text: str, org: tuple[int, int], color: tuple[int, int, int]
) -> None:
    # A dark outline under the text keeps it readable over any camera image.
    text = _ascii(text)
    for ink, thickness in ((BLACK, _OUTLINE_THICKNESS), (color, 1)):
        cv2.putText(
            image,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            _FONT_SCALE,
            ink,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )


def _text_width(cv2: Any, text: str) -> int:
    (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, _OUTLINE_THICKNESS)
    return int(width)


def _fit(cv2: Any, text: str, max_width: int) -> str:
    """``text`` in ASCII, cut short with "..." if needed to be at most ``max_width``
    pixels wide as drawn (empty if not even "..." fits)."""
    text = _ascii(text)
    if _text_width(cv2, text) <= max_width:
        return text
    # The longest prefix that still fits with "..." after it. Widths grow with the
    # prefix, so a binary search finds it in a few getTextSize calls per frame.
    lo, hi = -1, len(text) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _text_width(cv2, text[:mid].rstrip() + "...") <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return "" if lo < 0 else text[:lo].rstrip() + "..."


def _dot(cv2: Any, image: Any, point: Pixel, radius: int, color: tuple[int, int, int]) -> None:
    cv2.circle(image, _pt(point), radius, color, thickness=cv2.FILLED, lineType=cv2.LINE_AA)


def _draw_eye(cv2: Any, image: Any, eye: EyeView) -> None:
    cv2.line(image, _pt(eye.outer_px), _pt(eye.inner_px), YELLOW, thickness=1, lineType=cv2.LINE_AA)
    for corner in (eye.outer_px, eye.inner_px):
        _dot(cv2, image, corner, 2, YELLOW)
    for lid in (eye.top_px, eye.bottom_px):
        _dot(cv2, image, lid, 2, GRAY)
    _dot(cv2, image, eye.mid_px, 2, WHITE)
    # The pupil marker: a ring around the iris centre, and a dot on it.
    cv2.circle(
        image, _pt(eye.iris_px), _PUPIL_RING_RADIUS, MAGENTA, thickness=1, lineType=cv2.LINE_AA
    )
    _dot(cv2, image, eye.iris_px, 2, MAGENTA)
    # The shift, from the corner midpoint towards the iris, stretched by the gain.
    cv2.arrowedLine(
        image,
        _pt(eye.mid_px),
        _pt(eye.arrow_end_px),
        CYAN,
        thickness=2,
        line_type=cv2.LINE_AA,
        tipLength=0.3,
    )


def _offset_text(name: str, eye: EyeView | None) -> str:
    if eye is None:
        return f"{name} --"
    return f"{name} {eye.offset[0]:+.2f} {eye.offset[1]:+.2f}"


def _draw_minimap(cv2: Any, image: Any, minimap: MiniMap) -> None:
    x0, y0, x1, y1 = minimap.rect
    cv2.rectangle(image, (x0, y0), (x1, y1), GRAY, thickness=1, lineType=cv2.LINE_AA)
    _text(cv2, image, f"screen {minimap.screen[0]}x{minimap.screen[1]}", (x0, y1 + 14), GRAY)
    if minimap.cursor_px is None:
        return
    if minimap.cursor_state == "blocked":
        # Where the gaze is, though the gates kept the cursor from following it.
        cv2.circle(
            image,
            _pt(minimap.cursor_px),
            _CURSOR_RADIUS,
            RED,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        _text(cv2, image, "blocked", (x0, y1 + 14 + _LINE_H), RED)  # under the map
    else:
        color = GREEN if minimap.cursor_state == "allowed" else CYAN
        _dot(cv2, image, minimap.cursor_px, _CURSOR_RADIUS, color)


def _draw(cv2: Any, image: Any, view: ViewGeometry, gain: float) -> None:
    width, height = view.image_size
    for eye in (view.left, view.right):
        if eye is not None:
            _draw_eye(cv2, image, eye)
    if not view.face_found:
        _text(cv2, image, "no face", (width // 2 - 28, height // 2), RED)
    else:
        _text(
            cv2,
            image,
            "iris offset (eye half-widths): "
            f"{_offset_text('L', view.left)}  {_offset_text('R', view.right)}",
            (8, height - 8 - _LINE_H),
            WHITE,
        )
    _text(cv2, image, f"arrow = pupil shift x{gain:g}", (8, height - 8), CYAN)
    # The gate labels run along the same rows as the map: each one is cut short before
    # it, and the map is drawn last, so nothing covers its cursor marker.
    labels_end = (view.minimap.rect[0] if view.minimap is not None else width) - _MAP_MARGIN_PX
    for light, center in zip(view.gates, gate_light_centers(), strict=False):
        _dot(cv2, image, center, _LIGHT_RADIUS, GREEN if light.ok else RED)
        org = (center[0] + 12, center[1] + 4)
        label = _fit(cv2, f"{light.label}: {light.reason}", labels_end - org[0])
        if label:
            _text(cv2, image, label, org, WHITE)
    if view.minimap is not None:
        _draw_minimap(cv2, image, view.minimap)


def annotate(
    image: Any,
    landmarks: FaceLandmarks | None,
    cursor_px: ScreenPoint | None = None,
    screen: tuple[int, int] | None = None,
    gain: float = 4.0,
) -> Any:
    """Draw the pupil markers, their shift and (given ``screen``) the mini map on a copy
    of a BGR image, and return the copy. ``cursor_px`` (screen pixels) marks a point on
    the map. There are no gate lights: a still image has no gate decision."""
    cv2 = _import_cv2("annotate()")
    canvas = image.copy()
    view = compute_view(landmarks, _image_size(canvas), None, screen, gain, cursor_px=cursor_px)
    _draw(cv2, canvas, view, gain)
    return canvas


class TrackingView:
    """The "EyeOS tracking" window. Pass ``update`` as the Pipeline's ``on_frame``.

    Pressing q in the window, or closing it, only stops the drawing: the session keeps
    running, and the kill hotkey (or the end of the input) still ends it. The view
    only reads what the gates decided; it never changes it.
    """

    def __init__(
        self,
        screen: tuple[int, int] | None,
        gain: float = 4.0,
        window_name: str = WINDOW_NAME,
    ) -> None:
        self._cv2 = _import_cv2("The tracking view")
        problem = _display_problem(sys.platform, os.environ)
        if problem is not None:
            raise RuntimeError(problem)
        qt_platform = _qt_platform_override(sys.platform, os.environ)
        if qt_platform is not None:
            # Read by Qt when OpenCV creates its first window, which hasn't happened yet.
            os.environ["QT_QPA_PLATFORM"] = qt_platform
        self.screen = screen
        self.gain = gain
        self.window_name = window_name
        self._shown = False
        self._seen_visible = False
        self._closed = False
        # Open the window and close it again at once, so an OpenCV that can't show one
        # fails now, before the session or a calibration starts, not at the first frame.
        # (A Qt that can't start still aborts the process, but before anything else
        # has been opened.) imshow() opens the window again with the first frame.
        try:
            self._cv2.namedWindow(self.window_name)
        except self._cv2.error as e:
            raise _window_error(e) from e
        self._destroy_window()

    @property
    def closed(self) -> bool:
        return self._closed

    def update(self, frame: Frame, frame_input: FrameInput, status: GateStatus) -> None:
        """Draw one frame. Does nothing once the window has been closed.

        ``frame_input`` is part of the Pipeline's on_frame signature; the view draws
        from the frame's landmarks and the gates' status alone."""
        if self._closed:
            return
        cv2 = self._cv2
        canvas = _blank_canvas() if frame.image is None else frame.image.copy()
        view = compute_view(frame.landmarks, _image_size(canvas), status, self.screen, self.gain)
        _draw(cv2, canvas, view, self.gain)
        try:
            cv2.imshow(self.window_name, canvas)
        except cv2.error as e:
            if self._shown:
                raise
            raise _window_error(e) from e
        self._shown = True
        # waitKeyEx, not waitKey: waitKey keeps only the low byte of the key code, and
        # on X11 the Left arrow (keysym 0xFF51) would then read as "Q".
        key = cv2.waitKeyEx(1)
        if key in _QUIT_KEYS:
            self.close()
        elif self._closed_by_user():
            self.close()

    def _closed_by_user(self) -> bool:
        """Whether the window is gone: reported visible once, and no longer.

        Until it has been reported visible, "not visible" means "not on screen yet" (a
        window manager can map it late), and a backend that can't report visibility
        returns -1 throughout. Either way only q closes the view then."""
        cv2 = self._cv2
        try:
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return True  # e.g. Qt: "NULL guiReceiver" once no HighGUI window is left
        if visible:
            self._seen_visible = True
            return False
        return self._seen_visible

    def close(self) -> None:
        """Destroy the window once. Safe to call again, or after the user closed it."""
        if self._closed:
            return
        self._closed = True
        if self._shown:
            self._destroy_window()

    def _destroy_window(self) -> None:
        cv2 = self._cv2
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass  # already gone
        cv2.waitKey(1)  # HighGUI removes the window while it handles events
