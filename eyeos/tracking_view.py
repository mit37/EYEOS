"""A live view of what EyeOS tracks: a marker on each pupil, its shift from the eye's
corner midpoint (the number EyeOS turns into the cursor), a mini map of the screen with
the cursor on it, and the four gate lights, drawn over the camera image.

``compute_view`` is the pure geometry: it turns one frame's landmarks and gate status
into pixel positions and needs no OpenCV, so it is tested directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from eyeos.features import extract_features
from eyeos.gates import GateStatus, ScreenPoint
from eyeos.landmarks import FaceLandmarks, Point3D

Pixel = tuple[float, float]
CursorState = Literal["allowed", "blocked", "none"]

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
    return EyeView(
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


def _cursor(status: GateStatus | None) -> tuple[ScreenPoint | None, CursorState]:
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
) -> ViewGeometry:
    """Everything the tracking view draws for one frame, in image pixels.

    ``image_size`` and ``screen`` are (width, height). ``gain`` stretches each eye's
    shift arrow, since a real iris moves only a few pixels. Without ``screen`` there is
    no mini map; without ``status`` there are no gate lights and no cursor marker.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {image_size}")
    if screen is not None and (screen[0] <= 0 or screen[1] <= 0):
        raise ValueError(f"screen size must be positive, got {screen}")

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
        point, state = _cursor(status)
        minimap = _minimap((width, height), screen, point, state)

    return ViewGeometry(
        image_size=(width, height),
        left=left,
        right=right,
        iris_offsets=iris_offsets,
        minimap=minimap,
        gates=_gate_lights(status),
    )
