"""Fake ``cv2`` and ``mediapipe`` modules, so the capture input sources run in CI
without OpenCV, MediaPipe, a camera or a video file.

Install them per test with ``install_fakes(monkeypatch, cv2=..., mediapipe=...)`` so that
monkeypatch removes them again afterwards. Each fake records what the source did to it.
The OpenCV semantics are hardcoded (by default POS_MSEC is the timestamp of the last frame
read, ``index * 1000 / fps``), so these prove EyeOS's plumbing, not OpenCV's or MediaPipe's
behaviour.

The fake ``cv2`` also records HighGUI drawing and window calls (``drawn`` and
``window_calls``) for the tracking view. It draws nothing: it checks each call's argument
types the way OpenCV parses them (points must be pairs of ints) and keeps the arguments.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

N_FACE_MESH_LANDMARKS = 478  # 468 mesh points + 10 iris points with refine_landmarks=True


@dataclass(frozen=True)
class FakeImage:
    color: str  # "bgr" straight from read(), "rgb" after cvtColor
    index: int
    shape: tuple[int, ...] | None = None  # (height, width, channels), like a numpy image

    def copy(self) -> FakeImage:
        """A new image equal to this one, like ``ndarray.copy()``."""
        return replace(self)


class FakeCv2Error(Exception):
    """Stands in for ``cv2.error``."""


@dataclass
class FakeCv2Call:
    """One drawing call: which function, on which image, with which arguments."""

    name: str
    image: Any
    params: dict[str, Any] = field(default_factory=dict)


def _check_point(name: str, point: object) -> None:
    # OpenCV refuses float coordinates ("Can't parse 'center'. Sequence item with index 0
    # has a wrong type"), so a view that forgot to round must fail here too.
    if not (
        isinstance(point, tuple)
        and len(point) == 2
        and all(isinstance(c, int) and not isinstance(c, bool) for c in point)
    ):
        raise FakeCv2Error(f"Can't parse '{name}': expected a pair of ints, got {point!r}")


def _check_color(color: object) -> None:
    if not (
        isinstance(color, tuple) and len(color) == 3 and all(isinstance(c, int) for c in color)
    ):
        raise FakeCv2Error(f"Can't parse 'color': expected 3 ints (BGR), got {color!r}")


def _check_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise FakeCv2Error(f"Argument '{name}' must be an int, got {value!r}")


class FakeVideoCapture:
    def __init__(
        self,
        source: object,
        *,
        n_frames: int,
        opened: bool,
        fps: float,
        pos_msec: float | Callable[[int], float] | None,
        frame_shape: tuple[int, ...] | None = None,
    ) -> None:
        self.source = source
        self.n_frames = n_frames
        self.opened = opened
        self.fps = fps
        self.pos_msec = pos_msec
        self.frame_shape = frame_shape
        self.frames_read = 0
        self.release_calls = 0
        self.calls: list[str] = []

    def isOpened(self) -> bool:
        self.calls.append("isOpened")
        return self.opened and self.release_calls == 0

    def read(self) -> tuple[bool, FakeImage | None]:
        self.calls.append("read")
        if not self.opened or self.release_calls or self.frames_read >= self.n_frames:
            return False, None
        self.frames_read += 1
        return True, FakeImage("bgr", self.frames_read - 1, self.frame_shape)

    def get(self, prop: int) -> float:
        self.calls.append("get")
        if prop == FakeCv2.CAP_PROP_FPS:
            return self.fps
        if prop != FakeCv2.CAP_PROP_POS_MSEC:
            return 0.0
        last_index = self.frames_read - 1
        if callable(self.pos_msec):
            return float(self.pos_msec(last_index))
        if self.pos_msec is not None:
            return float(self.pos_msec)
        return max(last_index, 0) * 1000.0 / self.fps

    def release(self) -> None:
        self.calls.append("release")
        self.release_calls += 1


class FakeCv2(types.ModuleType):
    CAP_PROP_POS_MSEC = 0
    CAP_PROP_FPS = 5
    COLOR_BGR2RGB = 4
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16
    FILLED = -1
    WND_PROP_VISIBLE = 4
    WINDOW_AUTOSIZE = 1
    error = FakeCv2Error

    def __init__(
        self,
        *,
        n_frames: int = 0,
        opened: bool = True,
        fps: float = 30.0,
        pos_msec: float | Callable[[int], float] | None = None,
        frame_shape: tuple[int, ...] | None = None,
        keys: Sequence[int] = (),
        window_visible: float | Callable[[int], float] = 1.0,
        imshow_error: Exception | None = None,
        named_window_error: Exception | None = None,
        window_property_error: Exception | None = None,
        destroy_error: Exception | None = None,
    ) -> None:
        """``keys``: the key codes successive waitKeyEx() or waitKey() calls see (then
        -1, no key), in full as waitKeyEx() returns them; waitKey() keeps only the low
        byte, as OpenCV does (so the X11 Left arrow, 0xFF51, reads as 0x51, "Q").
        ``window_visible``: what getWindowProperty(WND_PROP_VISIBLE) returns, or a
        function of how many images imshow() has shown so far (a user closing the
        window). The ``*_error`` arguments make that call raise instead."""
        super().__init__("cv2")
        self.n_frames = n_frames
        self.opened = opened
        self.fps = fps
        self.pos_msec = pos_msec
        self.frame_shape = frame_shape
        self.captures: list[FakeVideoCapture] = []
        self.keys = list(keys)
        self.window_visible = window_visible
        self.imshow_error = imshow_error
        self.named_window_error = named_window_error
        self.window_property_error = window_property_error
        self.destroy_error = destroy_error
        self.drawn: list[FakeCv2Call] = []
        # ("namedWindow", name), ("imshow", name, image), ("waitKey", delay),
        # ("waitKeyEx", delay), ("getWindowProperty", name, prop), ("destroyWindow",
        # name), in call order.
        self.window_calls: list[tuple[Any, ...]] = []

    def VideoCapture(self, source: object) -> FakeVideoCapture:
        cap = FakeVideoCapture(
            source,
            n_frames=self.n_frames,
            opened=self.opened,
            fps=self.fps,
            pos_msec=self.pos_msec,
            frame_shape=self.frame_shape,
        )
        self.captures.append(cap)
        return cap

    def cvtColor(self, image: FakeImage, code: int) -> FakeImage:
        assert code == self.COLOR_BGR2RGB and image.color == "bgr"
        return FakeImage("rgb", image.index, image.shape)

    # -- drawing (signatures and keyword names as in OpenCV's Python bindings) ----------
    def circle(self, img, center, radius, color, thickness=1, lineType=8, shift=0):
        _check_point("center", center)
        _check_int("radius", radius)
        _check_color(color)
        _check_int("thickness", thickness)
        self.drawn.append(
            FakeCv2Call(
                "circle",
                img,
                dict(center=center, radius=radius, color=color, thickness=thickness),
            )
        )
        return img

    def line(self, img, pt1, pt2, color, thickness=1, lineType=8, shift=0):
        _check_point("pt1", pt1)
        _check_point("pt2", pt2)
        _check_color(color)
        _check_int("thickness", thickness)
        self.drawn.append(
            FakeCv2Call("line", img, dict(pt1=pt1, pt2=pt2, color=color, thickness=thickness))
        )
        return img

    def arrowedLine(self, img, pt1, pt2, color, thickness=1, line_type=8, shift=0, tipLength=0.1):
        _check_point("pt1", pt1)
        _check_point("pt2", pt2)
        _check_color(color)
        _check_int("thickness", thickness)
        self.drawn.append(
            FakeCv2Call(
                "arrowedLine",
                img,
                dict(pt1=pt1, pt2=pt2, color=color, thickness=thickness, tipLength=tipLength),
            )
        )
        return img

    def rectangle(self, img, pt1, pt2, color, thickness=1, lineType=8, shift=0):
        _check_point("pt1", pt1)
        _check_point("pt2", pt2)
        _check_color(color)
        _check_int("thickness", thickness)
        self.drawn.append(
            FakeCv2Call("rectangle", img, dict(pt1=pt1, pt2=pt2, color=color, thickness=thickness))
        )
        return img

    def putText(
        self,
        img,
        text,
        org,
        fontFace,
        fontScale,
        color,
        thickness=1,
        lineType=8,
        bottomLeftOrigin=False,
    ):
        if not isinstance(text, str):
            raise FakeCv2Error(f"Can't parse 'text': {text!r}")
        _check_point("org", org)
        _check_color(color)
        _check_int("thickness", thickness)
        self.drawn.append(
            FakeCv2Call(
                "putText",
                img,
                dict(text=text, org=org, fontScale=fontScale, color=color, thickness=thickness),
            )
        )
        return img

    def getTextSize(self, text, fontFace, fontScale, thickness):
        """``((width, height), baseline)`` from a fixed metric of 17 * fontScale px per
        character (OpenCV 4.11's Hershey simplex averages about 6.5 px per character at
        scale 0.4 and thickness 3; this gives 6.8), plus the thickness."""
        if not isinstance(text, str):
            raise FakeCv2Error(f"Can't parse 'text': {text!r}")
        _check_int("thickness", thickness)
        width = round(len(text) * 17 * fontScale) + thickness
        return (width, round(25 * fontScale)), round(12 * fontScale)

    # -- HighGUI windows -----------------------------------------------------------------
    @property
    def shown(self) -> list[Any]:
        """Every image imshow() was given, in order."""
        return [call[2] for call in self.window_calls if call[0] == "imshow"]

    def namedWindow(self, winname: str, flags: int = WINDOW_AUTOSIZE) -> None:
        if self.named_window_error is not None:
            raise self.named_window_error
        self.window_calls.append(("namedWindow", winname))

    def imshow(self, winname: str, mat: Any) -> None:
        if self.imshow_error is not None:
            raise self.imshow_error
        self.window_calls.append(("imshow", winname, mat))

    def waitKeyEx(self, delay: int = 0) -> int:
        self.window_calls.append(("waitKeyEx", delay))
        return self.keys.pop(0) if self.keys else -1

    def waitKey(self, delay: int = 0) -> int:
        self.window_calls.append(("waitKey", delay))
        code = self.keys.pop(0) if self.keys else -1
        return code if code == -1 else code & 0xFF

    def getWindowProperty(self, winname: str, prop_id: int) -> float:
        self.window_calls.append(("getWindowProperty", winname, prop_id))
        if self.window_property_error is not None:
            raise self.window_property_error
        if callable(self.window_visible):
            return float(self.window_visible(len(self.shown)))
        return float(self.window_visible)

    def destroyWindow(self, winname: str) -> None:
        self.window_calls.append(("destroyWindow", winname))
        if self.destroy_error is not None:
            raise self.destroy_error


def fake_face(x: float = 0.5, y: float = 0.5, z: float = 0.0) -> types.SimpleNamespace:
    """One face as the legacy Solutions API returns it: ``face.landmark[i].x/.y/.z``."""
    return types.SimpleNamespace(
        landmark=[types.SimpleNamespace(x=x, y=y, z=z) for _ in range(N_FACE_MESH_LANDMARKS)]
    )


class FakeFaceMesh:
    def __init__(self, faces_per_frame: int | Sequence[int], **kwargs: object) -> None:
        self.kwargs = kwargs
        self.faces_per_frame = faces_per_frame
        self.processed: list[FakeImage] = []
        self.close_calls = 0

    def process(self, image: FakeImage) -> types.SimpleNamespace:
        index = len(self.processed)
        self.processed.append(image)
        if isinstance(self.faces_per_frame, int):
            n_faces = self.faces_per_frame
        else:
            n_faces = self.faces_per_frame[index]
        faces = [fake_face() for _ in range(n_faces)]
        return types.SimpleNamespace(multi_face_landmarks=faces or None)

    def close(self) -> None:
        self.close_calls += 1


class FakeMediapipe(types.ModuleType):
    """``solutions=False`` mimics mediapipe>=0.10.30, which dropped the Solutions API."""

    def __init__(
        self,
        *,
        solutions: bool = True,
        version: str | None = "0.10.21",
        faces_per_frame: int | Sequence[int] = 1,
        face_mesh_error: Exception | None = None,
    ) -> None:
        super().__init__("mediapipe")
        if version is not None:
            self.__version__ = version
        self.faces_per_frame = faces_per_frame
        self.face_mesh_error = face_mesh_error
        self.face_meshes: list[FakeFaceMesh] = []
        if solutions:
            self.solutions = types.SimpleNamespace(
                face_mesh=types.SimpleNamespace(FaceMesh=self._new_face_mesh)
            )

    def _new_face_mesh(self, **kwargs: object) -> FakeFaceMesh:
        if self.face_mesh_error is not None:
            raise self.face_mesh_error
        mesh = FakeFaceMesh(self.faces_per_frame, **kwargs)
        self.face_meshes.append(mesh)
        return mesh


def install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cv2: types.ModuleType | None = None,
    mediapipe: types.ModuleType | None = None,
) -> None:
    if cv2 is not None:
        monkeypatch.setitem(sys.modules, "cv2", cv2)
    if mediapipe is not None:
        monkeypatch.setitem(sys.modules, "mediapipe", mediapipe)
