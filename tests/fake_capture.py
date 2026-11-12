"""Fake ``cv2`` and ``mediapipe`` modules, so the capture input sources run in CI
without OpenCV, MediaPipe, a camera or a video file.

Install them per test with ``install_fakes(monkeypatch, cv2=..., mediapipe=...)`` so that
monkeypatch removes them again afterwards. Each fake records what the source did to it.
The OpenCV semantics are hardcoded (by default POS_MSEC is the timestamp of the last frame
read, ``index * 1000 / fps``), so these prove EyeOS's plumbing, not OpenCV's or MediaPipe's
behaviour.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pytest

N_FACE_MESH_LANDMARKS = 478  # 468 mesh points + 10 iris points with refine_landmarks=True


@dataclass(frozen=True)
class FakeImage:
    color: str  # "bgr" straight from read(), "rgb" after cvtColor
    index: int


class FakeVideoCapture:
    def __init__(
        self,
        source: object,
        *,
        n_frames: int,
        opened: bool,
        fps: float,
        pos_msec: float | Callable[[int], float] | None,
    ) -> None:
        self.source = source
        self.n_frames = n_frames
        self.opened = opened
        self.fps = fps
        self.pos_msec = pos_msec
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
        return True, FakeImage("bgr", self.frames_read - 1)

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

    def __init__(
        self,
        *,
        n_frames: int = 0,
        opened: bool = True,
        fps: float = 30.0,
        pos_msec: float | Callable[[int], float] | None = None,
    ) -> None:
        super().__init__("cv2")
        self.n_frames = n_frames
        self.opened = opened
        self.fps = fps
        self.pos_msec = pos_msec
        self.captures: list[FakeVideoCapture] = []

    def VideoCapture(self, source: object) -> FakeVideoCapture:
        cap = FakeVideoCapture(
            source,
            n_frames=self.n_frames,
            opened=self.opened,
            fps=self.fps,
            pos_msec=self.pos_msec,
        )
        self.captures.append(cap)
        return cap

    def cvtColor(self, image: FakeImage, code: int) -> FakeImage:
        assert code == self.COLOR_BGR2RGB and image.color == "bgr"
        return FakeImage("rgb", image.index)


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
