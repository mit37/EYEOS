"""A tiny fake MPIIFaceGaze on disk, and a fake face detector that reads it.

The layout matches what eval/public/mpiifacegaze.py expects of the real dataset
(``pXX/pXX.txt`` with 28 fields a line, ``pXX/Calibration/*.mat``), but each "image" is
a small JSON file naming the iris offset to draw. ``FakeDetector`` turns that into
``eval.generate_synthetic.make_landmarks``, so a test controls exactly what the
"MediaPipe" output is and no face images, OpenCV or MediaPipe are involved.

The 3D gaze targets are written with explicit rotation matrices about one axis, not
with the harness's own Rodrigues code, so the screen-model check is not checked against
itself.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eval.generate_synthetic import make_landmarks
from eyeos.landmarks import FaceLandmarks, Point3D


def rotation_about_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def offsets_for_target(x: float, y: float, width: float, height: float) -> tuple[float, float]:
    """The noiseless iris offset for an on-screen target: linear in the screen fraction,
    so EyeOS's affine calibration can recover it exactly."""
    return (x - width / 2) / (width / 2), (y - height / 2) / (height / 2)


@dataclass
class FakePerson:
    name: str
    screen_px: tuple[int, int] = (1280, 800)
    screen_mm: tuple[float, float] = (331.0, 207.0)
    # The screen hangs below the camera and faces the user: roughly a half turn about y.
    theta: float = math.pi - 0.15
    tvec: tuple[float, float, float] = (165.0, 9.0, 4.0)
    face_center: tuple[float, float, float] = (12.0, 70.0, 560.0)
    n_images: int = 240
    seed: int = 0
    noise: float = 0.0  # Gaussian sigma added to each iris offset
    iris_bias: tuple[float, float] = (0.0, 0.0)  # added to every iris offset: a per-user quirk
    faces: dict[int, int] = field(default_factory=dict)  # image index -> face count
    closed: set[int] = field(default_factory=set)  # image indices with both eyes shut
    mirror_gt_x: bool = False  # write gt as if the screen's x axis were mirrored
    gt_offset_mm: float = 0.0  # added to every gt coordinate: a screen model that fits nothing
    write_mat: bool = True
    targets: list[tuple[float, float]] = field(default_factory=list)

    def gaze_target_mm(self, x: float, y: float) -> np.ndarray:
        w_px, h_px = self.screen_px
        w_mm, h_mm = self.screen_mm
        x_mm = x * w_mm / w_px
        if self.mirror_gt_x:
            x_mm = w_mm - x_mm
        local = np.array([x_mm, y * h_mm / h_px, 0.0])
        return rotation_about_y(self.theta) @ local + np.array(self.tvec) + self.gt_offset_mm


def annotation_line(
    image: str,
    target: tuple[float, float],
    face_center: tuple[float, float, float],
    gaze_target: tuple[float, float, float],
    eye: str = "left",
) -> str:
    landmarks = [500.0 + 10 * i for i in range(12)]  # six (x, y) points, unused by EyeOS
    head_rvec = [0.1, -0.2, 0.05]
    head_tvec = [-20.0, 30.0, 600.0]
    numbers = [*target, *landmarks, *head_rvec, *head_tvec, *face_center, *gaze_target]
    return " ".join([image, *(repr(float(v)) for v in numbers), eye])


def write_person(root: Path, person: FakePerson) -> FakePerson:
    rng = random.Random(person.seed)
    w, h = person.screen_px
    folder = root / person.name
    (folder / "day01").mkdir(parents=True, exist_ok=True)
    lines = []
    person.targets = []
    for i in range(person.n_images):
        x, y = round(rng.uniform(0, w - 1), 1), round(rng.uniform(0, h - 1), 1)
        person.targets.append((x, y))
        image = f"day01/{i:04d}.jpg"
        ox, oy = offsets_for_target(x, y, w, h)
        ox, oy = ox + person.iris_bias[0], oy + person.iris_bias[1]
        if person.noise:
            ox += rng.gauss(0.0, person.noise)
            oy += rng.gauss(0.0, person.noise)
        spec = {
            "ox": ox,
            "oy": oy,
            "faces": person.faces.get(i, 1),
            "closed": i in person.closed,
        }
        (folder / image).write_text(json.dumps(spec))
        gt = person.gaze_target_mm(x, y)
        lines.append(
            annotation_line(
                image, (x, y), person.face_center, (float(gt[0]), float(gt[1]), float(gt[2]))
            )
        )
    (folder / f"{person.name}.txt").write_text("\n".join(lines) + "\n")
    if person.write_mat:
        write_calibration(folder / "Calibration", person)
    return person


def write_calibration(folder: Path, person: FakePerson) -> None:
    from scipy.io import savemat

    folder.mkdir(parents=True, exist_ok=True)
    w_px, h_px = person.screen_px
    w_mm, h_mm = person.screen_mm
    savemat(
        folder / "screenSize.mat",
        {
            "width_pixel": np.array([[float(w_px)]]),
            "height_pixel": np.array([[float(h_px)]]),
            "width_mm": np.array([[w_mm]]),
            "height_mm": np.array([[h_mm]]),
        },
    )
    savemat(
        folder / "monitorPose.mat",
        {
            "rvects": np.array([[0.0], [person.theta], [0.0]]),
            "tvecs": np.array(person.tvec, dtype=float).reshape(3, 1),
        },
    )
    savemat(
        folder / "Camera.mat",
        {"cameraMatrix": np.eye(3), "distCoeffs": np.zeros((1, 5))},
    )


def write_dataset(root: Path, persons: list[FakePerson]) -> list[FakePerson]:
    return [write_person(root, p) for p in persons]


class FakeDetector:
    """Stands in for MediaPipe FaceMesh on a fake image: returns make_landmarks for the
    image's iris offset, ``None`` for zero faces, and shut lids when ``closed``."""

    def __init__(self) -> None:
        self.calls: list[Path] = []
        self.closed = False

    def __call__(self, image_path: Path) -> FaceLandmarks | None:
        self.calls.append(image_path)
        spec = json.loads(Path(image_path).read_text())
        if spec["faces"] == 0:
            return None
        landmarks = make_landmarks(spec["ox"], spec["oy"], face_count=spec["faces"])
        if spec["closed"]:
            landmarks = dataclasses.replace(
                landmarks,
                left_eye_top=Point3D(0.38, 0.40),
                left_eye_bottom=Point3D(0.38, 0.40),
                right_eye_top=Point3D(0.62, 0.40),
                right_eye_bottom=Point3D(0.62, 0.40),
            )
        return landmarks

    def close(self) -> None:
        self.closed = True


def fake_detector_factory() -> FakeDetector:
    """Module-level, so a process pool can pickle it by reference."""
    return FakeDetector()
