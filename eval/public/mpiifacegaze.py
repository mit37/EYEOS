"""MPIIFaceGaze: reading the dataset and checking its screen model.

Fetch the data first (CC BY-NC-SA 4.0; see docs/PUBLIC_DATASETS.md):

    python scripts/download_mpiifacegaze.py --accept-license

Expected layout (our best knowledge of the format; every reader here fails loudly, naming
the file and showing the offending line or keys, when the data does not fit it):

    MPIIFaceGaze/pXX/pXX.txt                    one line per image, 28 fields
    MPIIFaceGaze/pXX/Calibration/screenSize.mat width_pixel, height_pixel, width_mm, height_mm
    MPIIFaceGaze/pXX/Calibration/monitorPose.mat rvects, tvecs (screen -> camera)

Camera.mat (the camera intrinsics) is not needed: the protocol works on MediaPipe
landmarks and on-screen targets, and the angular error uses only the 3D points the
annotation file already gives.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

DATASET = "MPIIFaceGaze"
LICENSE = "CC BY-NC-SA 4.0"
CITATION = (
    "X. Zhang, Y. Sugano, M. Fritz, A. Bulling. \"It's Written All Over Your Face: "
    'Full-Face Appearance-Based Gaze Estimation." CVPR Workshops (CVPRW), 2017.',
    "X. Zhang, Y. Sugano, M. Fritz, A. Bulling. "
    '"Appearance-Based Gaze Estimation in the Wild." CVPR, 2015.',
)
PERSON_PATTERN = re.compile(r"p\d\d")
N_FIELDS = 28
SCREEN_MODEL_TOLERANCE_MM = 5.0

Vec3 = tuple[float, float, float]


class DatasetFormatError(ValueError):
    """The data on disk does not match the expected MPIIFaceGaze format."""


# ---------------------------------------------------------------------------
# pXX/pXX.txt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Annotation:
    """One line of pXX.txt."""

    image: str  # relative to the person's folder, e.g. day01/0005.jpg
    target_px: tuple[float, float]  # the on-screen gaze target, in screen pixels
    landmarks_px: tuple[float, ...]  # six (x, y) facial landmarks in the image; unused
    head_rvec: Vec3  # the dataset's head pose (Rodrigues); unused, EyeOS estimates its own
    head_tvec: Vec3
    face_center_mm: Vec3  # fc, camera coordinates
    gaze_target_mm: Vec3  # gt, camera coordinates
    eye: str  # 'left' or 'right'


def _vec3(values: list[float]) -> Vec3:
    return (values[0], values[1], values[2])


def parse_annotation_line(line: str, where: str) -> Annotation:
    """Parse one non-blank line; ``where`` (``path:line``) prefixes every error."""
    fields = line.split()
    shown = line.strip()
    if len(fields) != N_FIELDS:
        raise DatasetFormatError(
            f"{where}: expected {N_FIELDS} whitespace-separated fields, got {len(fields)}: "
            f"{shown!r}"
        )
    image = fields[0]
    path = PurePosixPath(image)
    if path.is_absolute() or ".." in path.parts or re.match(r"^[A-Za-z]:", image):
        raise DatasetFormatError(
            f"{where}: field 1 must be a relative image path inside the person's folder, "
            f"got {image!r}: {shown!r}"
        )
    numbers: list[float] = []
    for index, text in enumerate(fields[1:27], start=2):
        try:
            value = float(text)
        except ValueError:
            value = math.nan
        if not math.isfinite(value):
            raise DatasetFormatError(
                f"{where}: field {index} must be a finite number, got {text!r}: {shown!r}"
            )
        numbers.append(value)
    eye = fields[27]
    if eye not in ("left", "right"):
        raise DatasetFormatError(
            f"{where}: field 28 must be 'left' or 'right', got {eye!r}: {shown!r}"
        )
    return Annotation(
        image=image,
        target_px=(numbers[0], numbers[1]),
        landmarks_px=tuple(numbers[2:14]),
        head_rvec=_vec3(numbers[14:17]),
        head_tvec=_vec3(numbers[17:20]),
        face_center_mm=_vec3(numbers[20:23]),
        gaze_target_mm=_vec3(numbers[23:26]),
        eye=eye,
    )


def annotation_path(root: Path, person: str) -> Path:
    return Path(root) / person / f"{person}.txt"


def load_annotations(root: Path, person: str) -> list[Annotation]:
    path = annotation_path(root, person)
    if not path.is_file():
        raise DatasetFormatError(
            f"missing {path}: is --root the extracted MPIIFaceGaze folder "
            "(the one holding p00 ... p14)?"
        )
    annotations = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                annotations.append(parse_annotation_line(line, f"{path}:{line_number}"))
    if not annotations:
        raise DatasetFormatError(f"{path} has no annotations")
    return annotations


def find_persons(root: Path) -> list[str]:
    """The pXX folders under ``root`` that have an annotation file, sorted."""
    root = Path(root)
    if not root.is_dir():
        raise DatasetFormatError(f"{root} is not a directory")
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir()
        and PERSON_PATTERN.fullmatch(p.name)
        and annotation_path(root, p.name).is_file()
    )


# ---------------------------------------------------------------------------
# pXX/Calibration/*.mat
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScreenSize:
    width_px: float
    height_px: float
    width_mm: float
    height_mm: float


@dataclass(frozen=True)
class MonitorPose:
    """Screen -> camera: a point (x_mm, y_mm, 0) on the screen is R(rvec) @ p + tvec."""

    rvec: np.ndarray  # shape (3,)
    tvec: np.ndarray  # shape (3,)


def _load_mat(path: Path) -> dict[str, Any]:
    try:
        from scipy.io import loadmat
    except ImportError as e:  # pragma: no cover - scipy is in the dev and public-eval extras
        raise RuntimeError(
            "reading MPIIFaceGaze's .mat files needs scipy: pip install 'eyeos[public-eval]'"
        ) from e
    if not path.is_file():
        raise DatasetFormatError(f"missing {path}")
    try:
        raw = loadmat(path)
    except Exception as e:  # scipy raises several types for a malformed file
        raise DatasetFormatError(f"{path} is not a readable .mat file: {e}") from e
    return {k: np.squeeze(np.asarray(v)) for k, v in raw.items() if not k.startswith("__")}


def _get(data: dict[str, Any], path: Path, *names: str) -> Any:
    for name in names:
        if name in data:
            return data[name]
    raise DatasetFormatError(
        f"{path} has no {' or '.join(repr(n) for n in names)}; its keys are " f"{sorted(data)}"
    )


def _positive_scalar(data: dict[str, Any], path: Path, name: str) -> float:
    value = np.asarray(_get(data, path, name), dtype=float)
    if value.size != 1 or not math.isfinite(float(value)) or float(value) <= 0:
        raise DatasetFormatError(f"{path}: {name} must be one positive number, got {value!r}")
    return float(value)


def _vector3(data: dict[str, Any], path: Path, *names: str) -> np.ndarray:
    value = np.asarray(_get(data, path, *names), dtype=float).reshape(-1)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise DatasetFormatError(
            f"{path}: {'/'.join(names)} must be 3 numbers, got {value.tolist()!r}"
        )
    return value


def load_screen_size(root: Path, person: str) -> ScreenSize:
    path = Path(root) / person / "Calibration" / "screenSize.mat"
    data = _load_mat(path)
    return ScreenSize(
        width_px=_positive_scalar(data, path, "width_pixel"),
        height_px=_positive_scalar(data, path, "height_pixel"),
        width_mm=_positive_scalar(data, path, "width_mm"),
        height_mm=_positive_scalar(data, path, "height_mm"),
    )


def load_monitor_pose(root: Path, person: str) -> MonitorPose:
    path = Path(root) / person / "Calibration" / "monitorPose.mat"
    data = _load_mat(path)
    return MonitorPose(
        rvec=_vector3(data, path, "rvects", "rvecs"),
        tvec=_vector3(data, path, "tvecs", "tvects"),
    )


# ---------------------------------------------------------------------------
# The screen model: on-screen pixels -> camera coordinates (mm)
# ---------------------------------------------------------------------------


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Rotation matrix for an axis-angle vector (OpenCV's Rodrigues), in numpy."""
    r = np.asarray(rvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * kx + (1.0 - math.cos(theta)) * (kx @ kx)


@dataclass(frozen=True)
class ScreenModelVariant:
    """One reading of the screen -> camera transform. ``default`` is the documented one:
    R(rvects) @ [x_px * width_mm / width_pixel, y_px * height_mm / height_pixel, 0] + tvecs.
    The others flip an axis, transpose R or negate t, in case the files mean something
    slightly different."""

    mirror_x: bool = False
    mirror_y: bool = False
    transpose_r: bool = False
    negate_t: bool = False

    @property
    def name(self) -> str:
        parts = [
            label
            for flag, label in (
                (self.mirror_x, "mirror-x"),
                (self.mirror_y, "mirror-y"),
                (self.transpose_r, "transpose-R"),
                (self.negate_t, "negate-t"),
            )
            if flag
        ]
        return "+".join(parts) or "default"


SCREEN_MODELS: dict[str, ScreenModelVariant] = {
    v.name: v
    for v in (
        ScreenModelVariant(mx, my, tr, nt)
        for tr in (False, True)
        for nt in (False, True)
        for my in (False, True)
        for mx in (False, True)
    )
}


def screen_to_camera(
    points_px: np.ndarray, screen: ScreenSize, pose: MonitorPose, model: str = "default"
) -> np.ndarray:
    """Map on-screen pixels (n, 2) to camera coordinates in mm (n, 3)."""
    variant = SCREEN_MODELS[model]
    points = np.asarray(points_px, dtype=float).reshape(-1, 2)
    x_mm = points[:, 0] * screen.width_mm / screen.width_px
    y_mm = points[:, 1] * screen.height_mm / screen.height_px
    if variant.mirror_x:
        x_mm = screen.width_mm - x_mm
    if variant.mirror_y:
        y_mm = screen.height_mm - y_mm
    local = np.column_stack([x_mm, y_mm, np.zeros(len(points))])
    rotation = rodrigues(pose.rvec)
    if variant.transpose_r:
        rotation = rotation.T
    tvec = -pose.tvec if variant.negate_t else pose.tvec
    result: np.ndarray = local @ rotation.T + tvec
    return result


@dataclass(frozen=True)
class ScreenModelCheck:
    model: str | None  # the screen model to use for angles; None if none fits
    median_error_mm: float  # of the chosen model (the best candidate if none fits)
    default_median_error_mm: float
    candidates_median_mm: dict[str, float]  # filled in only when the default misses
    fits: bool
    reason: str | None


def _median_error_mm(
    model: str, targets: np.ndarray, gts: np.ndarray, screen: ScreenSize, pose: MonitorPose
) -> float:
    predicted = screen_to_camera(targets, screen, pose, model)
    return float(np.median(np.linalg.norm(predicted - gts, axis=1)))


def check_screen_model(
    targets_px: np.ndarray,
    gaze_targets_mm: np.ndarray,
    screen: ScreenSize,
    pose: MonitorPose,
    tolerance_mm: float = SCREEN_MODEL_TOLERANCE_MM,
) -> ScreenModelCheck:
    """Rebuild each 3D gaze target from its on-screen pixel and compare with the file.

    The documented model is used if its median error is within ``tolerance_mm``.
    Otherwise every variant in SCREEN_MODELS is tried and the best one is used if it is
    within tolerance. If none is, angular errors cannot be trusted and are not reported.
    """
    targets = np.asarray(targets_px, dtype=float).reshape(-1, 2)
    gts = np.asarray(gaze_targets_mm, dtype=float).reshape(-1, 3)
    default_error = _median_error_mm("default", targets, gts, screen, pose)
    if default_error <= tolerance_mm:
        return ScreenModelCheck("default", default_error, default_error, {}, True, None)
    candidates = {
        name: _median_error_mm(name, targets, gts, screen, pose) for name in SCREEN_MODELS
    }
    best = min(candidates, key=lambda name: (candidates[name], name))
    if candidates[best] <= tolerance_mm:
        return ScreenModelCheck(best, candidates[best], default_error, candidates, True, None)
    reason = (
        f"no screen model reproduces the annotated 3D gaze targets within {tolerance_mm:g} mm "
        f"(best: {best}, median {candidates[best]:.1f} mm), so angular error is not reported"
    )
    return ScreenModelCheck(None, candidates[best], default_error, candidates, False, reason)
