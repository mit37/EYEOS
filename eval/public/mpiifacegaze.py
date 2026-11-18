"""MPIIFaceGaze: run EyeOS's landmark pipeline and calibration on a public gaze dataset.

Fetch the data first (CC BY-NC-SA 4.0; see docs/PUBLIC_DATASETS.md), then extract
MediaPipe landmarks once (slow, cached, restartable) and evaluate (fast, repeatable):

    python scripts/download_mpiifacegaze.py --accept-license
    python -m eval.public.mpiifacegaze extract --root datasets/MPIIFaceGaze \\
        --cache datasets/cache/mpiifacegaze [--persons p00,p01] [--limit N] [--workers N]
    python -m eval.public.mpiifacegaze evaluate --root datasets/MPIIFaceGaze \\
        --cache datasets/cache/mpiifacegaze --out eval/results/mpiifacegaze.json \\
        [--variants NAMES]

The protocol itself (eval/public/protocol.py) and the models it compares
(eval/public/models.py) know nothing about MPIIFaceGaze; this module only reads the
dataset, checks its screen model and caches the landmarks.

Expected layout (checked against the real files on 2026-11-18; every reader here fails
loudly, naming the file and showing the offending line or keys, when the data does not
fit it):

    MPIIFaceGaze/pXX/pXX.txt                    one line per image, 28 fields
    MPIIFaceGaze/pXX/Calibration/screenSize.mat width_pixel, height_pixel, width_mm, height_mm
    MPIIFaceGaze/pXX/Calibration/monitorPose.mat rvects, tvecs (screen -> camera)

Camera.mat (the camera intrinsics) is not needed: the protocol works on MediaPipe
landmarks and on-screen targets, and the angular error uses only the 3D points the
annotation file already gives.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from eval.public import protocol
from eval.public.models import VARIANT_NAMES, build_variants
from eyeos.config import Config
from eyeos.input_sources._mediapipe_common import landmarks_from_mediapipe
from eyeos.landmarks import FaceLandmarks
from eyeos.synthetic_format import CSV_FIELDNAMES, landmarks_to_row, row_to_frame

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


class MissingDependencyError(RuntimeError):
    """An optional library (scipy, OpenCV, MediaPipe) is missing or too new."""


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
        raise MissingDependencyError(
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
    slightly different, and ``inverse_pose`` reads the pose as camera -> screen, so a
    screen point p is at R^T (p - t)."""

    mirror_x: bool = False
    mirror_y: bool = False
    transpose_r: bool = False
    negate_t: bool = False
    inverse_pose: bool = False

    @property
    def name(self) -> str:
        parts = [
            label
            for flag, label in (
                (self.mirror_x, "mirror-x"),
                (self.mirror_y, "mirror-y"),
                (self.transpose_r, "transpose-R"),
                (self.negate_t, "negate-t"),
                (self.inverse_pose, "inverse-pose"),
            )
            if flag
        ]
        return "+".join(parts) or "default"


SCREEN_MODELS: dict[str, ScreenModelVariant] = {
    v.name: v
    for v in (
        ScreenModelVariant(mx, my, tr, nt, ip)
        for ip in (False, True)
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
    if variant.inverse_pose:
        inverse: np.ndarray = (local - tvec) @ rotation  # each row: R^T (p - t)
        return inverse
    result: np.ndarray = local @ rotation.T + tvec  # each row: R p + t
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


# ---------------------------------------------------------------------------
# extract: images -> landmarks, cached as one CSV per person
# ---------------------------------------------------------------------------

# eyeos.synthetic_format's columns (so eyeos.synthetic_format.row_to_frame reads a row back
# as the FaceLandmarks the live pipeline would have seen), plus the dataset's labels.
CACHE_COLUMNS: list[str] = [
    *CSV_FIELDNAMES,
    "image",
    "target_x",
    "target_y",
    "fc_x",
    "fc_y",
    "fc_z",
    "gt_x",
    "gt_y",
    "gt_z",
]

Detector = Callable[[Path], "FaceLandmarks | None"]
DetectorFactory = Callable[[], Detector]

_MISSING_DEPS = (
    "extracting landmarks needs OpenCV and MediaPipe: pip install 'eyeos[public-eval]' "
    "(Python 3.10-3.12)"
)


def face_mesh_settings(
    min_detection_confidence: float = Config().gates.presence_confidence_threshold,
    max_num_faces: int = 2,
) -> dict[str, Any]:
    """The FaceMesh arguments MediaPipeImageDetector runs with; the cache manifest
    records them."""
    return {
        "static_image_mode": True,  # every photo is detected from scratch, no tracking
        "max_num_faces": max_num_faces,
        "refine_landmarks": True,  # required for the iris landmarks
        "min_detection_confidence": min_detection_confidence,
    }


class MediaPipeImageDetector:
    """One still image -> FaceLandmarks, the way the live webcam source does it.

    Uses the same legacy FaceMesh as eyeos/input_sources/_capture_base.py, but with
    ``static_image_mode=True`` (every photo is detected from scratch, no tracking), and
    the same conversion (landmarks_from_mediapipe with x rescaled by width / height).
    Faces below Gate 1's confidence threshold are dropped inside MediaPipe, as live.
    """

    def __init__(
        self,
        min_detection_confidence: float = Config().gates.presence_confidence_threshold,
        max_num_faces: int = 2,
    ) -> None:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as e:
            raise MissingDependencyError(_MISSING_DEPS) from e
        if not hasattr(mp, "solutions"):
            version = getattr(mp, "__version__", None) or "?"
            raise MissingDependencyError(
                f"installed mediapipe {version} has no legacy Solutions API (FaceMesh); "
                "it needs mediapipe<0.10.30: pip install 'eyeos[public-eval]'"
            )
        self._cv2: Any = cv2
        self.min_detection_confidence = min_detection_confidence
        self._mesh: Any = mp.solutions.face_mesh.FaceMesh(
            **face_mesh_settings(min_detection_confidence, max_num_faces)
        )

    def __call__(self, image_path: Path) -> FaceLandmarks | None:
        image = self._cv2.imread(str(image_path))
        if image is None:
            raise DatasetFormatError(f"cannot read image {image_path}")
        rgb = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        faces = self._mesh.process(rgb).multi_face_landmarks or []
        if not faces:
            return None
        return landmarks_from_mediapipe(
            faces[0].landmark,
            face_count=len(faces),
            detection_confidence=self.min_detection_confidence,
            aspect=width / height,
        )

    def close(self) -> None:
        self._mesh.close()


@dataclass(frozen=True)
class ExtractSummary:
    person: str
    images: int  # rows in the cache when extract_person returns
    with_face: int
    skipped: bool
    path: Path
    note: str | None = None  # why an existing cache that is not exactly the request was kept


def cache_path(cache_dir: Path, person: str) -> Path:
    return Path(cache_dir) / f"{person}.csv"


def _cached_images(path: Path) -> list[str] | None:
    """The image column of a complete cache file, or None if there is no usable one."""
    if not path.is_file():
        return None
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CACHE_COLUMNS:
            return None
        return [row["image"] for row in reader]


def _with_face(path: Path) -> int:
    with path.open(newline="") as f:
        return sum(1 for row in csv.DictReader(f) if int(row["face_count"]) > 0)


def _fmt(value: float) -> str:
    return repr(float(value))


def extract_person(
    root: Path,
    person: str,
    cache_dir: Path,
    detector_factory: DetectorFactory,
    limit: int | None = None,
    force: bool = False,
) -> ExtractSummary:
    """Detect landmarks in one person's images and cache them as ``<cache>/<person>.csv``.

    The file is written under a temporary name and renamed when complete, so an
    interrupted run never leaves a cache that looks finished. Unless ``force`` is set, an
    existing complete cache is kept when it starts with the requested images (the first
    ``limit``, or all), and is never replaced by a shorter one: a quick ``limit`` trial
    run after a full extraction must not throw away hours of detection.
    """
    annotations = load_annotations(root, person)[:limit]
    requested = [a.image for a in annotations]
    out = cache_path(cache_dir, person)
    cached = None if force else _cached_images(out)
    if cached is not None:
        note = None
        if len(cached) > len(requested):
            n, m = len(cached), len(requested)
            if cached[:m] == requested:
                note = (
                    f"kept the cache of {n} images, which starts with the {m} requested "
                    "(--force replaces it)"
                )
            else:
                note = (
                    f"kept the cache of {n} images rather than replace it with {m}, although "
                    f"its images differ from the first {m} in {person}.txt (--force replaces it)"
                )
        if cached == requested or note is not None:
            return ExtractSummary(person, len(cached), _with_face(out), True, out, note)

    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    detector = detector_factory()
    with_face = 0
    try:
        with partial.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CACHE_COLUMNS)
            writer.writeheader()
            for index, annotation in enumerate(annotations):
                landmarks = detector(Path(root) / person / annotation.image)
                with_face += landmarks is not None
                row = landmarks_to_row(float(index), landmarks)
                row["image"] = annotation.image
                row["target_x"], row["target_y"] = map(_fmt, annotation.target_px)
                for prefix, vec in (
                    ("fc", annotation.face_center_mm),
                    ("gt", annotation.gaze_target_mm),
                ):
                    for axis, value in zip("xyz", vec, strict=True):
                        row[f"{prefix}_{axis}"] = _fmt(value)
                writer.writerow(row)
    finally:
        close = getattr(detector, "close", None)
        if close is not None:
            close()
    partial.replace(out)
    return ExtractSummary(person, len(annotations), with_face, False, out)


def _extract_person_job(args: tuple[Any, ...]) -> ExtractSummary:
    return extract_person(*args)


# ---------------------------------------------------------------------------
# library versions and the cache manifest
# ---------------------------------------------------------------------------

OPENCV_DISTRIBUTIONS = (
    "opencv-contrib-python",
    "opencv-python",
    "opencv-contrib-python-headless",
    "opencv-python-headless",
)
MANIFEST_NAME = "manifest.json"
MANIFEST_NOTE = (
    "What each person's landmark cache was extracted with, written by "
    "`python -m eval.public.mpiifacegaze extract` when it (re)extracts that person; "
    "`evaluate` copies the evaluated persons' entries into its results."
)


def package_version(distribution: str) -> str | None:
    """The installed version of a distribution (importlib.metadata); None if absent."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def opencv_version() -> tuple[str | None, str | None]:
    """(distribution, version) of the first installed OpenCV wheel; (None, None) if none."""
    for distribution in OPENCV_DISTRIBUTIONS:
        version = package_version(distribution)
        if version is not None:
            return distribution, version
    return None, None


def _qualified_name(factory: object) -> str:
    module = getattr(factory, "__module__", None) or type(factory).__module__
    if module == "__main__":  # python -m eval.public.mpiifacegaze
        spec = getattr(sys.modules.get("__main__"), "__spec__", None)
        module = spec.name if spec is not None else module
    name = getattr(factory, "__qualname__", None) or type(factory).__qualname__
    return f"{module}.{name}"


def manifest_entry(detector_factory: DetectorFactory, summary: ExtractSummary) -> dict[str, Any]:
    """What one person's cache was just extracted with. No paths: evaluate copies it into
    a results file that may be committed."""
    distribution, opencv = opencv_version()
    return {
        "extracted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "images": summary.images,
        "with_face": summary.with_face,
        "detector": _qualified_name(detector_factory),
        "face_mesh": face_mesh_settings() if detector_factory is MediaPipeImageDetector else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {"mediapipe": package_version("mediapipe"), "opencv": opencv},
        "opencv_distribution": distribution,
    }


def manifest_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / MANIFEST_NAME


def read_manifest(cache_dir: Path) -> dict[str, Any] | None:
    """``<cache>/manifest.json``, or None for a cache without one (extracted before
    extract wrote manifests)."""
    path = manifest_path(cache_dir)
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise DatasetFormatError(
            f"{path} is not valid JSON ({e}); delete it, or rerun `extract --force`"
        ) from e
    if not isinstance(manifest, dict) or not isinstance(manifest.get("persons"), dict):
        raise DatasetFormatError(
            f"{path} has no 'persons' mapping; delete it, or rerun `extract --force`"
        )
    return manifest


def _record_in_manifest(cache_dir: Path, person: str, entry: dict[str, Any]) -> None:
    manifest = read_manifest(cache_dir) or {"note": MANIFEST_NOTE, "persons": {}}
    manifest["persons"][person] = entry
    manifest["persons"] = dict(sorted(manifest["persons"].items()))
    path = manifest_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(manifest, indent=2) + "\n")
    partial.replace(path)


def extract(
    root: Path,
    cache_dir: Path,
    persons: Sequence[str] | None = None,
    limit: int | None = None,
    workers: int = 1,
    force: bool = False,
    detector_factory: DetectorFactory = MediaPipeImageDetector,
) -> list[ExtractSummary]:
    """Extract every person (default: all pXX under ``root``), in parallel across persons
    when ``workers > 1``. Each worker builds its own detector from ``detector_factory``,
    which must therefore be picklable (a module-level class or function).

    As each person's cache is written, ``<cache>/manifest.json`` records what it was
    extracted with (manifest_entry); persons kept from an earlier run keep their entry.
    """
    names = list(persons) if persons is not None else find_persons(root)
    jobs = [(Path(root), p, Path(cache_dir), detector_factory, limit, force) for p in names]

    def record(summary: ExtractSummary) -> ExtractSummary:
        if not summary.skipped:
            _record_in_manifest(
                cache_dir, summary.person, manifest_entry(detector_factory, summary)
            )
        return summary

    if workers <= 1 or len(jobs) <= 1:
        return [record(_extract_person_job(job)) for job in jobs]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = [pool.submit(_extract_person_job, job) for job in jobs]
        for future in as_completed(futures):
            if future.exception() is None:
                record(future.result())
        return [future.result() for future in futures]  # raises the first failure


# ---------------------------------------------------------------------------
# evaluate: the cached landmarks through the EyeOS protocol (eval/public/protocol.py)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

NOTES = {
    "static_photos": (
        "MPIIFaceGaze is still photos, not live video, so the One Euro filter, dwell (Gate 3) "
        "and click confirmation are not exercised: every error is one frame's calibration "
        "output, rounded and clamped to the screen like the cursor pixel."
    ),
    "calibration_set": (
        "For each of EyeOS's 9 calibration dots, the K valid images whose true target is "
        "nearest the dot stand in for looking at it, and are added with their true targets. "
        "calibration_mean_distance_px says how far from the dots those images were."
    ),
    "screens": (
        "Each participant's own laptop screen (screenSize.mat) is the screen, not the user's: "
        "pixel errors are in that screen's pixels, and % of the diagonal compares across "
        "screens."
    ),
    "valid_frames": (
        "valid = exactly one face and both irises visible, what Gate 1 needs; detection_rate "
        "= valid / images. Invalid frames are in neither the calibration nor the test set."
    ),
    "test_sets": (
        "the test set is every valid image not used for calibration, so a variant with more "
        "images per dot is tested on slightly fewer images."
    ),
    "angular_error": (
        "the angle between (P_pred - fc) and (gt - fc), with the predicted pixel P_pred "
        "mapped to camera coordinates by the screen model recorded under screen_model; null "
        "for a person whose screen model does not reproduce the annotated gt within the "
        "tolerance."
    ),
    "head_pose": (
        "EyeOS's own simplified head pose (eyeos.features), not the dataset's: the share of "
        "test frames within Gate 2's head-pose limit of the calibration pose."
    ),
    "validation_error_px": (
        "Calibration's own leave-one-out RMS error for the eyeos.calibration variants, and the "
        "same leave-one-out definition for the other models; Gate 2 passes below the limit."
    ),
}


def read_cache(cache_dir: Path, person: str) -> list[dict[str, str]]:
    path = cache_path(cache_dir, person)
    if not path.is_file():
        raise DatasetFormatError(f"missing {path}: run extract for {person} first")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CACHE_COLUMNS:
            raise DatasetFormatError(
                f"{path} is not an extract cache in this version's format (its columns start "
                f"{list(reader.fieldnames or [])[:4]}); rerun `extract --force --persons {person}`"
            )
        rows = list(reader)
    if not rows:
        raise DatasetFormatError(f"{path} has no rows; rerun `extract --force --persons {person}`")
    return rows


def cached_persons(cache_dir: Path) -> list[str]:
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return []
    return sorted(
        p.stem for p in cache_dir.glob("*.csv") if p.is_file() and PERSON_PATTERN.fullmatch(p.stem)
    )


def _floats(rows: list[dict[str, str]], columns: Sequence[str]) -> np.ndarray:
    return np.array([[float(row[c]) for c in columns] for row in rows], dtype=float)


def load_person(
    root: Path, cache_dir: Path, person: str
) -> tuple[protocol.PersonData, ScreenModelCheck]:
    """One person's cached landmarks and labels, plus their checked screen model."""
    rows = read_cache(cache_dir, person)
    screen = load_screen_size(root, person)
    pose = load_monitor_pose(root, person)
    targets = _floats(rows, ("target_x", "target_y"))
    face_centers = _floats(rows, ("fc_x", "fc_y", "fc_z"))
    gaze_targets = _floats(rows, ("gt_x", "gt_y", "gt_z"))
    check = check_screen_model(targets, gaze_targets, screen, pose)

    to_camera = None
    if check.model is not None:
        model = check.model

        def to_camera(points_px: np.ndarray) -> np.ndarray:
            return screen_to_camera(points_px, screen, pose, model)

    data = protocol.PersonData(
        person=person,
        screen_px=(screen.width_px, screen.height_px),
        targets_px=targets,
        landmarks=[row_to_frame(row)[1] for row in rows],
        face_centers_mm=face_centers,
        gaze_targets_mm=gaze_targets,
        to_camera_mm=to_camera,
        angular_note=check.reason,
    )
    return data, check


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def environment(cache_dir: Path, persons: Sequence[str]) -> dict[str, Any]:
    """Where evaluate runs (Python, platform, library versions; None = not installed),
    and under ``extract`` what each evaluated person's cache was extracted with, from
    ``<cache>/manifest.json``: None for a cache without a manifest, and None for a person
    the manifest does not list."""
    distribution, opencv = opencv_version()
    manifest = read_manifest(cache_dir)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "opencv": opencv,
            "mediapipe": package_version("mediapipe"),
        },
        "opencv_distribution": distribution,
        "extract": (None if manifest is None else {p: manifest["persons"].get(p) for p in persons}),
    }


def evaluate_dataset(
    root: Path,
    cache_dir: Path,
    persons: Sequence[str] | None = None,
    variants: Sequence[str] | None = None,
    k_per_dot: int = protocol.DEFAULT_K_PER_DOT,
) -> dict[str, Any]:
    """Run the protocol on the cached persons (default: every pXX.csv in ``cache_dir``)
    for the named variants (default: all) and return the results document."""
    names = list(persons) if persons is not None else cached_persons(cache_dir)
    if not names:
        raise DatasetFormatError(
            f"no pXX.csv caches in {cache_dir}: run `python -m eval.public.mpiifacegaze "
            "extract` first"
        )
    registry = build_variants()
    chosen = list(variants) if variants is not None else list(VARIANT_NAMES)
    unknown = [v for v in chosen if v not in registry]
    if unknown:
        raise ValueError(f"unknown variant(s) {unknown}; choose from {list(VARIANT_NAMES)}")

    loaded = [load_person(root, cache_dir, name) for name in names]
    limits = protocol.Limits()
    results = protocol.evaluate(
        [data for data, _ in loaded], [registry[v] for v in chosen], k_per_dot, limits
    )
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "dataset": DATASET,
        "license": LICENSE,
        "citation": list(CITATION),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if dirty is None else bool(dirty),
        "environment": environment(cache_dir, names),
        "config": {
            "root": str(root),
            "cache": str(cache_dir),
            "persons": names,
            "variants": chosen,
            "k_per_dot": k_per_dot,
            "calibration_grid": "eyeos.calibration.target_points: 3x3 at 10/50/90% of the screen",
            "gate2_max_validation_error_px": limits.gate2_max_error_px,
            "max_head_pose_delta_deg": limits.max_head_pose_delta_deg,
            "screen_model_tolerance_mm": SCREEN_MODEL_TOLERANCE_MM,
        },
        "screen_model": {data.person: asdict(check) for data, check in loaded},
        "detection": results["detection"],
        "variants": results["variants"],
        "notes": NOTES,
    }


def _fmt_or_dash(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def summary_table(results: dict[str, Any]) -> str:
    """A short text table of the per-variant aggregates, for the terminal."""
    lines = [
        f"{'variant':<16}{'persons':>8}{'mean px':>10}{'% diag':>8}{'mean deg':>10}{'gate2':>8}"
    ]
    for name, variant in results["variants"].items():
        agg = variant["aggregate"]
        px = agg["pixel_error_px"] or {}
        deg = agg["angular_error_deg"] or {}
        gate2 = f"{agg['gate2_passed']}/{agg['persons_evaluated']}"
        lines.append(
            f"{name:<16}{agg['persons_evaluated']:>8}"
            f"{_fmt_or_dash(px.get('mean_of_person_means'), '.1f'):>10}"
            f"{_fmt_or_dash(agg['pixel_error_pct_diagonal_mean_of_person_means'], '.2f'):>8}"
            f"{_fmt_or_dash(deg.get('mean_of_person_means'), '.2f'):>10}"
            f"{gate2:>8}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


def _persons_arg(
    available: list[str], text: str | None, parser: argparse.ArgumentParser, where: str
) -> list[str] | None:
    """The --persons list, checked against what is available; None means all."""
    if text is None:
        return None
    names = [p.strip() for p in text.split(",") if p.strip()]
    unknown = [p for p in names if p not in available]
    if unknown:
        parser.error(
            f"--persons: {', '.join(unknown)} not found in {where} "
            f"(available: {', '.join(available) or 'none'})"
        )
    return names


def _variants_arg(text: str | None, parser: argparse.ArgumentParser) -> list[str] | None:
    if text is None:
        return None
    names = [v.strip() for v in text.split(",") if v.strip()]
    unknown = [v for v in names if v not in VARIANT_NAMES]
    if unknown:
        parser.error(
            f"--variants: unknown {', '.join(unknown)} (choose from {', '.join(VARIANT_NAMES)})"
        )
    return names


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.public.mpiifacegaze",
        description="Evaluate EyeOS's calibration on MPIIFaceGaze (see docs/PUBLIC_DATASETS.md).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ext = sub.add_parser("extract", help="detect MediaPipe landmarks once and cache them")
    ext.add_argument("--root", required=True, type=Path, help="the extracted MPIIFaceGaze folder")
    ext.add_argument("--cache", required=True, type=Path, help="where the per-person CSVs go")
    ext.add_argument("--persons", help="comma-separated, e.g. p00,p01 (default: all)")
    ext.add_argument("--limit", type=_positive_int, help="only the first N images per person")
    ext.add_argument("--workers", type=_positive_int, default=1, help="persons in parallel")
    ext.add_argument("--force", action="store_true", help="re-extract cached persons")

    ev = sub.add_parser("evaluate", help="run the EyeOS protocol on the cached landmarks")
    ev.add_argument("--root", required=True, type=Path, help="the extracted MPIIFaceGaze folder")
    ev.add_argument("--cache", required=True, type=Path, help="the extract cache folder")
    ev.add_argument("--out", required=True, type=Path, help="results JSON to write")
    ev.add_argument(
        "--variants", help=f"comma-separated (default: all): {', '.join(VARIANT_NAMES)}"
    )
    ev.add_argument("--persons", help="comma-separated, e.g. p00,p01 (default: all cached)")
    ev.add_argument(
        "--k-per-dot",
        type=_positive_int,
        default=protocol.DEFAULT_K_PER_DOT,
        help="images per calibration dot for variants that do not set their own (default: 5)",
    )
    return parser


def _run_extract(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    detector_factory: DetectorFactory | None,
) -> int:
    available = find_persons(args.root)
    if not available:
        raise DatasetFormatError(
            f"no pXX folders with a pXX.txt under {args.root}: is --root the extracted "
            "MPIIFaceGaze folder?"
        )
    persons = _persons_arg(available, args.persons, parser, str(args.root))
    summaries = extract(
        args.root,
        args.cache,
        persons=persons,
        limit=args.limit,
        workers=args.workers,
        force=args.force,
        detector_factory=detector_factory or MediaPipeImageDetector,
    )
    for s in summaries:
        state = "extracted"
        if s.skipped:
            state = s.note or "cached already, skipped (--force redoes it)"
        print(f"{s.person}: {s.images} images, {s.with_face} with a face; {state} -> {s.path}")
    return 0


def _run_evaluate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    persons = _persons_arg(cached_persons(args.cache), args.persons, parser, str(args.cache))
    variants = _variants_arg(args.variants, parser)
    results = evaluate_dataset(
        args.root, args.cache, persons=persons, variants=variants, k_per_dot=args.k_per_dot
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(summary_table(results))
    print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None, detector_factory: DetectorFactory | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            return _run_evaluate(args, parser)
        return _run_extract(args, parser, detector_factory)
    except (DatasetFormatError, MissingDependencyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
