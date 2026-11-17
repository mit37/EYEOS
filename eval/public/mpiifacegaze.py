"""MPIIFaceGaze: run EyeOS's landmark pipeline and calibration on a public gaze dataset.

Fetch the data first (CC BY-NC-SA 4.0; see docs/PUBLIC_DATASETS.md), then extract
MediaPipe landmarks once (slow, cached, restartable) and evaluate (fast, repeatable):

    python scripts/download_mpiifacegaze.py --accept-license
    python -m eval.public.mpiifacegaze extract --root datasets/MPIIFaceGaze \\
        --cache datasets/cache/mpiifacegaze [--persons p00,p01] [--limit N] [--workers N]

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

import argparse
import csv
import math
import re
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from eyeos.config import Config
from eyeos.input_sources._mediapipe_common import landmarks_from_mediapipe
from eyeos.landmarks import FaceLandmarks
from eyeos.synthetic_format import CSV_FIELDNAMES, landmarks_to_row

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
            static_image_mode=True,
            max_num_faces=max_num_faces,
            refine_landmarks=True,  # required for the iris landmarks
            min_detection_confidence=min_detection_confidence,
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
    images: int
    with_face: int
    skipped: bool
    path: Path


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
    interrupted run never leaves a cache that looks finished. An existing cache is reused
    when it holds exactly the requested images (the first ``limit``, or all).
    """
    annotations = load_annotations(root, person)[:limit]
    out = cache_path(cache_dir, person)
    if not force and _cached_images(out) == [a.image for a in annotations]:
        return ExtractSummary(person, len(annotations), _with_face(out), True, out)

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
    which must therefore be picklable (a module-level class or function)."""
    names = list(persons) if persons is not None else find_persons(root)
    jobs = [(Path(root), p, Path(cache_dir), detector_factory, limit, force) for p in names]
    if workers <= 1 or len(jobs) <= 1:
        return [_extract_person_job(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        return list(pool.map(_extract_person_job, jobs))


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


def _persons_arg(root: Path, text: str | None, parser: argparse.ArgumentParser) -> list[str]:
    available = find_persons(root)
    if text is None:
        if not available:
            parser.error(f"no pXX folders with annotations under {root}")
        return available
    names = [p.strip() for p in text.split(",") if p.strip()]
    unknown = [p for p in names if p not in available]
    if unknown:
        parser.error(
            f"--persons: {', '.join(unknown)} not found under {root} "
            f"(available: {', '.join(available) or 'none'})"
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
    return parser


def _run_extract(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    detector_factory: DetectorFactory | None,
) -> int:
    persons = _persons_arg(args.root, args.persons, parser)
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
        state = "cached already, skipped (--force redoes it)" if s.skipped else "extracted"
        print(f"{s.person}: {s.images} images, {s.with_face} with a face; {state} -> {s.path}")
    return 0


def main(argv: list[str] | None = None, detector_factory: DetectorFactory | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run_extract(args, parser, detector_factory)
    except (DatasetFormatError, MissingDependencyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
