"""The MPIIFaceGaze public-dataset harness: the downloader, the annotation parser, the
screen model, extraction and the EyeOS-faithful evaluation protocol.

Everything runs on a small fake dataset written to tmp_path. The fake "images" are text
files that a fake detector turns into landmarks, so neither OpenCV nor MediaPipe is
needed, and no network is used. The .mat calibration files are written with scipy; the
tests that need them skip when scipy is missing (it is in the ``dev`` extra, so CI runs
them).
"""

from __future__ import annotations

import csv
import dataclasses
import importlib.util
import io
import json
import math
import os
import random
import re
import shlex
import shutil
import socket
import subprocess
import sys
import types
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from packaging.requirements import Requirement

from eval.generate_synthetic import make_landmarks
from eval.public import models, protocol
from eval.public import mpiifacegaze as mpii
from eyeos.calibration import Calibration, target_points
from eyeos.config import Config, ScreenConfig
from eyeos.input_sources.synthetic import SyntheticInputSource
from eyeos.landmarks import FaceLandmarks, Point3D
from eyeos.output.dry_run import DryRunOutputAdapter
from eyeos.pipeline import Pipeline
from eyeos.synthetic_format import row_to_frame
from tests.fake_capture import FakeCv2, FakeImage, FakeMediapipe, install_fakes
from tests.fake_mpiifacegaze import (
    FakeDetector,
    FakePerson,
    annotation_line,
    fake_detector_factory,
    offsets_for_target,
    rotation_about_y,
    write_person,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# packaging and repo hygiene
# ---------------------------------------------------------------------------


def _extras() -> dict[str, list[str]]:
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    with open(ROOT / "pyproject.toml", "rb") as f:
        extras: dict[str, list[str]] = tomllib.load(f)["project"]["optional-dependencies"]
    return extras


def test_public_eval_extra_has_scipy_opencv_and_a_solutions_api_mediapipe():
    requirements = {r.name: r for r in map(Requirement, _extras()["public-eval"])}

    assert set(requirements) == {"scipy", "opencv-contrib-python", "mediapipe"}
    mediapipe = requirements["mediapipe"].specifier
    assert mediapipe.contains("0.10.21")
    assert not mediapipe.contains("0.10.30")


def test_dev_extra_has_scipy_so_ci_runs_the_mat_file_tests():
    requirements = {r.name: r for r in map(Requirement, _extras()["dev"])}

    assert "scipy" in requirements
    assert requirements["scipy"].specifier.contains("1.10.0")


def test_datasets_directory_is_gitignored():
    lines = {line.strip() for line in (ROOT / ".gitignore").read_text().splitlines()}

    assert "datasets/" in lines


# ---------------------------------------------------------------------------
# scripts/download_mpiifacegaze.py
# ---------------------------------------------------------------------------

_DOWNLOADER = ROOT / "scripts" / "download_mpiifacegaze.py"


def _load_downloader() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("download_mpiifacegaze", _DOWNLOADER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _no_network(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Make every way out to the network fail loudly, and record any attempt."""
    attempts: list[object] = []

    def refuse(*args: object, **kwargs: object) -> object:
        attempts.append(args)
        raise AssertionError(f"unexpected network call: {args!r}")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    return attempts


class _FakeResponse:
    def __init__(self, body: bytes, status: int, headers: dict[str, str]) -> None:
        self._body = io.BytesIO(body)
        self.status = status
        self.headers = headers

    def read(self, n: int = -1) -> bytes:
        return self._body.read(n)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self._body.close()


class _FakeServer:
    """Serves one file over a fake ``urllib.request.urlopen``, honouring Range requests
    (unless ``honour_range`` is False) and optionally cutting the body short."""

    def __init__(
        self, payload: bytes, honour_range: bool = True, truncate_to: int | None = None
    ) -> None:
        self.payload = payload
        self.honour_range = honour_range
        self.truncate_to = truncate_to
        self.requests: list[urllib.request.Request] = []

    def urlopen(
        self, request: urllib.request.Request, timeout: float | None = None
    ) -> _FakeResponse:
        self.requests.append(request)
        start = 0
        range_header = request.get_header("Range")
        if range_header and self.honour_range:
            start = int(range_header.removeprefix("bytes=").rstrip("-"))
        body = self.payload[start:]
        headers = {"Content-Length": str(len(body))}
        status = 200
        if start:
            status = 206
            headers["Content-Range"] = f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}"
        if self.truncate_to is not None:
            body = body[: self.truncate_to]
        return _FakeResponse(body, status, headers)


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buffer.getvalue()


_FAKE_ZIP_MEMBERS = {
    "MPIIFaceGaze/p00/p00.txt": b"day01/0001.jpg 1 2\n",
    "MPIIFaceGaze/p00/Calibration/screenSize.mat": b"not really a mat file",
}


def test_downloader_refuses_without_accept_license_and_makes_no_network_call(
    tmp_path, monkeypatch, capsys
):
    downloader = _load_downloader()
    attempts = _no_network(monkeypatch)

    code = downloader.main(["--dest", str(tmp_path / "datasets")])

    assert code == 2
    out = capsys.readouterr()
    text = out.out + out.err
    assert "CC BY-NC-SA 4.0" in text
    assert "It's Written All Over Your Face" in text
    assert "Appearance-Based Gaze Estimation in the Wild" in text
    assert "--accept-license" in text
    assert attempts == []
    assert not (tmp_path / "datasets").exists()


def test_downloader_script_exits_2_without_accept_license(tmp_path):
    result = subprocess.run(
        [sys.executable, str(_DOWNLOADER), "--dest", str(tmp_path / "d")],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 2
    assert "CC BY-NC-SA 4.0" in result.stdout + result.stderr
    assert not (tmp_path / "d").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_downloader_shell_wrapper_passes_arguments_through(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "download_mpiifacegaze.sh"), "--dest", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHON": sys.executable},
    )

    assert result.returncode == 2
    assert "--accept-license" in result.stdout + result.stderr


def test_downloader_streams_verifies_and_extracts_then_skips_a_second_run(
    tmp_path, monkeypatch, capsys
):
    downloader = _load_downloader()
    server = _FakeServer(_zip_bytes(_FAKE_ZIP_MEMBERS))
    monkeypatch.setattr(urllib.request, "urlopen", server.urlopen)
    dest = tmp_path / "datasets"

    assert downloader.main(["--accept-license", "--dest", str(dest)]) == 0

    assert (dest / "MPIIFaceGaze.zip").read_bytes() == server.payload
    assert (dest / "MPIIFaceGaze" / "p00" / "p00.txt").read_bytes() == b"day01/0001.jpg 1 2\n"
    assert not list(dest.glob("*.part"))
    assert len(server.requests) == 1
    assert server.requests[0].full_url == downloader.URL

    attempts = _no_network(monkeypatch)
    capsys.readouterr()
    assert downloader.main(["--accept-license", "--dest", str(dest)]) == 0
    assert attempts == []
    assert "already" in capsys.readouterr().out


def test_downloader_extracts_a_zip_without_a_top_level_folder_into_mpiifacegaze(
    tmp_path, monkeypatch
):
    downloader = _load_downloader()
    server = _FakeServer(_zip_bytes({"p03/p03.txt": b"x"}))
    monkeypatch.setattr(urllib.request, "urlopen", server.urlopen)

    assert downloader.main(["--accept-license", "--dest", str(tmp_path)]) == 0

    assert (tmp_path / "MPIIFaceGaze" / "p03" / "p03.txt").read_bytes() == b"x"


def test_downloader_does_not_nest_the_dataset_when_the_zip_has_other_top_level_files(
    tmp_path, monkeypatch
):
    downloader = _load_downloader()
    members = {**_FAKE_ZIP_MEMBERS, "readme.txt": b"about", "__MACOSX/._p00": b"junk"}
    server = _FakeServer(_zip_bytes(members))
    monkeypatch.setattr(urllib.request, "urlopen", server.urlopen)

    assert downloader.main(["--accept-license", "--dest", str(tmp_path)]) == 0

    target = tmp_path / "MPIIFaceGaze"
    assert (target / "p00" / "p00.txt").read_bytes() == b"day01/0001.jpg 1 2\n"
    assert not (target / "MPIIFaceGaze").exists()
    assert (target / downloader.DONE_MARKER).exists()
    assert mpii.find_persons(target) == ["p00"]


def test_downloader_fails_and_marks_nothing_done_when_no_participant_folder_comes_out(
    tmp_path, monkeypatch, capsys
):
    downloader = _load_downloader()
    server = _FakeServer(_zip_bytes({"Data/p00/p00.txt": b"x", "readme.txt": b"about"}))
    monkeypatch.setattr(urllib.request, "urlopen", server.urlopen)

    assert downloader.main(["--accept-license", "--dest", str(tmp_path)]) == 1

    err = capsys.readouterr().err
    assert "pXX/pXX.txt" in err and "'Data'" in err and "'readme.txt'" in err
    assert not (tmp_path / "MPIIFaceGaze" / downloader.DONE_MARKER).exists()


def test_downloader_resumes_a_partial_download_with_a_range_request(tmp_path, monkeypatch):
    downloader = _load_downloader()
    server = _FakeServer(_zip_bytes(_FAKE_ZIP_MEMBERS))
    monkeypatch.setattr(urllib.request, "urlopen", server.urlopen)
    (tmp_path / "MPIIFaceGaze.zip.part").write_bytes(server.payload[:100])

    assert downloader.main(["--accept-license", "--dest", str(tmp_path)]) == 0

    assert server.requests[0].get_header("Range") == "bytes=100-"
    assert (tmp_path / "MPIIFaceGaze.zip").read_bytes() == server.payload


def test_downloader_restarts_when_the_server_ignores_the_range_request(tmp_path, monkeypatch):
    downloader = _load_downloader()
    server = _FakeServer(_zip_bytes(_FAKE_ZIP_MEMBERS), honour_range=False)
    monkeypatch.setattr(urllib.request, "urlopen", server.urlopen)
    (tmp_path / "MPIIFaceGaze.zip.part").write_bytes(b"garbage from an older attempt")

    assert downloader.main(["--accept-license", "--dest", str(tmp_path)]) == 0

    assert (tmp_path / "MPIIFaceGaze.zip").read_bytes() == server.payload


def test_downloader_accepts_a_part_file_the_server_says_is_already_complete(tmp_path, monkeypatch):
    downloader = _load_downloader()
    payload = _zip_bytes(_FAKE_ZIP_MEMBERS)
    (tmp_path / "MPIIFaceGaze.zip.part").write_bytes(payload)

    def range_not_satisfiable(request: urllib.request.Request, timeout: float | None = None):
        raise urllib.error.HTTPError(request.full_url, 416, "Range Not Satisfiable", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", range_not_satisfiable)

    assert downloader.main(["--accept-license", "--dest", str(tmp_path)]) == 0

    assert (tmp_path / "MPIIFaceGaze.zip").read_bytes() == payload
    assert (tmp_path / "MPIIFaceGaze" / "p00" / "p00.txt").exists()


def test_downloader_keeps_a_cut_off_download_as_part_and_fails(tmp_path, monkeypatch, capsys):
    downloader = _load_downloader()
    server = _FakeServer(_zip_bytes(_FAKE_ZIP_MEMBERS), truncate_to=50)
    monkeypatch.setattr(urllib.request, "urlopen", server.urlopen)

    assert downloader.main(["--accept-license", "--dest", str(tmp_path)]) == 1

    assert (tmp_path / "MPIIFaceGaze.zip.part").read_bytes() == server.payload[:50]
    assert not (tmp_path / "MPIIFaceGaze.zip").exists()
    assert "rerun" in capsys.readouterr().err


def test_downloader_rejects_a_corrupt_zip_and_deletes_it(tmp_path, monkeypatch, capsys):
    downloader = _load_downloader()
    good = _zip_bytes({"MPIIFaceGaze/p00/p00.txt": bytes(range(256)) * 8})
    # Flip a byte inside the compressed member data so its CRC no longer matches.
    corrupt = bytearray(good)
    corrupt[len("MPIIFaceGaze/p00/p00.txt") + 40] ^= 0xFF
    server = _FakeServer(bytes(corrupt))
    monkeypatch.setattr(urllib.request, "urlopen", server.urlopen)

    assert downloader.main(["--accept-license", "--dest", str(tmp_path)]) == 1

    assert "corrupt" in capsys.readouterr().err
    assert not (tmp_path / "MPIIFaceGaze.zip").exists()
    assert not (tmp_path / "MPIIFaceGaze").exists()


def test_downloader_refuses_a_zip_member_that_escapes_the_destination(
    tmp_path, monkeypatch, capsys
):
    downloader = _load_downloader()
    server = _FakeServer(_zip_bytes({"MPIIFaceGaze/../../evil.txt": b"x"}))
    monkeypatch.setattr(urllib.request, "urlopen", server.urlopen)
    dest = tmp_path / "a" / "b"

    assert downloader.main(["--accept-license", "--dest", str(dest)]) == 1

    assert "outside" in capsys.readouterr().err
    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path / "a" / "evil.txt").exists()


# ---------------------------------------------------------------------------
# annotation and calibration-file parsing
# ---------------------------------------------------------------------------


@pytest.fixture
def scipy_io() -> types.ModuleType:
    return pytest.importorskip("scipy.io")


def test_parse_reads_all_28_fields_of_a_well_formed_line(tmp_path):
    person = write_person(tmp_path, FakePerson("p04", n_images=3, write_mat=False))

    annotations = mpii.load_annotations(tmp_path, "p04")

    assert [a.image for a in annotations] == ["day01/0000.jpg", "day01/0001.jpg", "day01/0002.jpg"]
    first = annotations[0]
    assert first.target_px == person.targets[0]
    assert first.landmarks_px == tuple(500.0 + 10 * i for i in range(12))
    assert first.head_rvec == (0.1, -0.2, 0.05)
    assert first.head_tvec == (-20.0, 30.0, 600.0)
    assert first.face_center_mm == person.face_center
    expected_gt = person.gaze_target_mm(*person.targets[0])
    assert first.gaze_target_mm == pytest.approx(tuple(expected_gt))
    assert first.eye == "left"


def test_parse_skips_blank_lines_and_accepts_tabs_and_crlf(tmp_path):
    folder = tmp_path / "p00"
    folder.mkdir()
    line = annotation_line("day01/0001.jpg", (1.0, 2.0), (0.0, 0.0, 500.0), (5.0, 6.0, 0.0))
    (folder / "p00.txt").write_text("\n" + line.replace(" ", "\t", 3) + "\r\n\n")

    (annotation,) = mpii.load_annotations(tmp_path, "p00")

    assert annotation.target_px == (1.0, 2.0)
    assert annotation.eye == "left"


@pytest.mark.parametrize(
    ("mutate", "problem"),
    [
        (lambda f: f[:27], "expected 28"),
        (lambda f: [*f, "extra"], "expected 28"),
        (lambda f: [f[0], "abc", *f[2:]], "field 2"),
        (lambda f: [*f[:5], "nan", *f[6:]], "field 6"),
        (lambda f: [*f[:27], "both"], "'left' or 'right'"),
        (lambda f: ["/etc/passwd", *f[1:]], "relative"),
        (lambda f: ["../p01/day01/0001.jpg", *f[1:]], "relative"),
    ],
)
def test_parse_fails_loudly_on_a_malformed_line_and_shows_it(tmp_path, mutate, problem):
    folder = tmp_path / "p00"
    folder.mkdir()
    good = annotation_line("day01/0001.jpg", (1.0, 2.0), (0.0, 0.0, 500.0), (5.0, 6.0, 0.0))
    bad = " ".join(mutate(good.split()))
    (folder / "p00.txt").write_text(good + "\n" + bad + "\n")

    with pytest.raises(mpii.DatasetFormatError) as excinfo:
        mpii.load_annotations(tmp_path, "p00")

    message = str(excinfo.value)
    assert "p00.txt:2" in message
    assert problem in message
    assert bad[:40] in message


def test_parse_fails_loudly_on_a_missing_or_empty_annotation_file(tmp_path):
    with pytest.raises(mpii.DatasetFormatError, match="p07.txt"):
        mpii.load_annotations(tmp_path, "p07")

    (tmp_path / "p07").mkdir()
    (tmp_path / "p07" / "p07.txt").write_text("\n\n")
    with pytest.raises(mpii.DatasetFormatError, match="no annotations"):
        mpii.load_annotations(tmp_path, "p07")


def test_calibration_files_load_and_squeeze(tmp_path, scipy_io):
    person = write_person(tmp_path, FakePerson("p02", n_images=2))

    screen = mpii.load_screen_size(tmp_path, "p02")
    pose = mpii.load_monitor_pose(tmp_path, "p02")

    assert screen == mpii.ScreenSize(1280.0, 800.0, 331.0, 207.0)
    assert pose.rvec.shape == (3,) and pose.tvec.shape == (3,)
    assert pose.rvec.tolist() == [0.0, person.theta, 0.0]
    assert pose.tvec.tolist() == list(person.tvec)


def test_monitor_pose_accepts_the_rvecs_spelling_too(tmp_path, scipy_io):
    folder = tmp_path / "p01" / "Calibration"
    folder.mkdir(parents=True)
    scipy_io.savemat(
        folder / "monitorPose.mat",
        {"rvecs": np.zeros((3, 1)), "tvecs": np.ones((1, 3))},
    )

    pose = mpii.load_monitor_pose(tmp_path, "p01")

    assert pose.rvec.tolist() == [0.0, 0.0, 0.0]
    assert pose.tvec.tolist() == [1.0, 1.0, 1.0]


def test_calibration_files_fail_loudly_on_missing_keys_or_bad_shapes(tmp_path, scipy_io):
    folder = tmp_path / "p01" / "Calibration"
    folder.mkdir(parents=True)
    scipy_io.savemat(folder / "screenSize.mat", {"width_pixel": 1280.0, "height_px": 800.0})
    scipy_io.savemat(folder / "monitorPose.mat", {"rvects": np.zeros(2), "tvecs": np.ones(3)})

    with pytest.raises(mpii.DatasetFormatError, match="height_pixel.*height_px"):
        mpii.load_screen_size(tmp_path, "p01")
    with pytest.raises(mpii.DatasetFormatError, match="rvects.*3 numbers"):
        mpii.load_monitor_pose(tmp_path, "p01")
    with pytest.raises(mpii.DatasetFormatError, match="p05"):
        mpii.load_screen_size(tmp_path, "p05")


# ---------------------------------------------------------------------------
# Rodrigues and the screen model
# ---------------------------------------------------------------------------


def _axis_rotation(axis: int, theta: float) -> np.ndarray:
    """Textbook rotation about the x (0), y (1) or z (2) axis, written out by hand."""
    c, s = math.cos(theta), math.sin(theta)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return rotation_about_y(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("theta", [0.3, -1.2, math.pi - 0.15, math.pi])
def test_rodrigues_matches_the_textbook_single_axis_rotations(axis, theta):
    rvec = np.zeros(3)
    rvec[axis] = theta

    assert mpii.rodrigues(rvec) == pytest.approx(_axis_rotation(axis, theta), abs=1e-12)


def test_rodrigues_of_zero_is_the_identity():
    assert mpii.rodrigues(np.zeros(3)) == pytest.approx(np.eye(3))
    assert mpii.rodrigues(np.full(3, 1e-14)) == pytest.approx(np.eye(3))


def test_rodrigues_of_an_arbitrary_axis_is_a_rotation_by_the_vector_length_about_it():
    rvec = np.array([0.4, -1.1, 0.7])
    angle = float(np.linalg.norm(rvec))

    rotation = mpii.rodrigues(rvec)

    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert rotation @ rvec == pytest.approx(rvec)  # the axis stays put
    assert np.trace(rotation) == pytest.approx(1 + 2 * math.cos(angle))


def test_rodrigues_matches_opencv_when_it_is_installed():
    cv2 = pytest.importorskip("cv2")
    rvec = np.array([0.4, -1.1, 0.7])

    expected, _ = cv2.Rodrigues(rvec.reshape(3, 1))

    assert mpii.rodrigues(rvec) == pytest.approx(expected, abs=1e-9)


def _screen_and_pose(person: FakePerson) -> tuple[mpii.ScreenSize, mpii.MonitorPose]:
    screen = mpii.ScreenSize(
        float(person.screen_px[0]),
        float(person.screen_px[1]),
        person.screen_mm[0],
        person.screen_mm[1],
    )
    pose = mpii.MonitorPose(np.array([0.0, person.theta, 0.0]), np.array(person.tvec))
    return screen, pose


def _targets_and_gts(person: FakePerson) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    targets = rng.uniform([0, 0], person.screen_px, size=(50, 2))
    gts = np.array([person.gaze_target_mm(x, y) for x, y in targets])
    return targets, gts


def test_screen_model_reproduces_consistent_3d_targets_to_about_0_mm():
    person = FakePerson("p00")
    screen, pose = _screen_and_pose(person)
    targets, gts = _targets_and_gts(person)

    check = mpii.check_screen_model(targets, gts, screen, pose)

    assert check.model == "default"
    assert check.fits
    assert check.median_error_mm < 1e-9
    assert check.default_median_error_mm < 1e-9
    assert check.reason is None
    assert mpii.screen_to_camera(targets, screen, pose, check.model) == pytest.approx(gts)


def test_screen_model_falls_back_to_the_best_alternative_and_records_it():
    person = FakePerson("p00", mirror_gt_x=True)
    screen, pose = _screen_and_pose(person)
    targets, gts = _targets_and_gts(person)

    check = mpii.check_screen_model(targets, gts, screen, pose)

    assert check.default_median_error_mm > 5.0
    assert check.model == "mirror-x"
    assert check.fits
    assert check.median_error_mm < 1e-9
    assert set(check.candidates_median_mm) == set(mpii.SCREEN_MODELS)


def test_screen_model_reports_no_fit_with_a_reason_when_nothing_is_within_5_mm():
    person = FakePerson("p00")
    screen, pose = _screen_and_pose(person)
    targets, gts = _targets_and_gts(person)

    check = mpii.check_screen_model(targets, gts + 40.0, screen, pose)

    assert not check.fits
    assert check.model is None
    assert "5" in check.reason and "angular" in check.reason


# ---------------------------------------------------------------------------
# extract: images -> MediaPipe landmarks -> one cached CSV per person
# ---------------------------------------------------------------------------


class _FailingDetector(FakeDetector):
    def __init__(self, fail_on_call: int) -> None:
        super().__init__()
        self.fail_on_call = fail_on_call

    def __call__(self, image_path: Path) -> FaceLandmarks | None:
        if len(self.calls) + 1 == self.fail_on_call:
            raise RuntimeError("interrupted")
        return super().__call__(image_path)


class _CountingFactory:
    """A detector factory that remembers every detector it made."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.detectors: list[FakeDetector] = []
        self.fail_on_call = fail_on_call

    def __call__(self) -> FakeDetector:
        detector = (
            FakeDetector() if self.fail_on_call is None else _FailingDetector(self.fail_on_call)
        )
        self.detectors.append(detector)
        return detector

    @property
    def images_seen(self) -> int:
        return sum(len(d.calls) for d in self.detectors)


def _read_cache(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_extract_caches_landmarks_targets_and_3d_points_per_person(tmp_path):
    root, cache = tmp_path / "MPIIFaceGaze", tmp_path / "cache"
    person = write_person(
        root, FakePerson("p00", n_images=6, faces={1: 0, 2: 2}, closed={3}, write_mat=False)
    )
    factory = _CountingFactory()

    (summary,) = mpii.extract(root, cache, detector_factory=factory)

    rows = _read_cache(cache / "p00.csv")
    assert list(rows[0]) == mpii.CACHE_COLUMNS
    assert [r["image"] for r in rows] == [f"day01/{i:04d}.jpg" for i in range(6)]
    assert [int(r["face_count"]) for r in rows] == [1, 0, 2, 1, 1, 1]
    annotations = mpii.load_annotations(root, "p00")
    for row, annotation, target in zip(rows, annotations, person.targets, strict=True):
        assert (float(row["target_x"]), float(row["target_y"])) == target
        assert tuple(float(row[f"fc_{a}"]) for a in "xyz") == annotation.face_center_mm
        assert tuple(float(row[f"gt_{a}"]) for a in "xyz") == annotation.gaze_target_mm
    _, landmarks = row_to_frame(rows[0])
    assert landmarks == FakeDetector()(root / "p00" / "day01" / "0000.jpg")
    assert row_to_frame(rows[1])[1] is None
    assert summary.person == "p00" and summary.images == 6 and summary.with_face == 5
    assert not summary.skipped
    assert all(d.closed for d in factory.detectors)


def test_extract_skips_cached_persons_unless_forced(tmp_path):
    root, cache = tmp_path / "MPIIFaceGaze", tmp_path / "cache"
    write_person(root, FakePerson("p00", n_images=4, write_mat=False))
    write_person(root, FakePerson("p01", n_images=4, write_mat=False))
    mpii.extract(root, cache, detector_factory=_CountingFactory())
    before = (cache / "p00.csv").read_text()

    again = _CountingFactory()
    summaries = mpii.extract(root, cache, detector_factory=again)
    assert again.images_seen == 0
    assert [s.skipped for s in summaries] == [True, True]
    assert (cache / "p00.csv").read_text() == before

    forced = _CountingFactory()
    mpii.extract(root, cache, persons=["p01"], force=True, detector_factory=forced)
    assert forced.images_seen == 4


def test_extract_limit_takes_the_first_n_images_and_a_later_full_run_redoes_them(tmp_path):
    root, cache = tmp_path / "MPIIFaceGaze", tmp_path / "cache"
    write_person(root, FakePerson("p00", n_images=7, write_mat=False))

    mpii.extract(root, cache, limit=3, detector_factory=_CountingFactory())
    assert [r["image"] for r in _read_cache(cache / "p00.csv")] == [
        f"day01/{i:04d}.jpg" for i in range(3)
    ]

    full = _CountingFactory()
    (summary,) = mpii.extract(root, cache, detector_factory=full)
    assert not summary.skipped and full.images_seen == 7
    assert len(_read_cache(cache / "p00.csv")) == 7


def test_a_limited_trial_run_keeps_a_complete_cache_that_starts_with_those_images(tmp_path, capsys):
    root, cache = tmp_path / "MPIIFaceGaze", tmp_path / "cache"
    write_person(root, FakePerson("p00", n_images=7, write_mat=False))
    mpii.extract(root, cache, detector_factory=_CountingFactory())
    before = (cache / "p00.csv").read_text()
    capsys.readouterr()

    trial = _CountingFactory()
    code = mpii.main(
        ["extract", "--root", str(root), "--cache", str(cache), "--limit", "3"],
        detector_factory=trial,
    )

    assert code == 0
    assert trial.detectors == []
    assert (cache / "p00.csv").read_text() == before
    assert len(_read_cache(cache / "p00.csv")) == 7
    out = capsys.readouterr().out
    assert "p00: 7 images" in out and "kept" in out


def test_extract_never_replaces_a_cache_with_a_shorter_one_without_force(tmp_path, capsys):
    root, cache = tmp_path / "MPIIFaceGaze", tmp_path / "cache"
    write_person(root, FakePerson("p00", n_images=7, write_mat=False))
    mpii.extract(root, cache, detector_factory=_CountingFactory())
    # A cache whose first image is not the annotation file's first, e.g. from another copy.
    path = cache / "p00.csv"
    path.write_text(path.read_text().replace("day01/0000.jpg", "day02/0000.jpg", 1))
    before = path.read_text()

    trial = _CountingFactory()
    (summary,) = mpii.extract(root, cache, limit=3, detector_factory=trial)

    assert summary.skipped and trial.detectors == []
    assert path.read_text() == before
    assert summary.images == 7
    assert summary.note is not None and "--force" in summary.note

    forced = _CountingFactory()
    (summary,) = mpii.extract(root, cache, limit=3, force=True, detector_factory=forced)
    assert not summary.skipped and forced.images_seen == 3
    assert [r["image"] for r in _read_cache(path)] == [f"day01/{i:04d}.jpg" for i in range(3)]


def test_an_interrupted_extract_leaves_no_cache_and_the_rerun_completes_it(tmp_path):
    root, cache = tmp_path / "MPIIFaceGaze", tmp_path / "cache"
    write_person(root, FakePerson("p00", n_images=5, write_mat=False))

    with pytest.raises(RuntimeError, match="interrupted"):
        mpii.extract(root, cache, detector_factory=_CountingFactory(fail_on_call=3))
    assert not (cache / "p00.csv").exists()

    (summary,) = mpii.extract(root, cache, detector_factory=_CountingFactory())
    assert not summary.skipped
    assert len(_read_cache(cache / "p00.csv")) == 5


def test_extract_in_parallel_matches_a_sequential_run(tmp_path):
    root = tmp_path / "MPIIFaceGaze"
    for name in ("p00", "p01", "p02"):
        write_person(root, FakePerson(name, n_images=5, seed=int(name[1:]), write_mat=False))

    mpii.extract(root, tmp_path / "seq", detector_factory=fake_detector_factory)
    summaries = mpii.extract(
        root, tmp_path / "par", workers=3, detector_factory=fake_detector_factory
    )

    assert [s.person for s in summaries] == ["p00", "p01", "p02"]
    for name in ("p00", "p01", "p02"):
        assert (tmp_path / "par" / f"{name}.csv").read_text() == (
            tmp_path / "seq" / f"{name}.csv"
        ).read_text()


def test_extract_cli_parses_persons_limit_and_workers(tmp_path, capsys):
    root, cache = tmp_path / "MPIIFaceGaze", tmp_path / "cache"
    for name in ("p00", "p01", "p02"):
        write_person(root, FakePerson(name, n_images=4, write_mat=False))

    code = mpii.main(
        ["extract", "--root", str(root), "--cache", str(cache), "--persons", "p00,p02"]
        + ["--limit", "2", "--workers", "1"],
        detector_factory=fake_detector_factory,
    )

    assert code == 0
    assert sorted(p.name for p in cache.iterdir()) == ["p00.csv", "p02.csv"]
    assert len(_read_cache(cache / "p02.csv")) == 2
    assert "p02" in capsys.readouterr().out


def test_extract_cli_rejects_an_unknown_person(tmp_path, capsys):
    root = tmp_path / "MPIIFaceGaze"
    write_person(root, FakePerson("p00", n_images=2, write_mat=False))

    with pytest.raises(SystemExit) as excinfo:
        mpii.main(
            ["extract", "--root", str(root), "--cache", str(tmp_path / "c"), "--persons", "p9"],
            detector_factory=fake_detector_factory,
        )

    assert excinfo.value.code == 2
    assert "p9" in capsys.readouterr().err


def test_extract_cli_reports_a_dataset_problem_as_one_line_and_exit_1(tmp_path, capsys):
    root = tmp_path / "MPIIFaceGaze"
    (root / "p00").mkdir(parents=True)
    (root / "p00" / "p00.txt").write_text("day01/0001.jpg 1 2 3\n")

    code = mpii.main(
        ["extract", "--root", str(root), "--cache", str(tmp_path / "c")],
        detector_factory=fake_detector_factory,
    )

    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "p00.txt:1" in err
    assert "Traceback" not in err


@pytest.mark.parametrize("make_root", [False, True])
def test_extract_cli_fails_on_a_root_without_participants(tmp_path, capsys, make_root):
    root = tmp_path / "not-the-dataset"
    if make_root:
        (root / "notes").mkdir(parents=True)

    code = mpii.main(
        ["extract", "--root", str(root), "--cache", str(tmp_path / "c")],
        detector_factory=fake_detector_factory,
    )

    assert code == 1
    assert "not-the-dataset" in capsys.readouterr().err
    assert not (tmp_path / "c").exists()


class _ImreadCv2(FakeCv2):
    """fake_capture's cv2 plus imread, returning a frame of the given (h, w, c) shape."""

    def __init__(self, shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.shape = shape
        self.read_paths: list[str] = []

    def imread(self, path: str) -> FakeImage | None:
        self.read_paths.append(path)
        if path.endswith("missing.jpg"):
            return None
        return FakeImage("bgr", len(self.read_paths) - 1, self.shape)


def test_mediapipe_image_detector_runs_static_face_mesh_and_rescales_x(monkeypatch, tmp_path):
    cv2 = _ImreadCv2((720, 1280, 3))
    mediapipe = FakeMediapipe(faces_per_frame=[1, 0, 2])
    install_fakes(monkeypatch, cv2=cv2, mediapipe=mediapipe)

    detector = mpii.MediaPipeImageDetector()
    one, none, two = (detector(tmp_path / f"{i}.jpg") for i in range(3))
    detector.close()

    (mesh,) = mediapipe.face_meshes
    assert mesh.kwargs["static_image_mode"] is True
    assert mesh.kwargs["max_num_faces"] == 2
    assert mesh.kwargs["refine_landmarks"] is True
    assert mesh.kwargs["min_detection_confidence"] == 0.5
    assert [image.color for image in mesh.processed] == ["rgb", "rgb", "rgb"]
    assert none is None
    assert one is not None and one.face_count == 1
    assert two is not None and two.face_count == 2
    # fake_face() puts every landmark at (0.5, 0.5); x is rescaled by width / height.
    assert one.left_iris_center.x == pytest.approx(0.5 * 1280 / 720)
    assert one.left_iris_center.y == pytest.approx(0.5)
    assert mesh.close_calls == 1


def test_mediapipe_image_detector_fails_loudly_on_an_unreadable_image(monkeypatch, tmp_path):
    install_fakes(monkeypatch, cv2=_ImreadCv2((10, 10, 3)), mediapipe=FakeMediapipe())

    detector = mpii.MediaPipeImageDetector()
    with pytest.raises(mpii.DatasetFormatError, match="missing.jpg"):
        detector(tmp_path / "missing.jpg")


def test_mediapipe_image_detector_explains_a_mediapipe_without_solutions(monkeypatch):
    install_fakes(
        monkeypatch,
        cv2=_ImreadCv2((10, 10, 3)),
        mediapipe=FakeMediapipe(solutions=False, version="0.10.30"),
    )

    with pytest.raises(RuntimeError, match=r"public-eval.*0\.10\.30|0\.10\.30.*public-eval"):
        mpii.MediaPipeImageDetector()


def test_mediapipe_image_detector_finds_no_face_in_a_blank_image_with_real_libraries(tmp_path):
    """Optional smoke test of the real OpenCV + MediaPipe calls; skipped without them."""
    cv2 = pytest.importorskip("cv2")
    mediapipe = pytest.importorskip("mediapipe")
    if not hasattr(mediapipe, "solutions"):
        pytest.skip("mediapipe without the legacy Solutions API")
    image = tmp_path / "blank.png"
    assert cv2.imwrite(str(image), np.full((72, 128, 3), 127, dtype=np.uint8))

    detector = mpii.MediaPipeImageDetector()
    try:
        assert detector(image) is None
    finally:
        detector.close()


# ---------------------------------------------------------------------------
# the evaluation protocol's building blocks (dataset-agnostic, no scipy needed)
# ---------------------------------------------------------------------------


def test_calibration_set_takes_the_k_nearest_images_to_each_grid_point():
    grid = target_points(1000.0, 1000.0)
    targets = []
    for gx, gy in grid:
        targets += [(gx + d, gy) for d in (4.0, 1.0, 3.0, 2.0)]  # 4 images near each dot
    targets += [(300.0, 300.0), (700.0, 700.0)]  # far from every dot

    selected, mean_distance = protocol.select_calibration_set(np.array(targets), grid, k=3)

    assert selected == [[4 * i + 1, 4 * i + 3, 4 * i + 2] for i in range(9)]
    assert mean_distance == pytest.approx([2.0] * 9)


def test_calibration_set_never_reuses_an_image_and_breaks_ties_by_order():
    targets = np.array([(0.5, 0.0), (0.6, 0.0), (5.0, 0.0), (5.0, 0.0), (-5.0, 0.0)])

    selected, _ = protocol.select_calibration_set(targets, [(0.0, 0.0), (1.0, 0.0)], k=2)

    # The second dot's nearest images (0.6 and 0.5) are taken already; of the two
    # equally near ones left (5.0 twice), the first in order wins.
    assert selected == [[0, 1], [2, 3]]


def test_calibration_set_needs_k_images_per_grid_point():
    with pytest.raises(ValueError, match="needs 18"):
        protocol.select_calibration_set(np.zeros((17, 2)), target_points(10.0, 10.0), k=2)


def test_predictions_are_rounded_and_clamped_exactly_like_the_pipelines_cursor_pixel():
    config = Config(screen=ScreenConfig(width_px=1280, height_px=800))
    pipeline = Pipeline(
        config, SyntheticInputSource(ROOT / "unused.csv"), DryRunOutputAdapter(echo=None)
    )
    points = np.array([[-5.2, 3.6], [1279.6, 799.5], [640.5, 400.49], [2000.0, -1.0], [10.5, 11.5]])

    clamped = protocol.clamp_to_screen(points, 1280.0, 800.0)

    assert clamped.tolist() == [list(pipeline._to_screen_pixel((x, y))) for x, y in points]


def test_angular_error_is_the_angle_between_the_predicted_and_true_gaze_rays():
    face_centers = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    true_targets = np.array([[0.0, 0.0, 100.0], [10.0, 10.0, 110.0]])
    predicted = np.array([[100.0, 0.0, 100.0], [10.0, 10.0, 110.0]])

    angles = protocol.angular_errors_deg(predicted, face_centers, true_targets)

    assert angles == pytest.approx([45.0, 0.0])


def test_valid_frames_need_exactly_one_face_with_both_irises_visible():
    assert protocol.is_valid(make_landmarks(0.1, 0.2))
    assert not protocol.is_valid(None)
    assert not protocol.is_valid(make_landmarks(0.1, 0.2, face_count=2))
    closed = dataclasses.replace(
        make_landmarks(0.0, 0.0),
        left_eye_top=Point3D(0.38, 0.40),
        left_eye_bottom=Point3D(0.38, 0.40),
    )
    assert not protocol.is_valid(closed)


# ---------------------------------------------------------------------------
# evaluate: the EyeOS-faithful protocol end to end on a fake dataset
# ---------------------------------------------------------------------------


def _extracted(tmp_path: Path, persons: list[FakePerson]) -> tuple[Path, Path]:
    root, cache = tmp_path / "MPIIFaceGaze", tmp_path / "cache"
    for person in persons:
        write_person(root, person)
    mpii.extract(root, cache, detector_factory=fake_detector_factory)
    return root, cache


def _second_screen(name: str, seed: int, **kwargs: Any) -> FakePerson:
    return FakePerson(name, seed=seed, screen_px=(1440, 900), screen_mm=(300.0, 188.0), **kwargs)


def test_noise_free_evaluate_recovers_the_targets_to_about_0_px_and_0_deg(tmp_path, scipy_io):
    root, cache = _extracted(tmp_path, [FakePerson("p00"), _second_screen("p01", seed=1)])

    results = mpii.evaluate_dataset(root, cache, variants=["eyeos"])

    eyeos = results["variants"]["eyeos"]
    assert sorted(eyeos["persons"]) == ["p00", "p01"]
    for person in eyeos["persons"].values():
        assert person["n_calibration"] == 45
        assert person["n_test"] == 240 - 45
        assert person["calibration_mean_distance_px"] > 0
        # Rounding to a whole pixel, as the cursor does, is the only error left.
        assert person["pixel_error_px"]["mean"] < 0.75
        assert person["pixel_error_px"]["p90"] < 0.75
        assert person["angular_error_deg"]["mean"] < 0.02
        assert person["validation_error_px"] < 0.5
        assert person["passes_gate2"] is True
        assert person["head_pose_within_limit_share"] == 1.0
    aggregate = eyeos["aggregate"]
    assert aggregate["persons_evaluated"] == 2
    assert aggregate["pixel_error_px"]["mean_of_person_means"] < 0.75
    assert aggregate["pixel_error_px"]["pooled_p90"] < 0.75
    assert aggregate["angular_error_deg"]["mean_of_person_means"] < 0.02
    assert aggregate["gate2_passed"] == 2
    for name in ("p00", "p01"):
        assert results["screen_model"][name]["model"] == "default"
        assert results["screen_model"][name]["median_error_mm"] < 1e-6
        assert results["detection"]["persons"][name]["detection_rate"] == 1.0


def test_noisy_evaluate_reports_a_proportionate_error_and_gate2_catches_a_bad_one(
    tmp_path, scipy_io
):
    root, cache = _extracted(
        tmp_path, [FakePerson("p00", noise=0.05), FakePerson("p01", seed=1, noise=0.2)]
    )

    results = mpii.evaluate_dataset(root, cache, variants=["eyeos"])

    persons = results["variants"]["eyeos"]["persons"]
    mild, bad = persons["p00"], persons["p01"]
    px = mild["pixel_error_px"]
    assert 10 < px["mean"] < 100
    assert px["median"] <= px["p90"]
    assert mild["pixel_error_pct_diagonal"]["mean"] == pytest.approx(
        100 * px["mean"] / math.hypot(1280, 800)
    )
    assert mild["angular_error_deg"]["mean"] > 0.5
    assert 10 < mild["validation_error_px"] < 60 and mild["passes_gate2"] is True
    assert bad["pixel_error_px"]["mean"] > px["mean"]
    assert bad["validation_error_px"] > 60 and bad["passes_gate2"] is False
    aggregate = results["variants"]["eyeos"]["aggregate"]
    assert aggregate["gate2_passed"] == 1
    assert aggregate["pixel_error_px"]["mean_of_person_means"] == pytest.approx(
        (px["mean"] + bad["pixel_error_px"]["mean"]) / 2
    )


def test_frames_without_exactly_one_open_eyed_face_are_counted_and_left_out(tmp_path, scipy_io):
    faces = {i: 0 for i in range(20)} | {i: 2 for i in range(20, 30)}
    root, cache = _extracted(tmp_path, [FakePerson("p00", faces=faces, closed=set(range(30, 35)))])

    results = mpii.evaluate_dataset(root, cache, variants=["eyeos"])

    assert results["detection"]["persons"]["p00"] == {
        "images": 240,
        "valid": 205,
        "no_face": 20,
        "multiple_faces": 10,
        "irises_not_visible": 5,
        "detection_rate": pytest.approx(205 / 240),
    }
    assert results["detection"]["aggregate"]["detection_rate_mean"] == pytest.approx(205 / 240)
    person = results["variants"]["eyeos"]["persons"]["p00"]
    assert person["n_calibration"] + person["n_test"] == 205


def _with_invalid_frames(person: FakePerson) -> FakePerson:
    """Every 7th image faceless and every 11th (from 3) shut-eyed, as real data has some."""
    faceless = set(range(0, person.n_images, 7))
    return dataclasses.replace(
        person,
        faces={i: 0 for i in faceless},
        closed=set(range(3, person.n_images, 11)) - faceless,
    )


def test_noise_free_evaluate_keeps_each_frame_with_its_own_target_around_invalid_frames(
    tmp_path, scipy_io
):
    persons = [
        _with_invalid_frames(FakePerson("p00")),
        _with_invalid_frames(_second_screen("p01", seed=1)),
    ]
    root, cache = _extracted(tmp_path, persons)

    results = mpii.evaluate_dataset(root, cache, variants=["eyeos", "global-lopo"])

    for person in persons:
        invalid = len(person.faces) + len(person.closed)
        assert invalid == 35 + 19
        assert results["detection"]["persons"][person.name]["valid"] == 240 - invalid
    for variant in ("eyeos", "global-lopo"):
        assert results["variants"][variant]["aggregate"]["persons_evaluated"] == 2, variant
        for name, person in results["variants"][variant]["persons"].items():
            # Any slip between a valid frame and its target or 3D labels shows up as
            # hundreds of pixels or degrees, not the rounding error left here.
            assert person["pixel_error_px"]["mean"] < 0.75, (variant, name)
            assert person["pixel_error_px"]["p90"] < 0.75, (variant, name)
            assert person["angular_error_deg"]["mean"] < 0.02, (variant, name)
            assert person["validation_error_px"] < 0.5, (variant, name)


def test_angular_error_is_null_with_the_reason_when_no_screen_model_fits(tmp_path, scipy_io):
    root, cache = _extracted(tmp_path, [FakePerson("p00", gt_offset_mm=40.0)])

    results = mpii.evaluate_dataset(root, cache, variants=["eyeos"])

    assert results["screen_model"]["p00"]["fits"] is False
    person = results["variants"]["eyeos"]["persons"]["p00"]
    assert person["angular_error_deg"] is None
    assert "5 mm" in person["angular_error_note"]
    assert person["pixel_error_px"]["mean"] < 0.75
    assert results["variants"]["eyeos"]["aggregate"]["angular_error_deg"] is None


def test_a_person_with_too_few_valid_frames_is_skipped_with_the_reason(tmp_path, scipy_io):
    root, cache = _extracted(tmp_path, [FakePerson("p00", n_images=40), FakePerson("p01")])

    results = mpii.evaluate_dataset(root, cache, variants=["eyeos"])

    eyeos = results["variants"]["eyeos"]
    assert "40 valid" in eyeos["persons"]["p00"]["skipped"]
    assert eyeos["aggregate"]["persons_evaluated"] == 1
    assert eyeos["aggregate"]["persons_skipped"] == ["p00"]


def test_evaluate_runs_every_variant_by_default_and_is_deterministic(tmp_path, scipy_io):
    root, cache = _extracted(
        tmp_path,
        [
            FakePerson("p00", noise=0.05),
            _second_screen("p01", seed=1, noise=0.05),
            FakePerson("p02", seed=2, noise=0.05),
        ],
    )

    first = mpii.evaluate_dataset(root, cache)
    second = mpii.evaluate_dataset(root, cache)

    assert list(first["variants"]) == list(models.VARIANT_NAMES)
    assert first["config"]["variants"] == list(models.VARIANT_NAMES)
    for results in (first, second):
        results.pop("generated")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    for name, variant in first["variants"].items():
        aggregate = variant["aggregate"]
        assert aggregate["persons_evaluated"] == 3, name
        assert math.isfinite(aggregate["pixel_error_px"]["mean_of_person_means"]), name
        assert math.isfinite(aggregate["angular_error_deg"]["mean_of_person_means"]), name


def test_evaluate_fails_loudly_on_a_cache_from_another_format(tmp_path, scipy_io):
    root, cache = _extracted(tmp_path, [FakePerson("p00", n_images=3)])
    (cache / "p00.csv").write_text("image,target_x\nday01/0000.jpg,1\n")

    with pytest.raises(mpii.DatasetFormatError, match="extract --force"):
        mpii.evaluate_dataset(root, cache, variants=["eyeos"])


def test_evaluate_cli_writes_the_results_json(tmp_path, scipy_io, capsys):
    root, cache = _extracted(tmp_path, [FakePerson("p00"), _second_screen("p01", seed=1)])
    out = tmp_path / "results" / "mpiifacegaze.json"

    code = mpii.main(
        ["evaluate", "--root", str(root), "--cache", str(cache), "--out", str(out)]
        + ["--variants", "eyeos"]
    )

    assert code == 0
    results = json.loads(out.read_text())
    assert results["dataset"] == "MPIIFaceGaze"
    assert results["license"] == "CC BY-NC-SA 4.0"
    assert len(results["citation"]) == 2
    assert results["generated"]
    assert "git_commit" in results
    assert results["config"]["k_per_dot"] == 5
    assert results["config"]["persons"] == ["p00", "p01"]
    assert results["config"]["gate2_max_validation_error_px"] == 60.0
    assert results["config"]["max_head_pose_delta_deg"] == 15.0
    assert list(results["variants"]) == ["eyeos"]
    assert set(results["screen_model"]) == {"p00", "p01"}
    assert "eyeos" in capsys.readouterr().out


def test_evaluate_cli_rejects_an_unknown_variant(tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        mpii.main(
            ["evaluate", "--root", str(tmp_path), "--cache", str(tmp_path)]
            + ["--out", str(tmp_path / "r.json"), "--variants", "eyeos,nope"]
        )

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "nope" in err and "eyeos" in err


def test_evaluate_cli_reports_an_empty_cache(tmp_path, capsys):
    code = mpii.main(
        ["evaluate", "--root", str(tmp_path), "--cache", str(tmp_path / "empty")]
        + ["--out", str(tmp_path / "r.json")]
    )

    assert code == 1
    assert "extract" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# variants: refit and compare models on the same data (no scipy needed)
# ---------------------------------------------------------------------------


def _person_data(tmp_path: Path, person: FakePerson) -> protocol.PersonData:
    """A fake person straight to PersonData through the fake detector: no .mat files."""
    write_person(tmp_path, dataclasses.replace(person, write_mat=False))
    annotations = mpii.load_annotations(tmp_path, person.name)
    detector = FakeDetector()
    return protocol.PersonData(
        person=person.name,
        screen_px=(float(person.screen_px[0]), float(person.screen_px[1])),
        targets_px=np.array([a.target_px for a in annotations]),
        landmarks=[detector(tmp_path / person.name / a.image) for a in annotations],
    )


def _person_data_3d(tmp_path: Path, person: FakePerson) -> protocol.PersonData:
    """_person_data plus the fake's 3D labels and its own screen model, still no scipy."""
    data = _person_data(tmp_path, person)

    def to_camera_mm(points_px: np.ndarray) -> np.ndarray:
        return np.array([person.gaze_target_mm(x, y) for x, y in points_px])

    return dataclasses.replace(
        data,
        face_centers_mm=np.tile(person.face_center, (len(data.targets_px), 1)),
        gaze_targets_mm=to_camera_mm(data.targets_px),
        to_camera_mm=to_camera_mm,
    )


def test_noise_free_protocol_pairs_valid_frames_with_their_targets_and_3d_labels(tmp_path):
    persons = [
        _person_data_3d(tmp_path, _with_invalid_frames(FakePerson("p00"))),
        _person_data_3d(tmp_path, _with_invalid_frames(_second_screen("p01", seed=1))),
    ]
    variants = models.build_variants()

    results = protocol.evaluate(persons, [variants["eyeos"], variants["global-lopo"]])

    for name in ("p00", "p01"):
        assert results["detection"]["persons"][name]["valid"] == 240 - 35 - 19
    for variant in ("eyeos", "global-lopo"):
        for name, person in results["variants"][variant]["persons"].items():
            assert person["n_calibration"] + person["n_test"] == 240 - 35 - 19
            assert person["pixel_error_px"]["mean"] < 0.75, (variant, name)
            assert person["angular_error_deg"]["mean"] < 0.02, (variant, name)
            assert person["validation_error_px"] < 0.5, (variant, name)


def _three_screens(tmp_path: Path, **kwargs: Any) -> list[protocol.PersonData]:
    return [
        _person_data(tmp_path, FakePerson("p00", **kwargs)),
        _person_data(tmp_path, _second_screen("p01", seed=1, **kwargs)),
        _person_data(
            tmp_path,
            FakePerson("p02", seed=2, screen_px=(1920, 1080), screen_mm=(344.0, 194.0), **kwargs),
        ),
    ]


def _mean_px(results: dict[str, Any], variant: str) -> float:
    mean: float = results["variants"][variant]["aggregate"]["pixel_error_px"][
        "mean_of_person_means"
    ]
    return mean


def test_the_variant_list_is_the_documented_one():
    assert models.VARIANT_NAMES == (
        "eyeos",
        "eyeos-poly2",
        "ridge-0.001",
        "ridge-0.1",
        "ridge-1",
        "ridge-10",
        "per-dot-1",
        "per-dot-5",
        "per-dot-10",
        "per-dot-20",
        "with-head-pose",
        "global-lopo",
    )


_DOC = ROOT / "docs" / "PUBLIC_DATASETS.md"


def test_the_doc_lists_every_variant_in_its_table():
    table_rows = [line for line in _DOC.read_text().splitlines() if line.startswith("| `")]
    # The first cell of each row names the variant(s); later cells may mention them again.
    first_cells = [row.split("|")[1] for row in table_rows]
    documented = [name for cell in first_cells for name in re.findall(r"`([a-z0-9.-]+)`", cell)]

    assert [n for n in documented if n in models.VARIANT_NAMES] == list(models.VARIANT_NAMES)


def test_the_doc_commands_parse_with_the_real_command_line():
    text = _DOC.read_text().replace("\\\n", " ")
    commands = [
        shlex.split(line.strip())
        for line in text.splitlines()
        if line.strip().startswith("python -m eval.public.mpiifacegaze ")
    ]

    assert {c[3] for c in commands} == {"extract", "evaluate"}
    for command in commands:
        args = mpii.build_parser().parse_args(command[3:])
        assert args.root == Path("datasets/MPIIFaceGaze")
        assert args.cache == Path("datasets/cache/mpiifacegaze")


def test_every_variant_runs_and_the_noise_free_ones_recover_the_targets(tmp_path):
    persons = _three_screens(tmp_path)

    results = protocol.evaluate(persons, list(models.build_variants().values()))

    assert list(results["variants"]) == list(models.VARIANT_NAMES)
    for name, variant in results["variants"].items():
        assert variant["aggregate"]["persons_evaluated"] == 3, name
        for person in variant["persons"].values():
            assert person["n_calibration"] == 9 * variant["k_per_dot"], name
            assert person["validation_error_px"] is not None, name
            assert person["angular_error_deg"] is None  # no 3D labels in this PersonData
    for name in (
        "eyeos",
        "ridge-0.001",
        "per-dot-1",
        "per-dot-5",
        "per-dot-10",
        "per-dot-20",
        "with-head-pose",
        "global-lopo",
    ):
        assert _mean_px(results, name) < 0.75, name
    assert [results["variants"][f"per-dot-{k}"]["k_per_dot"] for k in (1, 5, 10, 20)] == [
        1,
        5,
        10,
        20,
    ]


def test_variants_that_restate_the_shipping_calibration_reproduce_it_exactly(tmp_path):
    persons = _three_screens(tmp_path, noise=0.05)
    variants = models.build_variants()

    results = protocol.evaluate(
        persons, [variants[n] for n in ("eyeos", "ridge-0.001", "per-dot-5")]
    )

    eyeos = results["variants"]["eyeos"]["persons"]
    assert results["variants"]["ridge-0.001"]["persons"] == eyeos
    assert results["variants"]["per-dot-5"]["persons"] == eyeos


def test_a_stronger_ridge_pulls_predictions_toward_the_middle(tmp_path):
    persons = _three_screens(tmp_path)
    variants = models.build_variants()

    results = protocol.evaluate(
        persons, [variants[f"ridge-{lam}"] for lam in ("0.001", "0.1", "1", "10")]
    )

    errors = [_mean_px(results, f"ridge-{lam}") for lam in ("0.001", "0.1", "1", "10")]
    assert errors == sorted(errors)
    assert errors[-1] > 10 * max(errors[0], 0.1)


def _yaw_person(name: str = "p00", n: int = 240, seed: int = 0) -> protocol.PersonData:
    """Iris offsets that depend on head yaw as well as the target: turning the head
    toward a target leaves the eyes less to rotate."""
    rng = random.Random(seed)
    width, height = 1280.0, 800.0
    targets, landmarks = [], []
    for _ in range(n):
        x, y = rng.uniform(0, width - 1), rng.uniform(0, height - 1)
        nose_x = rng.uniform(0.45, 0.55)
        yaw_deg = (nose_x - 0.5) / 0.2 * 60.0  # what eyeos.features reads back
        ox, oy = offsets_for_target(x, y, width, height)
        landmarks.append(make_landmarks(ox - 0.01 * yaw_deg, oy, nose_x=nose_x))
        targets.append((x, y))
    return protocol.PersonData(name, (width, height), np.array(targets), landmarks)


def test_with_head_pose_learns_a_gaze_shift_that_follows_the_head(tmp_path):
    variants = models.build_variants()

    results = protocol.evaluate([_yaw_person()], [variants["eyeos"], variants["with-head-pose"]])

    shipping = results["variants"]["eyeos"]["persons"]["p00"]
    with_pose = results["variants"]["with-head-pose"]["persons"]["p00"]
    assert shipping["pixel_error_px"]["mean"] > 20
    assert shipping["validation_error_px"] > 20
    assert with_pose["pixel_error_px"]["mean"] < 0.75
    assert with_pose["validation_error_px"] < 0.5


def test_the_generic_leave_one_out_matches_calibrations_own(tmp_path):
    person = protocol.prepare(_person_data(tmp_path, FakePerson("p00", noise=0.05)))
    features = person.features[:45]
    targets = person.data.targets_px[person.valid_rows][:45]
    calibration = Calibration(1280.0, 800.0)
    for f, (x, y) in zip(features, targets, strict=True):
        calibration.add_sample((x, y), f)
    calibration.fit()

    def fit_predict(train: np.ndarray, held_out: int) -> np.ndarray:
        fold = Calibration(1280.0, 800.0)
        for i in train:
            fold.add_sample(tuple(targets[i]), features[i])
        fold.fit()
        return np.array(fold.predict(features[held_out]))

    generic = models.leave_one_out_rms_px(targets, fit_predict)

    assert generic == pytest.approx(calibration.validation_error_px, rel=1e-9)
    assert models.leave_one_out_rms_px(targets[:5], fit_predict) == math.inf


class _SpyVariant:
    """Records which other persons the protocol lets it learn from, per fit."""

    def __init__(self, needs_other_persons: bool) -> None:
        self.name = "spy"
        self.description = "records its fit contexts"
        self.k_per_dot = None
        self.needs_other_persons = needs_other_persons
        self.seen: list[tuple[str, ...]] = []

    def fit(self, features, targets_px, context):
        self.seen.append(tuple(p.name for p in context.other_persons))
        return models.build_variants()["eyeos"].fit(features, targets_px, context)


def test_leave_one_person_out_never_trains_on_the_person_it_tests(tmp_path):
    persons = _three_screens(tmp_path)
    lopo, plain = _SpyVariant(needs_other_persons=True), _SpyVariant(needs_other_persons=False)

    protocol.evaluate(persons, [lopo])
    protocol.evaluate(persons, [plain])

    assert lopo.seen == [("p01", "p02"), ("p00", "p02"), ("p00", "p01")]
    assert plain.seen == [(), (), ()]


def test_the_lopo_training_set_is_every_other_persons_valid_frames_in_screen_fractions(
    tmp_path,
):
    p00 = protocol.prepare(_person_data(tmp_path, FakePerson("p00", faces={0: 0, 1: 2})))
    p01 = protocol.prepare(_person_data(tmp_path, _second_screen("p01", seed=1)))

    features, fractions = models.lopo_training_set([p00, p01])

    assert features.shape == (238 + 240, 7)
    assert fractions.shape == (238 + 240, 2)
    # p00's images 0 (no face) and 1 (two faces) are left out, so its rows start at 2.
    assert fractions[:238] == pytest.approx(p00.data.targets_px[2:] / np.array([1280.0, 800.0]))
    assert fractions[238:] == pytest.approx(p01.data.targets_px / np.array([1440.0, 900.0]))
    assert features[0] == pytest.approx(
        [*p00.features[0].iris_vector(), *p00.features[0].head_pose()]
    )


def test_global_lopo_needs_another_person(tmp_path):
    persons = [_person_data(tmp_path, FakePerson("p00"))]

    results = protocol.evaluate(persons, [models.build_variants()["global-lopo"]])

    person = results["variants"]["global-lopo"]["persons"]["p00"]
    assert person == {"skipped": "needs at least one other person with valid frames"}
    assert results["variants"]["global-lopo"]["aggregate"]["pixel_error_px"] is None


def test_global_lopo_skips_instead_of_training_on_others_without_valid_frames(tmp_path):
    no_faces = {i: 0 for i in range(240)}
    persons = [
        _person_data(tmp_path, FakePerson("p00")),
        _person_data(tmp_path, FakePerson("p01", seed=1, faces=no_faces)),
    ]
    spy = _SpyVariant(needs_other_persons=True)

    results = protocol.evaluate(persons, [models.build_variants()["global-lopo"], spy])

    lopo = results["variants"]["global-lopo"]["persons"]
    assert lopo["p00"] == {"skipped": "needs at least one other person with valid frames"}
    assert "0 valid frames" in lopo["p01"]["skipped"]
    assert spy.seen == []


def test_global_lopo_per_user_correction_absorbs_a_quirk_the_others_do_not_share(tmp_path):
    persons = _three_screens(tmp_path)[:2] + [
        _person_data(tmp_path, FakePerson("p02", seed=2, iris_bias=(0.3, -0.2)))
    ]

    results = protocol.evaluate(persons, [models.build_variants()["global-lopo"]])

    for person in results["variants"]["global-lopo"]["persons"].values():
        assert person["pixel_error_px"]["mean"] < 0.75
        assert person["validation_error_px"] < 0.5
