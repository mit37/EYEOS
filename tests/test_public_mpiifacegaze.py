"""The MPIIFaceGaze public-dataset harness: the downloader, the annotation parser, the
screen model, extraction and the EyeOS-faithful evaluation protocol.

Everything runs on a small fake dataset written to tmp_path. The fake "images" are text
files that a fake detector turns into landmarks, so neither OpenCV nor MediaPipe is
needed, and no network is used. The .mat calibration files are written with scipy; the
tests that need them skip when scipy is missing (it is in the ``dev`` extra, so CI runs
them).
"""

from __future__ import annotations

import importlib.util
import io
import math
import os
import shutil
import socket
import subprocess
import sys
import types
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pytest
from packaging.requirements import Requirement

from eval.public import mpiifacegaze as mpii
from tests.fake_mpiifacegaze import (
    FakePerson,
    annotation_line,
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
