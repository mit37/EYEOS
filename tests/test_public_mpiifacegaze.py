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

import pytest
from packaging.requirements import Requirement

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
