"""scripts/check_setup.py: the post-install check the setup scripts run.

OpenCV and MediaPipe are the fakes from tests/fake_capture.py, and pynput/scipy are
looked up through a patched metadata lookup, so no real library, camera or keyboard
is touched.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tests.fake_capture import FakeCv2, FakeMediapipe, install_fakes

REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = REPO_ROOT / "scripts" / "check_setup.py"
_spec = importlib.util.spec_from_file_location("check_setup", _SCRIPT)
assert _spec is not None and _spec.loader is not None
check_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_setup)


@pytest.fixture
def all_installed(monkeypatch):
    """A complete install: fake OpenCV + MediaPipe, and pynput + scipy present."""
    mediapipe = FakeMediapipe(faces_per_frame=0)
    install_fakes(monkeypatch, cv2=FakeCv2(), mediapipe=mediapipe)
    versions = {"pynput": "1.8.2", "scipy": "1.17.1"}
    monkeypatch.setattr(check_setup, "_dist_version", versions.get)
    monkeypatch.setattr(check_setup, "_python_version", lambda: (3, 12))
    return mediapipe


def _by_name(results):
    return {r.name: r for r in results}


@pytest.mark.parametrize("version", [(3, 10), (3, 11), (3, 12)])
def test_python_3_10_to_3_12_passes(version):
    assert check_setup.check_python(version).status == "ok"


@pytest.mark.parametrize("version", [(3, 9), (3, 13), (3, 14)])
def test_other_pythons_fail_and_say_why(version):
    result = check_setup.check_python(version)
    assert result.status == "fail"
    assert "3.10-3.12" in result.detail
    assert "mediapipe<0.10.30" in result.detail
    # Not "wheels for 3.10-3.12 only": mediapipe 0.10.21 also ships 3.9 wheels.
    assert "no wheels for Python 3.13 or newer" in result.detail


def test_a_complete_install_passes_and_runs_face_mesh_once(all_installed, capsys):
    assert check_setup.main([]) == 0

    out = capsys.readouterr().out
    assert "All required checks passed" in out
    assert "[FAIL]" not in out
    # FaceMesh was really built (with the iris landmarks) and given one blank frame.
    (mesh,) = all_installed.face_meshes
    assert mesh.kwargs["refine_landmarks"] is True
    assert len(mesh.processed) == 1
    assert getattr(mesh.processed[0], "shape", None) == (480, 640, 3)
    assert mesh.close_calls == 1


def test_mediapipe_without_the_solutions_api_fails_with_the_python_hint(
    all_installed, monkeypatch, capsys
):
    install_fakes(monkeypatch, mediapipe=FakeMediapipe(solutions=False, version="0.10.31"))

    assert check_setup.main([]) == 1

    out = capsys.readouterr().out
    assert "Solutions API" in out
    assert "0.10.31" in out
    assert "no wheels for Python 3.13 or newer" in out


@pytest.mark.parametrize("missing", ["cv2", "mediapipe"])
def test_missing_webcam_libraries_fail_with_the_install_command(
    all_installed, monkeypatch, capsys, missing
):
    monkeypatch.setitem(sys.modules, missing, None)

    assert check_setup.main([]) == 1

    assert 'pip install -e ".[webcam]"' in capsys.readouterr().out


def test_a_face_mesh_that_cannot_start_fails_with_its_error(all_installed, monkeypatch, capsys):
    install_fakes(
        monkeypatch, mediapipe=FakeMediapipe(face_mesh_error=RuntimeError("model file missing"))
    )

    assert check_setup.main([]) == 1

    assert "model file missing" in capsys.readouterr().out


def test_missing_pynput_fails_because_the_kill_switch_needs_it(all_installed, monkeypatch, capsys):
    monkeypatch.setattr(check_setup, "_dist_version", {"scipy": "1.17.1"}.get)

    assert check_setup.main([]) == 1

    out = capsys.readouterr().out
    assert "[FAIL] pynput" in out
    assert "windows-fallback" in out


def test_missing_scipy_only_warns(all_installed, monkeypatch, capsys):
    monkeypatch.setattr(check_setup, "_dist_version", {"pynput": "1.8.2"}.get)

    assert check_setup.main([]) == 0

    out = capsys.readouterr().out
    assert "[warn] scipy" in out
    assert "public-eval" in out


def test_a_wrong_python_fails_the_whole_check(all_installed, monkeypatch, capsys):
    monkeypatch.setattr(check_setup, "_python_version", lambda: (3, 13))

    assert check_setup.main([]) == 1
    assert "[FAIL] Python 3.13" in capsys.readouterr().out


def test_eyeos_from_another_checkout_warns(tmp_path):
    result = check_setup.check_eyeos(repo_root=tmp_path)
    assert result.status == "warn"
    assert "another checkout" in result.detail
    assert check_setup.check_eyeos(repo_root=REPO_ROOT).status == "ok"


def test_output_is_plain_ascii(all_installed, capsys):
    """Windows consoles and redirected output may not be UTF-8."""
    check_setup.main([])
    capsys.readouterr().out.encode("ascii")
