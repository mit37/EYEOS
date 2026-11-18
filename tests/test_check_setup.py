"""scripts/check_setup.py: the post-install check the setup scripts run.

OpenCV and MediaPipe are the fakes from tests/fake_capture.py, and pynput/scipy are
looked up through a patched metadata lookup, so no real library, camera or keyboard
is touched.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import types
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


@pytest.mark.parametrize(("module", "label"), [("cv2", "OpenCV"), ("mediapipe", "MediaPipe")])
def test_a_library_that_raises_on_import_fails_its_own_line(
    all_installed, monkeypatch, capsys, tmp_path, module, label
):
    """A broken install often raises something other than ImportError on import (the
    protobuf "Descriptors cannot be created directly" TypeError, a NumPy ABI
    AttributeError): it must become a [FAIL] line, not a traceback with no lines."""
    package = tmp_path / module
    package.mkdir()
    (package / "__init__.py").write_text(
        'raise TypeError("Descriptors cannot be created directly.")\n'
    )
    monkeypatch.delitem(sys.modules, module)
    monkeypatch.syspath_prepend(str(tmp_path))

    assert check_setup.main([]) == 1

    out = capsys.readouterr().out
    assert (
        f"[FAIL] {label}: installed but failed to import: "
        "TypeError: Descriptors cannot be created directly." in out
    )
    assert "[ok]   pynput 1.8.2" in out  # the checks after it still ran
    assert "[ok]   scipy 1.17.1" in out


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


NON_ASCII_PYTHON = "C:\\Users\\Jos\u00e9\\eyeos\\.venv\\Scripts\\python.exe"
NON_ASCII_EYEOS = "/home/Zo\u00eb/\u773c\u775b/eyeos/eyeos/__init__.py"


def _fake_eyeos(monkeypatch, file: str) -> None:
    """Make ``import eyeos`` (inside check_eyeos) find a package at ``file``."""
    module = types.ModuleType("eyeos")
    module.__file__ = file
    module.__version__ = "2.0.0"
    monkeypatch.setitem(sys.modules, "eyeos", module)


def test_output_is_plain_ascii_even_with_non_ascii_paths(all_installed, monkeypatch, capsys):
    """Windows consoles and redirected output may not be UTF-8, and the guide clones into
    $HOME\\eyeos, so the Python and eyeos paths it prints can hold any character."""
    monkeypatch.setattr(sys, "executable", NON_ASCII_PYTHON)
    _fake_eyeos(monkeypatch, NON_ASCII_EYEOS)

    check_setup.main([])

    out = capsys.readouterr().out
    out.encode("ascii")
    # Escaped, not dropped: the paths stay recognisable.
    assert "Jos\\xe9" in out and "Zo\\xeb" in out


def test_a_stdout_that_cannot_encode_the_paths_does_not_crash(all_installed, monkeypatch):
    """A redirected stdout in a legacy code page: print() itself must not raise."""
    monkeypatch.setattr(sys, "executable", NON_ASCII_PYTHON)
    _fake_eyeos(monkeypatch, NON_ASCII_EYEOS)
    raw = io.BytesIO()
    stdout = io.TextIOWrapper(raw, encoding="ascii", errors="strict")
    monkeypatch.setattr(sys, "stdout", stdout)

    status = check_setup.main([])

    stdout.flush()
    out = raw.getvalue().decode("ascii")
    assert status == 0
    assert "Jos\\xe9" in out.splitlines()[0]
    assert "[warn] eyeos 2.0.0: imported from /home/Zo\\xeb/" in out
    assert out.rstrip().endswith("All required checks passed.")


@pytest.mark.parametrize(
    "site_packages",
    [".venv/lib/python3.12/site-packages", ".venv/Lib/site-packages"],  # Linux/macOS, Windows
)
def test_eyeos_installed_as_a_copy_inside_the_checkout_warns(monkeypatch, tmp_path, site_packages):
    """pip install without -e puts a frozen copy in .venv, and .venv is inside the
    checkout: `eyeos run` would run that copy, not this folder's code after a git pull."""
    (tmp_path / "eyeos").mkdir()
    _fake_eyeos(monkeypatch, str(tmp_path / site_packages / "eyeos" / "__init__.py"))

    result = check_setup.check_eyeos(repo_root=tmp_path)

    assert result.status == "warn"
    assert "a copy" in result.detail and "without -e" in result.detail
    assert "pip install -e ." in result.detail


def test_eyeos_from_this_checkouts_eyeos_folder_is_ok(monkeypatch, tmp_path):
    _fake_eyeos(monkeypatch, str(tmp_path / "eyeos" / "__init__.py"))
    result = check_setup.check_eyeos(repo_root=tmp_path)
    assert result.status == "ok"
    assert str(tmp_path / "eyeos") in result.detail
