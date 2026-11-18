"""scripts/setup_unix.sh, driven in its dry-run mode against fake interpreters.

Each test copies the script into a throwaway "repo" and runs it with PATH holding only
fake ``python3.x`` executables (tiny sh scripts that print a version), so it never
touches a real Python, the network or this checkout's .venv. The real end-to-end run is
recorded in docs/LOCAL_SETUP.md.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
UNIX_SCRIPT = REPO_ROOT / "scripts" / "setup_unix.sh"
EXTRAS = "dev,webcam,windows-fallback,public-eval"
BASH = shutil.which("bash")

unix_only = pytest.mark.skipif(
    sys.platform == "win32" or BASH is None or not Path("/bin/sh").exists(),
    reason="setup_unix.sh is for Linux/macOS (needs bash and /bin/sh)",
)


def _fake_python(path: Path, version: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\necho {version}\n")
    path.chmod(0o755)
    return path


def _run(
    base: Path, pythons: dict[str, str], *, pyproject: bool = True, **env: str
) -> subprocess.CompletedProcess:
    """Run a copy of the script in base/repo (keeping anything already there, such as a
    .venv) with PATH = base/bin holding the given fake interpreters."""
    repo = base / "repo"
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy(UNIX_SCRIPT, repo / "scripts" / "setup_unix.sh")
    if pyproject:
        (repo / "pyproject.toml").write_text('[project]\nname = "eyeos"\n')
    bin_dir = base / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name, version in pythons.items():
        _fake_python(bin_dir / name, version)
    assert BASH is not None
    return subprocess.run(
        [BASH, str(repo / "scripts" / "setup_unix.sh")],
        env={"PATH": str(bin_dir), "EYEOS_SETUP_DRY_RUN": "1", **env},
        capture_output=True,
        text=True,
        timeout=60,
    )


def _commands(result: subprocess.CompletedProcess) -> list[str]:
    """The commands the dry run would have run, in order."""
    return [line[2:] for line in result.stdout.splitlines() if line.startswith("+ ")]


def _chosen(result: subprocess.CompletedProcess) -> str:
    """The interpreter the dry run would build .venv with."""
    for command in _commands(result):
        if command.endswith(" -m venv .venv"):
            return Path(command[: -len(" -m venv .venv")]).name
    raise AssertionError(f"no venv command in:\n{result.stdout}\n{result.stderr}")


@unix_only
def test_prefers_the_newest_supported_python(tmp_path):
    result = _run(tmp_path, {"python3.12": "3.12", "python3.11": "3.11", "python3.10": "3.10"})
    assert result.returncode == 0, result.stderr
    assert _chosen(result) == "python3.12"


@unix_only
def test_falls_back_to_3_11_then_3_10(tmp_path):
    both = _run(tmp_path / "a", {"python3.11": "3.11", "python3.10": "3.10"})
    assert _chosen(both) == "python3.11"
    assert _chosen(_run(tmp_path / "b", {"python3.10": "3.10"})) == "python3.10"


@unix_only
def test_skips_a_python3_that_is_too_new(tmp_path):
    result = _run(tmp_path, {"python3": "3.13", "python3.11": "3.11"})
    assert result.returncode == 0, result.stderr
    assert _chosen(result) == "python3.11"


@unix_only
def test_uses_plain_python3_when_it_is_supported(tmp_path):
    assert _chosen(_run(tmp_path, {"python3": "3.12"})) == "python3"


@unix_only
@pytest.mark.parametrize("pythons", [{"python3": "3.13", "python": "3.9"}, {}])
def test_no_supported_python_fails_and_says_why(tmp_path, pythons):
    result = _run(tmp_path, pythons)
    assert result.returncode != 0
    assert _commands(result) == []
    assert "3.10-3.12" in result.stderr
    assert "mediapipe<0.10.30" in result.stderr
    for name, version in pythons.items():
        assert f"{name} is {version}" in result.stderr


@unix_only
def test_eyeos_python_overrides_the_search_and_is_checked(tmp_path):
    chosen = _fake_python(tmp_path / "custom" / "my-python", "3.11")
    result = _run(tmp_path / "ok", {"python3.12": "3.12"}, EYEOS_PYTHON=str(chosen))
    assert result.returncode == 0, result.stderr
    assert _chosen(result) == "my-python"

    too_new = _fake_python(tmp_path / "custom" / "new-python", "3.13")
    result = _run(tmp_path / "bad", {"python3.12": "3.12"}, EYEOS_PYTHON=str(too_new))
    assert result.returncode != 0
    assert _commands(result) == []
    assert "Python 3.13" in result.stderr and "3.10-3.12" in result.stderr


@unix_only
def test_dry_run_prints_every_step_in_order(tmp_path):
    result = _run(tmp_path, {"python3.12": "3.12"})
    assert result.returncode == 0, result.stderr
    venv_python = ".venv/bin/python"
    assert _commands(result)[1:] == [
        f"{venv_python} -m pip install --upgrade pip",
        f'{venv_python} -m pip install -e ".[{EXTRAS}]"',
        f"{venv_python} scripts/check_setup.py",
        f"{venv_python} -m pytest -q",
    ]
    assert "Next steps" in result.stdout


@unix_only
def test_skip_tests_leaves_out_pytest(tmp_path):
    result = _run(tmp_path, {"python3.12": "3.12"}, EYEOS_SKIP_TESTS="1")
    assert result.returncode == 0, result.stderr
    assert not [c for c in _commands(result) if " -m pytest" in c]


@unix_only
def test_reuses_a_venv_made_with_a_supported_python(tmp_path):
    _fake_python(tmp_path / "repo" / ".venv" / "bin" / "python", "3.11")
    result = _run(tmp_path, {"python3.12": "3.12"})
    assert result.returncode == 0, result.stderr
    assert "reusing .venv (Python 3.11)" in result.stdout
    assert not [c for c in _commands(result) if " -m venv " in c]


@unix_only
def test_refuses_a_venv_made_with_an_unsupported_python(tmp_path):
    _fake_python(tmp_path / "repo" / ".venv" / "bin" / "python", "3.13")
    result = _run(tmp_path, {"python3.12": "3.12"})
    assert result.returncode != 0
    assert ".venv" in result.stderr and "3.13" in result.stderr
    assert _commands(result) == []


@unix_only
def test_refuses_to_run_outside_a_checkout(tmp_path):
    result = _run(tmp_path, {"python3.12": "3.12"}, pyproject=False)
    assert result.returncode != 0
    assert "pyproject.toml" in result.stderr
    assert _commands(result) == []
