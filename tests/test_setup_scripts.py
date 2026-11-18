"""The setup scripts.

scripts/setup_unix.sh is driven in its dry-run mode against fake interpreters: each
test copies the script into a throwaway "repo" and runs it with PATH holding only fake
``python3.x`` executables (tiny sh scripts that print a version), so it never touches a
real Python, the network or this checkout's .venv. The real end-to-end run is recorded
in docs/LOCAL_SETUP.md.

scripts/setup_windows.ps1 has never run (no PowerShell or Windows where it was
written), so it only gets static checks: that it stays plain ASCII and in step with
setup_unix.sh.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
UNIX_SCRIPT = REPO_ROOT / "scripts" / "setup_unix.sh"
WINDOWS_SCRIPT = REPO_ROOT / "scripts" / "setup_windows.ps1"
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
    assert "no wheels for Python 3.13 or newer" in result.stderr
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


def _fake_python_without_pip(path: Path, version: str) -> Path:
    """The .venv python a failed ``python -m venv`` leaves behind: it runs, but has no pip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then\n'
        '    echo "$0: No module named pip" >&2\n'
        "    exit 1\n"
        "fi\n"
        f"echo {version}\n"
    )
    path.chmod(0o755)
    return path


def _fake_debian_python(path: Path, version: str, venv_python: Path) -> Path:
    """Debian/Ubuntu's pythonX.Y without the pythonX.Y-venv package: ``-m venv DIR``
    makes DIR/bin/python, then fails because ensurepip is missing."""
    mkdir, cp = shutil.which("mkdir"), shutil.which("cp")
    assert mkdir is not None and cp is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then\n'
        f'    {shlex.quote(mkdir)} -p "$3/bin"\n'
        f'    {shlex.quote(cp)} {shlex.quote(str(venv_python))} "$3/bin/python"\n'
        '    echo "The virtual environment was not created successfully because ensurepip '
        'is not available."\n'
        "    exit 1\n"
        "fi\n"
        f"echo {version}\n"
    )
    path.chmod(0o755)
    return path


@unix_only
def test_refuses_a_reused_venv_without_pip(tmp_path):
    _fake_python_without_pip(tmp_path / "repo" / ".venv" / "bin" / "python", "3.12")
    result = _run(tmp_path, {"python3.12": "3.12"})
    assert result.returncode != 0
    assert "no pip" in result.stderr
    assert "rm -rf .venv" in result.stderr
    assert "no network" not in result.stderr
    assert _commands(result) == []  # not even the pip upgrade


@unix_only
def test_a_venv_left_by_a_failed_creation_is_not_reused(tmp_path):
    """The Debian/Ubuntu path: no python3.12-venv, then a rerun after installing it.
    Not a dry run: every interpreter here is a fake shell script."""
    template = _fake_python_without_pip(tmp_path / "template" / "python", "3.12")
    _fake_debian_python(tmp_path / "bin" / "python3.12", "3.12", template)

    first = _run(tmp_path, {}, EYEOS_SETUP_DRY_RUN="")
    assert first.returncode != 0
    assert "sudo apt install python3.12-venv" in first.stderr
    assert "rm -rf .venv" in first.stderr
    assert (tmp_path / "repo" / ".venv" / "bin" / "python").exists()  # half made

    _fake_python(tmp_path / "bin" / "python3.12", "3.12")  # python3.12-venv installed
    second = _run(tmp_path, {}, EYEOS_SETUP_DRY_RUN="")
    assert second.returncode != 0
    assert "reusing .venv" not in second.stdout
    assert "no pip" in second.stderr and "rm -rf .venv" in second.stderr
    assert "no network" not in second.stderr


@unix_only
def test_refuses_to_run_outside_a_checkout(tmp_path):
    result = _run(tmp_path, {"python3.12": "3.12"}, pyproject=False)
    assert result.returncode != 0
    assert "pyproject.toml" in result.stderr
    assert _commands(result) == []


# ---------------------------------------------------------------------------
# setup_windows.ps1: static checks only (it has never run)
# ---------------------------------------------------------------------------


def test_windows_script_is_plain_ascii():
    """Windows PowerShell 5.1 reads a script without a BOM in the ANSI code page, so a
    non-ASCII character (an em dash, a degree sign) could break a string."""
    WINDOWS_SCRIPT.read_bytes().decode("ascii")


def test_windows_script_header_says_it_has_never_run():
    header = WINDOWS_SCRIPT.read_text().split("#>", 1)[0]
    assert "UNTESTED" in header
    assert "setup_unix.sh" in header  # and points at the twin that has run


def test_both_scripts_install_the_same_extras():
    unix = re.search(r'^EXTRAS="([^"]+)"$', UNIX_SCRIPT.read_text(), re.MULTILINE)
    windows = re.search(r'^\$Extras = "([^"]+)"$', WINDOWS_SCRIPT.read_text(), re.MULTILINE)
    assert unix is not None and windows is not None
    assert unix.group(1) == windows.group(1) == EXTRAS


def test_windows_script_asks_the_py_launcher_for_3_12_then_3_11_then_3_10():
    text = WINDOWS_SCRIPT.read_text()
    candidates = re.search(r"^\$Candidates = @\(([^)]*)\)$", text, re.MULTILINE)
    assert candidates is not None
    assert re.findall(r'"([^"]+)"', candidates.group(1)) == ["3.12", "3.11", "3.10"]
    assert '& py "-$v"' in text


def test_windows_script_explains_the_python_limit_accurately():
    source = re.search(r"^\$Why312 = (.*?)\n\n", WINDOWS_SCRIPT.read_text(), re.M | re.S)
    assert source is not None
    # The message as PowerShell builds it: its "..." + "..." pieces joined.
    message = "".join(re.findall(r'"([^"]*)"', source.group(1)))
    assert "no wheels for Python 3.13 or newer" in message
    assert "3.10-3.12 only" not in message  # mediapipe 0.10.21 also ships 3.9 wheels


def test_windows_script_requires_a_64_bit_x86_python():
    """mediapipe<0.10.30's Windows wheels are win_amd64 only."""
    assert '-ne "win-amd64"' in WINDOWS_SCRIPT.read_text()


def test_windows_script_runs_the_same_steps_in_the_same_order():
    text = WINDOWS_SCRIPT.read_text()
    steps = [
        "-m venv .venv",
        "-m pip install --upgrade pip",
        '-m pip install -e ".[$Extras]"',
        r"scripts\check_setup.py",
        "-m pytest -q",
    ]
    # Each step is both shown ("+ ...") and run ("& ..."); take where it is run.
    positions = []
    for step in steps:
        runs = [m.start() for m in re.finditer(re.escape(step), text)]
        run_at = [p for p in runs if text.rfind("& ", 0, p) > text.rfind("\n", 0, p)]
        assert run_at, f"setup_windows.ps1 never runs: {step}"
        positions.append(run_at[0])
    assert positions == sorted(positions)
    # And every native step's exit code is checked.
    assert text.count("if ($LASTEXITCODE -ne 0)") >= len(steps)


def test_windows_script_refuses_a_reused_venv_without_pip():
    """Like setup_unix.sh: a .venv left by a failed venv creation runs but has no pip,
    and must not be reused (pip's failure would read as a network problem)."""
    text = WINDOWS_SCRIPT.read_text()
    start = text.index("if (Test-Path -LiteralPath $VenvPython) {")
    reuse = text[start : text.index('Say "reusing .venv', start)]
    assert "& $VenvPython -m pip --version" in reuse
    assert "no pip" in reuse and "Remove-Item -Recurse -Force .venv" in reuse
    created = text[text.index("-m venv .venv", start) : text.index("# --- 4.")]
    failed = created[created.index("if ($LASTEXITCODE -ne 0)") :]
    assert "Remove-Item -Recurse -Force .venv" in failed
