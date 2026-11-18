"""docs/LOCAL_SETUP.md: the commands and config snippets it tells people to use.

A guide that has never run on Windows can at least be held to this: every YAML snippet
loads as a real config, the one-line false-click counter really counts clicks in a
session log, the install command matches the setup scripts, and every ``eyeos run``
flag it uses exists.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pytest

from eyeos.cli import build_arg_parser
from eyeos.config import load_config
from eyeos.gates import GateStatus
from eyeos.logging_utils import SessionLogger

REPO_ROOT = Path(__file__).resolve().parent.parent
GUIDE = REPO_ROOT / "docs" / "LOCAL_SETUP.md"
DEMO = REPO_ROOT / "docs" / "DEMO.md"
INSTALL = 'pip install -e ".[dev,webcam,windows-fallback,public-eval]"'

# Built on parallel branches (agents "view" and "public") and documented from their
# shared interface. Once those branches are merged, nothing here is missing any more.
PENDING_RUN_FLAGS = {"--view"}
PENDING_EXTRAS = {"public-eval"}


def _guide() -> str:
    return GUIDE.read_text(encoding="utf-8")


def _code_blocks(text: str, lang: str | None = None) -> list[str]:
    blocks = re.findall(r"^```(\w*)\n(.*?)^```", text, re.MULTILINE | re.DOTALL)
    return [body for tag, body in blocks if lang is None or tag == lang]


def test_demo_links_the_guide_in_its_intro():
    intro = DEMO.read_text(encoding="utf-8").split("\n## ", 1)[0]
    assert "(LOCAL_SETUP.md)" in intro  # a relative link that works on GitHub


@pytest.mark.parametrize("index", range(5))
def test_yaml_snippets_are_valid_configs(tmp_path, index):
    blocks = _code_blocks(_guide(), "yaml")
    assert len(blocks) >= 5, "expected the safe, keypress, blink-hold, zone and reading configs"
    path = tmp_path / "config.yaml"
    path.write_text(blocks[index])
    load_config(path)  # raises on an unknown key or an invalid value


def test_every_yaml_snippet_is_checked():
    assert len(_code_blocks(_guide(), "yaml")) == 5  # bump the parametrize range with this


def _one_liner() -> str:
    """The python -c program in the block right after the false-click marker."""
    text = _guide()
    marker = text.index("<!-- false-click-count -->")
    block = _code_blocks(text[marker:])[0]
    match = re.search(r'^python -c "([^"]+)" (\S+)$', block.strip(), re.MULTILINE)
    assert match, f"no python -c one-liner in:\n{block}"
    return match.group(1)


def _status(click: bool, dwell: bool) -> GateStatus:
    return GateStatus(
        gate1_presence=True,
        gate2_calibration=True,
        gate3_stability=dwell,
        gate4_kill_and_limits=True,
        move_allowed=True,
        cursor_target_px=(10.0, 20.0),
        click_fired=click,
        reasons={"gate1": "ok"},
    )


def test_the_false_click_one_liner_counts_clicks_in_a_real_session_log(tmp_path):
    log = tmp_path / "reading.jsonl"
    frames = [  # (timestamp_s, click_fired, dwell satisfied): dwells outnumber clicks
        (2.0, False, True),
        (60.0, True, True),
        (150.0, False, True),
        (240.0, True, True),
        (302.4, False, False),
    ]
    with SessionLogger(log) as logger:
        for t, click, dwell in frames:
            logger.log(t, _status(click, dwell))

    out = subprocess.run(
        [sys.executable, "-c", _one_liner(), str(log)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout

    assert out.strip() == "2 clicks in 300 s over 5 frames"


def test_the_one_liner_survives_powershell_and_bash_quoting():
    """No double quotes, dollar signs, backticks or exclamation marks inside it."""
    program = _one_liner()
    assert not set(program) & set('"$`!')


def test_install_command_matches_the_setup_scripts():
    assert INSTALL in _guide()
    extras = INSTALL.split("[", 1)[1].split("]", 1)[0]
    for script in ("setup_unix.sh", "setup_windows.ps1"):
        assert extras in (REPO_ROOT / "scripts" / script).read_text()


def test_installed_extras_exist_in_pyproject():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        defined = set(tomllib.load(f)["project"]["optional-dependencies"])
    wanted = set(INSTALL.split("[", 1)[1].split("]", 1)[0].split(","))
    assert wanted - defined <= PENDING_EXTRAS


def test_every_eyeos_run_flag_in_the_guide_exists():
    subcommands = next(
        a for a in build_arg_parser()._actions if isinstance(a, argparse._SubParsersAction)
    )
    known = set(subcommands.choices["run"]._option_string_actions)
    used = set()
    for block in _code_blocks(_guide()):
        for line in block.splitlines():
            if line.lstrip().startswith("eyeos run"):
                used |= set(re.findall(r"(?<!\S)--[a-z][a-z-]*", line))
    assert {"--config", "--calibrate", "--overlay"} <= used
    assert used - known <= PENDING_RUN_FLAGS


def test_the_guide_says_what_has_not_run():
    text = _guide()
    assert "setup_windows.ps1" in text
    assert "never been run" in text  # the Windows script
    assert "Python 3.12" in text and "mediapipe<0.10.30" in text
