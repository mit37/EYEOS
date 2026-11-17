"""The MPIIFaceGaze public-dataset harness: the downloader, the annotation parser, the
screen model, extraction and the EyeOS-faithful evaluation protocol.

Everything runs on a small fake dataset written to tmp_path. The fake "images" are text
files that a fake detector turns into landmarks, so neither OpenCV nor MediaPipe is
needed, and no network is used. The .mat calibration files are written with scipy; the
tests that need them skip when scipy is missing (it is in the ``dev`` extra, so CI runs
them).
"""

from __future__ import annotations

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
