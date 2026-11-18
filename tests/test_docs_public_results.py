"""Hand-written docs about the MPIIFaceGaze run: they must not contradict the committed
results file (``eval/results/mpiifacegaze.json``).

The README's numbers are generated and drift-checked by ``scripts/update_readme.py``.
These tests guard the prose around them, which no script writes: facts about the
dataset that the results file records, and statements that went stale once a real run
was committed.
"""

from __future__ import annotations

import importlib.util
import json
import re
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results" / "mpiifacegaze.json"
README = ROOT / "README.md"
PUBLIC_DOC = ROOT / "docs" / "PUBLIC_DATASETS.md"
DOWNLOADER = ROOT / "scripts" / "download_mpiifacegaze.py"

needs_results = pytest.mark.skipif(
    not RESULTS.exists(), reason="no committed MPIIFaceGaze results file"
)


def _results() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(RESULTS.read_text(encoding="utf-8"))
    return data


def _downloader() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("download_mpiifacegaze", DOWNLOADER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _image_counts() -> tuple[int, int, int]:
    images = [p["images"] for p in _results()["detection"]["persons"].values()]
    return min(images), max(images), sum(images)


def _without_generated_regions(text: str) -> str:
    """README text with every ``<!-- name:start -->...<!-- name:end -->`` region cut out."""
    return re.sub(r"<!-- ([\w-]+):start -->.*?<!-- \1:end -->", "", text, flags=re.DOTALL)


@needs_results
def test_the_license_notice_gives_the_real_number_of_images_per_person():
    lo, hi, _ = _image_counts()
    notice = " ".join(_downloader().LICENSE_NOTICE.split())

    assert f"{lo:,} to {hi:,}" in notice
    assert "about 3000" not in notice


@needs_results
def test_the_dataset_doc_gives_the_real_number_of_images_per_person_and_in_all():
    lo, hi, total = _image_counts()
    section = PUBLIC_DOC.read_text(encoding="utf-8").split("## MPIIFaceGaze", 1)[1]
    intro = " ".join(section.split("**License", 1)[0].split())

    assert f"{lo:,} to {hi:,}" in intro
    assert f"{total:,} in all" in intro
    assert "about 3000" not in intro


@needs_results
@pytest.mark.parametrize(
    "stale",
    [
        # What's in it, before the real run: the harness had never met the real data.
        "never run on the real dataset",
        # What this does not do: the MPIIFaceGaze table in Results is real photos.
        "No real accuracy numbers yet",
        "Every number in Results comes from a synthetic fixture",
        # The format was checked against the real files on the run.
        "untested against the real files",
    ],
)
def test_the_readme_prose_has_no_claim_the_real_run_made_false(stale):
    prose = " ".join(_without_generated_regions(README.read_text(encoding="utf-8")).split())

    assert stale not in prose


def test_the_synthetic_caveat_sits_with_the_synthetic_table_not_under_the_real_one():
    text = README.read_text(encoding="utf-8")
    caveat = text.index("What the synthetic numbers mean")
    public_end = text.index("<!-- public-results:end -->")
    after_public = text[public_end : text.index("\n## ", public_end)]

    assert text.index("<!-- results:end -->") < caveat < text.index("<!-- public-results:start -->")
    assert "injected noise" not in after_public
