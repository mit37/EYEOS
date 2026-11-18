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
LOCAL_SETUP = ROOT / "docs" / "LOCAL_SETUP.md"
PLAN = ROOT / "docs" / "PLAN.md"
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


@pytest.mark.parametrize(
    ("doc", "unsupported"),
    [
        # The gap to live use was never measured, in either direction.
        (PUBLIC_DOC, "pessimistic for live use"),
        (PUBLIC_DOC, "how much better it is"),
        # The harness's validation error is not the statistic live Gate 2 computes.
        (PUBLIC_DOC, "A live EyeOS session on these photos would refuse"),
        # Setup notes and wall times were recorded by hand; only the metrics are in it.
        (PUBLIC_DOC, "Every number here is in"),
        # p08 ranks 5th of 15 by relative gain; the relation with head motion is loose.
        (PUBLIC_DOC, "biggest for the people whose head moved most"),
        # The real run timed the extraction.
        (LOCAL_SETUP, "not timed here"),
    ],
    ids=lambda v: v.name if isinstance(v, Path) else v,
)
def test_the_docs_make_no_claim_the_real_run_does_not_support(doc, unsupported):
    prose = " ".join(doc.read_text(encoding="utf-8").split())

    assert unsupported not in prose


@pytest.mark.parametrize(
    ("doc", "in_progress"),
    [
        # The baselines, eyeos-live9 and `diagnose` were written after the real run, on
        # branches merged with these docs: they exist but have not run on the real data.
        (PUBLIC_DOC, "being added"),
        (PLAN, "being added"),
        (PLAN, "separate branches"),
        # The results file of the real run predates the `environment` block.
        (PUBLIC_DOC, "and the JSON does not record library versions"),
    ],
    ids=lambda v: v.name if isinstance(v, Path) else v,
)
def test_the_docs_do_not_call_work_added_after_the_real_run_unfinished(doc, in_progress):
    prose = " ".join(doc.read_text(encoding="utf-8").split())

    assert in_progress not in prose


def _live_comparison_items() -> dict[str, str]:
    """The labelled sub-bullets of "How these numbers compare with live use", flattened."""
    text = PUBLIC_DOC.read_text(encoding="utf-8")
    start = text.index("**How these numbers compare with live use")
    block = text[start : text.index("\n- **", start)]
    items = {}
    for item in re.split(r"\n  - ", block)[1:]:
        label, _, body = item.partition(":")
        items[label.strip()] = " ".join(body.split())
    return items


def test_the_live_comparison_lists_static_image_detection_with_no_direction():
    # The doc's own Limits say only that tracking mode "can behave differently".
    items = _live_comparison_items()

    assert "static-image mode" in items["Not known either way"]
    assert "static" not in items["Harder here"]
    assert "from scratch" not in items["Harder here"]


def test_the_live_comparison_lists_the_shared_calibration_pool_as_easier_here():
    # Both protocol/F4 verifiers: calibration and test photos come from one multi-day
    # pool, while a live calibration never samples the drift later in its own sitting.
    easier = _live_comparison_items()["Easier here"]

    assert "same multi-day pool" in easier
    assert "drift later in the sitting" in easier


def test_the_license_precaution_covers_every_copy_of_the_mpiifacegaze_metrics():
    # If the metrics might be adapted material, so are their copies in the README and the
    # docs; marking only the JSON would leave the same numbers under MIT next to it.
    text = " ".join(PUBLIC_DOC.read_text(encoding="utf-8").split())
    start = text.index("**Its license.**")
    note = text[start : text.index("## ", start)]

    for place in ("eval/results/mpiifacegaze.json", "README.md", "this document"):
        assert place in note, place


def test_the_readme_writes_gate_2s_head_pose_limit_without_a_degree_sign():
    # Yaw and pitch are uncalibrated pseudo-degrees, so "15°" would claim true degrees.
    text = README.read_text(encoding="utf-8")

    assert not re.search(
        r"cfg:gates\.calibration_max_head_pose_delta_deg -->[^<]*<!-- /cfg -->\s*°", text
    )


@pytest.mark.parametrize(
    ("doc", "start"),
    [
        (README, "**The head-pose estimate is uncalibrated.**"),
        (PUBLIC_DOC, "**EyeOS's head pose is a simplified, uncalibrated estimate**"),
    ],
    ids=["README.md", "PUBLIC_DATASETS.md"],
)
def test_the_head_pose_units_note_says_roll_is_in_real_degrees(doc, start):
    # features.py takes roll from atan2 on the eye-centre line; only yaw and pitch use
    # the hand-picked scales, and Gate 2's limit applies to the largest of the three.
    text = " ".join(doc.read_text(encoding="utf-8").split())
    begin = text.index(start)
    end = re.search(r" (- \*\*|#+ )", text[begin:])
    assert end is not None
    note = text[begin : begin + end.start()]

    assert "roll" in note.lower()
    assert "real degrees" in note


def _table_headers(text: str) -> list[str]:
    """Every header cell of every Markdown table in ``text``."""
    lines = text.splitlines()
    cells: list[str] = []
    for header, rule in zip(lines, lines[1:], strict=False):
        if header.startswith("|") and re.match(r"\|\s*-{3,}", rule):
            cells += [cell.strip() for cell in header.strip().strip("|").split("|")]
    return cells


def test_every_column_the_readme_prose_names_is_in_a_generated_table():
    # The generator owns the tables and may rename a column; prose that names one by a
    # name it no longer has points the reader at nothing.
    text = README.read_text(encoding="utf-8")
    prose = " ".join(_without_generated_regions(text).split())
    headers = _table_headers(text)

    for name in re.findall(r"\b[Tt]he ([^.;:,()]{1,60}?) column\b", prose):
        assert any(name in header for header in headers), name


@pytest.mark.parametrize(
    "phrase",
    [
        "What the synthetic numbers mean",
        # The caveat's body, so that a copy of it without the heading is caught too.
        "function of the noise injected into the synthetic fixture",
        "the pipeline's math does what it should",
    ],
)
def test_the_synthetic_caveat_sits_with_the_synthetic_table_not_under_the_real_one(phrase):
    text = " ".join(README.read_text(encoding="utf-8").split())
    synthetic_end = text.index("<!-- results:end -->")
    public_start = text.index("<!-- public-results:start -->")

    assert text.count(phrase) == 1
    assert synthetic_end < text.index(phrase) < public_start
