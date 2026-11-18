#!/usr/bin/env python3
"""Check that this Python can run EyeOS live. The setup scripts run it after installing.

    python scripts/check_setup.py

It checks the Python version, where ``eyeos`` is imported from, OpenCV, MediaPipe's
legacy Solutions API (by running FaceMesh once on a blank frame), and that pynput
(the kill switch) and scipy (the MPIIFaceGaze eval) are installed. The test suite runs
OpenCV and MediaPipe only as fakes, so it passes even when the real ones are broken;
this check is what catches that.

It opens no camera, shows no window and starts no keyboard listener. Exit code 0 when
everything a live session needs is there, 1 otherwise. Output is plain ASCII, because a
Windows console or a redirected output may not be UTF-8.
"""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path
from typing import Any, NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_PYTHONS = ((3, 10), (3, 11), (3, 12))
WEBCAM_INSTALL = 'pip install -e ".[webcam]"'


class Result(NamedTuple):
    name: str
    status: str  # "ok" | "warn" | "fail"; only "fail" makes the check fail
    detail: str


def _python_version() -> tuple[int, int]:
    return sys.version_info[0], sys.version_info[1]


def _dist_version(dist: str) -> str | None:
    """The installed version of a distribution, without importing it."""
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_python(version: tuple[int, int]) -> Result:
    name = f"Python {version[0]}.{version[1]}"
    if version in SUPPORTED_PYTHONS:
        return Result(name, "ok", "inside 3.10-3.12")
    return Result(
        name,
        "fail",
        "EyeOS needs Python 3.10-3.12: its webcam extra needs mediapipe<0.10.30 (the "
        "last versions with the legacy Solutions API), which has no wheels for Python "
        "3.13 or newer. Make the .venv with Python 3.12 (Windows: py -3.12 -m venv .venv) "
        "and install again.",
    )


def check_eyeos(repo_root: Path = REPO_ROOT) -> Result:
    try:
        import eyeos
    except ImportError as e:
        return Result("eyeos", "fail", f"not importable ({e}): pip install -e . from {repo_root}")
    location = Path(eyeos.__file__).resolve().parent
    name = f"eyeos {getattr(eyeos, '__version__', '?')}"
    if not location.is_relative_to(repo_root.resolve()):
        return Result(
            name,
            "warn",
            f"imported from {location}, another checkout, not {repo_root}: run "
            "pip install -e . from this folder to use this checkout's code",
        )
    return Result(name, "ok", f"from {location}")


def _import_error(label: str, module: str, e: ImportError) -> Result:
    if isinstance(e, ModuleNotFoundError) and e.name == module:
        return Result(label, "fail", f"not installed: {WEBCAM_INSTALL} (Python 3.10-3.12)")
    # Installed, but it (or something it imports) failed to load.
    return Result(label, "fail", f"installed but failed to import: {e}")


def check_webcam_stack() -> list[Result]:
    """OpenCV and MediaPipe, and one FaceMesh run with the settings EyeOS uses."""
    try:
        import cv2
    except ImportError as e:
        cv2_result = _import_error("OpenCV", "cv2", e)
    else:
        cv2_result = Result(f"OpenCV {getattr(cv2, '__version__', '?')}", "ok", "imports")

    try:
        import mediapipe as mp
    except ImportError as e:
        return [cv2_result, _import_error("MediaPipe", "mediapipe", e)]
    version = getattr(mp, "__version__", "?")
    name = f"MediaPipe {version}"
    if not hasattr(mp, "solutions"):
        return [
            cv2_result,
            Result(
                name,
                "fail",
                f"mediapipe {version} has no legacy Solutions API (removed in 0.10.30). "
                "EyeOS needs mediapipe<0.10.30, which has no wheels for Python 3.13 or "
                f"newer: recreate .venv with Python 3.12 and run {WEBCAM_INSTALL} again.",
            ),
        ]
    return [cv2_result, _run_face_mesh(mp, name)]


def _run_face_mesh(mp: Any, name: str) -> Result:
    import numpy as np

    try:
        mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=2, refine_landmarks=True
        )
    except Exception as e:  # anything MediaPipe raises here means no live session
        return Result(name, "fail", f"FaceMesh could not start: {type(e).__name__}: {e}")
    try:
        result = mesh.process(np.zeros((480, 640, 3), dtype=np.uint8))
        faces = len(getattr(result, "multi_face_landmarks", None) or [])
    except Exception as e:
        return Result(name, "fail", f"FaceMesh failed on a blank frame: {type(e).__name__}: {e}")
    finally:
        mesh.close()
    return Result(name, "ok", f"FaceMesh ran on a blank frame ({faces} faces)")


def check_pynput() -> Result:
    version = _dist_version("pynput")
    if version is None:
        return Result(
            "pynput",
            "fail",
            "not installed. The kill switch (Esc) and confirm key need it, and eyeos "
            'refuses to move the real cursor without it: pip install -e ".[windows-fallback]"',
        )
    return Result(
        f"pynput {version}", "ok", "installed (its listener starts only inside eyeos run)"
    )


def check_scipy() -> Result:
    version = _dist_version("scipy")
    if version is None:
        return Result(
            "scipy",
            "warn",
            'not installed; only the MPIIFaceGaze eval needs it: pip install -e ".[public-eval]"',
        )
    return Result(f"scipy {version}", "ok", "installed")


def run_checks() -> list[Result]:
    return [
        check_python(_python_version()),
        check_eyeos(),
        *check_webcam_stack(),
        check_pynput(),
        check_scipy(),
    ]


def main(argv: list[str] | None = None) -> int:
    del argv  # no options; the parameter keeps the scripts' usual main(argv) shape
    print(f"EyeOS setup check ({sys.executable})")
    results = run_checks()
    for r in results:
        tag = {"ok": "[ok]  ", "warn": "[warn]", "fail": "[FAIL]"}[r.status]
        print(f"  {tag} {r.name}: {r.detail}")
    failed = [r for r in results if r.status == "fail"]
    if failed:
        print(f"{len(failed)} required check(s) failed; see docs/LOCAL_SETUP.md, Troubleshooting.")
        return 1
    print("All required checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
