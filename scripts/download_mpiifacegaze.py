#!/usr/bin/env python3
"""Download and unpack the MPIIFaceGaze dataset for eval/public/mpiifacegaze.py.

    python scripts/download_mpiifacegaze.py --accept-license [--dest datasets]

It prints the dataset's license and citation and refuses to download anything until you
pass --accept-license (exit code 2). Then it streams the zip to <dest>/MPIIFaceGaze.zip
with progress, resumes a cut-off download with an HTTP Range request, checks every member
with zipfile.testzip, and extracts it to <dest>/MPIIFaceGaze, failing unless at least one
participant's pXX/pXX.txt came out. A rerun skips whatever is already done. datasets/ is
gitignored: the images show real people's faces and must never be committed.

Standard library only, so it runs the same way on Windows (python
scripts\\download_mpiifacegaze.py --accept-license). scripts/download_mpiifacegaze.sh is
a thin wrapper around it.
"""

from __future__ import annotations

import argparse
import http.client
import re
import sys
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path
from typing import Any

URL = "https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip"
ZIP_NAME = "MPIIFaceGaze.zip"
DATASET_DIR = "MPIIFaceGaze"
# Written after a complete, verified extraction; a rerun that finds it does nothing.
DONE_MARKER = ".eyeos-extracted"

_CHUNK_BYTES = 1 << 20
_TIMEOUT_S = 60.0
_IGNORED_PREFIXES = ("__MACOSX/",)

LICENSE_NOTICE = """\
MPIIFaceGaze (Max Planck Institute for Informatics)
  15 participants, about 3000 laptop-webcam face images each, each labelled with
  the on-screen point the person was looking at.

License: CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  International, https://creativecommons.org/licenses/by-nc-sa/4.0/).
  Non-commercial use only; credit the authors; share anything derived from the data
  under the same license. The images show real people's faces: keep them, and any
  cache derived from them, out of version control (datasets/ is gitignored).

If you use it, cite:
  X. Zhang, Y. Sugano, M. Fritz, A. Bulling.
    "It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation."
    CVPR Workshops (CVPRW), 2017.
  X. Zhang, Y. Sugano, M. Fritz, A. Bulling.
    "Appearance-Based Gaze Estimation in the Wild." CVPR, 2015.
"""


class DownloadError(Exception):
    """A failure the user can act on; main() prints it and exits 1."""


def _content_length(headers: Any) -> int | None:
    value = headers.get("Content-Length") if headers is not None else None
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _progress(done: int, total: int | None) -> None:
    mb = done / 1e6
    if total:
        line = f"\r  {mb:,.1f} / {total / 1e6:,.1f} MB ({100.0 * done / total:5.1f}%)"
    else:
        line = f"\r  {mb:,.1f} MB"
    print(line, end="", file=sys.stderr, flush=True)


def download(url: str, part: Path, final: Path) -> None:
    """Stream ``url`` into ``part``, then rename it to ``final`` once it is complete.

    An existing ``part`` is resumed with a Range request. A server that answers the
    Range request with the whole file (status 200) is restarted from byte 0. A stream
    that ends before its Content-Length leaves ``part`` in place for the next run.
    """
    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "eyeos-dataset-downloader"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    print(f"Downloading {url}" + (f" (resuming at byte {offset:,})" if offset else ""))
    progress_line_open = False
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            if offset and getattr(response, "status", 200) != 206:
                offset = 0  # the server ignored the Range header: start over
            length = _content_length(response.headers)
            total = offset + length if length is not None else None
            written = offset
            last_shown = -1
            with part.open("ab" if offset else "wb") as f:
                while True:
                    chunk = response.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
                    if written // (8 * _CHUNK_BYTES) != last_shown:
                        last_shown = written // (8 * _CHUNK_BYTES)
                        _progress(written, total)
                        progress_line_open = True
            _progress(written, total)
            print(file=sys.stderr)
            progress_line_open = False
    except urllib.error.HTTPError as e:
        if e.code == 416 and offset:
            # Range Not Satisfiable: the partial file already holds every byte.
            part.replace(final)
            return
        raise DownloadError(f"{url}: HTTP {e.code} {e.reason}") from e
    except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
        # OSError covers a reset or timed-out connection; HTTPException covers a chunked
        # response cut off mid-stream (IncompleteRead).
        if progress_line_open:
            print(file=sys.stderr)  # end the progress line, so the error has its own
        raise DownloadError(
            f"{url}: {e}. The partial download is kept at {part}; rerun to resume."
        ) from e
    if total is not None and written != total:
        raise DownloadError(
            f"the download stopped at {written:,} of {total:,} bytes. The partial download "
            f"is kept at {part}; rerun to resume."
        )
    part.replace(final)


def verify(zip_path: Path) -> None:
    """CRC-check every member; delete the zip if it is corrupt, so a rerun refetches it."""
    print(f"Verifying {zip_path} ...")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
    except (zipfile.BadZipFile, zlib.error, EOFError, ValueError) as e:
        bad = f"<archive: {e}>"
    if bad is not None:
        zip_path.unlink()
        raise DownloadError(
            f"{zip_path} is corrupt (first bad member: {bad}); deleted it. Rerun to "
            "download it again."
        )


def _members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    return [i for i in zf.infolist() if not i.filename.startswith(_IGNORED_PREFIXES)]


def _has_participants(target: Path) -> bool:
    """Does ``target`` hold at least one ``pXX/pXX.txt``?"""
    return target.is_dir() and any(
        re.fullmatch(r"p\d\d", p.name) and (p / f"{p.name}.txt").is_file() for p in target.iterdir()
    )


def extract(zip_path: Path, dest: Path) -> Path:
    """Extract to ``dest/MPIIFaceGaze``, whether or not the zip has that top-level folder.

    A zip with an ``MPIIFaceGaze/`` folder is extracted into ``dest`` (anything else at
    its top level lands next to the folder); one without it is extracted into
    ``dest/MPIIFaceGaze``. Every member is checked before anything is written: one whose
    path would land outside the target (an absolute path or ``..``) aborts the whole
    extraction. The extraction is marked done only if a ``pXX/pXX.txt`` came out.
    """
    target = dest / DATASET_DIR
    with zipfile.ZipFile(zip_path) as zf:
        members = _members(zf)
        nested = any(i.filename.startswith(DATASET_DIR + "/") for i in members)
        base = dest if nested else target
        base_resolved = base.resolve()
        for info in members:
            name = info.filename
            out = (base / name).resolve()
            if Path(name).is_absolute() or not out.is_relative_to(base_resolved):
                raise DownloadError(
                    f"refusing to extract {zip_path}: member {name!r} would land outside {base}"
                )
        print(f"Extracting {len(members):,} files to {target} ...")
        for info in members:
            zf.extract(info, base)
        top_level = sorted({i.filename.split("/", 1)[0] for i in members})
    if not _has_participants(target):
        raise DownloadError(
            f"{zip_path} did not produce any pXX/pXX.txt under {target} (the zip's top-level "
            f"entries are {', '.join(repr(n) for n in top_level) or 'none'}). Move the "
            f"participant folders p00 ... p14 into {target} by hand."
        )
    (target / DONE_MARKER).write_text(f"extracted from {zip_path.name}\n")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download MPIIFaceGaze (CC BY-NC-SA 4.0) for eval/public/mpiifacegaze.py."
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="confirm you accept the CC BY-NC-SA 4.0 terms printed above",
    )
    parser.add_argument("--dest", default="datasets", help="download directory (default: datasets)")
    parser.add_argument("--url", default=URL, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    print(LICENSE_NOTICE)
    if not args.accept_license:
        print(
            "Refusing to download: rerun with --accept-license to confirm you accept these "
            "terms.",
            file=sys.stderr,
        )
        return 2

    dest = Path(args.dest)
    target = dest / DATASET_DIR
    if (target / DONE_MARKER).exists():
        print(f"MPIIFaceGaze is already downloaded and extracted in {target}; nothing to do.")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / ZIP_NAME
    part = dest / (ZIP_NAME + ".part")
    try:
        if zip_path.exists():
            print(f"{zip_path} is already downloaded; skipping the download.")
        else:
            download(args.url, part, zip_path)
        verify(zip_path)
        extract(zip_path, dest)
    except DownloadError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"Done: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
