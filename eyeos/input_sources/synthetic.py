"""Reads a committed CSV of pre-computed landmarks. No camera, no MediaPipe — this is
what CI, the eval harness, and the dry-run pipeline test all run against."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from eyeos.input_sources import Frame, InputSource
from eyeos.synthetic_format import row_to_frame


class SyntheticInputSource(InputSource):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def frames(self) -> Iterator[Frame]:
        with self.path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamp_s, landmarks = row_to_frame(row)
                yield Frame(timestamp_s=timestamp_s, landmarks=landmarks)
