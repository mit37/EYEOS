"""The InputSource interface: webcam | video file | synthetic landmarks (PRD §3/§6.1).

Everything downstream of an InputSource (features, calibration, gates) only ever sees
``Frame`` objects, so the pipeline doesn't care whether frames came from a live camera,
a recorded video, or a committed CSV fixture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

from eyeos.landmarks import FaceLandmarks


@dataclass(frozen=True)
class Frame:
    timestamp_s: float
    landmarks: FaceLandmarks | None  # None means "no face detected this frame"


class InputSource(ABC):
    """A source of timestamped face landmarks."""

    @abstractmethod
    def frames(self) -> Iterator[Frame]:
        """Yield frames in timestamp order. May be called once per source instance."""

    def close(self) -> None:
        """Release any underlying resource (camera handle, file handle, ...)."""
        return None

    def __enter__(self) -> InputSource:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
