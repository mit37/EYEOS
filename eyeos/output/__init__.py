"""The OutputAdapter interface: dry-run | Windows SendInput | pynput fallback (PRD §3/§6.1)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class OutputAdapter(ABC):
    # Fail-closed default: anything that might move the real cursor must be driven
    # with a kill switch (Pipeline refuses otherwise). Only the dry-run backend opts out.
    moves_real_cursor: bool = True

    @abstractmethod
    def move_cursor(self, x: int, y: int) -> None: ...

    @abstractmethod
    def click(self) -> None: ...

    def close(self) -> None:
        return None
