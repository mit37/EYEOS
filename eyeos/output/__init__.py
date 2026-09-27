"""The OutputAdapter interface: dry-run | Windows SendInput | pynput fallback (PRD §3/§6.1)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class OutputAdapter(ABC):
    @abstractmethod
    def move_cursor(self, x: int, y: int) -> None: ...

    @abstractmethod
    def click(self) -> None: ...

    def close(self) -> None:
        return None
