"""Prints actions instead of moving the mouse. The default backend, and the only one
exercised by this project's tests and CI (see docs/PLAN.md)."""

from __future__ import annotations

from collections.abc import Callable

from eyeos.output import OutputAdapter


class DryRunOutputAdapter(OutputAdapter):
    def __init__(self, echo: Callable[[str], None] | None = print) -> None:
        self.actions: list[tuple[str, int, int] | tuple[str]] = []
        self._echo = echo

    def move_cursor(self, x: int, y: int) -> None:
        self.actions.append(("move", x, y))
        if self._echo is not None:
            self._echo(f"[dry-run] move_cursor({x}, {y})")

    def click(self) -> None:
        self.actions.append(("click",))
        if self._echo is not None:
            self._echo("[dry-run] click()")
