"""A cross-platform fallback output backend using pynput. Documented as less tested
than the Windows SendInput path (PRD §3) — it's the "should work everywhere" option,
not the primary target. Requires ``pip install eyeos[windows-fallback]``.
"""

from __future__ import annotations

from typing import Any

from eyeos.output import OutputAdapter


class PynputOutputAdapter(OutputAdapter):
    def __init__(self) -> None:
        try:
            from pynput.mouse import Button, Controller
        except ImportError as e:
            raise RuntimeError(
                "PynputOutputAdapter requires pynput: pip install 'eyeos[windows-fallback]'"
            ) from e
        self._controller: Any = Controller()
        self._button = Button.left

    def move_cursor(self, x: int, y: int) -> None:
        self._controller.position = (x, y)

    def click(self) -> None:
        self._controller.click(self._button, 1)
