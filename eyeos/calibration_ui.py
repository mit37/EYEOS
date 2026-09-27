"""A fullscreen 9-dot calibration screen (PRD §4: "a calibration UI (a fullscreen
Tkinter/PySide6 dots screen)"). Needs a display — untested here, same as overlay.py
(and guards the same two failure modes: tkinter missing entirely, or installed with
no $DISPLAY); see docs/PLAN.md.

Usage: show one target at a time; the caller presses space when they're looking
steadily at the dot, which calls ``capture_sample(target_px)`` (wired by the caller to
pull the current FeatureVector out of the running pipeline and add it to a
Calibration). Escape cancels.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from eyeos.calibration import target_points

_DOT_RADIUS = 12


class CalibrationUI:
    def __init__(
        self,
        screen_width_px: int,
        screen_height_px: int,
        capture_sample: Callable[[tuple[float, float]], None],
    ) -> None:
        try:
            import tkinter as tk
        except ImportError as e:
            raise RuntimeError(
                "CalibrationUI needs a display (tkinter is not available in this "
                "environment). Run it on a machine with a desktop session."
            ) from e

        self._tk = tk
        self.capture_sample = capture_sample
        self.targets = target_points(screen_width_px, screen_height_px)
        self._index = 0
        self._cancelled = False

        try:
            self.root: Any = tk.Tk()
        except tk.TclError as e:
            raise RuntimeError(
                "CalibrationUI needs a display (tkinter is installed, but there is "
                "no $DISPLAY / desktop session here). Run it on a machine with a "
                "desktop session."
            ) from e
        self.root.attributes("-fullscreen", True)
        self.canvas: Any = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.root.bind("<space>", self._on_space)
        self.root.bind("<Escape>", self._on_escape)

        self._draw_current_target()

    def _draw_current_target(self) -> None:
        self.canvas.delete("all")
        if self._index >= len(self.targets):
            return
        x, y = self.targets[self._index]
        r = _DOT_RADIUS
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="")
        self.canvas.create_text(
            20,
            20,
            anchor="nw",
            fill="white",
            text=f"Point {self._index + 1}/{len(self.targets)} — space to capture, esc to cancel",
        )

    def _on_space(self, _event: object) -> None:
        if self._index >= len(self.targets):
            return
        self.capture_sample(self.targets[self._index])
        self._index += 1
        if self._index >= len(self.targets):
            self.root.destroy()
        else:
            self._draw_current_target()

    def _on_escape(self, _event: object) -> None:
        self._cancelled = True
        self.root.destroy()

    def run(self) -> bool:
        """Blocks until calibration finishes or is cancelled. Returns True on success."""
        self.root.mainloop()
        return not self._cancelled
