"""A fullscreen 9-dot calibration screen (PRD §4: "a calibration UI (a fullscreen
Tkinter/PySide6 dots screen)"). Needs a display; tested headless against a fake
tkinter (tests/fake_tk.py), and guards both no-display failure modes (tkinter missing,
or installed with no $DISPLAY).

Usage: show one target at a time; the caller presses space when they're looking
steadily at the dot, which calls ``capture_sample(target_px)`` (wired by the caller to
pull the current FeatureVector out of the running pipeline and add it to a
Calibration). If it returns False (no usable sample), the same dot is asked for again
instead of being silently skipped; it may call ``cancel()`` first to give up. Escape
cancels, and so does closing the window (Alt+F4, the title-bar X): ``run()`` is True
only once every dot has been captured.
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
        capture_sample: Callable[[tuple[float, float]], bool | None],
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
        self._closed = False

        try:
            self.root: Any = tk.Tk()
        except tk.TclError as e:
            raise RuntimeError(
                "CalibrationUI needs a display (tkinter is installed, but there is "
                "no $DISPLAY / desktop session here). Run it on a machine with a "
                "desktop session."
            ) from e
        actual = (int(self.root.winfo_screenwidth()), int(self.root.winfo_screenheight()))
        if actual != (int(screen_width_px), int(screen_height_px)):
            self.root.destroy()
            raise ValueError(
                f"config screen.width_px/height_px is {screen_width_px}x{screen_height_px} but "
                f"the screen reports {actual[0]}x{actual[1]}: set them to the reported size "
                "(Windows display scaling changes it, e.g. 1920x1080 at 125% reports 1536x864)"
            )
        self.root.attributes("-fullscreen", True)
        self.canvas: Any = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.root.bind("<space>", self._on_space)
        self.root.bind("<Escape>", self.cancel)
        # Tk's default close handler only destroys the window, which would end run()
        # looking like a finished calibration. Closing it must cancel, like Esc.
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)

        self._draw_current_target()

    def _draw_current_target(self, retry: bool = False) -> None:
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
        if retry:
            self.canvas.create_text(
                20,
                50,
                anchor="nw",
                fill="orange",
                text="No usable sample (face/eyes not found). Look at the dot, press space again.",
            )

    def _on_space(self, _event: object) -> None:
        if self._closed or self._index >= len(self.targets):
            return
        captured = self.capture_sample(self.targets[self._index])
        if self._closed:  # capture_sample cancelled the screen (e.g. the camera failed)
            return
        if captured is False:
            self._draw_current_target(retry=True)
            return
        self._index += 1
        if self._index >= len(self.targets):
            self._close()
        else:
            self._draw_current_target()

    def cancel(self, _event: object = None) -> None:
        """Stop without a calibration. Esc and the window's close button land here, and
        the caller may call it too (from ``capture_sample``, say)."""
        self._cancelled = True
        self._close()

    def _close(self) -> None:
        if not self._closed:
            self._closed = True
            self.root.destroy()

    def run(self) -> bool:
        """Blocks until calibration finishes or is cancelled.

        Returns True only if nothing cancelled it and every dot was captured.
        """
        self.root.mainloop()
        return not self._cancelled and self._index >= len(self.targets)
