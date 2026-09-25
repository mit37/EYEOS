"""A small always-on-top window showing the four gate lights live (PRD §2: "the
overlay shows the four gate lights live"). Needs a display, which no cloud CI runner
has — ``tkinter`` itself may or may not be installed (it isn't in this project's local
build container; it is, but with no ``$DISPLAY``, on GitHub's own runners), so this
guards both failure modes and only fails clearly, never exercised for real; see
docs/PLAN.md.
"""

from __future__ import annotations

from typing import Any

from eyeos.gates import GateStatus

_GATE_LABELS = ("1 Presence", "2 Calibration", "3 Stability", "4 Kill/limits")
_GREEN = "#2ecc71"
_RED = "#e74c3c"


class GateOverlay:
    def __init__(self) -> None:
        try:
            import tkinter as tk
        except ImportError as e:
            raise RuntimeError(
                "GateOverlay needs a display (tkinter is not available in this "
                "environment). Run it on a machine with a desktop session."
            ) from e

        self._tk = tk
        try:
            self.root: Any = tk.Tk()
        except tk.TclError as e:
            raise RuntimeError(
                "GateOverlay needs a display (tkinter is installed, but there is no "
                "$DISPLAY / desktop session here). Run it on a machine with a "
                "desktop session."
            ) from e
        self.root.title("EyeOS gates")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)

        self._lights: list[Any] = []
        self._reason_labels: list[Any] = []
        for label in _GATE_LABELS:
            row = tk.Frame(self.root)
            row.pack(fill="x", padx=8, pady=2)
            canvas = tk.Canvas(row, width=16, height=16, highlightthickness=0)
            oval = canvas.create_oval(2, 2, 14, 14, fill=_RED, outline="")
            canvas.pack(side="left")
            tk.Label(row, text=label, width=14, anchor="w").pack(side="left")
            reason = tk.Label(row, text="", anchor="w")
            reason.pack(side="left")
            self._lights.append((canvas, oval))
            self._reason_labels.append(reason)

    def update(self, status: GateStatus) -> None:
        passes = (
            status.gate1_presence,
            status.gate2_calibration,
            status.gate3_stability,
            status.gate4_kill_and_limits,
        )
        reasons = (
            status.reasons.get("gate1", ""),
            status.reasons.get("gate2", ""),
            status.reasons.get("gate3", ""),
            status.reasons.get("gate4", ""),
        )
        for (canvas, oval), ok, reason, label in zip(
            self._lights, passes, reasons, self._reason_labels, strict=True
        ):
            canvas.itemconfig(oval, fill=_GREEN if ok else _RED)
            label.config(text=reason)
        self.root.update_idletasks()
        self.root.update()

    def close(self) -> None:
        self.root.destroy()
