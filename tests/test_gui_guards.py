"""overlay.py and calibration_ui.py need a display; this container has none — not
even ``tkinter`` is importable here (see docs/PLAN.md). These tests confirm both
modules fail with a clear, guarded error rather than an unhandled ImportError/crash.
"""

from __future__ import annotations

import pytest

from eyeos.calibration_ui import CalibrationUI
from eyeos.overlay import GateOverlay


def test_overlay_raises_clear_error_without_a_display():
    with pytest.raises(RuntimeError, match="display"):
        GateOverlay()


def test_calibration_ui_raises_clear_error_without_a_display():
    with pytest.raises(RuntimeError, match="display"):
        CalibrationUI(1920, 1080, capture_sample=lambda _pt: None)
