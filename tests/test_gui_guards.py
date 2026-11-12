"""overlay.py and calibration_ui.py, tested headless against a fake ``tkinter``.

The guard tests cover both ways a machine can lack a display: tkinter not installed
at all (this project's build container), and tkinter installed with no $DISPLAY
(GitHub's CI runners).
"""

from __future__ import annotations

import sys

import pytest

from eyeos.calibration import target_points
from eyeos.calibration_ui import CalibrationUI
from eyeos.gates import GateStatus
from eyeos.overlay import GateOverlay
from tests.fake_tk import TclError, make_fake_tk


@pytest.fixture
def fake_tk(monkeypatch):
    module = make_fake_tk()
    monkeypatch.setitem(sys.modules, "tkinter", module)
    return module


@pytest.mark.parametrize("cls", [GateOverlay, CalibrationUI])
def test_clear_error_when_tkinter_is_missing(monkeypatch, cls):
    monkeypatch.setitem(sys.modules, "tkinter", None)
    with pytest.raises(RuntimeError, match="display"):
        cls() if cls is GateOverlay else cls(1920, 1080, capture_sample=lambda _pt: True)


@pytest.mark.parametrize("cls", [GateOverlay, CalibrationUI])
def test_clear_error_when_there_is_no_display(monkeypatch, cls):
    monkeypatch.setitem(
        sys.modules, "tkinter", make_fake_tk(fail_with=TclError("no display name and no $DISPLAY"))
    )
    with pytest.raises(RuntimeError, match="display"):
        cls() if cls is GateOverlay else cls(1920, 1080, capture_sample=lambda _pt: True)


def _status(g1, g2, g3, g4):
    return GateStatus(
        gate1_presence=g1,
        gate2_calibration=g2,
        gate3_stability=g3,
        gate4_kill_and_limits=g4,
        move_allowed=g1 and g2 and g4,
        cursor_target_px=None,
        click_fired=False,
        reasons={"gate1": "r1", "gate2": "r2", "gate3": "r3", "gate4": "r4"},
    )


def test_overlay_lights_follow_each_gate(fake_tk):
    overlay = GateOverlay()

    overlay.update(_status(True, False, True, False))

    fills = [canvas.items[oval]["fill"] for canvas, oval in overlay._lights]
    assert fills == ["#2ecc71", "#e74c3c", "#2ecc71", "#e74c3c"]
    assert [label.options["text"] for label in overlay._reason_labels] == ["r1", "r2", "r3", "r4"]
    assert overlay.root.updates == 1


def test_overlay_close_destroys_the_window(fake_tk):
    overlay = GateOverlay()
    overlay.close()
    assert overlay.root.destroyed


def test_calibration_ui_captures_all_nine_dots_in_order(fake_tk):
    captured = []
    ui = CalibrationUI(1920, 1080, capture_sample=lambda pt: captured.append(pt) or True)
    ui.root.on_mainloop = lambda root: [root.fire("<space>") for _ in range(9)]

    assert ui.run() is True
    assert captured == target_points(1920, 1080)
    assert ui.root.destroyed


def test_calibration_ui_escape_cancels(fake_tk):
    ui = CalibrationUI(1920, 1080, capture_sample=lambda _pt: True)
    ui.root.on_mainloop = lambda root: root.fire("<Escape>")

    assert ui.run() is False
    assert ui.root.destroyed


def test_calibration_ui_closing_the_window_cancels(fake_tk):
    # Alt+F4 after 6 dots: Tk's default close handler would just destroy the window.
    captured = []
    ui = CalibrationUI(1920, 1080, capture_sample=lambda pt: captured.append(pt) or True)

    def six_dots_then_close(root):
        for _ in range(6):
            root.fire("<space>")
        root.close_window()

    ui.root.on_mainloop = six_dots_then_close

    assert ui.run() is False
    assert len(captured) == 6
    assert ui.root.destroyed


def test_calibration_ui_fails_unless_every_dot_was_captured(fake_tk):
    # However the main loop ends early, 6 of 9 dots is not a calibration.
    ui = CalibrationUI(1920, 1080, capture_sample=lambda _pt: True)
    ui.root.on_mainloop = lambda root: [root.fire("<space>") for _ in range(6)]

    assert ui.run() is False


def test_calibration_ui_reprompts_a_dot_whose_capture_failed(fake_tk):
    attempts = []

    def capture(pt):
        attempts.append(pt)
        return len(attempts) != 1  # the very first capture sees no face

    ui = CalibrationUI(1920, 1080, capture_sample=capture)
    ui.root.on_mainloop = lambda root: [root.fire("<space>") for _ in range(10)]

    assert ui.run() is True
    assert attempts[0] == attempts[1]  # same dot asked again
    assert attempts[1:] == target_points(1920, 1080)


def test_calibration_ui_refuses_a_config_screen_that_does_not_match(monkeypatch):
    # Windows at 125% scaling reports a 1920x1080 panel as 1536x864.
    monkeypatch.setitem(sys.modules, "tkinter", make_fake_tk(screen_size=(1536, 864)))
    with pytest.raises(ValueError, match="1536x864"):
        CalibrationUI(1920, 1080, capture_sample=lambda _pt: True)

    ui = CalibrationUI(1536, 864, capture_sample=lambda _pt: True)
    assert all(0 <= x < 1536 and 0 <= y < 864 for x, y in ui.targets)
