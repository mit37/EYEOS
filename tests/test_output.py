"""Output adapters: dry-run is fully tested; Windows SendInput and the pynput
fallback are only tested on their guarded-missing-dependency / wrong-platform paths
(this container is Linux with no pynput installed — see docs/PLAN.md)."""

from __future__ import annotations

import sys

import pytest

from eyeos.output.dry_run import DryRunOutputAdapter
from eyeos.output.pynput_fallback import PynputOutputAdapter
from eyeos.output.windows_sendinput import WindowsSendInputOutput


def test_dry_run_records_moves_and_clicks_in_order():
    adapter = DryRunOutputAdapter(echo=None)
    adapter.move_cursor(100, 200)
    adapter.click()
    adapter.move_cursor(300, 400)
    assert adapter.actions == [("move", 100, 200), ("click",), ("move", 300, 400)]


def test_dry_run_echoes_by_default(capsys):
    adapter = DryRunOutputAdapter()
    adapter.move_cursor(1, 2)
    adapter.click()
    out = capsys.readouterr().out
    assert "move_cursor(1, 2)" in out
    assert "click()" in out


def test_dry_run_echo_can_be_silenced():
    seen = []
    adapter = DryRunOutputAdapter(echo=seen.append)
    adapter.move_cursor(1, 2)
    assert seen == ["[dry-run] move_cursor(1, 2)"]


def test_dry_run_close_is_a_harmless_no_op():
    adapter = DryRunOutputAdapter(echo=None)
    adapter.close()  # must not raise


@pytest.mark.skipif(sys.platform == "win32", reason="this test is for the non-Windows guard")
def test_windows_sendinput_raises_off_windows():
    with pytest.raises(RuntimeError, match="Windows"):
        WindowsSendInputOutput()


def test_pynput_adapter_raises_clear_error_without_pynput_installed():
    with pytest.raises(RuntimeError, match="pynput"):
        PynputOutputAdapter()


class _FakeUser32:
    def __init__(self, size):
        self.size = size
        self.sent = []

    def GetSystemMetrics(self, index):
        return self.size[index]

    def SendInput(self, count, pointer, size):
        mi = pointer.contents.union.mi
        self.sent.append((mi.dx, mi.dy, mi.dwFlags))
        return count


def _fake_windows(monkeypatch, size=(1536, 864)):
    import ctypes
    import types

    user32 = _FakeUser32(size)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(user32=user32), raising=False)
    return user32


def test_windows_sendinput_refuses_a_mismatched_screen(monkeypatch):
    _fake_windows(monkeypatch)
    with pytest.raises(ValueError, match="1536x864"):
        WindowsSendInputOutput(expected_size=(1920, 1080))


def test_windows_sendinput_clamps_absolute_coordinates(monkeypatch):
    user32 = _fake_windows(monkeypatch)
    out = WindowsSendInputOutput(expected_size=(1536, 864))

    out.move_cursor(5000, 5000)
    out.move_cursor(-10, -10)

    assert [(dx, dy) for dx, dy, _ in user32.sent] == [(65535, 65535), (0, 0)]
