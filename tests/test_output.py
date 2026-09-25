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
