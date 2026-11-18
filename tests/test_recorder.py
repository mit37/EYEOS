"""``eyeos record`` (eyeos/recorder.py), tested headless against a fake ``tkinter``
(tests/fake_tk.py) and synthetic frames. No camera, display or keyboard."""

from __future__ import annotations

import sys

import pytest

from tests.fake_tk import make_fake_tk


@pytest.fixture
def fake_tk(monkeypatch):
    module = make_fake_tk()
    monkeypatch.setitem(sys.modules, "tkinter", module)
    return module


# ---------------------------------------------------------------------------
# The fake tkinter's event loop: after() callbacks, polled user actions, bell()
# ---------------------------------------------------------------------------


def test_fake_tk_mainloop_runs_after_callbacks_in_due_order_until_destroyed(fake_tk):
    root = fake_tk.Tk()
    ran = []

    def tick(n):
        ran.append(n)
        if n < 3:
            root.after(10, tick, n + 1)
        else:
            root.destroy()

    root.after(50, ran.append, "late")
    root.after(0, tick, 0)
    root.mainloop()

    # tick 0..3 are due at 0, 10, 20, 30 ms; the window is gone before 50 ms.
    assert ran == [0, 1, 2, 3]
    assert root.destroyed


def test_fake_tk_after_cancel_and_poll_hook(fake_tk):
    root = fake_tk.Tk()
    ran = []
    cancelled = root.after(5, ran.append, "cancelled")
    root.after(10, ran.append, "kept")
    root.after_cancel(cancelled)
    polls = []
    root.on_poll = lambda r: polls.append(len(ran))

    root.mainloop()  # returns once no callback is left, like a test-sized event loop

    assert ran == ["kept"]
    assert polls == [0]  # the user gets a turn before each callback


def test_fake_tk_reports_an_after_callback_error_like_real_tk(fake_tk, capsys):
    root = fake_tk.Tk()

    def broken():
        raise ValueError("boom")

    root.after(0, broken)
    root.mainloop()

    assert isinstance(root.callback_errors[0], ValueError)
    assert "Exception in Tkinter callback" in capsys.readouterr().err


def test_fake_tk_bell_counts_and_is_strict_after_destroy(fake_tk):
    from tests.fake_tk import TclError

    root = fake_tk.Tk()
    root.bell()
    assert root.bells == 1
    root.destroy()
    with pytest.raises(TclError):
        root.bell()
    with pytest.raises(TclError):
        root.after(0, lambda: None)
