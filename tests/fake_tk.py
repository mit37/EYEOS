"""A tiny fake ``tkinter`` for testing overlay.py and calibration_ui.py headless.

Install with ``monkeypatch.setitem(sys.modules, "tkinter", make_fake_tk())``. Widgets
record what was drawn; ``Tk.fire(sequence)`` simulates a key binding firing, and
``Tk.close_window()`` the window manager's close button (Alt+F4, the title-bar X).
``Tk.on_mainloop`` and ``Tk.on_update`` script what the user does while Tk is
processing events.

Like real Tk, a destroyed window stays destroyed: a second ``destroy()``, and any call
on it or on one of its widgets, raises TclError. Events (``fire``, ``close_window``)
sent after that are dropped, as no window is left to receive them. Also like real Tk,
an exception raised by an event handler is not raised to the caller: it goes to
``report_callback_exception``, which prints it and records it in ``callback_errors``.
"""

from __future__ import annotations

import sys
import traceback
import types
from typing import Any


class TclError(Exception):
    pass


class _Widget:
    def __init__(self, master: Any = None, **kwargs: Any) -> None:
        self.master = master
        self.options = dict(kwargs)
        if master is not None:
            master._check_alive(type(self).__name__.lower())

    def _root(self) -> Any:
        widget: Any = self
        while widget.master is not None:
            widget = widget.master
        return widget

    def _check_alive(self, command: str) -> None:
        if getattr(self._root(), "destroyed", False):
            if isinstance(self, Tk):
                raise TclError(f'can\'t invoke "{command}" command: application has been destroyed')
            raise TclError(f'invalid command name ".!{type(self).__name__.lower()}"')

    def pack(self, **_kwargs: Any) -> None:
        self._check_alive("pack")

    def config(self, **kwargs: Any) -> None:
        self._check_alive("configure")
        self.options.update(kwargs)

    configure = config


class Tk(_Widget):
    fail_with: Exception | None = None  # set to TclError(...) to simulate "no $DISPLAY"
    screen_size: tuple[int, int] = (1920, 1080)

    def __init__(self) -> None:
        if type(self).fail_with is not None:
            raise type(self).fail_with
        super().__init__()
        self.bindings: dict[str, Any] = {}
        self.protocols: dict[str, Any] = {}
        self.attrs: dict[str, Any] = {}
        self.destroyed = False
        self.destroy_calls = 0
        self.updates = 0
        self.mainloop_calls = 0
        self.on_mainloop: Any = None
        self.on_update: Any = None  # called inside update(), as Tk handles events there
        self.callback_errors: list[BaseException] = []

    def winfo_screenwidth(self) -> int:
        self._check_alive("winfo")
        return self.screen_size[0]

    def winfo_screenheight(self) -> int:
        self._check_alive("winfo")
        return self.screen_size[1]

    def title(self, text: str) -> None:
        self._check_alive("wm")
        self.options["title"] = text

    def attributes(self, name: str, value: Any) -> None:
        self._check_alive("wm")
        self.attrs[name] = value

    def resizable(self, *_args: Any) -> None:
        self._check_alive("wm")

    def bind(self, sequence: str, handler: Any) -> None:
        self._check_alive("bind")
        self.bindings[sequence] = handler

    def _dispatch(self, handler: Any, *args: Any) -> None:
        """Run an event handler the way Tk does: report an exception, don't raise it."""
        try:
            handler(*args)
        except Exception:
            self.report_callback_exception(*sys.exc_info())

    def report_callback_exception(self, exc: Any, val: Any, tb: Any) -> None:
        self.callback_errors.append(val)
        print("Exception in Tkinter callback", file=sys.stderr)
        traceback.print_exception(exc, val, tb, file=sys.stderr)

    def fire(self, sequence: str) -> None:
        if not self.destroyed:
            self._dispatch(self.bindings[sequence], None)

    def protocol(self, name: str, func: Any = None) -> None:
        self._check_alive("wm")
        self.protocols[name] = func

    def close_window(self) -> None:
        """The user closes the window. Like real Tk, with no WM_DELETE_WINDOW handler
        registered the window is simply destroyed."""
        if self.destroyed:
            return
        handler = self.protocols.get("WM_DELETE_WINDOW")
        self._dispatch(handler if handler is not None else self.destroy)

    def update(self) -> None:
        self._check_alive("update")
        self.updates += 1
        if self.on_update is not None:
            self.on_update(self)

    def update_idletasks(self) -> None:
        self._check_alive("update")

    def mainloop(self) -> None:
        self.mainloop_calls += 1
        if self.on_mainloop is not None:
            self.on_mainloop(self)

    def destroy(self) -> None:
        self._check_alive("destroy")
        self.destroy_calls += 1
        self.destroyed = True


class Frame(_Widget):
    pass


class Label(_Widget):
    pass


class Canvas(_Widget):
    def __init__(self, master: Any = None, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.items: dict[int, dict[str, Any]] = {}
        self._next_id = 1

    def _add(self, kind: str, coords: tuple[float, ...], **kwargs: Any) -> int:
        self._check_alive(f"create {kind}")
        item_id = self._next_id
        self._next_id += 1
        self.items[item_id] = {"kind": kind, "coords": coords, **kwargs}
        return item_id

    def create_oval(self, *coords: float, **kwargs: Any) -> int:
        return self._add("oval", coords, **kwargs)

    def create_text(self, *coords: float, **kwargs: Any) -> int:
        return self._add("text", coords, **kwargs)

    def itemconfig(self, item_id: int, **kwargs: Any) -> None:
        self._check_alive("itemconfigure")
        self.items[item_id].update(kwargs)

    def delete(self, what: str) -> None:
        self._check_alive("delete")
        if what == "all":
            self.items.clear()

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [item for item in self.items.values() if item["kind"] == kind]


def make_fake_tk(
    fail_with: Exception | None = None, screen_size: tuple[int, int] = (1920, 1080)
) -> types.ModuleType:
    module = types.ModuleType("tkinter")
    tk_class = type("Tk", (Tk,), {"fail_with": fail_with, "screen_size": screen_size})
    module.Tk = tk_class  # type: ignore[attr-defined]
    module.Frame = Frame  # type: ignore[attr-defined]
    module.Label = Label  # type: ignore[attr-defined]
    module.Canvas = Canvas  # type: ignore[attr-defined]
    module.TclError = TclError  # type: ignore[attr-defined]
    return module
