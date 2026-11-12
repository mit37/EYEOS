"""A tiny fake ``tkinter`` for testing overlay.py and calibration_ui.py headless.

Install with ``monkeypatch.setitem(sys.modules, "tkinter", make_fake_tk())``. Widgets
record what was drawn; ``Tk.fire(sequence)`` simulates a key binding firing.
"""

from __future__ import annotations

import types
from typing import Any


class TclError(Exception):
    pass


class _Widget:
    def __init__(self, master: Any = None, **kwargs: Any) -> None:
        self.master = master
        self.options = dict(kwargs)

    def pack(self, **_kwargs: Any) -> None:
        return None

    def config(self, **kwargs: Any) -> None:
        self.options.update(kwargs)

    configure = config


class Tk(_Widget):
    fail_with: Exception | None = None  # set to TclError(...) to simulate "no $DISPLAY"

    def __init__(self) -> None:
        if type(self).fail_with is not None:
            raise type(self).fail_with
        super().__init__()
        self.bindings: dict[str, Any] = {}
        self.attrs: dict[str, Any] = {}
        self.destroyed = False
        self.updates = 0
        self.mainloop_calls = 0
        self.on_mainloop: Any = None

    def title(self, text: str) -> None:
        self.options["title"] = text

    def attributes(self, name: str, value: Any) -> None:
        self.attrs[name] = value

    def resizable(self, *_args: Any) -> None:
        return None

    def bind(self, sequence: str, handler: Any) -> None:
        self.bindings[sequence] = handler

    def fire(self, sequence: str) -> None:
        self.bindings[sequence](None)

    def update(self) -> None:
        self.updates += 1

    def update_idletasks(self) -> None:
        return None

    def mainloop(self) -> None:
        self.mainloop_calls += 1
        if self.on_mainloop is not None:
            self.on_mainloop(self)

    def destroy(self) -> None:
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
        item_id = self._next_id
        self._next_id += 1
        self.items[item_id] = {"kind": kind, "coords": coords, **kwargs}
        return item_id

    def create_oval(self, *coords: float, **kwargs: Any) -> int:
        return self._add("oval", coords, **kwargs)

    def create_text(self, *coords: float, **kwargs: Any) -> int:
        return self._add("text", coords, **kwargs)

    def itemconfig(self, item_id: int, **kwargs: Any) -> None:
        self.items[item_id].update(kwargs)

    def delete(self, what: str) -> None:
        if what == "all":
            self.items.clear()

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [item for item in self.items.values() if item["kind"] == kind]


def make_fake_tk(fail_with: Exception | None = None) -> types.ModuleType:
    module = types.ModuleType("tkinter")
    tk_class = type("Tk", (Tk,), {"fail_with": fail_with})
    module.Tk = tk_class  # type: ignore[attr-defined]
    module.Frame = Frame  # type: ignore[attr-defined]
    module.Label = Label  # type: ignore[attr-defined]
    module.Canvas = Canvas  # type: ignore[attr-defined]
    module.TclError = TclError  # type: ignore[attr-defined]
    return module
