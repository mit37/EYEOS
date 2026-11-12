"""Keyboard signals for the pipeline: the kill hotkey (PRD §1: "one hotkey kills it
instantly"; Gate 4: "kill hotkey not pressed") and an optional confirm key.

The kill switch LATCHES: one press stops the session for good (restart to resume).
A key source is behind a tiny interface so tests drive it with ``FakeKeySource``; the
real one is an event-driven pynput listener thread, so a tap shorter than one camera
frame is still caught.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from eyeos.config import Config

KeyCallback = Callable[[str], None]

_NAMED_KEYS = {
    "esc": "esc",
    "escape": "esc",
    "space": "space",
    "enter": "enter",
    "return": "enter",
    "tab": "tab",
    **{f"f{i}": f"f{i}" for i in range(1, 13)},
}

_PYNPUT_HINT = "the kill hotkey needs pynput and a keyboard: pip install 'eyeos[windows-fallback]'"


def parse_key_name(name: str) -> str:
    """Canonical name for a single key, e.g. 'Escape' -> 'esc'. Chords are rejected."""
    lowered = name.strip().lower()
    if lowered in _NAMED_KEYS:
        return _NAMED_KEYS[lowered]
    if len(lowered) == 1 and lowered.isprintable() and not lowered.isspace():
        return lowered
    valid = ", ".join(sorted(set(_NAMED_KEYS.values()), key=lambda k: (len(k), k)))
    raise ValueError(f"unknown key {name!r}: use a single character or one of {valid}")


class KeySource(Protocol):
    def start(self, on_press: KeyCallback, on_release: KeyCallback) -> None: ...

    def stop(self) -> None: ...


class FakeKeySource:
    """In-memory key source for tests: ``press``/``release`` call the callbacks directly."""

    def __init__(self) -> None:
        self._on_press: KeyCallback | None = None
        self._on_release: KeyCallback | None = None
        self.running = False

    def start(self, on_press: KeyCallback, on_release: KeyCallback) -> None:
        self._on_press, self._on_release = on_press, on_release
        self.running = True

    def stop(self) -> None:
        self.running = False

    def press(self, name: str) -> None:
        if self.running and self._on_press is not None:
            self._on_press(parse_key_name(name))

    def release(self, name: str) -> None:
        if self.running and self._on_release is not None:
            self._on_release(parse_key_name(name))


def _pynput_key_name(key: Any) -> str | None:
    char = getattr(key, "char", None)
    if char:
        return char.lower()
    name = getattr(key, "name", None)
    if isinstance(name, str) and name.lower() in _NAMED_KEYS:
        return _NAMED_KEYS[name.lower()]
    return None


class PynputKeySource:
    """Global keyboard listener (pynput). Does not swallow the keys it watches."""

    def __init__(self) -> None:
        self._listener: Any = None

    def start(self, on_press: KeyCallback, on_release: KeyCallback) -> None:
        try:
            from pynput import keyboard
        except ImportError as e:
            raise RuntimeError(_PYNPUT_HINT) from e

        def handle(callback: KeyCallback) -> Callable[[Any], None]:
            def inner(key: Any) -> None:
                name = _pynput_key_name(key)
                if name is not None:
                    callback(name)

            return inner

        try:
            self._listener = keyboard.Listener(
                on_press=handle(on_press), on_release=handle(on_release)
            )
            self._listener.start()
        except Exception as e:  # e.g. no X display / no input permission
            raise RuntimeError(f"{_PYNPUT_HINT} ({e})") from e

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


class KeySignals:
    """``external_signals`` for Pipeline: returns (killed, confirm_held) every frame.

    The timestamp argument is ignored on purpose: key events arrive on the wall clock,
    camera/video frames on their own clock.
    """

    def __init__(self, kill_key: str, source: KeySource, confirm_key: str | None = None) -> None:
        self.kill_key = parse_key_name(kill_key)
        self.confirm_key = parse_key_name(confirm_key) if confirm_key is not None else None
        self._killed = threading.Event()
        self._confirm_held = threading.Event()
        self._source = source
        source.start(self._on_press, self._on_release)

    def _on_press(self, name: str) -> None:
        if name == self.kill_key:
            self._killed.set()
        elif name == self.confirm_key:
            self._confirm_held.set()

    def _on_release(self, name: str) -> None:
        if name == self.confirm_key:
            self._confirm_held.clear()

    @property
    def killed(self) -> bool:
        return self._killed.is_set()

    def __call__(self, _timestamp_s: float) -> tuple[bool, bool]:
        return self._killed.is_set(), self._confirm_held.is_set()

    def close(self) -> None:
        self._source.stop()

    def __enter__(self) -> KeySignals:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def build_key_signals(config: Config, source: KeySource | None = None) -> KeySignals:
    """Start listening for ``config.output.kill_hotkey``. Raises RuntimeError if no
    keyboard backend can start, so callers can refuse to drive the real cursor."""
    return KeySignals(config.output.kill_hotkey, source or PynputKeySource())
