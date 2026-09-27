"""Moves the real cursor on Windows via the Win32 ``SendInput`` API (ctypes, no extra
dependency). Untested in this cloud container (no Windows desktop) — see
docs/PLAN.md. ``SendInput`` is used instead of the simpler ``SetCursorPos`` because it
injects at the same level as real hardware input, which plays more reliably with
apps/games that ignore ``SetCursorPos``.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from typing import Any

from eyeos.output import OutputAdapter

_INPUT_MOUSE = 0
_MOUSEEVENTF_MOVE = 0x0001
_MOUSEEVENTF_ABSOLUTE = 0x8000
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_SM_CXSCREEN = 0
_SM_CYSCREEN = 1


class _MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _InputUnion(ctypes.Union):
    _fields_ = [("mi", _MouseInput)]


class _Input(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _InputUnion)]


class WindowsSendInputOutput(OutputAdapter):
    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError(
                "WindowsSendInputOutput only works on Windows (sys.platform == 'win32')"
            )
        self._user32: Any = ctypes.windll.user32
        self._screen_w: int = self._user32.GetSystemMetrics(_SM_CXSCREEN)
        self._screen_h: int = self._user32.GetSystemMetrics(_SM_CYSCREEN)

    def _send(self, dx: int, dy: int, flags: int) -> None:
        extra = ctypes.c_ulong(0)
        mi = _MouseInput(dx, dy, 0, flags, 0, ctypes.pointer(extra))
        inp = _Input(_INPUT_MOUSE, _InputUnion(mi=mi))
        self._user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))

    def move_cursor(self, x: int, y: int) -> None:
        # Absolute mouse-move coordinates are normalized to [0, 65535] across the
        # primary screen, per the SendInput/MOUSEEVENTF_ABSOLUTE contract.
        norm_x = int(x * 65535 / max(self._screen_w - 1, 1))
        norm_y = int(y * 65535 / max(self._screen_h - 1, 1))
        self._send(norm_x, norm_y, _MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE)

    def click(self) -> None:
        self._send(0, 0, _MOUSEEVENTF_LEFTDOWN)
        self._send(0, 0, _MOUSEEVENTF_LEFTUP)
