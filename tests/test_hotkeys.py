from __future__ import annotations

import sys

import pytest

from eyeos.config import load_config
from eyeos.hotkeys import (
    FakeKeySource,
    KeySignals,
    PynputKeySource,
    build_key_signals,
    parse_key_name,
)


@pytest.mark.parametrize(
    "name, canonical",
    [("esc", "esc"), ("Escape", "esc"), ("F12", "f12"), ("a", "a"), ("space", "space")],
)
def test_parse_key_name_accepts_single_keys(name, canonical):
    assert parse_key_name(name) == canonical


@pytest.mark.parametrize("name", ["this-key-does-not-exist", "ctrl+c", "", " "])
def test_parse_key_name_rejects_unknown_keys_and_chords(name):
    with pytest.raises(ValueError):
        parse_key_name(name)


def test_kill_latches_after_release_regardless_of_timestamps():
    source = FakeKeySource()
    keys = KeySignals("esc", source)
    assert keys(0.0) == (False, False)
    source.press("esc")
    source.release("esc")
    assert keys(1.0) == (True, False)
    assert keys(0.5) == (True, False)  # frame clocks may repeat or go backwards
    assert keys.killed


def test_unrelated_key_does_not_kill():
    source = FakeKeySource()
    keys = KeySignals("esc", source)
    source.press("q")
    source.release("q")
    assert keys(0.0) == (False, False)


def test_confirm_key_reports_true_only_while_held():
    source = FakeKeySource()
    keys = KeySignals("esc", source, confirm_key="space")
    source.press("space")
    assert keys(0.0) == (False, True)
    source.release("space")
    assert keys(0.1) == (False, False)


def test_close_stops_the_source():
    source = FakeKeySource()
    with KeySignals("esc", source):
        assert source.running
    assert not source.running


def test_pynput_source_without_pynput_raises_with_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynput", None)
    with pytest.raises(RuntimeError, match="pip install 'eyeos\\[windows-fallback\\]'"):
        PynputKeySource().start(lambda _k: None, lambda _k: None)


def test_build_key_signals_uses_configured_kill_key(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("output:\n  kill_hotkey: F12\n")
    source = FakeKeySource()
    keys = build_key_signals(load_config(cfg_path), source=source)
    source.press("esc")
    assert not keys.killed
    source.press("f12")
    assert keys.killed


def test_load_config_rejects_unknown_kill_hotkey(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("output:\n  kill_hotkey: ctrl+c\n")
    with pytest.raises(ValueError, match="output.kill_hotkey"):
        load_config(cfg_path)


def test_build_key_signals_wires_the_configured_confirm_key(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("output:\n  confirm_key: space\n")
    source = FakeKeySource()
    keys = build_key_signals(load_config(cfg_path), source=source)
    source.press("space")
    assert keys(0.0) == (False, True)


# ---------------------------------------------------------------------------
# The real PynputKeySource, against a fake ``pynput`` module
# ---------------------------------------------------------------------------


class _FakeListener:
    def __init__(self, on_press=None, on_release=None, fail_with=None):
        self.on_press, self.on_release = on_press, on_release
        self.started = self.stopped = False
        self._fail_with = fail_with

    def start(self):
        if self._fail_with is not None:
            raise self._fail_with
        self.started = True

    def stop(self):
        self.stopped = True


def _install_fake_pynput(monkeypatch, fail_with=None):
    """pynput reports named keys as Key members (with .name) and printable keys as
    KeyCode objects (with .char); these stand-ins have just those attributes."""
    import types

    listeners = []

    def listener(on_press=None, on_release=None):
        made = _FakeListener(on_press, on_release, fail_with)
        listeners.append(made)
        return made

    pynput = types.ModuleType("pynput")
    pynput.keyboard = types.SimpleNamespace(Listener=listener)
    monkeypatch.setitem(sys.modules, "pynput", pynput)
    return listeners


def _key(name):
    import types

    return types.SimpleNamespace(name=name)


def _keycode(char):
    import types

    return types.SimpleNamespace(char=char)


@pytest.mark.parametrize(
    "kill_hotkey, event",
    [("esc", _key("esc")), ("F12", _key("f12")), ("q", _keycode("Q"))],
)
def test_real_pynput_source_delivers_the_kill_key(monkeypatch, tmp_path, kill_hotkey, event):
    listeners = _install_fake_pynput(monkeypatch)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(f"output:\n  kill_hotkey: {kill_hotkey}\n")

    keys = build_key_signals(load_config(cfg_path))
    (listener,) = listeners
    assert listener.started

    listener.on_press(_key("shift"))  # not a key EyeOS knows: ignored
    listener.on_press(_keycode(None))  # a KeyCode with no character: ignored
    listener.on_press(_keycode("x"))
    assert not keys.killed

    listener.on_press(event)
    listener.on_release(event)
    assert keys.killed  # latched after release

    keys.close()
    assert listener.stopped


def test_real_pynput_source_reports_the_confirm_key_while_held(monkeypatch, tmp_path):
    listeners = _install_fake_pynput(monkeypatch)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("output:\n  confirm_key: space\n")
    keys = build_key_signals(load_config(cfg_path))
    (listener,) = listeners

    listener.on_press(_key("space"))
    assert keys(0.0) == (False, True)
    listener.on_release(_key("space"))
    assert keys(0.0) == (False, False)


def test_real_pynput_listener_that_cannot_start_raises_with_install_hint(monkeypatch):
    _install_fake_pynput(monkeypatch, fail_with=OSError("no input permission"))
    with pytest.raises(RuntimeError, match="no input permission"):
        PynputKeySource().start(lambda _k: None, lambda _k: None)
