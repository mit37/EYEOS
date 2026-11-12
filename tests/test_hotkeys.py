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
