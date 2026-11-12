"""Config file loading. See config.example.yaml for the full schema."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from eyeos.hotkeys import parse_key_name


@dataclass(frozen=True)
class ScreenConfig:
    width_px: int = 1920
    height_px: int = 1080


@dataclass(frozen=True)
class InputConfig:
    source: str = "synthetic"
    device_index: int = 0
    video_path: str | None = None
    synthetic_path: str | None = "eval/fixtures/synthetic_session.csv"


@dataclass(frozen=True)
class CalibrationConfig:
    points: int = 9


@dataclass(frozen=True)
class FilterConfig:
    min_cutoff: float = 1.0
    beta: float = 0.02
    d_cutoff: float = 1.0


@dataclass(frozen=True)
class NoClickZone:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


CONFIRMATION_MODES = ("either", "blink_hold", "keypress")


@dataclass(frozen=True)
class GatesConfig:
    presence_confidence_threshold: float = 0.5
    calibration_max_error_px: float = 60.0
    calibration_max_head_pose_delta_deg: float = 15.0
    dwell_time_s: float = 0.8
    dwell_jitter_px: float = 20.0
    max_cursor_speed_px_s: float = 4000.0
    max_clicks_per_window: int = 5
    click_window_s: float = 10.0
    confirmation_hold_s: float = 0.4
    # PRD §2: "a blink-hold >= 400 ms *or* a keypress, configurable".
    confirmation_mode: str = "either"  # either | blink_hold | keypress
    # A longer gap between frames (a camera stall) fails Gate 1 for that frame and
    # restarts the dwell, so the gap never counts as observed time.
    max_frame_gap_s: float = 0.25
    no_click_zones: tuple[NoClickZone, ...] = ()


@dataclass(frozen=True)
class OutputConfig:
    backend: str = "dry_run"
    kill_hotkey: str = "esc"
    confirm_key: str | None = None


@dataclass(frozen=True)
class LoggingConfig:
    session_log_dir: str | None = "logs"  # null disables the default session log


@dataclass(frozen=True)
class Config:
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    input: InputConfig = field(default_factory=InputConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    gates: GatesConfig = field(default_factory=GatesConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _check_keys(section: str, data: Any, cls: type) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{section} must be a mapping of keys, got {type(data).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            f"unknown config key(s) {', '.join(f'{section}.{k}' for k in unknown)}; "
            f"{section} accepts: {', '.join(sorted(known))}"
        )


def _dc(cls: type, data: dict[str, Any] | None, section: str) -> Any:
    data = data or {}
    _check_keys(section, data, cls)
    return cls(**data)


def load_config(path: str | Path | None) -> Config:
    """Load a Config from a YAML file, falling back to defaults for missing keys.

    ``path=None`` returns the all-defaults Config (useful for tests).
    """
    raw: Any = {}
    if path is not None:
        text = Path(path).read_text()
        raw = yaml.safe_load(text) or {}

    _check_keys("config", raw, Config)
    gates_raw = raw.get("gates") or {}
    _check_keys("gates", gates_raw, GatesConfig)
    gates_raw = dict(gates_raw)
    zones_raw = gates_raw.pop("no_click_zones", []) or []
    if not isinstance(zones_raw, list):
        raise ValueError("gates.no_click_zones must be a list of zones")
    for i, zone in enumerate(zones_raw):
        _check_keys(f"gates.no_click_zones[{i}]", zone, NoClickZone)
    zones = tuple(NoClickZone(**z) for z in zones_raw)

    output = _dc(OutputConfig, raw.get("output"), "output")
    try:
        kill_key = parse_key_name(output.kill_hotkey)
    except ValueError as e:
        raise ValueError(f"output.kill_hotkey: {e}") from e
    if output.confirm_key is not None:
        try:
            confirm_key = parse_key_name(output.confirm_key)
        except ValueError as e:
            raise ValueError(f"output.confirm_key: {e}") from e
        if confirm_key == kill_key:
            raise ValueError("output.confirm_key must differ from output.kill_hotkey")

    calibration = _dc(CalibrationConfig, raw.get("calibration"), "calibration")
    if calibration.points != 9:
        raise ValueError("calibration.points: PRD §3 fixes a 9-point calibration")

    gates = GatesConfig(**gates_raw, no_click_zones=zones)
    if not 0.0 <= gates.presence_confidence_threshold <= 1.0:
        raise ValueError("gates.presence_confidence_threshold must be between 0 and 1")
    if not gates.max_frame_gap_s > 0:
        raise ValueError("gates.max_frame_gap_s must be positive")
    if gates.confirmation_mode not in CONFIRMATION_MODES:
        raise ValueError(
            f"gates.confirmation_mode must be one of {', '.join(CONFIRMATION_MODES)}, "
            f"not {gates.confirmation_mode!r}"
        )
    if gates.confirmation_mode == "keypress" and output.confirm_key is None:
        raise ValueError(
            "gates.confirmation_mode 'keypress' needs output.confirm_key: without a "
            "confirm key no click can ever be confirmed"
        )

    return Config(
        screen=_dc(ScreenConfig, raw.get("screen"), "screen"),
        input=_dc(InputConfig, raw.get("input"), "input"),
        calibration=calibration,
        filter=_dc(FilterConfig, raw.get("filter"), "filter"),
        gates=gates,
        output=output,
        logging=_dc(LoggingConfig, raw.get("logging"), "logging"),
    )
