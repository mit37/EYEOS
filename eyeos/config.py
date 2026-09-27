"""Config file loading. See config.example.yaml for the full schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


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
    no_click_zones: tuple[NoClickZone, ...] = ()


@dataclass(frozen=True)
class OutputConfig:
    backend: str = "dry_run"
    kill_hotkey: str = "esc"


@dataclass(frozen=True)
class LoggingConfig:
    session_log_dir: str = "logs"


@dataclass(frozen=True)
class Config:
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    input: InputConfig = field(default_factory=InputConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    gates: GatesConfig = field(default_factory=GatesConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _dc(cls: type, data: dict[str, Any] | None) -> Any:
    return cls(**(data or {}))


def load_config(path: str | Path | None) -> Config:
    """Load a Config from a YAML file, falling back to defaults for missing keys.

    ``path=None`` returns the all-defaults Config (useful for tests).
    """
    raw: dict[str, Any] = {}
    if path is not None:
        text = Path(path).read_text()
        raw = yaml.safe_load(text) or {}

    gates_raw = dict(raw.get("gates") or {})
    zones_raw = gates_raw.pop("no_click_zones", []) or []
    zones = tuple(NoClickZone(**z) for z in zones_raw)

    return Config(
        screen=_dc(ScreenConfig, raw.get("screen")),
        input=_dc(InputConfig, raw.get("input")),
        calibration=_dc(CalibrationConfig, raw.get("calibration")),
        filter=_dc(FilterConfig, raw.get("filter")),
        gates=GatesConfig(**gates_raw, no_click_zones=zones),
        output=_dc(OutputConfig, raw.get("output")),
        logging=_dc(LoggingConfig, raw.get("logging")),
    )
