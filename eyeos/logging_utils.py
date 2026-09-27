"""Session logs: one JSON line per frame, carrying every gate's decision and reason
(PRD §2: "every gate decision is logged with its reason")."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eyeos.gates import GateStatus


class SessionLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w")

    def log(self, timestamp_s: float, status: GateStatus) -> None:
        record: dict[str, Any] = {
            "timestamp_s": timestamp_s,
            "gate1_presence": status.gate1_presence,
            "gate2_calibration": status.gate2_calibration,
            "gate3_stability": status.gate3_stability,
            "gate4_kill_and_limits": status.gate4_kill_and_limits,
            "move_allowed": status.move_allowed,
            "cursor_target_px": status.cursor_target_px,
            "click_fired": status.click_fired,
            "reasons": status.reasons,
        }
        self._fh.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> SessionLogger:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
