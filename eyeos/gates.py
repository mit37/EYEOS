"""The four-gate safety state machine — the interesting part of EyeOS (PRD §2).

A pure state machine: it consumes one ``FrameInput`` per tick (a frame's worth of
already-filtered pipeline output — see ``pipeline.py``) and returns a ``GateStatus``.
All timing comes from ``FrameInput.timestamp_s``, which the caller controls — there is
no wall-clock read anywhere in this module, so tests drive it with an entirely fake
clock just by choosing timestamps.

    1. Presence    -- one face, confidence >= threshold, both irises visible.
                      Fails -> cursor frozen.
    2. Calibration -- a calibration exists, its error < X px, pose within Y deg.
                      Fails -> "recalibrate"; cursor frozen.
    3. Stability   -- gaze has stayed in the same region for the dwell time.
                      Fails -> no click (movement still OK).
    4. Kill/limits -- kill hotkey up; speed/click-rate within limits; not a
                      no-click zone. Fails -> all output stops.

Movement needs gates 1, 2, 4. Click needs all four plus a confirmation.

Design decision: gate 4 is evaluated as one bundled boolean, exactly as the PRD table
states it — kill switch, speed limit, click-rate limit and no-click zones all gate
*both* movement and clicking. A gaze that is momentarily over a no-click zone, or that
has used up its click budget, freezes the cursor too, not just the click. That is
stricter than a typical UI would need, but this project's whole pitch is a
conservative, legible safety model, and "one bundled gate, one clear reason" is easier
to reason about (and to log, and to show as one overlay light) than four independently
wired sub-gates. See the README's Design decisions section.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from eyeos.calibration import CalibrationStatus
from eyeos.config import GatesConfig
from eyeos.features import head_pose_delta_deg

ScreenPoint = tuple[float, float]
HeadPose = tuple[float, float, float]


@dataclass(frozen=True)
class FrameInput:
    """One tick of already-filtered pipeline output, ready for gating."""

    timestamp_s: float
    face_count: int
    detection_confidence: float
    both_irises_visible: bool
    gaze_point_px: ScreenPoint | None
    head_pose_deg: HeadPose
    kill_switch_pressed: bool = False
    # Continuous confirmation gesture (e.g. blink-hold), seconds held so far this
    # frame; None if no such gesture is currently active.
    confirm_hold_s: float | None = None
    # Instantaneous confirmation (e.g. a keypress) this frame.
    confirm_keypress: bool = False


@dataclass(frozen=True)
class GateStatus:
    gate1_presence: bool
    gate2_calibration: bool
    gate3_stability: bool
    gate4_kill_and_limits: bool
    move_allowed: bool
    cursor_target_px: ScreenPoint | None
    click_fired: bool
    reasons: dict[str, str] = field(default_factory=dict)


def _distance(a: ScreenPoint, b: ScreenPoint) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class GateStateMachine:
    def __init__(self, config: GatesConfig | None = None) -> None:
        self.config = config or GatesConfig()
        self._calibration = CalibrationStatus.none()
        self._dwell_anchor: ScreenPoint | None = None
        self._dwell_start_s: float | None = None
        self._click_timestamps: list[float] = []
        self._awaiting_confirmation_release = False
        self._prev_gaze_point: ScreenPoint | None = None
        self._prev_timestamp_s: float | None = None

    def set_calibration(self, status: CalibrationStatus) -> None:
        self._calibration = status

    # -- gate 1 ---------------------------------------------------------------
    def _gate1(self, frame: FrameInput) -> tuple[bool, str]:
        if frame.face_count != 1:
            return False, f"expected exactly one face, saw {frame.face_count}"
        if frame.detection_confidence < self.config.presence_confidence_threshold:
            return False, (
                f"detection confidence {frame.detection_confidence:.2f} below "
                f"threshold {self.config.presence_confidence_threshold:.2f}"
            )
        if not frame.both_irises_visible:
            return False, "both irises not visible"
        return True, "ok"

    # -- gate 2 ---------------------------------------------------------------
    def _gate2(self, frame: FrameInput) -> tuple[bool, str]:
        if not self._calibration.exists:
            return False, "no calibration for this session — recalibrate"
        if self._calibration.validation_error_px >= self.config.calibration_max_error_px:
            return False, (
                f"calibration error {self._calibration.validation_error_px:.1f}px >= "
                f"{self.config.calibration_max_error_px:.1f}px — recalibrate"
            )
        delta = head_pose_delta_deg(self._calibration.calibration_head_pose, frame.head_pose_deg)
        if delta > self.config.calibration_max_head_pose_delta_deg:
            return False, (
                f"head pose drifted {delta:.1f}deg from calibration pose "
                f"(limit {self.config.calibration_max_head_pose_delta_deg:.1f}deg) — recalibrate"
            )
        return True, "ok"

    # -- gate 4 -----------------------------------------------------------
    def _gate4(self, frame: FrameInput) -> tuple[bool, str]:
        if frame.kill_switch_pressed:
            return False, "kill switch pressed"

        speed_px_s = 0.0
        if (
            frame.gaze_point_px is not None
            and self._prev_gaze_point is not None
            and self._prev_timestamp_s is not None
        ):
            dt = frame.timestamp_s - self._prev_timestamp_s
            if dt > 0:
                speed_px_s = _distance(frame.gaze_point_px, self._prev_gaze_point) / dt
        if speed_px_s > self.config.max_cursor_speed_px_s:
            return False, f"cursor speed {speed_px_s:.0f}px/s exceeds limit"

        recent_clicks = [
            t for t in self._click_timestamps if frame.timestamp_s - t < self.config.click_window_s
        ]
        if len(recent_clicks) >= self.config.max_clicks_per_window:
            return False, (
                f"click limit reached ({len(recent_clicks)}/{self.config.max_clicks_per_window}"
                f" in {self.config.click_window_s:.0f}s)"
            )

        if frame.gaze_point_px is not None:
            for zone in self.config.no_click_zones:
                if zone.contains(*frame.gaze_point_px):
                    return False, "gaze is inside a no-click zone"

        return True, "ok"

    # -- gate 3 (dwell) -------------------------------------------------------
    def _update_dwell(self, frame: FrameInput, eligible: bool) -> bool:
        if not eligible or frame.gaze_point_px is None:
            self._dwell_anchor = None
            self._dwell_start_s = None
            return False

        if self._dwell_anchor is None or self._dwell_start_s is None:
            self._dwell_anchor = frame.gaze_point_px
            self._dwell_start_s = frame.timestamp_s
        elif _distance(frame.gaze_point_px, self._dwell_anchor) > self.config.dwell_jitter_px:
            self._dwell_anchor = frame.gaze_point_px
            self._dwell_start_s = frame.timestamp_s

        return (frame.timestamp_s - self._dwell_start_s) >= self.config.dwell_time_s

    def step(self, frame: FrameInput) -> GateStatus:
        reasons: dict[str, str] = {}

        gate1, reasons["gate1"] = self._gate1(frame)
        gate2, reasons["gate2"] = self._gate2(frame)
        gate4, reasons["gate4"] = self._gate4(frame)

        move_allowed = gate1 and gate2 and gate4
        gate3 = self._update_dwell(frame, eligible=move_allowed)
        reasons["gate3"] = (
            "dwell satisfied"
            if gate3
            else (
                "dwell not eligible (gates 1/2/4 not all passing)"
                if not move_allowed
                else "dwell in progress"
            )
        )

        confirmed = frame.confirm_keypress or (
            frame.confirm_hold_s is not None
            and frame.confirm_hold_s >= self.config.confirmation_hold_s
        )
        confirmation_active = frame.confirm_keypress or frame.confirm_hold_s is not None

        click_fired = False
        if move_allowed and gate3 and confirmed and not self._awaiting_confirmation_release:
            click_fired = True
            self._click_timestamps.append(frame.timestamp_s)
            self._awaiting_confirmation_release = True
            reasons["click"] = "fired"
        elif not confirmation_active:
            self._awaiting_confirmation_release = False
            reasons["click"] = "no confirmation gesture"
        elif self._awaiting_confirmation_release:
            reasons["click"] = "confirmation already used — release and re-confirm"
        else:
            reasons["click"] = "not all conditions met"

        # Trim the click-rate window so it never grows unbounded over a long session.
        self._click_timestamps = [
            t for t in self._click_timestamps if frame.timestamp_s - t < self.config.click_window_s
        ]

        self._prev_gaze_point = frame.gaze_point_px
        self._prev_timestamp_s = frame.timestamp_s

        return GateStatus(
            gate1_presence=gate1,
            gate2_calibration=gate2,
            gate3_stability=gate3,
            gate4_kill_and_limits=gate4,
            move_allowed=move_allowed,
            cursor_target_px=frame.gaze_point_px if move_allowed else None,
            click_fired=click_fired,
            reasons=reasons,
        )
