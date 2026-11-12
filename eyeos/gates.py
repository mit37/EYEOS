"""The four-gate safety state machine — the interesting part of EyeOS (PRD §2).

A pure state machine: it consumes one ``FrameInput`` per tick (a frame's worth of
already-filtered pipeline output — see ``pipeline.py``) and returns a ``GateStatus``.
All timing comes from ``FrameInput.timestamp_s``, which the caller controls — there is
no wall-clock read anywhere in this module, so tests drive it with an entirely fake
clock just by choosing timestamps.

    1. Presence    -- one face, confidence >= threshold, both irises visible, and a
                      live clock (finite, advancing, no gap > max_frame_gap_s).
                      Fails -> cursor frozen.
    2. Calibration -- a calibration exists, its error < X px, pose within Y deg.
                      Fails -> "recalibrate"; cursor frozen.
    3. Stability   -- gaze has stayed in the same region for the dwell time, counting
                      only observed frames (not ``gaze_held`` ones). A click uses the
                      dwell up: the next click needs a fresh one.
                      Fails -> no click (movement still OK).
    4. Kill/limits -- kill hotkey up; cursor speed and click rate within limits;
                      not a no-click zone. Fails -> all output stops.

Movement needs gates 1, 2, 4. Click needs all four plus a confirmation.

Design decision: gate 4 is evaluated as one bundled boolean, exactly as the PRD table
states it — kill switch, speed limit, click-rate limit and no-click zones all gate
*both* movement and clicking. A gaze that is momentarily over a no-click zone, or that
has used up its click budget, freezes the cursor too, not just the click. That is
stricter than a typical UI would need, but this project's whole pitch is a
conservative, legible safety model, and "one bundled gate, one clear reason" is easier
to reason about (and to log, and to show as one overlay light) than four independently
wired sub-gates. See the README's Design decisions section.

Cursor speed is the average speed from the last position actually emitted to the
current gaze, so a frozen frame or a dropout cannot reset it. A large jump keeps gate 4
failing until that average is within the limit (about distance / max_cursor_speed_px_s),
then the cursor relocates in one step. The emitted cursor is never rate-limited toward
the gaze: a cursor still in transit could click somewhere the zone check never looked.
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
    # True when gaze_point_px was carried over from an earlier frame (eyes closed in the
    # blink-hold grace window) rather than observed in this one. Gate 3 doesn't count it.
    gaze_held: bool = False


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
    # The filtered gaze point fed to the gates, set even when movement is blocked.
    smoothed_gaze_px: ScreenPoint | None = None


def _distance(a: ScreenPoint, b: ScreenPoint) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class GateStateMachine:
    def __init__(self, config: GatesConfig | None = None) -> None:
        self.config = config or GatesConfig()
        self._calibration = CalibrationStatus.none()
        self._dwell_anchor: ScreenPoint | None = None
        self._dwell_start_s: float | None = None
        self._dwell_paused_at_s: float | None = None
        self._dwell_held_s = 0.0
        self._dwell_satisfied_when_observed = False
        self._click_timestamps: list[float] = []
        self._awaiting_confirmation_release = False
        self._last_emitted_px: ScreenPoint | None = None
        self._last_emitted_timestamp_s: float | None = None
        self._last_frame_timestamp_s: float | None = None

    def set_calibration(self, status: CalibrationStatus) -> None:
        self._calibration = status

    # -- gate 1 ---------------------------------------------------------------
    def _gate1(self, frame: FrameInput) -> tuple[bool, str]:
        # A clock that doesn't advance makes every timing check (dwell, speed, blink-hold)
        # meaningless, so it fails closed rather than pinning the cursor silently.
        if not math.isfinite(frame.timestamp_s):
            return False, "timestamp is not finite"
        last_t = self._last_frame_timestamp_s
        if last_t is not None and not frame.timestamp_s > last_t:
            return False, f"timestamp not advancing (t={frame.timestamp_s:.3f} <= {last_t:.3f})"
        # A stalled camera must not count as observed dwell or blink-hold time.
        if last_t is not None and frame.timestamp_s - last_t > self.config.max_frame_gap_s:
            return False, (
                f"frame gap {frame.timestamp_s - last_t:.2f}s exceeds "
                f"{self.config.max_frame_gap_s:.2f}s"
            )
        if not math.isfinite(frame.detection_confidence):
            return False, "detection confidence is not a number"
        if frame.face_count != 1:
            return False, f"expected exactly one face, saw {frame.face_count}"
        if frame.detection_confidence < self.config.presence_confidence_threshold:
            return False, (
                f"detection confidence {frame.detection_confidence:.2f} below "
                f"threshold {self.config.presence_confidence_threshold:.2f}"
            )
        if not frame.both_irises_visible:
            return False, "both irises not visible"
        if frame.confirm_hold_s is not None:
            return True, f"ok (eyes closed {frame.confirm_hold_s:.2f}s: blink-hold grace)"
        return True, "ok"

    # -- gate 2 ---------------------------------------------------------------
    def _gate2(self, frame: FrameInput) -> tuple[bool, str]:
        if not self._calibration.exists:
            return False, "no calibration for this session — recalibrate"
        if not all(math.isfinite(a) for a in frame.head_pose_deg):
            return False, "head pose is not finite"
        if not math.isfinite(self._calibration.validation_error_px):
            return False, "calibration not validated — recalibrate"
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
        if frame.gaze_point_px is not None and not all(
            math.isfinite(c) for c in frame.gaze_point_px
        ):
            return False, "gaze point is not finite"

        speed_px_s = 0.0
        if (
            frame.gaze_point_px is not None
            and self._last_emitted_px is not None
            and self._last_emitted_timestamp_s is not None
        ):
            dt = frame.timestamp_s - self._last_emitted_timestamp_s
            if dt > 0:
                speed_px_s = _distance(frame.gaze_point_px, self._last_emitted_px) / dt
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
    def _reset_dwell(self) -> None:
        self._dwell_anchor = None
        self._dwell_start_s = None
        self._dwell_paused_at_s = None
        self._dwell_held_s = 0.0
        self._dwell_satisfied_when_observed = False

    def _update_dwell(self, frame: FrameInput, eligible: bool) -> bool:
        if not eligible or frame.gaze_point_px is None:
            self._reset_dwell()
            return False

        if frame.gaze_held:
            # A held point is not an observation. Gate 3 keeps the answer from the last
            # observed frame, and the held time doesn't count toward the dwell, so a
            # glance followed by closed eyes can't satisfy it.
            if self._dwell_start_s is not None and self._dwell_paused_at_s is None:
                self._dwell_paused_at_s = frame.timestamp_s
            satisfied = self._dwell_satisfied_when_observed
            if (
                frame.confirm_hold_s is not None
                and frame.confirm_hold_s >= self.config.confirmation_hold_s
            ):
                # A confirm-length closure ends the dwell either way: it clicks on this
                # frame, or not at all. Otherwise single-frame glimpses spread over a
                # long, fluttering closure could add up to a new dwell.
                self._reset_dwell()
            return satisfied

        if self._dwell_paused_at_s is not None and self._dwell_start_s is not None:
            held = frame.timestamp_s - self._dwell_paused_at_s
            self._dwell_start_s += held
            self._dwell_held_s += held
            # Within one dwell the eyes must be open more than they are closed. Otherwise
            # short glimpses through a fluttering closure could add up to a dwell. (The
            # tolerance keeps a tie from flipping on float rounding.)
            if self._dwell_held_s >= frame.timestamp_s - self._dwell_start_s - 1e-9:
                self._reset_dwell()
        self._dwell_paused_at_s = None

        if self._dwell_anchor is None or self._dwell_start_s is None:
            self._dwell_anchor = frame.gaze_point_px
            self._dwell_start_s = frame.timestamp_s
        elif _distance(frame.gaze_point_px, self._dwell_anchor) > self.config.dwell_jitter_px:
            # A new anchor is a new dwell: blinks during the old one don't count against it.
            self._dwell_anchor = frame.gaze_point_px
            self._dwell_start_s = frame.timestamp_s
            self._dwell_held_s = 0.0

        satisfied = (frame.timestamp_s - self._dwell_start_s) >= self.config.dwell_time_s
        self._dwell_satisfied_when_observed = satisfied
        return satisfied

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

        mode = self.config.confirmation_mode
        key_down = frame.confirm_keypress if mode in ("either", "keypress") else False
        hold_s = frame.confirm_hold_s if mode in ("either", "blink_hold") else None
        ignored = (frame.confirm_keypress and not key_down) or (
            frame.confirm_hold_s is not None and hold_s is None
        )

        # A key confirms only what the eyes are observed on now: not a point held through
        # a blink or a reopening, where the eyes may already be somewhere else. While it
        # is down it still counts as the active gesture, so holding it through a blink
        # is not a release.
        keypress = key_down and not frame.gaze_held
        confirmed = keypress or (hold_s is not None and hold_s >= self.config.confirmation_hold_s)
        confirmation_active = key_down or hold_s is not None

        click_fired = False
        if move_allowed and gate3 and confirmed and not self._awaiting_confirmation_release:
            click_fired = True
            self._click_timestamps.append(frame.timestamp_s)
            self._awaiting_confirmation_release = True
            # A click uses the dwell up. Otherwise a dwell met once could confirm click
            # after click, e.g. across eyelid flutter in a long closure.
            self._reset_dwell()
            reasons["click"] = "fired"
        elif not confirmation_active:
            self._awaiting_confirmation_release = False
            reasons["click"] = (
                f"confirmation ignored (confirmation_mode={mode})"
                if ignored
                else "no confirmation gesture"
            )
        elif self._awaiting_confirmation_release:
            reasons["click"] = "confirmation already used — release and re-confirm"
        else:
            reasons["click"] = "not all conditions met"

        # Trim the click-rate window so it never grows unbounded over a long session.
        # A non-finite timestamp is never stored and never used to trim.
        if math.isfinite(frame.timestamp_s):
            self._click_timestamps = [
                t
                for t in self._click_timestamps
                if frame.timestamp_s - t < self.config.click_window_s
            ]

        if move_allowed and frame.gaze_point_px is not None:
            self._last_emitted_px = frame.gaze_point_px
            self._last_emitted_timestamp_s = frame.timestamp_s
        if math.isfinite(frame.timestamp_s) and (
            self._last_frame_timestamp_s is None or frame.timestamp_s > self._last_frame_timestamp_s
        ):
            self._last_frame_timestamp_s = frame.timestamp_s

        return GateStatus(
            gate1_presence=gate1,
            gate2_calibration=gate2,
            gate3_stability=gate3,
            gate4_kill_and_limits=gate4,
            move_allowed=move_allowed,
            cursor_target_px=frame.gaze_point_px if move_allowed else None,
            click_fired=click_fired,
            reasons=reasons,
            smoothed_gaze_px=frame.gaze_point_px,
        )
