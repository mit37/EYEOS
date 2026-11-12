"""Wires input source -> features -> calibration -> One Euro filter -> gates -> output
(PRD §3). This is the "dry-run mode runs the whole pipeline on a video file" milestone:
every InputSource and OutputAdapter is interchangeable, so the exact same Pipeline
runs identically over a webcam, a recorded video, or a committed synthetic-landmark
CSV — only the source/backend objects passed to the constructor differ.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from eyeos.calibration import Calibration
from eyeos.config import Config
from eyeos.features import FeatureVector, extract_features
from eyeos.filtering import OneEuroFilter2D
from eyeos.gates import FrameInput, GateStateMachine, GateStatus, ScreenPoint
from eyeos.input_sources import Frame, InputSource
from eyeos.logging_utils import SessionLogger
from eyeos.output import OutputAdapter

# How much longer than the confirmation-hold threshold a closed-eye stretch is still
# treated as a deliberate "blink and hold to confirm" gesture, rather than a genuine
# eyes-away dropout. See Pipeline._resolve_blink_gesture's docstring for why this
# exists at all.
_BLINK_GESTURE_GRACE_S = 0.5
# After a closure, the eyes count as open again only once they have stayed open this
# long. A shorter glimpse (lid flutter while resting the eyes) is part of the closure:
# it neither ends the blink-hold gesture nor counts as looking.
_MIN_REOPEN_S = 0.1

ExternalSignals = Callable[[float], tuple[bool, bool]]


def _no_external_signals(_timestamp_s: float) -> tuple[bool, bool]:
    return False, False


@dataclass
class PipelineRunResult:
    statuses: list[GateStatus] = field(default_factory=list)
    killed: bool = False

    @property
    def frame_count(self) -> int:
        return len(self.statuses)

    @property
    def total_clicks(self) -> int:
        return sum(1 for s in self.statuses if s.click_fired)

    @property
    def total_moves(self) -> int:
        return sum(1 for s in self.statuses if s.move_allowed)


class Pipeline:
    def __init__(
        self,
        config: Config,
        input_source: InputSource,
        output: OutputAdapter,
        calibration: Calibration | None = None,
        logger: SessionLogger | None = None,
        external_signals: ExternalSignals | None = None,
        on_status: Callable[[GateStatus], None] | None = None,
    ) -> None:
        if output.moves_real_cursor and external_signals is None:
            raise ValueError(
                "refusing to drive the real cursor without a kill switch; "
                "pass external_signals=build_key_signals(config) (eyeos.hotkeys)"
            )
        self.config = config
        self.input_source = input_source
        self.output = output
        self.calibration = calibration
        self.logger = logger
        self.external_signals = external_signals or _no_external_signals
        # Called with every frame's GateStatus, e.g. to drive a live overlay
        # (PRD §2: "the overlay shows the four gate lights live") — see docs/DEMO.md.
        self.on_status = on_status

        self.gates = GateStateMachine(config.gates)
        if calibration is not None:
            self.gates.set_calibration(calibration.status())

        self._filter: OneEuroFilter2D | None = None
        self._blink_hold_start_s: float | None = None
        self._reopen_since_s: float | None = None
        # The last point the gates emitted for an observed (eyes open, trusted) frame;
        # None once an observed frame fails any gate. Held frames reuse it.
        self._last_open_gaze: ScreenPoint | None = None
        self._kill_seen = False

    def _resolve_blink_gesture(
        self, timestamp_s: float, raw_eyes_open: bool
    ) -> tuple[bool, float | None, bool]:
        """Turn a raw per-frame eye-openness reading into (visible_for_gate1, confirm_hold_s).

        PRD §2 wants a "blink-hold >= 400ms" click confirmation, but Gate 1 (PRD §2's
        own table) requires both irises visible to allow movement at all — and a
        deliberate blink *is* both eyes closed, the same signal Gate 1 uses. Taken
        literally, Gate 1 would freeze the cursor (and Gate 3's dwell) the instant the
        confirmation gesture starts, so the click could never fire.

        This method is the pipeline's resolution: while eyes have been closed for no
        longer than ``confirmation_hold_s + grace``, treat it as an in-progress
        deliberate confirm gesture — report the eyes as still "visible" to Gate 1 (so
        movement/dwell state isn't lost) and report the accumulated hold duration as
        ``confirm_hold_s``. Past that grace period, it stops looking like a deliberate
        blink and starts looking like the user genuinely looked away or closed their
        eyes — from then on it's reported honestly (not visible), so Gate 1 correctly
        freezes the cursor.

        Returns ``(visible_for_gate1, confirm_hold_s, eyes_open)``. ``eyes_open`` is
        False on closed frames and on the first ``_MIN_REOPEN_S`` of a reopening, which
        still belong to the closure.
        """
        if raw_eyes_open:
            if self._blink_hold_start_s is None:
                return True, None, True
            if self._reopen_since_s is None:
                self._reopen_since_s = timestamp_s
            if timestamp_s - self._reopen_since_s >= _MIN_REOPEN_S - 1e-9:
                self._blink_hold_start_s = None
                self._reopen_since_s = None
                return True, None, True
            # Not open long enough yet: still part of the closure (below).
        else:
            self._reopen_since_s = None
        if self.config.gates.confirmation_mode == "keypress":
            # No blink gesture to wait for: closed eyes fail Gate 1 now.
            return False, None, False

        if self._blink_hold_start_s is None:
            self._blink_hold_start_s = timestamp_s
        held_s = timestamp_s - self._blink_hold_start_s

        max_hold_s = self.config.gates.confirmation_hold_s + _BLINK_GESTURE_GRACE_S
        if held_s <= max_hold_s:
            return True, held_s, False
        return False, None, False

    def _frame_input(self, frame: Frame) -> FrameInput:
        kill_switch_pressed, confirm_keypress = self.external_signals(frame.timestamp_s)
        self._kill_seen = self._kill_seen or kill_switch_pressed

        if not math.isfinite(frame.timestamp_s):
            # Gate 1 rejects it. Touch no timing state and don't feed the filter.
            return FrameInput(
                timestamp_s=frame.timestamp_s,
                face_count=frame.landmarks.face_count if frame.landmarks else 0,
                detection_confidence=0.0,
                both_irises_visible=False,
                gaze_point_px=None,
                head_pose_deg=(0.0, 0.0, 0.0),
                kill_switch_pressed=kill_switch_pressed,
                confirm_keypress=confirm_keypress,
            )

        if frame.landmarks is None:
            self._blink_hold_start_s = None
            self._reopen_since_s = None
            self._last_open_gaze = None
            return FrameInput(
                timestamp_s=frame.timestamp_s,
                face_count=0,
                detection_confidence=0.0,
                both_irises_visible=False,
                gaze_point_px=None,
                head_pose_deg=(0.0, 0.0, 0.0),
                kill_switch_pressed=kill_switch_pressed,
                confirm_keypress=confirm_keypress,
            )

        landmarks = frame.landmarks
        raw_eyes_open = landmarks.both_irises_visible
        visible_for_gate, confirm_hold_s, eyes_open = self._resolve_blink_gesture(
            frame.timestamp_s, raw_eyes_open
        )
        features = extract_features(landmarks)

        # Closed-eye landmarks are never mapped to the screen or fed to the filter: the
        # iris estimate under a closed lid is MediaPipe's guess. During a blink-hold the
        # gaze holds the last point the gates emitted while the eyes were open (that is
        # where the click must land); past the grace window there is no gaze at all. A
        # held point is marked as such, so Gate 3 doesn't count it as dwell. The head pose
        # is always this frame's: it comes from the nose, face outline and eye corners,
        # not the lids, and Gate 2 must still see a head turn mid-blink.
        head_pose = features.head_pose()
        gaze_point: ScreenPoint | None
        gaze_held = False
        if eyes_open:
            gaze_point = self._map_gaze(frame.timestamp_s, features)
        elif confirm_hold_s is not None:
            gaze_point, gaze_held = self._last_open_gaze, True
        else:
            # Past the grace window (or in keypress mode) there is no gaze, and the
            # held point is dropped after gating (see _process_frame).
            gaze_point = None

        return FrameInput(
            timestamp_s=frame.timestamp_s,
            face_count=landmarks.face_count,
            detection_confidence=landmarks.detection_confidence,
            both_irises_visible=visible_for_gate,
            gaze_point_px=gaze_point,
            head_pose_deg=head_pose,
            kill_switch_pressed=kill_switch_pressed,
            confirm_hold_s=confirm_hold_s,
            confirm_keypress=confirm_keypress,
            gaze_held=gaze_held,
        )

    def _map_gaze(self, timestamp_s: float, features: FeatureVector) -> ScreenPoint | None:
        if self.calibration is None or not self.calibration.is_fitted:
            return None
        raw_point = self.calibration.predict(features)
        if not all(math.isfinite(c) for c in raw_point):
            return None  # never seed or feed the filter a NaN/inf: it would stick forever
        if self._filter is None:
            self._filter = OneEuroFilter2D(
                timestamp_s,
                raw_point,
                min_cutoff=self.config.filter.min_cutoff,
                beta=self.config.filter.beta,
                d_cutoff=self.config.filter.d_cutoff,
            )
            smoothed = raw_point
        else:
            smoothed = self._filter(timestamp_s, raw_point)
        return self._to_screen_pixel(smoothed)

    def _to_screen_pixel(self, point: ScreenPoint) -> ScreenPoint:
        """The exact on-screen pixel the cursor will be sent to.

        Gate 4's no-click zones and speed limit must judge this pixel, not the float
        gaze it came from, or a click can land inside a zone after rounding/clamping.
        """
        width, height = self.config.screen.width_px, self.config.screen.height_px
        x = min(max(round(point[0]), 0), width - 1)
        y = min(max(round(point[1]), 0), height - 1)
        return float(x), float(y)

    def _process_frame(self, frame: Frame) -> GateStatus:
        frame_input = self._frame_input(frame)
        status = self.gates.step(frame_input)
        if not frame_input.gaze_held:
            # Only a point the gates allowed may be held through a later blink. A frame
            # they rejected (a second face, a blocked jump, a lost face) drops it.
            self._last_open_gaze = status.cursor_target_px if status.move_allowed else None

        if status.move_allowed and status.cursor_target_px is not None:
            x, y = status.cursor_target_px
            self.output.move_cursor(int(round(x)), int(round(y)))
        if status.click_fired:
            self.output.click()
        if self.logger is not None:
            self.logger.log(frame.timestamp_s, status)
        if self.on_status is not None:
            self.on_status(status)

        return status

    def run(
        self, max_frames: int | None = None, frames: Iterator[Frame] | None = None
    ) -> PipelineRunResult:
        """Run until the source ends, ``max_frames`` is reached, or the kill switch trips.

        The kill frame itself is still processed (Gate 4 fails, so nothing is emitted and
        the log/overlay show why); then the session ends. Restart to resume.

        ``frames`` lets a caller that already started reading the source (e.g. an
        in-process calibration) continue on the same stream instead of reopening it.
        """
        result = PipelineRunResult()
        with self.input_source:
            stream = frames if frames is not None else self.input_source.frames()
            for i, frame in enumerate(stream):
                if max_frames is not None and i >= max_frames:
                    break
                result.statuses.append(self._process_frame(frame))
                if self._kill_seen:
                    result.killed = True
                    break
        return result
