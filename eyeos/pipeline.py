"""Wires input source -> features -> calibration -> One Euro filter -> gates -> output
(PRD §3). This is the "dry-run mode runs the whole pipeline on a video file" milestone:
every InputSource and OutputAdapter is interchangeable, so the exact same Pipeline
runs identically over a webcam, a recorded video, or a committed synthetic-landmark
CSV — only the source/backend objects passed to the constructor differ.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from eyeos.calibration import Calibration
from eyeos.config import Config
from eyeos.features import extract_features
from eyeos.filtering import OneEuroFilter2D
from eyeos.gates import FrameInput, GateStateMachine, GateStatus
from eyeos.input_sources import Frame, InputSource
from eyeos.logging_utils import SessionLogger
from eyeos.output import OutputAdapter

# How much longer than the confirmation-hold threshold a closed-eye stretch is still
# treated as a deliberate "blink and hold to confirm" gesture, rather than a genuine
# eyes-away dropout. See Pipeline._resolve_blink_gesture's docstring for why this
# exists at all.
_BLINK_GESTURE_GRACE_S = 0.5

ExternalSignals = Callable[[float], tuple[bool, bool]]


def _no_external_signals(_timestamp_s: float) -> tuple[bool, bool]:
    return False, False


@dataclass
class PipelineRunResult:
    statuses: list[GateStatus] = field(default_factory=list)

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

    def _resolve_blink_gesture(
        self, timestamp_s: float, raw_eyes_open: bool
    ) -> tuple[bool, float | None]:
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
        """
        if raw_eyes_open:
            self._blink_hold_start_s = None
            return True, None

        if self._blink_hold_start_s is None:
            self._blink_hold_start_s = timestamp_s
        held_s = timestamp_s - self._blink_hold_start_s

        max_hold_s = self.config.gates.confirmation_hold_s + _BLINK_GESTURE_GRACE_S
        if held_s <= max_hold_s:
            return True, held_s
        return False, None

    def _frame_input(self, frame: Frame) -> FrameInput:
        kill_switch_pressed, confirm_keypress = self.external_signals(frame.timestamp_s)

        if frame.landmarks is None:
            self._blink_hold_start_s = None
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
        features = extract_features(landmarks)

        raw_point = None
        if self.calibration is not None and self.calibration.is_fitted:
            raw_point = self.calibration.predict(features)

        smoothed_point = None
        if raw_point is not None:
            if self._filter is None:
                self._filter = OneEuroFilter2D(
                    frame.timestamp_s,
                    raw_point,
                    min_cutoff=self.config.filter.min_cutoff,
                    beta=self.config.filter.beta,
                    d_cutoff=self.config.filter.d_cutoff,
                )
                smoothed_point = raw_point
            else:
                smoothed_point = self._filter(frame.timestamp_s, raw_point)

        visible_for_gate, confirm_hold_s = self._resolve_blink_gesture(
            frame.timestamp_s, landmarks.both_irises_visible
        )

        return FrameInput(
            timestamp_s=frame.timestamp_s,
            face_count=landmarks.face_count,
            detection_confidence=landmarks.detection_confidence,
            both_irises_visible=visible_for_gate,
            gaze_point_px=smoothed_point,
            head_pose_deg=features.head_pose(),
            kill_switch_pressed=kill_switch_pressed,
            confirm_hold_s=confirm_hold_s,
            confirm_keypress=confirm_keypress,
        )

    def _process_frame(self, frame: Frame) -> GateStatus:
        frame_input = self._frame_input(frame)
        status = self.gates.step(frame_input)

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

    def run(self, max_frames: int | None = None) -> PipelineRunResult:
        result = PipelineRunResult()
        with self.input_source:
            for i, frame in enumerate(self.input_source.frames()):
                if max_frames is not None and i >= max_frames:
                    break
                result.statuses.append(self._process_frame(frame))
        return result
