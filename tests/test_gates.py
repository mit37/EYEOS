"""The four-gate state machine, tested one transition at a time.

Every timestamp below is a value we chose, not a wall-clock read — the state machine
has no clock of its own (see gates.py's module docstring), so controlling
``FrameInput.timestamp_s`` from these tests *is* the fake clock.
"""

from __future__ import annotations

from eyeos.calibration import CalibrationStatus
from eyeos.config import GatesConfig, NoClickZone
from eyeos.gates import FrameInput, GateStateMachine

VALID_CALIBRATION = CalibrationStatus(
    exists=True, validation_error_px=10.0, calibration_head_pose=(0.0, 0.0, 0.0)
)


def make_config(**overrides) -> GatesConfig:
    defaults = dict(
        presence_confidence_threshold=0.5,
        calibration_max_error_px=60.0,
        calibration_max_head_pose_delta_deg=15.0,
        dwell_time_s=0.5,
        dwell_jitter_px=20.0,
        max_cursor_speed_px_s=2000.0,
        max_clicks_per_window=2,
        click_window_s=5.0,
        confirmation_hold_s=0.4,
        no_click_zones=(NoClickZone(1800, 0, 1920, 50),),
    )
    defaults.update(overrides)
    return GatesConfig(**defaults)


def make_frame(**overrides) -> FrameInput:
    defaults = dict(
        timestamp_s=0.0,
        face_count=1,
        detection_confidence=0.9,
        both_irises_visible=True,
        gaze_point_px=(500.0, 500.0),
        head_pose_deg=(0.0, 0.0, 0.0),
        kill_switch_pressed=False,
        confirm_hold_s=None,
        confirm_keypress=False,
    )
    defaults.update(overrides)
    return FrameInput(**defaults)


def machine(config: GatesConfig | None = None, calibrated: bool = True) -> GateStateMachine:
    m = GateStateMachine(config or make_config())
    if calibrated:
        m.set_calibration(VALID_CALIBRATION)
    return m


# ---------------------------------------------------------------------------
# Gate 1: presence
# ---------------------------------------------------------------------------


def test_gate1_passes_with_one_confident_face_both_irises():
    status = machine().step(make_frame())
    assert status.gate1_presence is True
    assert status.reasons["gate1"] == "ok"


def test_gate1_fails_with_zero_faces():
    status = machine().step(make_frame(face_count=0))
    assert status.gate1_presence is False
    assert "face" in status.reasons["gate1"]


def test_gate1_fails_with_two_faces():
    status = machine().step(make_frame(face_count=2))
    assert status.gate1_presence is False


def test_gate1_fails_below_confidence_threshold():
    status = machine().step(make_frame(detection_confidence=0.49))
    assert status.gate1_presence is False
    assert "confidence" in status.reasons["gate1"]


def test_gate1_passes_at_confidence_threshold_boundary():
    status = machine().step(make_frame(detection_confidence=0.5))
    assert status.gate1_presence is True


def test_gate1_fails_when_irises_not_visible():
    status = machine().step(make_frame(both_irises_visible=False))
    assert status.gate1_presence is False
    assert "iris" in status.reasons["gate1"]


# ---------------------------------------------------------------------------
# Gate 2: calibration validity
# ---------------------------------------------------------------------------


def test_gate2_fails_with_no_calibration():
    status = machine(calibrated=False).step(make_frame())
    assert status.gate2_calibration is False
    assert "recalibrate" in status.reasons["gate2"]


def test_gate2_passes_with_valid_fresh_calibration():
    status = machine().step(make_frame())
    assert status.gate2_calibration is True


def test_gate2_fails_when_validation_error_too_high():
    m = machine(calibrated=False)
    m.set_calibration(
        CalibrationStatus(exists=True, validation_error_px=61.0, calibration_head_pose=(0, 0, 0))
    )
    status = m.step(make_frame())
    assert status.gate2_calibration is False
    assert "error" in status.reasons["gate2"]


def test_gate2_fails_at_validation_error_boundary():
    # Config threshold is 60.0px; the check is strict "error < threshold to pass".
    m = machine(calibrated=False)
    m.set_calibration(
        CalibrationStatus(exists=True, validation_error_px=60.0, calibration_head_pose=(0, 0, 0))
    )
    status = m.step(make_frame())
    assert status.gate2_calibration is False


def test_gate2_fails_when_head_pose_drifted_too_far():
    status = machine().step(make_frame(head_pose_deg=(20.0, 0.0, 0.0)))
    assert status.gate2_calibration is False
    assert "pose" in status.reasons["gate2"]


def test_gate2_passes_at_head_pose_boundary():
    status = machine().step(make_frame(head_pose_deg=(15.0, 0.0, 0.0)))
    assert status.gate2_calibration is True


def test_gate2_fails_just_past_head_pose_boundary():
    status = machine().step(make_frame(head_pose_deg=(15.1, 0.0, 0.0)))
    assert status.gate2_calibration is False


# ---------------------------------------------------------------------------
# Gate 4: kill switch & limits
# ---------------------------------------------------------------------------


def test_gate4_passes_by_default():
    status = machine().step(make_frame())
    assert status.gate4_kill_and_limits is True


def test_gate4_fails_when_kill_switch_pressed():
    status = machine().step(make_frame(kill_switch_pressed=True))
    assert status.gate4_kill_and_limits is False
    assert "kill switch" in status.reasons["gate4"]


def test_gate4_first_frame_never_fails_on_speed_with_no_prior_point():
    status = machine().step(make_frame(gaze_point_px=(1900.0, 1000.0)))
    assert status.gate4_kill_and_limits is True


def test_gate4_fails_when_cursor_speed_exceeds_limit():
    m = machine()
    m.step(make_frame(timestamp_s=0.0, gaze_point_px=(0.0, 0.0)))
    # 3000px in 0.1s = 30,000px/s, far over the 2000px/s test limit.
    status = m.step(make_frame(timestamp_s=0.1, gaze_point_px=(3000.0, 0.0)))
    assert status.gate4_kill_and_limits is False
    assert "speed" in status.reasons["gate4"]


def test_gate4_passes_when_cursor_speed_within_limit():
    m = machine()
    m.step(make_frame(timestamp_s=0.0, gaze_point_px=(0.0, 0.0)))
    # 100px in 0.1s = 1000px/s, under the 2000px/s test limit.
    status = m.step(make_frame(timestamp_s=0.1, gaze_point_px=(100.0, 0.0)))
    assert status.gate4_kill_and_limits is True


def test_gate4_fails_inside_no_click_zone():
    status = machine().step(make_frame(gaze_point_px=(1850.0, 20.0)))
    assert status.gate4_kill_and_limits is False
    assert "no-click zone" in status.reasons["gate4"]


def test_gate4_passes_outside_no_click_zone():
    status = machine().step(make_frame(gaze_point_px=(500.0, 500.0)))
    assert status.gate4_kill_and_limits is True


def test_gate4_no_click_zone_check_handles_missing_gaze_point():
    status = machine().step(make_frame(gaze_point_px=None))
    assert status.gate4_kill_and_limits is True


def test_gate4_fails_once_click_rate_limit_reached():
    # max_clicks_per_window=2 in the test config.
    m = machine()
    t = 0.0
    # Dwell in, confirm, click #1.
    for _ in range(4):
        m.step(make_frame(timestamp_s=t, gaze_point_px=(500.0, 500.0)))
        t += 0.2
    m.step(make_frame(timestamp_s=t, gaze_point_px=(500.0, 500.0), confirm_keypress=True))
    t += 0.2
    # Release, move to a new spot, dwell again, click #2.
    m.step(make_frame(timestamp_s=t, gaze_point_px=(700.0, 500.0)))
    t += 0.2
    for _ in range(3):
        m.step(make_frame(timestamp_s=t, gaze_point_px=(700.0, 500.0)))
        t += 0.2
    m.step(make_frame(timestamp_s=t, gaze_point_px=(700.0, 500.0), confirm_keypress=True))
    t += 0.2
    # A third click attempt should now be blocked by the rate limit.
    m.step(make_frame(timestamp_s=t, gaze_point_px=(900.0, 500.0)))
    t += 0.2
    for _ in range(3):
        m.step(make_frame(timestamp_s=t, gaze_point_px=(900.0, 500.0)))
        t += 0.2
    status = m.step(make_frame(timestamp_s=t, gaze_point_px=(900.0, 500.0), confirm_keypress=True))
    assert status.gate4_kill_and_limits is False
    assert "click limit" in status.reasons["gate4"]
    assert status.click_fired is False


def test_gate4_click_rate_limit_forgets_clicks_outside_the_window():
    m = machine(make_config(click_window_s=1.0, dwell_time_s=0.1, max_clicks_per_window=1))
    m.step(make_frame(timestamp_s=0.0, gaze_point_px=(500.0, 500.0)))
    status = m.step(
        make_frame(timestamp_s=0.2, gaze_point_px=(500.0, 500.0), confirm_keypress=True)
    )
    assert status.click_fired is True

    # Well past the 1s window: the rate limit should no longer see that old click.
    m.step(make_frame(timestamp_s=5.0, gaze_point_px=(500.0, 500.0)))
    status = m.step(
        make_frame(timestamp_s=5.2, gaze_point_px=(500.0, 500.0), confirm_keypress=True)
    )
    assert status.gate4_kill_and_limits is True
    assert status.click_fired is True


# ---------------------------------------------------------------------------
# Movement: needs gates 1, 2, 4 (not 3)
# ---------------------------------------------------------------------------


def test_move_allowed_when_gates_1_2_4_pass():
    status = machine().step(make_frame())
    assert status.move_allowed is True
    assert status.cursor_target_px == (500.0, 500.0)


def test_move_blocked_when_gate1_fails():
    status = machine().step(make_frame(face_count=0))
    assert status.move_allowed is False
    assert status.cursor_target_px is None


def test_move_blocked_when_gate2_fails():
    status = machine(calibrated=False).step(make_frame())
    assert status.move_allowed is False
    assert status.cursor_target_px is None


def test_move_blocked_when_gate4_fails():
    status = machine().step(make_frame(kill_switch_pressed=True))
    assert status.move_allowed is False
    assert status.cursor_target_px is None


def test_move_allowed_even_when_gate3_has_not_been_reached_yet():
    # Movement doesn't need dwell — only clicking does.
    status = machine().step(make_frame())
    assert status.gate3_stability is False
    assert status.move_allowed is True


# ---------------------------------------------------------------------------
# Gate 3: dwell / stability
# ---------------------------------------------------------------------------


def test_gate3_false_immediately_on_entering_a_region():
    status = machine().step(make_frame(timestamp_s=0.0))
    assert status.gate3_stability is False


def test_gate3_true_after_dwell_time_elapses_in_place():
    m = machine()
    m.step(make_frame(timestamp_s=0.0, gaze_point_px=(500.0, 500.0)))
    m.step(make_frame(timestamp_s=0.3, gaze_point_px=(505.0, 500.0)))
    status = m.step(make_frame(timestamp_s=0.6, gaze_point_px=(500.0, 505.0)))
    assert status.gate3_stability is True


def test_gate3_resets_when_gaze_jumps_past_jitter_threshold():
    m = machine()
    m.step(make_frame(timestamp_s=0.0, gaze_point_px=(500.0, 500.0)))
    m.step(make_frame(timestamp_s=0.3, gaze_point_px=(500.0, 500.0)))
    # Jump 100px, well past the 20px jitter threshold: dwell restarts.
    status = m.step(make_frame(timestamp_s=0.4, gaze_point_px=(600.0, 500.0)))
    assert status.gate3_stability is False


def test_gate3_does_not_reset_the_clock_for_small_jitter():
    m = machine()
    m.step(make_frame(timestamp_s=0.0, gaze_point_px=(500.0, 500.0)))
    # Small 5px jitters, all inside the 20px threshold — dwell keeps counting from t=0.
    m.step(make_frame(timestamp_s=0.2, gaze_point_px=(503.0, 498.0)))
    m.step(make_frame(timestamp_s=0.4, gaze_point_px=(497.0, 502.0)))
    status = m.step(make_frame(timestamp_s=0.51, gaze_point_px=(500.0, 500.0)))
    assert status.gate3_stability is True


def test_gate3_never_true_while_move_is_not_allowed():
    m = machine()
    for t in (0.0, 0.3, 0.6, 0.9):
        status = m.step(make_frame(timestamp_s=t, face_count=0))
    assert status.gate3_stability is False


def test_gate3_resets_when_gaze_point_is_missing():
    m = machine()
    m.step(make_frame(timestamp_s=0.0, gaze_point_px=(500.0, 500.0)))
    m.step(make_frame(timestamp_s=0.3, gaze_point_px=None, both_irises_visible=False))
    status = m.step(make_frame(timestamp_s=0.6, gaze_point_px=(500.0, 500.0)))
    # Dwell had to restart after the dropout, so 0.3s after the restart isn't enough.
    assert status.gate3_stability is False


def test_gate3_restarts_cleanly_after_recovering_from_a_dropout():
    m = machine()
    m.step(make_frame(timestamp_s=0.0, gaze_point_px=(500.0, 500.0)))
    m.step(make_frame(timestamp_s=0.3, gaze_point_px=None, both_irises_visible=False))
    m.step(make_frame(timestamp_s=0.6, gaze_point_px=(500.0, 500.0)))
    status = m.step(make_frame(timestamp_s=1.2, gaze_point_px=(500.0, 500.0)))
    assert status.gate3_stability is True


# ---------------------------------------------------------------------------
# Click firing: needs all four gates + dwell + confirmation
# ---------------------------------------------------------------------------


def _dwell_in(m: GateStateMachine, point=(500.0, 500.0), start_t=0.0, dwell=0.5):
    m.step(make_frame(timestamp_s=start_t, gaze_point_px=point))
    return start_t + dwell + 0.01


def test_click_fires_with_keypress_confirmation_after_dwell():
    m = machine()
    t = _dwell_in(m)
    status = m.step(make_frame(timestamp_s=t, gaze_point_px=(500.0, 500.0), confirm_keypress=True))
    assert status.click_fired is True
    assert status.reasons["click"] == "fired"


def test_click_fires_with_blink_hold_confirmation_meeting_threshold():
    m = machine()
    t = _dwell_in(m)
    status = m.step(make_frame(timestamp_s=t, gaze_point_px=(500.0, 500.0), confirm_hold_s=0.4))
    assert status.click_fired is True


def test_click_does_not_fire_when_blink_hold_is_too_short():
    m = machine()
    t = _dwell_in(m)
    status = m.step(make_frame(timestamp_s=t, gaze_point_px=(500.0, 500.0), confirm_hold_s=0.39))
    assert status.click_fired is False
    assert status.reasons["click"] != "fired"


def test_click_does_not_fire_before_dwell_is_satisfied():
    m = machine()
    m.step(make_frame(timestamp_s=0.0, gaze_point_px=(500.0, 500.0)))
    status = m.step(
        make_frame(timestamp_s=0.1, gaze_point_px=(500.0, 500.0), confirm_keypress=True)
    )
    assert status.click_fired is False


def test_click_does_not_fire_when_gate1_fails():
    m = machine()
    t = _dwell_in(m)
    status = m.step(make_frame(timestamp_s=t, face_count=0, confirm_keypress=True))
    assert status.click_fired is False


def test_click_does_not_fire_when_gate2_fails():
    m = machine(calibrated=False)
    t = _dwell_in(m)
    status = m.step(make_frame(timestamp_s=t, gaze_point_px=(500.0, 500.0), confirm_keypress=True))
    assert status.click_fired is False


def test_click_does_not_fire_when_kill_switch_pressed():
    m = machine()
    t = _dwell_in(m)
    status = m.step(
        make_frame(
            timestamp_s=t,
            gaze_point_px=(500.0, 500.0),
            kill_switch_pressed=True,
            confirm_keypress=True,
        )
    )
    assert status.click_fired is False


def test_click_does_not_fire_inside_no_click_zone():
    m = machine()
    t = _dwell_in(m, point=(1850.0, 20.0))
    status = m.step(make_frame(timestamp_s=t, gaze_point_px=(1850.0, 20.0), confirm_keypress=True))
    assert status.click_fired is False
    assert "no-click zone" in status.reasons["gate4"]


def test_click_requires_confirmation_release_before_firing_again():
    m = machine()
    t = _dwell_in(m)
    first = m.step(make_frame(timestamp_s=t, gaze_point_px=(500.0, 500.0), confirm_keypress=True))
    assert first.click_fired is True
    # Holding the same confirmation gesture on the very next frame must not re-fire.
    second = m.step(
        make_frame(timestamp_s=t + 0.05, gaze_point_px=(500.0, 500.0), confirm_hold_s=0.45)
    )
    assert second.click_fired is False
    assert "release" in second.reasons["click"]


def test_click_can_fire_again_after_confirmation_is_released_and_reapplied():
    m = machine()
    t = _dwell_in(m)
    first = m.step(make_frame(timestamp_s=t, gaze_point_px=(500.0, 500.0), confirm_keypress=True))
    assert first.click_fired is True
    # Release the confirmation gesture.
    m.step(make_frame(timestamp_s=t + 0.1, gaze_point_px=(500.0, 500.0)))
    # Re-confirm.
    second = m.step(
        make_frame(timestamp_s=t + 0.2, gaze_point_px=(500.0, 500.0), confirm_keypress=True)
    )
    assert second.click_fired is True


def test_awaiting_confirmation_flag_clears_once_gesture_ends_without_a_click():
    # A confirmation gesture held while dwell hasn't been reached yet never fires,
    # but releasing it should still clear internal state cleanly (no crash / stuck flag).
    m = machine()
    m.step(make_frame(timestamp_s=0.0, gaze_point_px=(500.0, 500.0), confirm_keypress=True))
    status = m.step(make_frame(timestamp_s=0.1, gaze_point_px=(500.0, 500.0)))
    assert status.reasons["click"] == "no confirmation gesture"


def test_reasons_dict_has_all_four_gates_and_click():
    status = machine().step(make_frame())
    assert set(status.reasons) == {"gate1", "gate2", "gate3", "gate4", "click"}


# ---------------------------------------------------------------------------
# End-to-end integration
# ---------------------------------------------------------------------------


def test_full_sequence_look_dwell_confirm_click_exactly_once():
    m = machine()
    statuses = []
    t = 0.0
    # Look at a spot and hold for longer than the dwell time.
    for _ in range(6):
        statuses.append(m.step(make_frame(timestamp_s=t, gaze_point_px=(400.0, 300.0))))
        t += 0.1
    assert not any(s.click_fired for s in statuses)
    assert statuses[-1].gate3_stability is True

    # Confirm with a keypress: exactly one click.
    click_status = m.step(
        make_frame(timestamp_s=t, gaze_point_px=(400.0, 300.0), confirm_keypress=True)
    )
    assert click_status.click_fired is True

    # Keep holding — must not double-fire.
    t += 0.1
    again = m.step(make_frame(timestamp_s=t, gaze_point_px=(400.0, 300.0), confirm_keypress=True))
    assert again.click_fired is False


def test_kill_switch_freezes_mid_session_and_recovers_after_release():
    m = machine()
    t = 0.0
    for _ in range(6):
        m.step(make_frame(timestamp_s=t, gaze_point_px=(400.0, 300.0)))
        t += 0.1
    killed = m.step(
        make_frame(timestamp_s=t, gaze_point_px=(400.0, 300.0), kill_switch_pressed=True)
    )
    assert killed.move_allowed is False
    assert killed.gate3_stability is False  # dwell must not survive a kill event

    t += 0.1
    recovered = m.step(make_frame(timestamp_s=t, gaze_point_px=(400.0, 300.0)))
    assert recovered.move_allowed is True
    assert recovered.gate3_stability is False  # has to dwell again from scratch
