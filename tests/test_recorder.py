"""``eyeos record`` (eyeos/recorder.py), tested headless against a fake ``tkinter``
(tests/fake_tk.py) and synthetic frames. No camera, display or keyboard."""

from __future__ import annotations

import ast
import csv
import errno
import math
import shlex
import sys
from pathlib import Path

import pytest

from tests.fake_tk import TclError, make_fake_tk

W, H = 1920, 1080
DT = 1 / 32  # exact in binary, so frame timestamps add up without rounding drift


@pytest.fixture
def fake_tk(monkeypatch):
    module = make_fake_tk()
    monkeypatch.setitem(sys.modules, "tkinter", module)
    return module


def _tracking_roots(module):
    """Make ``module.Tk`` remember every window it creates."""
    roots = []
    base = module.Tk

    class TrackedTk(base):  # type: ignore[misc, valid-type]
        def __init__(self):
            super().__init__()
            roots.append(self)

    module.Tk = TrackedTk
    return roots


# ---------------------------------------------------------------------------
# The fake tkinter's event loop: after() callbacks, polled user actions, bell()
# ---------------------------------------------------------------------------


def test_fake_tk_mainloop_runs_after_callbacks_in_due_order_until_destroyed(fake_tk):
    root = fake_tk.Tk()
    ran = []

    def tick(n):
        ran.append(n)
        if n < 3:
            root.after(10, tick, n + 1)
        else:
            root.destroy()

    root.after(50, ran.append, "late")
    root.after(0, tick, 0)
    root.mainloop()

    # tick 0..3 are due at 0, 10, 20, 30 ms; the window is gone before 50 ms.
    assert ran == [0, 1, 2, 3]
    assert root.destroyed


def test_fake_tk_after_cancel_and_poll_hook(fake_tk):
    root = fake_tk.Tk()
    ran = []
    cancelled = root.after(5, ran.append, "cancelled")
    root.after(10, ran.append, "kept")
    root.after_cancel(cancelled)
    polls = []
    root.on_poll = lambda r: polls.append(len(ran))

    root.mainloop()  # returns once no callback is left, like a test-sized event loop

    assert ran == ["kept"]
    assert polls == [0]  # the user gets a turn before each callback


def test_fake_tk_reports_an_after_callback_error_like_real_tk(fake_tk, capsys):
    root = fake_tk.Tk()

    def broken():
        raise ValueError("boom")

    root.after(0, broken)
    root.mainloop()

    assert isinstance(root.callback_errors[0], ValueError)
    assert "Exception in Tkinter callback" in capsys.readouterr().err


def test_fake_tk_bell_counts_and_is_strict_after_destroy(fake_tk):
    root = fake_tk.Tk()
    root.bell()
    assert root.bells == 1
    root.destroy()
    with pytest.raises(TclError):
        root.bell()
    with pytest.raises(TclError):
        root.after(0, lambda: None)


# ---------------------------------------------------------------------------
# The stage script
# ---------------------------------------------------------------------------


def test_steps_follow_the_stage_order():
    from eyeos.calibration import target_points
    from eyeos.recorder import AWAY_LEAD_IN_S, AWAY_S, build_steps

    steps = build_steps(W, H)

    assert [s.stage for s in steps] == (
        ["calibration"] * 9 + ["validation"] * 12 + ["pause", "looking_away"] * 3
    ) + ["pause", "reading"]
    assert [s.target for s in steps[:9]] == target_points(W, H)
    assert {s.segment for s in steps[:9]} == {"calibration"}
    assert all(s.duration_s == 2.0 and s.settle_s == 0.6 for s in steps[:21])

    validation = [s.target for s in steps[9:21]]
    assert {s.segment for s in steps[9:21]} == {"on_target"}
    assert len(set(validation)) == 12
    assert not set(validation) & set(target_points(W, H))  # off the calibration grid
    assert all(0 < x < W and 0 < y < H for x, y in validation)

    away = [s for s in steps if s.stage == "looking_away"]
    assert [s.away_mode for s in away] == ["off_screen", "eyes_closed", "no_face"]
    for s in away:
        assert s.segment == "looking_away" and s.target is None and s.bell_at_end
        assert s.settle_s == AWAY_LEAD_IN_S and s.duration_s == AWAY_LEAD_IN_S + AWAY_S
    pauses = [s for s in steps if s.stage == "pause"]
    assert all(s.duration_s is None and s.segment == "settle" for s in pauses)

    reading = steps[-1]
    assert (reading.segment, reading.duration_s, reading.settle_s) == ("just_reading", 60.0, 0.6)
    assert reading.target is None


def test_no_reading_ends_after_the_last_look_away_prompt():
    from eyeos.recorder import build_steps

    steps = build_steps(W, H, reading=False, dot_s=1.5, settle_s=0.3)

    assert steps[-1].stage == "looking_away"
    assert all(s.stage != "reading" for s in steps)
    assert all(s.duration_s == 1.5 and s.settle_s == 0.3 for s in steps[:21])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"dot_s": 0.6, "settle_s": 0.6}, "settle"),
        ({"dot_s": 2.0, "settle_s": -0.1}, "settle"),
        ({"dot_s": float("nan")}, "dot"),
        ({"reading_s": 0.5, "settle_s": 0.6}, "reading"),
        ({"reading_s": float("inf")}, "reading"),
    ],
)
def test_build_steps_rejects_timings_that_leave_nothing_to_measure(kwargs, match):
    from eyeos.recorder import build_steps

    with pytest.raises(ValueError, match=match):
        build_steps(W, H, **kwargs)


def test_a_short_reading_is_fine_when_reading_is_skipped():
    from eyeos.recorder import build_steps

    build_steps(W, H, reading=False, reading_s=0.0)


# ---------------------------------------------------------------------------
# A scripted user in front of the recorder
# ---------------------------------------------------------------------------


def _closed_eyes(landmarks):
    import dataclasses

    from eyeos.landmarks import Point3D

    return dataclasses.replace(
        landmarks,
        left_eye_top=Point3D(0.38, 0.40),
        left_eye_bottom=Point3D(0.38, 0.40),
        right_eye_top=Point3D(0.62, 0.40),
        right_eye_bottom=Point3D(0.62, 0.40),
    )


class FakeUser:
    """A noise-free user who does what the screen asks.

    Each frame is made when the recorder asks for it, from what the fake canvas shows
    at that moment: the user looks at the dot that is drawn, does what a look-away
    prompt asks, and looks at the screen centre otherwise. ``lag_s`` keeps the eyes on
    the previous dot for that long after a new one appears, like a real saccade.
    """

    def __init__(self, dt=DT, lag_s=0.0, max_frames=20_000, fail_at=None, error=None):
        self.dt = dt
        self.lag_s = lag_s
        self.max_frames = max_frames
        self.fail_at = fail_at
        self.error = error
        self.recorder = None
        self.shown: dict[float, tuple[float, float] | None] = {}  # timestamp -> dot drawn
        self._dot = None
        self._prev_dot = None
        self._dot_since = 0.0

    def dot_on_screen(self):
        ovals = self.recorder.canvas.of_kind("oval")
        if not ovals:
            return None
        x0, y0, x1, y1 = ovals[0]["coords"]
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    def _look(self, t):
        from eval.generate_synthetic import make_landmarks, offset_for_target

        dot = self.dot_on_screen()
        self.shown[t] = dot
        if dot != self._dot:
            self._prev_dot, self._dot, self._dot_since = self._dot, dot, t
        if dot is not None:
            looking_at = dot
            if self._prev_dot is not None and t - self._dot_since < self.lag_s:
                looking_at = self._prev_dot
            return make_landmarks(*offset_for_target(*looking_at))
        step = self.recorder.current_step
        mode = step.away_mode if step is not None else ""
        if mode == "off_screen":
            return make_landmarks(1.5, 0.0, nose_x=0.6)
        if mode == "eyes_closed":
            return _closed_eyes(make_landmarks(0.0, 0.0))
        if mode == "no_face":
            return None
        return make_landmarks(0.0, 0.0)

    def frames(self):
        from eyeos.input_sources import Frame

        for i in range(self.max_frames):
            if self.fail_at is not None and i == self.fail_at:
                raise self.error
            t = i * self.dt
            yield Frame(timestamp_s=t, landmarks=self._look(t))


def _recorder(tmp_path, user, frames=None, out_name="session.csv", **kwargs):
    from eyeos.recorder import SessionRecorder

    stream = frames if frames is not None else user.frames()
    recorder = SessionRecorder(W, H, stream, tmp_path / out_name, **kwargs)
    user.recorder = recorder
    return recorder


def _press_space_when_asked(root):
    root.fire("<space>")  # the recorder ignores space while a timed step runs


def _rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _blocks(rows):
    """Consecutive rows that share (segment, target, away_mode), as (key, rows)."""
    blocks: list[tuple[tuple[str, str, str, str], list[dict[str, str]]]] = []
    for row in rows:
        key = (row["segment"], row["target_x"], row["target_y"], row["away_mode"])
        if blocks and blocks[-1][0] == key:
            blocks[-1][1].append(row)
        else:
            blocks.append((key, [row]))
    return blocks


# ---------------------------------------------------------------------------
# A full recording
# ---------------------------------------------------------------------------


def test_a_session_records_every_stage_in_order(fake_tk, tmp_path):
    from eval.generate_synthetic import FIELDNAMES
    from eyeos.calibration import target_points
    from eyeos.recorder import build_steps

    user = FakeUser()
    recorder = _recorder(tmp_path, user, reading_s=3.0)
    read_on_the_intro: list[float] = []

    def poll(root):
        if recorder.phase == "intro":
            if len(user.shown) < 10:
                return  # the user reads the intro while the camera check runs
            read_on_the_intro.extend(sorted(user.shown))
        root.fire("<space>")

    recorder.root.on_poll = poll

    assert recorder.run() is True

    out = tmp_path / "session.csv"
    with out.open(newline="") as f:
        assert f.readline().rstrip("\r\n").split(",") == FIELDNAMES
    rows = _rows(out)
    assert recorder.frame_count == len(rows)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["session.csv"]  # no temp file left

    # The intro's frames feed the camera check but are not recorded: the file starts at
    # the first frame read after space, on the first calibration dot.
    assert len(read_on_the_intro) == 10
    first = rows[0]
    assert float(first["timestamp_s"]) == read_on_the_intro[-1] + DT
    assert first["segment"] == "settle"
    assert (float(first["target_x"]), float(first["target_y"])) == target_points(W, H)[0]
    assert all(float(r["timestamp_s"]) > read_on_the_intro[-1] for r in rows)

    # Measured rows come in the script's order, one block per step.
    measured = [key for key, _ in _blocks(rows) if key[0] != "settle"]
    expected = []
    for step in build_steps(W, H, reading_s=3.0):
        if step.stage == "pause":
            continue
        target = ("", "") if step.target is None else tuple(repr(v) for v in step.target)
        expected.append((step.segment, *target, step.away_mode))
    assert measured == expected

    # Every row's target is the dot that was on screen when its frame was taken.
    for row in rows:
        shown = user.shown[float(row["timestamp_s"])]
        if row["target_x"]:
            assert (float(row["target_x"]), float(row["target_y"])) == shown
        else:
            assert shown is None

    # Each dot: settle frames first (its first 0.6 s), then the measured ones.
    for key, block in _blocks(rows):
        if key[1] and key[0] != "settle":
            assert len(block) == 45  # elapsed 0.625 .. 2.0 s at 32 fps
    by_dot: dict[tuple[str, str], list[dict[str, str]]] = {}
    for r in rows:
        if r["target_x"]:
            by_dot.setdefault((r["target_x"], r["target_y"]), []).append(r)
    dots = list(by_dot.values())
    assert len(dots) == 21
    for i, dot in enumerate(dots):
        segments = [r["segment"] for r in dot]
        n_settle = segments.count("settle")
        assert segments == ["settle"] * n_settle + [segments[-1]] * (len(segments) - n_settle)
        # Only the first dot has a frame at elapsed 0. A later dot's clock starts at the
        # timestamp of the frame that ended the dot before it, so its first frame is
        # already one frame in.
        assert n_settle == (20 if i == 0 else 19), f"dot {i + 1}"
    # So every dot lasts exactly dot_s, counted from where the one before it ended.
    ends = [float(dot[-1]["timestamp_s"]) for dot in dots]
    starts = [float(dots[0][0]["timestamp_s"]), *ends[:-1]]
    assert [end - start for start, end in zip(starts, ends, strict=True)] == [2.0] * 21

    # Reading: 3 s with the first 0.6 s left out, and no target.
    reading = [r for r in rows if r["segment"] == "just_reading"]
    assert len(reading) == round((3.0 - 0.625) / DT) + 1
    assert all(r["target_x"] == "" for r in reading)
    assert recorder.root.bells == 3  # one beep at the end of each look-away prompt
    assert recorder.root.destroyed


def test_a_recorded_session_evaluates_to_near_zero_error(fake_tk, tmp_path):
    from eval.run_eval import compute_metrics

    recorder = _recorder(tmp_path, FakeUser(), reading_s=5.0)
    recorder.root.on_poll = _press_space_when_asked
    assert recorder.run() is True

    metrics = compute_metrics(tmp_path / "session.csv")

    assert metrics["calibration_validation_error_px"] < 1.0
    assert metrics["on_target_frame_count"] == 12 * 45
    assert metrics["mean_pixel_error_raw_px"] < 1.0
    assert metrics["mean_pixel_error_filtered_px"] < 1.0
    assert set(metrics["looking_away_by_mode"]) == {"off_screen", "eyes_closed", "no_face"}
    # 2.0 .. 6.0 s into each prompt, both ends included: 4 s at 32 fps, plus one frame.
    assert metrics["looking_away_by_mode"]["no_face"] == {"frames": 129, "false_accepts": 0}
    assert metrics["looking_away_frame_count"] == 129  # no_face is a Gate-1 mode
    assert metrics["just_reading_frame_count"] == round((5.0 - 0.625) / DT) + 1
    assert metrics["just_reading_false_click_count"] == 0


def test_settle_frames_are_left_out_of_the_eval(fake_tk, tmp_path):
    from eval.generate_synthetic import FIELDNAMES
    from eval.run_eval import compute_metrics

    # The eyes reach each new dot 0.3 s late: those frames still look at the last one.
    recorder = _recorder(tmp_path, FakeUser(lag_s=0.3), reading=False)
    recorder.root.on_poll = _press_space_when_asked
    assert recorder.run() is True
    session = tmp_path / "session.csv"

    metrics = compute_metrics(session)
    assert metrics["calibration_validation_error_px"] < 1.0
    assert metrics["mean_pixel_error_raw_px"] < 1.0

    # The check has teeth: label the settle frames as measured and the fit breaks.
    rows = _rows(session)
    for row in rows:
        if row["segment"] == "settle" and row["target_x"]:
            in_calibration = float(row["timestamp_s"]) < 9 * 2.0
            row["segment"] = "calibration" if in_calibration else "on_target"
    relabelled = tmp_path / "relabelled.csv"
    with relabelled.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    assert compute_metrics(relabelled)["calibration_validation_error_px"] > 50.0


def test_run_eval_ignores_a_segment_it_does_not_know(tmp_path):
    from eval.generate_synthetic import generate_rows, make_landmarks, write_csv
    from eval.run_eval import compute_metrics
    from eyeos.synthetic_format import session_row

    rows = generate_rows()
    plain = tmp_path / "plain.csv"
    write_csv(plain, rows)
    # A "settle" frame inside the first calibration dot, looking far away from it.
    t = (float(rows[2]["timestamp_s"]) + float(rows[3]["timestamp_s"])) / 2
    far_off = session_row(t, make_landmarks(0.9, -0.9), (192.0, 108.0), "settle")
    with_settle = tmp_path / "with_settle.csv"
    write_csv(with_settle, [*rows[:3], far_off, *rows[3:]])

    before, after = compute_metrics(plain), compute_metrics(with_settle)

    assert after["frame_count"] == before["frame_count"] + 1
    for key in ("calibration_validation_error_px", "on_target_frame_count"):
        assert after[key] == before[key], key


def test_the_bell_rings_at_the_end_of_each_look_away_prompt_and_only_then(fake_tk, tmp_path):
    recorder = _recorder(tmp_path, FakeUser(), reading_s=3.0)
    recorder.root.on_poll = _press_space_when_asked
    rang = []
    ring = recorder.root.bell

    def bell():
        step = recorder.current_step
        rang.append((step.stage, step.away_mode, recorder.frame_count))
        ring()

    recorder.root.bell = bell

    assert recorder.run() is True

    assert [(stage, mode) for stage, mode, _ in rang] == [
        ("looking_away", "off_screen"),
        ("looking_away", "eyes_closed"),
        ("looking_away", "no_face"),
    ]
    rows = _rows(tmp_path / "session.csv")
    for _stage, mode, n_rows in rang:  # it rings on the prompt's last frame
        assert rows[n_rows - 1]["away_mode"] == mode
        assert rows[n_rows]["away_mode"] != mode


def test_a_frame_with_a_bad_timestamp_is_kept_but_never_measured(fake_tk, tmp_path):
    from eyeos.input_sources import Frame

    user = FakeUser()

    def frames():
        for i, frame in enumerate(user.frames()):
            yield Frame(math.nan, frame.landmarks) if i == 40 else frame

    recorder = _recorder(tmp_path, user, frames=frames(), reading=False)
    recorder.root.on_poll = _press_space_when_asked
    assert recorder.run() is True

    rows = _rows(tmp_path / "session.csv")
    (bad,) = [r for r in rows if r["timestamp_s"] == "nan"]
    assert bad["segment"] == "settle"  # 1.25 s into the first dot, but unmeasurable
    assert sum(r["segment"] == "calibration" for r in rows) == 9 * 45 - 1


# ---------------------------------------------------------------------------
# Cancelling, and the done screen
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("how", ["escape", "close_window"])
@pytest.mark.parametrize("stage", ["intro", "calibration", "validation", "looking_away", "reading"])
def test_cancelling_at_any_stage_leaves_no_file(fake_tk, tmp_path, stage, how):
    user = FakeUser()
    recorder = _recorder(tmp_path, user, reading_s=3.0)
    polls_in_stage = []

    def poll(root):
        step = recorder.current_step
        current = recorder.phase if step is None else step.stage
        if current == stage:
            polls_in_stage.append(1)
            if stage == "intro" or len(polls_in_stage) == 10:
                root.fire("<Escape>") if how == "escape" else root.close_window()
                return
        root.fire("<space>")

    recorder.root.on_poll = poll

    assert recorder.run() is False
    assert list(tmp_path.iterdir()) == []
    assert recorder.root.destroyed
    assert recorder.root.callback_errors == []


def test_esc_on_the_done_screen_keeps_the_file(fake_tk, tmp_path):
    recorder = _recorder(tmp_path, FakeUser(), reading=False)
    recorder.root.on_poll = lambda root: root.fire(
        "<Escape>" if recorder.phase == "done" else "<space>"
    )

    assert recorder.run() is True
    assert (tmp_path / "session.csv").exists()


def test_the_done_screen_closes_by_itself(fake_tk, tmp_path):
    recorder = _recorder(tmp_path, FakeUser(), reading=False)
    done_at = []

    def poll(root):
        if recorder.phase == "done":
            done_at.append(root.now_ms)
        else:
            root.fire("<space>")

    recorder.root.on_poll = poll

    assert recorder.run() is True
    assert recorder.root.destroyed
    assert recorder.root.now_ms - done_at[0] >= 2000


# ---------------------------------------------------------------------------
# Screens it refuses, and camera failures
# ---------------------------------------------------------------------------


def test_a_config_screen_that_does_not_match_is_refused(monkeypatch, tmp_path):
    from eyeos.recorder import SessionRecorder

    # Windows at 125% scaling reports a 1920x1080 panel as 1536x864.
    module = make_fake_tk(screen_size=(1536, 864))
    roots = _tracking_roots(module)
    monkeypatch.setitem(sys.modules, "tkinter", module)

    with pytest.raises(ValueError, match="1536x864"):
        SessionRecorder(1920, 1080, iter(()), tmp_path / "session.csv")
    assert roots[0].destroyed

    recorder = SessionRecorder(1536, 864, iter(()), tmp_path / "session.csv")
    assert recorder.root.attrs["-fullscreen"] is True
    # Also sized and placed by hand, for when no window manager applies -fullscreen,
    # and focused, so space and Esc reach it.
    assert recorder.root.geometry_spec == "1536x864+0+0"
    assert recorder.root.focus_forced


@pytest.mark.parametrize(
    "window",
    [(1920, 0, 1920, 1080), (0, 0, 1920, 1040), (0, 0, 378, 265)],
    ids=["on-a-second-screen", "taskbar-left-visible", "never-went-fullscreen"],
)
def test_recording_only_starts_in_a_window_that_covers_the_screen(monkeypatch, tmp_path, window):
    # The CSV labels each frame with the dot's screen position, which is only where the
    # dot is if the canvas starts at the screen's top-left corner and covers it.
    module = make_fake_tk(window_geometry=window)
    monkeypatch.setitem(sys.modules, "tkinter", module)
    recorder = _recorder(tmp_path, FakeUser())
    recorder.root.on_poll = _press_space_when_asked

    with pytest.raises(RuntimeError, match="does not cover the screen") as e:
        recorder.run()

    x, y, w, h = window
    assert f"{w}x{h} at +{x}+{y}" in str(e.value)
    assert recorder.root.destroyed
    assert list(tmp_path.iterdir()) == []


def test_fake_tk_widgets_report_the_window_geometry(monkeypatch):
    module = make_fake_tk(screen_size=(1536, 864))
    root = module.Tk()
    canvas = module.Canvas(root)
    assert (canvas.winfo_rootx(), canvas.winfo_rooty()) == (0, 0)
    assert (canvas.winfo_width(), canvas.winfo_height()) == (1536, 864)

    small = make_fake_tk(window_geometry=(10, 20, 300, 200)).Tk()
    assert (small.winfo_rootx(), small.winfo_rooty()) == (10, 20)
    assert (small.winfo_width(), small.winfo_height()) == (300, 200)


@pytest.mark.parametrize(
    "tkinter_module",
    [None, make_fake_tk(fail_with=TclError("no display name and no $DISPLAY"))],
    ids=["tkinter-missing", "no-display"],
)
def test_clear_error_without_a_display(monkeypatch, tmp_path, tkinter_module):
    from eyeos.recorder import SessionRecorder

    monkeypatch.setitem(sys.modules, "tkinter", tkinter_module)
    with pytest.raises(RuntimeError, match="display"):
        SessionRecorder(W, H, iter(()), tmp_path / "session.csv")


def test_a_stream_that_ends_mid_run_raises_a_clear_error(fake_tk, tmp_path):
    # 21 s of frames: calibration takes 18 s, so it ends during the 2nd validation dot.
    recorder = _recorder(tmp_path, FakeUser(max_frames=int(21 / DT)))
    recorder.root.on_poll = _press_space_when_asked

    with pytest.raises(RuntimeError, match="camera stopped during recording") as e:
        recorder.run()

    assert "validation dot 2 of 12" in str(e.value)
    assert recorder.root.destroyed
    assert recorder.root.callback_errors == []  # raised to the caller, not lost in Tk
    assert list(tmp_path.iterdir()) == []


def test_a_stream_that_ends_before_space_is_pressed_says_so(fake_tk, tmp_path):
    recorder = _recorder(tmp_path, FakeUser(max_frames=5))

    with pytest.raises(RuntimeError, match="before the recording started"):
        recorder.run()


@pytest.mark.parametrize("error", [OSError("camera unplugged"), KeyboardInterrupt()])
def test_a_camera_error_mid_run_is_raised_once_the_screen_has_closed(fake_tk, tmp_path, error):
    recorder = _recorder(tmp_path, FakeUser(fail_at=200, error=error))
    recorder.root.on_poll = _press_space_when_asked

    with pytest.raises(type(error)) as raised:
        recorder.run()

    assert raised.value is error
    assert recorder.root.destroyed
    assert recorder.root.callback_errors == []
    assert list(tmp_path.iterdir()) == []


def test_a_write_that_fails_part_way_leaves_no_partial_file(fake_tk, tmp_path, monkeypatch):
    # The disk fills up after the header and the first row are written.
    on_disk_mid_write = []

    def disk_full(writer, rows):
        writer.writerow(rows[0])
        on_disk_mid_write.extend(p.name for p in tmp_path.iterdir())
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(csv.DictWriter, "writerows", disk_full)
    recorder = _recorder(tmp_path, FakeUser(), reading=False)
    recorder.root.on_poll = _press_space_when_asked

    with pytest.raises(OSError, match="No space left"):
        recorder.run()

    assert on_disk_mid_write  # the failure came after a file was started
    assert list(tmp_path.iterdir()) == []
    assert recorder.root.destroyed


def test_an_out_directory_that_cannot_be_made_is_an_error(fake_tk, tmp_path):
    (tmp_path / "not-a-dir").write_text("")
    user = FakeUser()
    recorder = _recorder(tmp_path, user, out_name="not-a-dir/session.csv", reading=False)
    recorder.root.on_poll = _press_space_when_asked

    with pytest.raises(OSError):
        recorder.run()

    assert [p.name for p in tmp_path.iterdir()] == ["not-a-dir"]
    assert recorder.root.destroyed


def test_the_recorder_listens_only_to_its_own_window_and_never_drives_the_cursor(fake_tk, tmp_path):
    import eyeos.recorder as recorder_module

    recorder = _recorder(tmp_path, FakeUser())
    assert set(recorder.root.bindings) == {"<space>", "<Escape>"}

    tree = ast.parse(Path(recorder_module.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = ("eyeos.output", "eyeos.hotkeys", "eyeos.pipeline", "pynput")
    assert not [m for m in imported if m.startswith(forbidden)]


# ---------------------------------------------------------------------------
# eyeos record (the CLI)
# ---------------------------------------------------------------------------


def _record_config(tmp_path, n_frames=1500, backend="dry_run"):
    """A synthetic input of a user looking at the screen centre, n_frames long."""
    from eval.generate_synthetic import make_landmarks
    from eyeos.synthetic_format import CSV_FIELDNAMES, landmarks_to_row

    frames_csv = tmp_path / "camera.csv"
    with frames_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for i in range(n_frames):
            writer.writerow(landmarks_to_row(i * DT, make_landmarks(0.0, 0.0)))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"input:\n  source: synthetic\n  synthetic_path: {frames_csv}\n"
        f"output:\n  backend: {backend}\n"
    )
    return config_path


@pytest.fixture
def scripted_user(monkeypatch, fake_tk):
    """Install a SessionRecorder in the CLI whose window ``action(recorder, root)``
    acts on before each tick."""
    import eyeos.cli as cli
    from eyeos.recorder import SessionRecorder

    made = []

    def install(action):
        class Scripted(SessionRecorder):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.root.on_poll = lambda root: action(self, root)
                made.append(self)

        monkeypatch.setattr(cli, "SessionRecorder", Scripted)
        return made

    return install


FAST = ["--dot-s", "0.5", "--settle-s", "0.1", "--reading-s", "1"]


def test_cli_record_writes_the_session_and_prints_the_next_command(tmp_path, capsys, scripted_user):
    from eval.generate_synthetic import FIELDNAMES
    from eval.run_eval import main as run_eval_main
    from eyeos.cli import console_main

    scripted_user(lambda rec, root: root.fire("<space>"))
    config = _record_config(tmp_path)
    out = tmp_path / "recordings" / "s1.csv"  # the directory doesn't exist yet

    assert console_main(["record", "--config", str(config), "--out", str(out), *FAST]) == 0

    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("recorded ") and str(out.resolve()) in lines[0]
    assert "landmarks only" in lines[0]
    with out.open(newline="") as f:
        assert f.readline().rstrip("\r\n").split(",") == FIELDNAMES

    # The printed command is the one to run next, and it works as printed.
    command = shlex.split(lines[-1])
    assert command[:3] == ["python", "-m", "eval.run_eval"]
    assert command[command.index("--config") + 1] == str(config.resolve())
    assert run_eval_main(command[3:]) == 0
    assert Path(command[command.index("--out") + 1]).exists()


def test_cli_record_returns_1_when_cancelled(tmp_path, capsys, scripted_user):
    from eyeos.cli import console_main

    scripted_user(lambda rec, root: root.fire("<Escape>"))
    out = tmp_path / "s1.csv"

    code = console_main(["record", "--config", str(_record_config(tmp_path)), "--out", str(out)])

    assert code == 1
    assert "cancelled" in capsys.readouterr().err
    assert not out.exists()


def test_cli_record_never_overwrites_a_session(tmp_path, capsys, scripted_user):
    from eyeos.cli import console_main

    made = scripted_user(lambda rec, root: root.fire("<space>"))
    out = tmp_path / "s1.csv"
    out.write_text("an earlier session\n")

    code = console_main(["record", "--config", str(_record_config(tmp_path)), "--out", str(out)])

    assert code == 2
    assert "already exists" in capsys.readouterr().err
    assert out.read_text() == "an earlier session\n"
    assert made == []  # refused before any window opened


def test_cli_record_fails_before_the_screen_when_the_camera_cannot_open(
    tmp_path, capsys, monkeypatch
):
    from eyeos.cli import console_main
    from tests.fake_capture import FakeCv2, FakeMediapipe, install_fakes

    install_fakes(monkeypatch, cv2=FakeCv2(opened=False), mediapipe=FakeMediapipe())
    # Opening the screen first would raise this, which console_main doesn't catch.
    monkeypatch.setitem(sys.modules, "tkinter", make_fake_tk(fail_with=AssertionError("screen")))
    config = tmp_path / "config.yaml"
    config.write_text("input:\n  source: webcam\n")
    out = tmp_path / "s1.csv"

    assert console_main(["record", "--config", str(config), "--out", str(out)]) == 2

    err = capsys.readouterr().err
    assert err.splitlines()[-1].startswith("eyeos: error: could not open webcam device 0")
    assert "not 'webcam'" not in err
    assert not out.exists()


def test_cli_record_reports_a_stream_that_ends_mid_run(tmp_path, capsys, scripted_user):
    from eyeos.cli import console_main

    made = scripted_user(lambda rec, root: root.fire("<space>"))
    config = _record_config(tmp_path, n_frames=50)
    out = tmp_path / "s1.csv"

    assert console_main(["record", "--config", str(config), "--out", str(out)]) == 2

    assert "camera stopped during recording" in capsys.readouterr().err
    assert made[0].root.destroyed
    assert not out.exists()


def test_cli_record_checks_its_timings_before_opening_the_camera(tmp_path, capsys, monkeypatch):
    from eyeos.cli import console_main
    from tests.fake_capture import FakeCv2, FakeMediapipe, install_fakes

    fake_cv2 = FakeCv2(opened=False)
    install_fakes(monkeypatch, cv2=fake_cv2, mediapipe=FakeMediapipe())
    config = tmp_path / "config.yaml"
    config.write_text("input:\n  source: webcam\n")

    argv = ["record", "--config", str(config), "--out", str(tmp_path / "s.csv")]
    assert console_main([*argv, "--dot-s", "0.5", "--settle-s", "0.5"]) == 2

    assert "settle" in capsys.readouterr().err
    assert fake_cv2.captures == []


def test_cli_record_needs_a_config_and_an_out_path(tmp_path, capsys):
    from eyeos.cli import console_main

    with pytest.raises(SystemExit) as e:
        console_main(["record", "--out", str(tmp_path / "s.csv")])
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        console_main(["record", "--config", str(_record_config(tmp_path))])


def test_cli_record_never_builds_an_output_or_a_key_listener(tmp_path, monkeypatch, scripted_user):
    import eyeos.cli as cli
    from eyeos.cli import console_main

    def refuse(_config):
        raise AssertionError("eyeos record must not move the cursor or listen to keys")

    monkeypatch.setattr(cli, "_build_output", refuse)
    monkeypatch.setattr(cli, "_build_key_signals", refuse)
    scripted_user(lambda rec, root: root.fire("<space>"))
    # Even with a real-cursor backend in the config.
    config = _record_config(tmp_path, backend="pynput")

    argv = ["record", "--config", str(config), "--out", str(tmp_path / "s.csv"), *FAST]
    assert console_main(argv) == 0


@pytest.mark.parametrize(
    "extra, says_300",
    [([], True), (["--reading-s", "300"], False), (["--no-reading"], False)],
)
def test_cli_record_says_when_reading_is_shorter_than_the_prd_asks(
    tmp_path, capsys, scripted_user, extra, says_300
):
    from eyeos.cli import console_main

    scripted_user(lambda rec, root: root.fire("<Escape>"))
    argv = ["record", "--config", str(_record_config(tmp_path)), "--out", str(tmp_path / "s.csv")]

    assert console_main([*argv, *extra]) == 1
    assert ("--reading-s 300" in capsys.readouterr().err) is says_300


def test_cli_record_warns_that_only_a_webcam_records_a_real_session(
    tmp_path, capsys, scripted_user
):
    from eyeos.cli import console_main

    scripted_user(lambda rec, root: root.fire("<Escape>"))
    argv = ["record", "--config", str(_record_config(tmp_path)), "--out", str(tmp_path / "s.csv")]

    assert console_main([*argv, "--no-reading"]) == 1
    err = capsys.readouterr().err
    assert "input.source is 'synthetic', not 'webcam'" in err
