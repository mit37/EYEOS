"""``eyeos record``: a fullscreen, labelled session recorder (PRD §5: "eval/ takes a
recorded video + the known on-screen target positions").

It walks the user through a fixed sequence of screens and writes one CSV row per
camera frame: the frame's face landmarks plus what was on screen when it was taken
(``target_x``/``target_y``, ``segment``, ``away_mode``), in the format
``eval/run_eval.py`` reads. Only landmark coordinates are written. No camera image is
kept, in memory or on disk.

The stages, in order:

1. intro: what is about to happen, and whether the camera finds a face. Space starts.
   Frames read here are not recorded.
2. calibration: the 9 dots of ``eyeos.calibration.target_points``, ``dot_s`` each.
   Segment ``calibration``.
3. validation: 12 dots off the calibration grid, ``dot_s`` each. Segment ``on_target``.
4. looking away: three prompts, each started with space: look off-screen (the head
   may turn), close your eyes, lean out of the camera's view. Each gives the user
   ``AWAY_LEAD_IN_S`` to do it, then records ``AWAY_S`` as segment ``looking_away``
   (``away_mode`` off_screen, eyes_closed, no_face), then rings the bell.
5. reading: a public-domain passage for ``reading_s``. Segment ``just_reading``, no
   target. Skipped with ``reading=False``.
6. done: the CSV is written, and the window closes after a moment.

Rows that belong to no measured stage get segment ``settle``, which run_eval ignores:
the first ``settle_s`` of each dot and of the reading screen (the eyes are still on
their way), the look-away lead-ins, and pauses that wait for space. They are kept so
the pipeline replays one continuous stream.

The window reads one frame per ``root.after`` tick, and every stage is timed by the
frames' own timestamps, not the wall clock: a frame belongs to the screen that was
showing when it was read. The frame that ends a timed screen is the last one of it,
and the next screen's clock starts at that frame's timestamp.

Each row's target is the dot's screen position, so recording only starts once the
canvas covers the whole screen from its top-left corner (the config's screen size).

The recorder never moves the cursor, has no output adapter, and listens to no global
key (not the kill hotkey or the confirm key): only space and Esc in its own window.
Esc, or closing the window, cancels, and nothing is written.

Needs a display. Tested headless against a fake tkinter (tests/fake_tk.py). Outside
CI it has run on real Tk 9.0 under Xvfb (a virtual X display with no window manager)
with a paced synthetic camera; it has not run on a physical screen, on Windows, or
with a real webcam.
"""

from __future__ import annotations

import csv
import math
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eyeos.calibration import target_points
from eyeos.calibration_ui import check_screen_size
from eyeos.input_sources import Frame
from eyeos.landmarks import FaceLandmarks
from eyeos.synthetic_format import SESSION_FIELDNAMES, session_row

ScreenPoint = tuple[float, float]

SETTLE_SEGMENT = "settle"
AWAY_LEAD_IN_S = 2.0  # time to look away, close the eyes or lean out before recording
AWAY_S = 4.0
DONE_CLOSE_MS = 3000
_TICK_MS = 1
_DOT_RADIUS = 12

AWAY_PROMPTS: tuple[tuple[str, str], ...] = (
    (
        "off_screen",
        "Look away from the screen, at something off to one side. Turning your head is fine.",
    ),
    ("eyes_closed", "Close your eyes and keep them closed."),
    ("no_face", "Lean out of the camera's view, so your face is out of the picture."),
)

# A 4x3 grid between the calibration rows and columns (which sit at 10/50/90%), listed
# in an order that jumps across the screen.
_VALIDATION_FRACS: tuple[tuple[float, float], ...] = (
    (0.2, 0.25),
    (0.6, 0.75),
    (0.8, 0.5),
    (0.4, 0.5),
    (0.8, 0.25),
    (0.2, 0.75),
    (0.6, 0.25),
    (0.4, 0.75),
    (0.2, 0.5),
    (0.8, 0.75),
    (0.4, 0.25),
    (0.6, 0.5),
)

READING_SOURCE = "Lewis Carroll, Alice's Adventures in Wonderland (1865), chapter I. Public domain."
READING_PASSAGE = (
    "Alice was beginning to get very tired of sitting by her sister on the bank, and of "
    "having nothing to do: once or twice she had peeped into the book her sister was "
    "reading, but it had no pictures or conversations in it, 'and what is the use of a "
    "book,' thought Alice 'without pictures or conversations?'\n\n"
    "So she was considering in her own mind (as well as she could, for the hot day made "
    "her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would "
    "be worth the trouble of getting up and picking the daisies, when suddenly a White "
    "Rabbit with pink eyes ran close by her.\n\n"
    "There was nothing so very remarkable in that; nor did Alice think it so very much "
    "out of the way to hear the Rabbit say to itself, 'Oh dear! Oh dear! I shall be "
    "late!' (when she thought it over afterwards, it occurred to her that she ought to "
    "have wondered at this, but at the time it all seemed quite natural); but when the "
    "Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then "
    "hurried on, Alice started to her feet, for it flashed across her mind that she had "
    "never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of "
    "it, and burning with curiosity, she ran across the field after it, and fortunately "
    "was just in time to see it pop down a large rabbit-hole under the hedge.\n\n"
    "In another moment down went Alice after it, never once considering how in the world "
    "she was to get out again.\n\n"
    "The rabbit-hole went straight on like a tunnel for some way, and then dipped "
    "suddenly down, so suddenly that Alice had not a moment to think about stopping "
    "herself before she found herself falling down a very deep well."
)

_BG = "black"
_FG = "white"
_DIM = "#9a9a9a"


@dataclass(frozen=True)
class Step:
    """One screen of the recording after the intro.

    ``duration_s`` None means the screen waits for space. Frames in its first
    ``settle_s`` are labelled ``settle``; the rest get ``segment``.
    """

    stage: str  # calibration | validation | pause | looking_away | reading
    segment: str
    duration_s: float | None
    settle_s: float = 0.0
    target: ScreenPoint | None = None
    away_mode: str = ""
    title: str = ""
    text: str = ""
    bell_at_end: bool = False


def validation_points(screen_width_px: float, screen_height_px: float) -> list[ScreenPoint]:
    """The 12 validation dots: off the 9-point calibration grid, inside its hull."""
    return [(fx * screen_width_px, fy * screen_height_px) for fx, fy in _VALIDATION_FRACS]


def check_timings(dot_s: float, settle_s: float, reading_s: float, reading: bool) -> None:
    """Raise ValueError for timings that would leave nothing of a stage to measure."""
    if not math.isfinite(dot_s) or dot_s <= 0:
        raise ValueError(f"dot time must be a positive number of seconds, not {dot_s}")
    if not math.isfinite(settle_s) or not 0 <= settle_s < dot_s:
        raise ValueError(
            f"settle time must be at least 0 and shorter than the dot time ({dot_s:g} s), "
            f"not {settle_s:g}: otherwise no frame of a dot is measured"
        )
    if reading and (not math.isfinite(reading_s) or reading_s <= settle_s):
        raise ValueError(
            f"reading time must be a finite number of seconds longer than the settle time "
            f"({settle_s:g} s), not {reading_s:g}"
        )


def build_steps(
    screen_width_px: float,
    screen_height_px: float,
    *,
    dot_s: float = 2.0,
    settle_s: float = 0.6,
    reading_s: float = 60.0,
    reading: bool = True,
) -> list[Step]:
    """The recording's screens after the intro, in order. Raises ValueError for timings
    that would leave nothing to measure."""
    check_timings(dot_s, settle_s, reading_s, reading)
    steps: list[Step] = []
    calibration = target_points(screen_width_px, screen_height_px)
    for i, target in enumerate(calibration, start=1):
        steps.append(
            Step(
                "calibration",
                "calibration",
                dot_s,
                settle_s,
                target=target,
                title=f"calibration dot {i} of {len(calibration)}",
            )
        )
    validation = validation_points(screen_width_px, screen_height_px)
    for i, target in enumerate(validation, start=1):
        steps.append(
            Step(
                "validation",
                "on_target",
                dot_s,
                settle_s,
                target=target,
                title=f"validation dot {i} of {len(validation)}",
            )
        )
    for i, (mode, instruction) in enumerate(AWAY_PROMPTS, start=1):
        title = f"look-away prompt {i} of {len(AWAY_PROMPTS)} ({mode})"
        steps.append(
            Step(
                "pause",
                SETTLE_SEGMENT,
                None,
                title=f"the pause before {title}",
                text=(
                    f"Look-away test {i} of {len(AWAY_PROMPTS)}\n\n{instruction}\n\n"
                    "Press space, then do it straight away. Keep it up until you hear "
                    f"a beep (about {AWAY_LEAD_IN_S + AWAY_S:g} s later; if your sound "
                    "is off, count to ten), then look back at the screen."
                ),
            )
        )
        steps.append(
            Step(
                "looking_away",
                "looking_away",
                AWAY_LEAD_IN_S + AWAY_S,
                AWAY_LEAD_IN_S,
                away_mode=mode,
                title=title,
                text=f"{instruction}\n\nKeep it up until the beep.",
                bell_at_end=True,
            )
        )
    if reading:
        steps.append(
            Step(
                "pause",
                SETTLE_SEGMENT,
                None,
                title="the pause before reading",
                text=(
                    "Last part: reading\n\n"
                    f"The next screen shows a passage of text for {reading_s:g} s. Read "
                    "it at your normal pace, as you would read anything on screen. If "
                    "you reach the end, start again from the top. Don't press anything: "
                    "it ends by itself.\n\nPress space to start."
                ),
            )
        )
        steps.append(
            Step(
                "reading",
                "just_reading",
                reading_s,
                settle_s,
                title="the reading screen",
                text=READING_PASSAGE,
            )
        )
    return steps


def face_status(landmarks: FaceLandmarks | None) -> str:
    """The intro screen's one-line camera check."""
    if landmarks is None:
        return "Camera: no face found. Move into the camera's view."
    if landmarks.face_count != 1:
        return f"Camera: {landmarks.face_count} faces found. Only you should be in view."
    if not landmarks.both_irises_visible:
        return "Camera: face found, but the eyes don't look open."
    return "Camera: face found, eyes open. Ready."


def write_session_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a session CSV with exactly SESSION_FIELDNAMES.

    It never replaces a file: if one is at ``path`` by now (say another recording
    saved there meanwhile), this raises FileExistsError and leaves it as it was. The
    rows go to a temporary file beside ``path``, which then gets the name ``path``
    only if nothing has it. If anything fails, nothing is left at ``path`` or beside
    it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    try:
        with partial.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SESSION_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        _publish(partial, path)
    except FileExistsError:
        raise FileExistsError(
            f"{path} was created while this session was being recorded. eyeos record "
            "never replaces a file, so that file was left as it was and this session "
            "was not saved."
        ) from None
    finally:
        partial.unlink(missing_ok=True)


def _publish(partial: Path, path: Path) -> None:
    """Give the finished file the name ``path`` unless something already has it
    (FileExistsError). os.replace would overwrite it; a hard link can't."""
    try:
        os.link(partial, path)
        return
    except FileExistsError:
        raise
    except OSError:
        pass  # no hard links on this drive (FAT, exFAT, some network shares)
    dst = path.open("xb")  # fails, rather than replacing a file that's there
    try:
        with dst, partial.open("rb") as src:
            shutil.copyfileobj(src, dst)
    except BaseException:
        path.unlink(missing_ok=True)  # only this call can have created it
        raise


class SessionRecorder:
    """The fullscreen recording window. ``run()`` returns True once the CSV is written,
    False if the user cancelled, and raises (after closing the window) if reading the
    frames or writing the file failed.

    ``frames`` is read one frame per tick, from the intro on. The caller should have
    read one frame already, so a camera that can't open fails before the window does.
    """

    def __init__(
        self,
        screen_width_px: int,
        screen_height_px: int,
        frames: Iterator[Frame],
        out_path: str | Path,
        *,
        dot_s: float = 2.0,
        settle_s: float = 0.6,
        reading_s: float = 60.0,
        reading: bool = True,
    ) -> None:
        # Bad timings fail before a window opens.
        self.steps = build_steps(
            screen_width_px,
            screen_height_px,
            dot_s=dot_s,
            settle_s=settle_s,
            reading_s=reading_s,
            reading=reading,
        )
        try:
            import tkinter as tk
        except ImportError as e:
            raise RuntimeError(
                "eyeos record needs a display (tkinter is not available in this "
                "environment). Run it on a machine with a desktop session."
            ) from e

        self._tk = tk
        self.width = int(screen_width_px)
        self.height = int(screen_height_px)
        self.out_path = Path(out_path)
        self._frames = frames
        self._dot_s = dot_s
        self._reading_s = reading_s if reading else 0.0
        self._rows: list[dict[str, str]] = []
        self._phase = "intro"
        self._index = 0
        self._step_start_s: float | None = None
        self._failures: list[BaseException] = []
        self._saved = False
        self._closed = False
        self._after_id: Any = None
        self._status_item: Any = None
        self._status_text = ""

        try:
            self.root: Any = tk.Tk()
        except tk.TclError as e:
            raise RuntimeError(
                "eyeos record needs a display (tkinter is installed, but there is no "
                "$DISPLAY / desktop session here). Run it on a machine with a desktop "
                "session."
            ) from e
        check_screen_size(self.root, self.width, self.height)
        # -fullscreen needs a window manager to act on it; the explicit size and place
        # cover the screen without one (e.g. under Xvfb). Focus, so space and Esc work.
        self.root.geometry(f"{self.width}x{self.height}+0+0")
        self.root.attributes("-fullscreen", True)
        self.canvas: Any = tk.Canvas(self.root, bg=_BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.root.focus_force()
        self.root.bind("<space>", self._on_space)
        self.root.bind("<Escape>", self.cancel)
        # Tk's default close handler would only destroy the window. Closing it must
        # cancel, like Esc.
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)
        self._draw()

    # -- state the caller (and tests) can read ---------------------------------------

    @property
    def phase(self) -> str:
        """``intro``, ``steps`` or ``done``."""
        return self._phase

    @property
    def current_step(self) -> Step | None:
        """The screen showing now, or None on the intro and done screens."""
        return self.steps[self._index] if self._phase == "steps" else None

    @property
    def frame_count(self) -> int:
        return len(self._rows)

    # -- running ---------------------------------------------------------------------

    def run(self) -> bool:
        """Blocks until the session is saved, cancelled or fails."""
        self._after_id = self.root.after(0, self._tick)
        try:
            self.root.mainloop()
        finally:
            self._close()
        if self._failures:
            raise self._failures[0]
        return self._saved

    def cancel(self, _event: object = None) -> None:
        """Esc and the window's close button land here. Before the file is written this
        throws the session away; on the done screen it only closes the window."""
        self._close()

    def _tick(self) -> None:
        # Runs inside Tk's event loop, where an exception would be printed behind the
        # fullscreen window and the loop would carry on without a frame source. Keep
        # it, close the window, and let run() raise it.
        self._after_id = None
        if self._closed or self._phase == "done":
            return
        try:
            frame = next(self._frames)
        except StopIteration:
            self._fail(
                RuntimeError(
                    "camera stopped during recording: the input source ran out of "
                    f"frames {self._where()}"
                )
            )
            return
        except (Exception, KeyboardInterrupt) as e:  # Tk would swallow even Ctrl+C here
            self._fail(e)
            return
        try:
            self._on_frame(frame)
            if not self._closed:
                # Canvas redraws are Tk idle tasks, which a busy after() loop starves.
                self.root.update_idletasks()
        except (Exception, KeyboardInterrupt) as e:
            self._fail(e)
            return
        if not self._closed and self._phase != "done":
            self._after_id = self.root.after(_TICK_MS, self._tick)

    def _on_frame(self, frame: Frame) -> None:
        if self._phase == "intro":
            self._show_face_status(frame.landmarks)
            return
        step = self.steps[self._index]
        t = frame.timestamp_s
        finite = math.isfinite(t)
        if self._step_start_s is None and finite:
            self._step_start_s = t
        elapsed = t - self._step_start_s if finite and self._step_start_s is not None else math.nan
        # NaN compares False: a frame with no usable timestamp is never measured.
        segment = step.segment if elapsed >= step.settle_s else SETTLE_SEGMENT
        self._rows.append(session_row(t, frame.landmarks, step.target, segment, step.away_mode))
        if step.duration_s is not None and elapsed >= step.duration_s:
            self._next_step(start_s=t)

    def _on_space(self, _event: object = None) -> None:
        if self._closed:
            return
        if self._phase == "done":
            self._close()
        elif self._phase == "intro":
            uncovered = self._uncovered_screen()
            if uncovered is not None:
                self._fail(RuntimeError(uncovered))
                return
            self._phase = "steps"
            self._index = 0
            self._step_start_s = None  # the clock starts at the next frame
            self._draw()
        elif self.steps[self._index].duration_s is None:
            self._next_step(start_s=None)

    def _next_step(self, start_s: float | None) -> None:
        if self.steps[self._index].bell_at_end:
            self.root.bell()
        self._index += 1
        self._step_start_s = start_s
        if self._index >= len(self.steps):
            self._finish()
        else:
            self._draw()

    def _finish(self) -> None:
        write_session_csv(self.out_path, self._rows)
        self._saved = True
        self._phase = "done"
        self._draw()
        self.root.after(DONE_CLOSE_MS, self._close)

    def _fail(self, error: BaseException) -> None:
        self._failures.append(error)
        self._close()

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except self._tk.TclError:
                pass
            self._after_id = None
        try:
            self.root.destroy()
        except self._tk.TclError:
            pass  # already destroyed

    def _uncovered_screen(self) -> str | None:
        """Why the canvas doesn't cover the screen from its top-left corner, or None.

        Each row's target is the dot's position on the screen, which is only where the
        dot is drawn if the canvas starts at (0, 0) and spans the whole screen."""
        self.root.update_idletasks()
        c = self.canvas
        x, y = int(c.winfo_rootx()), int(c.winfo_rooty())
        w, h = int(c.winfo_width()), int(c.winfo_height())
        if (x, y, w, h) == (0, 0, self.width, self.height):
            return None
        return (
            f"the recording window does not cover the screen: it is {w}x{h} at +{x}+{y}, "
            f"not {self.width}x{self.height} at +0+0, so the dots would not be where the "
            "session file says. Run it on the primary screen, with nothing (such as an "
            "always-on-top taskbar) keeping it from going fullscreen."
        )

    def _where(self) -> str:
        if self._phase == "intro":
            return "before the recording started"
        return f"at {self.steps[self._index].title}"

    # -- drawing ---------------------------------------------------------------------

    def _font(self, fraction_of_height: float) -> tuple[str, int]:
        # A negative Tk font size is in pixels, so text scales with the screen.
        return ("Helvetica", -max(12, round(self.height * fraction_of_height)))

    def _show_face_status(self, landmarks: FaceLandmarks | None) -> None:
        status = face_status(landmarks)
        if status != self._status_text and self._status_item is not None:
            self._status_text = status
            self.canvas.itemconfig(self._status_item, text=status)

    def _draw(self) -> None:
        c = self.canvas
        c.delete("all")
        w, h = self.width, self.height
        if self._phase == "intro":
            self._draw_intro()
            return
        if self._phase == "done":
            c.create_text(
                w / 2,
                h / 2,
                text=(
                    f"Done. Saved {self.frame_count} frames of face landmarks to\n"
                    f"{self.out_path}\n\nThis window closes by itself."
                ),
                fill=_FG,
                font=self._font(1 / 36),
                width=w * 0.8,
                justify="center",
            )
            return
        step = self.steps[self._index]
        if step.target is not None:
            x, y = step.target
            r = _DOT_RADIUS
            c.create_oval(x - r, y - r, x + r, y + r, fill=_FG, outline="")
            c.create_oval(x - 2, y - 2, x + 2, y + 2, fill=_BG, outline="")
            c.create_text(
                20,
                20,
                anchor="nw",
                fill=_DIM,
                font=self._font(1 / 60),
                text=f"Look at the dot ({step.title}). Esc cancels.",
            )
        elif step.stage == "reading":
            c.create_text(
                w / 2,
                h * 0.03,
                anchor="n",
                fill=_DIM,
                font=self._font(1 / 60),
                text=(
                    "Read at your normal pace. If you reach the end, start again from "
                    "the top. This ends by itself. Esc cancels."
                ),
            )
            c.create_text(
                w / 2,
                h * 0.09,
                anchor="n",
                fill=_FG,
                font=self._font(1 / 40),
                width=w * 0.7,
                justify="left",
                text=step.text,
            )
            c.create_text(
                w / 2,
                h * 0.97,
                anchor="s",
                fill=_DIM,
                font=self._font(1 / 70),
                text=READING_SOURCE,
            )
        else:
            c.create_text(
                w / 2,
                h / 2,
                fill=_FG,
                font=self._font(1 / 32),
                width=w * 0.7,
                justify="center",
                text=step.text + ("\n\nEsc cancels." if step.duration_s is None else ""),
            )

    def _draw_intro(self) -> None:
        c = self.canvas
        w, h = self.width, self.height
        n_dots = sum(1 for s in self.steps if s.target is not None)
        timed = sum(s.duration_s or 0.0 for s in self.steps)
        reading = (
            f"4. Read a passage of text for {self._reading_s:g} s.\n" if self._reading_s else ""
        )
        c.create_text(
            w / 2,
            h * 0.10,
            anchor="n",
            fill=_FG,
            font=self._font(1 / 22),
            text="EyeOS session recorder",
        )
        c.create_text(
            w / 2,
            h * 0.20,
            anchor="n",
            fill=_FG,
            font=self._font(1 / 40),
            width=w * 0.7,
            justify="left",
            text=(
                "This records where your face and eyes are (landmark coordinates only; "
                "no camera image is saved) while you follow the prompts, so EyeOS's "
                "accuracy can be measured on your own webcam. It never moves the mouse "
                f"or clicks. It takes about {math.ceil(timed / 60)} min.\n\n"
                f"1. Look at each dot until the next one appears ({n_dots} dots, "
                f"{self._dot_s:g} s each). Keep your head where it normally is.\n"
                "2. Three short look-away tests: look away, close your eyes, lean out "
                "of view. Each one starts when you press space.\n"
                "3. A beep tells you when each look-away test is over.\n"
                f"{reading}\n"
                "Esc cancels at any time, and then nothing is saved."
            ),
        )
        self._status_item = c.create_text(
            w / 2,
            h * 0.80,
            anchor="n",
            fill=_FG,
            font=self._font(1 / 36),
            text=self._status_text or "Camera: waiting for the first frame...",
        )
        c.create_text(
            w / 2,
            h * 0.88,
            anchor="n",
            fill=_FG,
            font=self._font(1 / 30),
            text="Press space to start.",
        )
