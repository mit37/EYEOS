"""``eyeos run`` — runs the full pipeline against a config file, dry-run by default.

Without ``--calibrate`` there is no calibration, so Gate 2 correctly refuses to move
the cursor; the safety gates and their logged reasons are still fully exercised.
``--calibrate`` shows the 9-dot screen first and fits on the same camera stream the
session then runs on; ``--overlay`` shows the four gate lights; ``--view`` shows the
live tracking view (eyeos/tracking_view.py). All three need a display.

``eyeos record`` shows a fullscreen sequence of dots and prompts and saves the face
landmarks of every frame, labelled with what was on screen, for eval/run_eval.py
(see docs/RECORDING.md). It never moves the cursor. Needs a display.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import os
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

from eyeos.calibration_session import calibrate_interactively
from eyeos.calibration_ui import CalibrationUI
from eyeos.config import Config, load_config
from eyeos.hotkeys import KeySignals, build_key_signals
from eyeos.input_sources import InputSource
from eyeos.logging_utils import SessionLogger
from eyeos.output import OutputAdapter
from eyeos.overlay import GateOverlay
from eyeos.pipeline import Pipeline
from eyeos.recorder import SessionRecorder, check_timings
from eyeos.tracking_view import TrackingView

# PRD §5's click false-trigger test uses a 5-minute "just reading" segment.
PRD_READING_S = 300.0


def _build_input_source(config: Config, keep_images: bool = False) -> InputSource:
    """``keep_images`` makes a camera source attach each frame's image (for --view). A
    synthetic source has no images, so it ignores it."""
    source = config.input.source
    if source == "synthetic":
        from eyeos.input_sources.synthetic import SyntheticInputSource

        if not config.input.synthetic_path:
            raise ValueError("input.source is 'synthetic' but input.synthetic_path is not set")
        return SyntheticInputSource(config.input.synthetic_path)
    if source == "video_file":
        from eyeos.input_sources.video_file import VideoFileInputSource

        if not config.input.video_path:
            raise ValueError("input.source is 'video_file' but input.video_path is not set")
        threshold = config.gates.presence_confidence_threshold
        return VideoFileInputSource(
            config.input.video_path,
            min_detection_confidence=threshold,
            min_tracking_confidence=threshold,
            keep_images=keep_images,
        )
    if source == "webcam":
        from eyeos.input_sources.webcam import WebcamInputSource

        threshold = config.gates.presence_confidence_threshold
        return WebcamInputSource(
            device_index=config.input.device_index,
            min_detection_confidence=threshold,
            min_tracking_confidence=threshold,
            keep_images=keep_images,
        )
    raise ValueError(f"unknown input.source: {source!r}")


OUTPUT_BACKENDS = ("dry_run", "windows_sendinput", "pynput")


def _build_output(config: Config) -> OutputAdapter:
    backend = config.output.backend
    if backend == "dry_run":
        from eyeos.output.dry_run import DryRunOutputAdapter

        return DryRunOutputAdapter()
    if backend == "windows_sendinput":
        from eyeos.output.windows_sendinput import WindowsSendInputOutput

        return WindowsSendInputOutput(
            expected_size=(config.screen.width_px, config.screen.height_px)
        )
    if backend == "pynput":
        from eyeos.output.pynput_fallback import PynputOutputAdapter

        return PynputOutputAdapter()
    raise ValueError(f"unknown output.backend: {backend!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eyeos", description="Webcam eye tracking as an input device."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline against a config file")
    run.add_argument(
        "--config", default=None, help="path to a config YAML (default: built-in defaults)"
    )
    run.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    run.add_argument(
        "--log",
        default=None,
        help="session log path (default: <logging.session_log_dir>/session-<UTC time>.jsonl)",
    )
    run.add_argument("--no-log", action="store_true", help="don't write a session log")
    run.add_argument(
        "--calibrate",
        action="store_true",
        help="run the 9-dot calibration screen first, on the same camera stream (needs a display)",
    )
    run.add_argument(
        "--overlay", action="store_true", help="show the four gate lights live (needs a display)"
    )
    run.add_argument(
        "--view",
        action="store_true",
        help="show the live tracking view: pupil markers, their shift, a mini screen map "
        "and the gate lights over the camera image (needs a display and OpenCV; "
        "q or closing the window hides it, the session keeps running)",
    )

    record = sub.add_parser(
        "record",
        help="record a labelled session for eval/run_eval.py (needs a display and a camera)",
        description=(
            "Show calibration and validation dots, three look-away prompts and a "
            "reading passage, fullscreen, and save each camera frame's face landmarks "
            "(no images) with what was on screen. See docs/RECORDING.md."
        ),
    )
    record.add_argument("--config", required=True, help="config YAML: input source and screen size")
    record.add_argument(
        "--out", required=True, help="session CSV to write (an existing file is never replaced)"
    )
    record.add_argument(
        "--dot-s", type=float, default=2.0, help="seconds each dot is shown (default 2.0)"
    )
    record.add_argument(
        "--settle-s",
        type=float,
        default=0.6,
        help="seconds at the start of each dot, and of reading, left out of the eval "
        "while the eyes get there (default 0.6)",
    )
    record.add_argument(
        "--reading-s",
        type=float,
        default=60.0,
        help="seconds of reading (default 60; 300 matches PRD §5's 5-minute segment)",
    )
    record.add_argument("--no-reading", action="store_true", help="skip the reading stage")

    return parser


def _session_log_path(args: argparse.Namespace, config: Config) -> Path | None:
    """PRD §2: every gate decision is logged, so a session log is on by default."""
    if args.no_log:
        return None
    if args.log:
        return Path(args.log)
    if config.logging.session_log_dir is None:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(config.logging.session_log_dir) / f"session-{stamp}.jsonl"
    n = 1
    while path.exists():  # a second run within the same second keeps both logs
        n += 1
        path = path.with_name(f"session-{stamp}-{n}.jsonl")
    return path


def _build_key_signals(config: Config) -> KeySignals:
    return build_key_signals(config)


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if config.output.backend not in OUTPUT_BACKENDS:
        raise ValueError(f"unknown output.backend: {config.output.backend!r}")

    if config.output.backend == "dry_run" and config.gates.confirmation_mode == "keypress":
        print(
            "eyeos: warning: dry runs don't listen to the keyboard, so in "
            "confirmation_mode 'keypress' no click can be confirmed",
            file=sys.stderr,
        )

    # Any backend that can move the real cursor gets the kill switch first: if no
    # keyboard backend can start, exit before an output adapter even exists.
    keys: KeySignals | None = None
    if config.output.backend != "dry_run":
        try:
            keys = _build_key_signals(config)
        except RuntimeError as e:
            print(
                f"eyeos: refusing to drive the cursor without a kill switch: {e}", file=sys.stderr
            )
            return 2

    # Every resource opened below is closed on the way out, in reverse order, even if
    # another one's close() raises. The kill switch was opened first, so it stops last.
    with contextlib.ExitStack() as cleanup:
        if keys is not None:
            cleanup.callback(keys.close)
        log_path = _session_log_path(args, config)
        input_source = _build_input_source(config, keep_images=args.view)
        cleanup.callback(input_source.close)
        view: TrackingView | None = None
        if args.view:
            # Built before calibrating, so a missing OpenCV or display stops the run
            # before the user has sat through the dots. The window opens with frame 1.
            view = TrackingView((config.screen.width_px, config.screen.height_px))
            cleanup.callback(view.close)
        frames = iter(input_source.frames())
        calibration = None
        if args.calibrate:
            # Open the source now. Opened lazily inside the dot screen's key handler, a
            # webcam or MediaPipe error would be swallowed by Tk behind the window.
            first_frame = next(frames, None)
            if first_frame is None:
                print(
                    "eyeos: the input source gave no frames, so there is nothing to "
                    "calibrate on",
                    file=sys.stderr,
                )
                return 1
            frames = itertools.chain([first_frame], frames)
            calibration = calibrate_interactively(config, frames, CalibrationUI)
            if calibration is None:
                print("eyeos: calibration cancelled", file=sys.stderr)
                return 1
            print(f"calibration validation error: {calibration.validation_error_px:.1f} px")
        overlay: GateOverlay | None = None
        if args.overlay:
            overlay = GateOverlay()
            cleanup.callback(overlay.close)
        output = _build_output(config)
        cleanup.callback(output.close)
        logger: SessionLogger | None = None
        if log_path is not None:
            logger = SessionLogger(log_path)
            cleanup.callback(logger.close)
        pipeline = Pipeline(
            config,
            input_source,
            output,
            calibration=calibration,
            logger=logger,
            external_signals=keys,
            on_status=overlay.update if overlay is not None else None,
            on_frame=view.update if view is not None else None,
        )
        result = pipeline.run(max_frames=args.max_frames, frames=frames)

    print(
        f"processed {result.frame_count} frames: "
        f"{result.total_moves} moves, {result.total_clicks} clicks"
    )
    if log_path is not None:
        print(f"session log: {log_path}")
    if result.killed:
        print(f"kill switch ({config.output.kill_hotkey}) pressed: session ended")
    return 0


_POWERSHELL_BARE = re.compile(r"[A-Za-z0-9_./:\\-]+")
# PowerShell ends a single-quoted string at any of these, not only at the ASCII one.
_POWERSHELL_SINGLE_QUOTES = re.compile("['‘’‚‛]")


def _shell_arg(value: str, *, powershell: bool | None = None) -> str:
    """Quote a path so the user's shell passes it on unchanged.

    On Windows that is PowerShell (the shell docs/LOCAL_SETUP.md uses; not cmd):
    anything but a plain path goes in single quotes, which PowerShell takes
    literally, with each single quote doubled. Elsewhere it is sh (bash, zsh, ...),
    via shlex.quote. The sh form is tested by running it through sh; the PowerShell
    form has not been run in a real PowerShell.
    """
    if powershell is None:
        powershell = os.name == "nt"
    if not powershell:
        return shlex.quote(value)
    if _POWERSHELL_BARE.fullmatch(value):
        return value
    return "'" + _POWERSHELL_SINGLE_QUOTES.sub(lambda m: m.group() * 2, value) + "'"


def _record(args: argparse.Namespace) -> int:
    reading = not args.no_reading
    # Bad timings, a bad config and an existing file all fail before the camera opens.
    check_timings(args.dot_s, args.settle_s, args.reading_s, reading)
    config = load_config(args.config)
    out = Path(args.out)
    if out.exists():
        raise ValueError(
            f"{out} already exists, and eyeos record never replaces a session: "
            "pass a new --out path"
        )
    if config.input.source != "webcam":
        print(
            f"eyeos: warning: input.source is {config.input.source!r}, not 'webcam': its "
            "frames come as fast as they can be read and don't follow the dots, so this "
            "is only good for testing",
            file=sys.stderr,
        )
    if reading and args.reading_s < PRD_READING_S:
        print(
            f"eyeos: note: reading lasts {args.reading_s:g} s; PRD §5's click "
            f"false-trigger test uses 5 minutes (--reading-s {PRD_READING_S:g})",
            file=sys.stderr,
        )

    with contextlib.ExitStack() as cleanup:
        input_source = _build_input_source(config)
        cleanup.callback(input_source.close)
        frames = iter(input_source.frames())
        # Open the source before the window: a camera or MediaPipe error raised inside
        # the window's frame loop would only surface once the screen had come up.
        first_frame = next(frames, None)
        if first_frame is None:
            raise RuntimeError("the input source gave no frames, so there is nothing to record")
        recorder = SessionRecorder(
            config.screen.width_px,
            config.screen.height_px,
            itertools.chain([first_frame], frames),
            out,
            dot_s=args.dot_s,
            settle_s=args.settle_s,
            reading_s=args.reading_s,
            reading=reading,
        )
        saved = recorder.run()

    if not saved:
        print("eyeos: recording cancelled; nothing was saved", file=sys.stderr)
        return 1
    results = out.with_name(f"{out.stem}-results.json")
    print(
        f"recorded {recorder.frame_count} frames to {out.resolve()} "
        "(face landmarks only; no camera images)"
    )
    print("evaluate it, from the repository root, with:")
    print(
        f"python -m eval.run_eval --session {_shell_arg(str(out.resolve()))} "
        f"--config {_shell_arg(str(Path(args.config).resolve()))} "
        f"--out {_shell_arg(str(results.resolve()))}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Errors propagate as exceptions (for tests and programmatic use)."""
    args = build_arg_parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "record":
        return _record(args)
    raise ValueError(f"unknown command: {args.command!r}")  # pragma: no cover


def console_main(argv: list[str] | None = None) -> int:
    """The ``eyeos`` command: like main(), but a config, input or OS error is printed
    as one line instead of a traceback. Every resource is already closed by then."""
    try:
        return main(argv)
    except KeyboardInterrupt:
        print("eyeos: interrupted", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, OSError) as e:
        print(f"eyeos: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(console_main())
