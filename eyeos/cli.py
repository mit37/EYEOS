"""``eyeos run`` — runs the full pipeline against a config file, dry-run by default.

Without a fitted calibration, Gate 2 correctly refuses to move the cursor (no
calibration exists yet) — that's expected on a first run; the safety gates and their
logged reasons are still fully exercised. See docs/DEMO.md for the calibration step.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from eyeos.config import Config, load_config
from eyeos.hotkeys import KeySignals, build_key_signals
from eyeos.input_sources import InputSource
from eyeos.logging_utils import SessionLogger
from eyeos.output import OutputAdapter
from eyeos.pipeline import Pipeline


def _build_input_source(config: Config) -> InputSource:
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
        return VideoFileInputSource(config.input.video_path)
    if source == "webcam":
        from eyeos.input_sources.webcam import WebcamInputSource

        return WebcamInputSource(device_index=config.input.device_index)
    raise ValueError(f"unknown input.source: {source!r}")


OUTPUT_BACKENDS = ("dry_run", "windows_sendinput", "pynput")


def _build_output(config: Config) -> OutputAdapter:
    backend = config.output.backend
    if backend == "dry_run":
        from eyeos.output.dry_run import DryRunOutputAdapter

        return DryRunOutputAdapter()
    if backend == "windows_sendinput":
        from eyeos.output.windows_sendinput import WindowsSendInputOutput

        return WindowsSendInputOutput()
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
    return Path(config.logging.session_log_dir) / f"session-{stamp}.jsonl"


def _build_key_signals(config: Config) -> KeySignals:
    return build_key_signals(config)


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if config.output.backend not in OUTPUT_BACKENDS:
        raise ValueError(f"unknown output.backend: {config.output.backend!r}")

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

    log_path = _session_log_path(args, config)
    output: OutputAdapter | None = None
    logger: SessionLogger | None = None
    try:
        input_source = _build_input_source(config)
        output = _build_output(config)
        logger = SessionLogger(log_path) if log_path is not None else None
        pipeline = Pipeline(config, input_source, output, logger=logger, external_signals=keys)
        result = pipeline.run(max_frames=args.max_frames)
    finally:
        if output is not None:
            output.close()
        if logger is not None:
            logger.close()
        if keys is not None:
            keys.close()

    print(
        f"processed {result.frame_count} frames: "
        f"{result.total_moves} moves, {result.total_clicks} clicks"
    )
    if log_path is not None:
        print(f"session log: {log_path}")
    if result.killed:
        print(f"kill switch ({config.output.kill_hotkey}) pressed: session ended")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    raise ValueError(f"unknown command: {args.command!r}")  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
