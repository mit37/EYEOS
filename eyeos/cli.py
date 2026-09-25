"""``eyeos run`` — runs the full pipeline against a config file, dry-run by default.

Without a fitted calibration, Gate 2 correctly refuses to move the cursor (no
calibration exists yet) — that's expected on a first run; the safety gates and their
logged reasons are still fully exercised. See docs/DEMO.md for the calibration step.
"""

from __future__ import annotations

import argparse
import sys

from eyeos.config import Config, load_config
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
    run.add_argument("--log", default=None, help="path to write a JSON-lines session log")

    return parser


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    input_source = _build_input_source(config)
    output = _build_output(config)
    logger = SessionLogger(args.log) if args.log else None

    pipeline = Pipeline(config, input_source, output, logger=logger)
    try:
        result = pipeline.run(max_frames=args.max_frames)
    finally:
        output.close()
        if logger is not None:
            logger.close()

    print(
        f"processed {result.frame_count} frames: "
        f"{result.total_moves} moves, {result.total_clicks} clicks"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    raise ValueError(f"unknown command: {args.command!r}")  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
