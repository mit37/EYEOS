# PLAN.md — EyeOS v2 rebuild

Restated from `PRD.md` §8, per `STANDARDS.md` §6 step 1.

## Definition of Done

- [x] Gate state machine fully tested (≥40 tests), including the kill switch and no-click zones
      — 49 tests in `tests/test_gates.py`, driven entirely by explicit timestamps (no wall clock).
- [x] Dry-run mode runs the whole pipeline on a video file
      — on a synthetic-landmark CSV in this container (no real video available); the exact
      same `Pipeline`/`VideoFileInputSource` code path works on a real video file once
      `opencv-python`/`mediapipe` are installed on real hardware (untested here, see below).
- [x] Eval numbers generated (synthetic in CI; real ones from Mitansh's recordings or marked pending)
      — `eval/run_eval.py`; real-recording numbers explicitly marked "not measured" pending
      Mitansh's own recordings.
- [x] README states "experiment, not an assistive-tech product"; CI green; tag v2.0.0

## Milestones (PRD §6), built in order

1. [x] Scaffold, CI, config, an input-source interface (webcam | video file | synthetic landmarks).
2. [x] Landmark → features + head pose (unit-tested on fixture landmarks).
3. [x] Calibration + regression + validation error.
4. [x] One Euro filter (tested against the reference behavior).
5. [x] The four gates as a pure state machine with a fake clock (the most tests: every gate transition).
6. [x] Output adapters (Windows SendInput + fallback) behind an interface; dry-run mode prints actions instead of moving the mouse.
7. [x] Overlay + calibration UI.
8. [x] Eval, README with gate diagram, `docs/DEMO.md`, tag v2.0.0.

## Cloud-instance constraints (PRD §7, STANDARDS §6.4)

This session has no webcam, no Windows desktop, and no display server (`tkinter` is not
even importable in this container — verified during scaffolding). Consequences:

- `WebcamInputSource` and `VideoFileInputSource` depend on `opencv-python` and `mediapipe`,
  which are **not installed** and are not required by the core package. They are behind the
  `InputSource` interface, with the imports lazy (inside `frames()`, not at module level),
  so importing `eyeos` never requires them. `tests/test_input_sources.py` exercises exactly
  what this container can: the guarded "dependency missing" `RuntimeError` path (both
  packages are genuinely absent here) and the MediaPipe-landmark-index mapping as a pure
  function on fake landmark objects. Real capture is untested here and must be verified by
  Mitansh on a real machine (see `docs/DEMO.md`).
- `WindowsSendInputOutput` depends on `ctypes.windll`, which only exists on Windows. It is
  behind the `OutputAdapter` interface and is guarded to raise a clear `RuntimeError` off
  Windows. It is untested here; `DryRunOutputAdapter` (fully tested) stands in for it in
  the pipeline test and in the eval harness.
- `overlay.py` and `calibration_ui.py` depend on `tkinter` *and* a real display, which
  this build container has neither of, and GitHub's own hosted CI runners have only the
  first (`tkinter` is installed there, but there's no `$DISPLAY`). Both modules guard
  both failure modes — `ImportError` (no tkinter) and `tkinter.TclError` (tkinter
  present, no display) — into the same clear `RuntimeError`, so importing the package
  never fails and CI fails clearly instead of with a raw Tcl traceback (this is exactly
  the bug CI itself caught on the first push of this branch: the guard only handled the
  `ImportError` case, which is all this local build container ever produces, and CI's
  `TclError` case slipped through untested until then). They are not otherwise unit
  tested (nothing to test without a display); `docs/DEMO.md` is the script for Mitansh
  to record them on his own machine.
- The eval numbers in the README come from a **generated synthetic landmark sequence**
  (`eval/fixtures/`, produced by `eval/generate_synthetic.py`, committed so CI needs no
  camera). Real-session numbers from recorded video are marked "not measured — pending
  Mitansh's recordings" until he supplies them.

Everything else (features, calibration, filter, gates, dry-run pipeline, eval on synthetic
data) runs and is tested in this container with no external hardware.
