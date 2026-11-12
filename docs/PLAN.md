# PLAN.md — EyeOS v2 rebuild

Restated from `PRD.md` §8, per `STANDARDS.md` §6 step 1.

## Definition of Done

- [x] Gate state machine fully tested (≥40 tests), including the kill switch and no-click
      zones. All of `tests/test_gates.py` is driven by explicit timestamps (no wall clock),
      and README's Testing section shows the generated count.
- [ ] (partial) Dry-run mode runs the whole pipeline on a video file
      - Done: the same `Pipeline` runs end to end on a synthetic-landmark CSV, and
        `eyeos run` runs end to end on a `video_file` config against fake
        OpenCV/MediaPipe modules (`tests/fake_capture.py`, used by `tests/test_cli.py`):
        container timestamps, face counts reaching Gate 1, the session log, and cleanup
        on a full run and on `--max-frames`.
      - Pending: decoding a real video file, which needs the `webcam` extra (Python
        3.10–3.12) and a real recording (see `docs/DEMO.md` §4).
- [x] Eval numbers generated (synthetic in CI; real ones marked pending)
      - `eval/run_eval.py` produces them, and `scripts/update_readme.py` writes them into
        README.md (CI fails on drift).
      - Real-recording numbers, including PRD §5's 5-minute reading segment, are
        marked "not measured".
- [x] README states "experiment, not an assistive-tech product"
- [ ] CI green on the commit to be tagged
- [ ] Tag v2.0.0. Pending: the owner tags `main` once the items above are met (this
      build's agent cannot push tags or edit repo settings).

## Milestones (PRD §6)

1. [x] Scaffold, CI, config, an input-source interface (webcam | video file | synthetic landmarks).
2. [x] Landmark → features + head pose (unit-tested on fixture landmarks).
3. [x] Calibration + regression + validation error.
4. [x] One Euro filter (tested against the reference behavior).
5. [x] The four gates as a pure state machine with a fake clock (the most tests: every gate transition).
6. [x] Output adapters (Windows SendInput + fallback) behind an interface; dry-run mode prints actions instead of moving the mouse.
7. [x] Overlay + calibration UI.
8. [x] Eval, README with gate diagram, `docs/DEMO.md`. [ ] Tag v2.0.0 (see above).

### How this was actually built (history note)

- **First build (2026-09-25).** Milestones 1–8 were built in one agent session on
  branch `claude/new-session-cawn9l` and committed as one commit (`170b0e2`). CI then
  caught a bug, fixed in `b051c61` (the tkinter `TclError` guard described below). The
  pair was squash-merged as PR #1 (`fd3858b`).
- **History split (2026-09-27).** At the owner's request, `main` was rewritten into
  12 per-milestone commits. They are a post-hoc split of the same final tree, so their
  timestamps (all within about a minute on 2026-09-27) record when the split was made,
  not when each milestone was written. The dates were not faked (STANDARDS §1.2), but
  the order does not show build order.
- **A known flaw in that split.** The "feat: gate overlay and fullscreen calibration
  UI" commit (`b2aae59`) already contains the `TclError` fix, so the later "fix: guard
  tkinter.TclError" commit (`d094fbe`) only changes this file. Correcting it would mean
  rewriting `main` again, which is the owner's call.
- **Audit and hardening (2026-11-12).** A multi-agent audit against PRD.md and
  STANDARDS.md confirmed 42 findings, several of them critical:
  - the kill hotkey was never wired;
  - `eyeos run` could never use a calibration;
  - blink-hold clicks could not fire;
  - the README's jitter "negative result" was an eval bug.

  They were fixed in individual commits.
- **Review of the hardening (2026-11-12).** A second multi-agent review, of the
  hardening diff itself, found more real bugs, each with a repro. Among them:
  - Gate 3's dwell kept building while the eyes were closed, so a glance followed by
    a closure clicked.
  - The blink-hold gaze could come from frames the gates had rejected.
  - Lids partway through a blink threw the cursor off target.
  - Camera stalls and NaN timestamps were not handled.
  - Closing the calibration window counted as a finished calibration.
  - Closing the overlay crashed the cleanup.
  - Camera errors during `--calibrate` were swallowed.

  Each fix came with a test that failed before it, and each was re-reviewed.
- **Review rounds 3-9 (2026-11-12).** The first fix for the half-closed-lid bug was
  itself wrong, and is recorded here as a negative result.
  - That fix held the gaze while the lids were under 80% of their usual opening.
    Looking down narrows the lids, so a confirm key pressed just after looking away
    clicked the target the gaze had left. Lid flutter also kept the hold alive
    without limit.
  - It was replaced by the root-cause fix: iris offsets and roll are now measured
    from the eye corners, in one unit on both axes, so lid movement can't move the
    gaze.
  - Lid flutter needed three attempts. Two dwell rules (a confirm-length closure
    ends the dwell; closed time may not exceed open time) were each beaten by a new
    flutter pattern in the next round. What finally held is a 0.1 s debounce on
    reopening: a shorter glimpse stays part of the closure. A click also uses the
    dwell up.
  - Round 6 found that the debounce itself let a 0.33 s blink click (its open frames
    advanced the blink-hold) and let a keypress click the old target just after the
    eyes reopened elsewhere. The hold now freezes at the last closed frame, and the
    debounce is skipped after a sub-0.1 s dropout or when the reopened gaze has
    moved.
  - Round 7 found that the hold still spanned glimpses (a blink plus dropouts
    clicked with 0.15-0.27 s of closed eyes) and that the gaze-moved check misfired
    on noisy gaze. The blink-hold now counts one unbroken run of closed frames, and
    the gaze-moved check was replaced by ignoring the confirm key during the
    debounce. One fail-closed gap is left and documented: very frequent dropouts
    right after a blink can keep Gate 1 failing.
  - Round 8 found that muting the key during the reopening counted as a release (a
    key held through a blink clicked twice), that a key pressed mid-blink clicked the
    point held from before it, and that a lost-face frame reset closure tracking. A
    key now confirms only on observed open-eye frames while its up/down state still
    drives the release latch, and a lost face no longer ends a closure.
  - Round 9 found that a lost-face or dropped frame, or a re-sent old timestamp,
    could stretch two glimpses into a 0.1 s "reopening". Unobserved time no longer
    counts toward an open run, and a non-advancing timestamp is rejected before it
    touches any blink state.
  - The blink path is the part of EyeOS that most needs real recordings: each round
    found a new noise pattern, and none of the thresholds is tuned on real data.
  - Eye openness is now judged in each eye's own frame, so it holds at any head
    roll.
  - Further fixes in these rounds: MediaPipe's per-axis normalisation broke roll on
    16:9 frames; the iris axes flipped at 90 deg of roll; Ctrl+C during calibration
    was misreported; bad YAML gave a traceback; and README config values could go
    stale.
- Still open: the owner-only items (tag, repo description and topics, branch
  protection) and hardware verification.

## Cloud-instance constraints (PRD §7, STANDARDS §6.4)

This build had no webcam, no Windows desktop, no keyboard and no display server
(`tkinter` isn't even importable in the build container). Every hardware-facing path
is therefore behind an interface and tested only against a fake:

- **Webcam and video file** (`WebcamInputSource`, `VideoFileInputSource`). OpenCV and
  MediaPipe are optional and imported lazily, so importing `eyeos` never needs them.
  - The frame loop, cleanup on failure, the confidence-threshold plumbing and
    host-clock timestamps are tested against fake `cv2`/`mediapipe` modules
    (`tests/fake_capture.py`). These fakes prove EyeOS's plumbing, not OpenCV's or
    MediaPipe's real behaviour.
  - Live use needs `mediapipe<0.10.30` (the Solutions API was removed upstream in
    0.10.30), so Python 3.10–3.12.
  - The claim that `CAP_PROP_POS_MSEC` is constant on live DirectShow/AVFoundation
    cameras was reasoned from OpenCV's source and not tested on hardware.
- **Windows output** (`WindowsSendInputOutput`) needs `ctypes.windll` and is guarded to
  raise off Windows. It has never run here; `DryRunOutputAdapter` stands in for it.
- **Keyboard** (`PynputKeySource`, the kill switch and confirm key). It is tested
  through `FakeKeySource`. The real pynput listener has never run here. Without it,
  `eyeos run` refuses to drive the real cursor.
- **Display** (`overlay.py`, `calibration_ui.py`). Both are tested against a fake
  tkinter (`tests/fake_tk.py`), and both no-display modes raise a clear `RuntimeError`.
  The first push of `claude/new-session-cawn9l` showed why both modes matter: the guard
  only handled `ImportError`, the only mode this container produces, and GitHub's
  runners (tkinter installed, no `$DISPLAY`) hit the unguarded `TclError`.
  - During the review, both windows also ran on real Tk 8.6 under Xvfb, a virtual X
    display, from a separate Python 3.12 environment. Those were ad-hoc repro scripts:
    closing the calibration window, closing the overlay, and a camera that fails
    before or during calibration.
  - That is not part of CI, and neither window has been shown on a physical screen.
- **Eval.** The README numbers come from a generated synthetic landmark sequence
  (`eval/fixtures/`, made by `eval/generate_synthetic.py`). Real-session numbers are
  marked "not measured", pending recordings, and so is the PRD's 5-minute reading
  false-click test. A real-accuracy eval also needs footage recorded with known
  on-screen targets (see `docs/DEMO.md` §4).

Everything else runs and is tested in this container with no external hardware:
features, calibration, filter, gates, hotkey latch logic, the dry-run pipeline and the
eval on synthetic data.
