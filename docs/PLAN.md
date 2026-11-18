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
      - Also run once with the real libraries (OpenCV 4.11, MediaPipe 0.10.21, Python
        3.11): `eyeos run` decoded a generated 45-frame video, took its timestamps from
        the container, and reported "no face" on every frame. That is a run outside CI,
        on a video with no face in it.
      - Pending: the same run on footage with a real face (see `docs/DEMO.md` §4).
- [x] Eval numbers generated (synthetic in CI; live-recording ones marked pending)
      - `eval/run_eval.py` produces them, and `scripts/update_readme.py` writes them into
        README.md (CI fails on drift).
      - Real-recording numbers, including PRD §5's 5-minute reading segment, are
        marked "not measured".
      - README also holds an MPIIFaceGaze table (real webcam photos, not a live
        session), which `scripts/update_readme.py` generates from the committed
        `eval/results/mpiifacegaze.json`. CI checks the README against that file but
        cannot rerun it: the dataset is never committed.
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
- **Review rounds 3-11 (2026-11-12).** The first fix for the half-closed-lid bug was
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
  - Round 10 found that a bystander's open eyes could end the user's closure, and
    that a dropped frame still counted toward the 0.4 s hold. Frames Gate 1 rejects
    for presence are now treated like a lost face, and a dropped frame restarts the
    closed run too.
  - Round 11 found that NaN eyelid points read as "closed", that frames with a bad
    timestamp didn't break an open run, and that a glimpse clearly on another point
    left the old point clickable by the next blink-hold. All three now fail closed.
  - The blink path is the part of EyeOS that most needs real recordings: each round
    found a new noise pattern, and none of the thresholds is tuned on real data.
  - Eye openness is now judged in each eye's own frame, so it holds at any head
    roll.
  - Further fixes in these rounds: MediaPipe's per-axis normalisation broke roll on
    16:9 frames; the iris axes flipped at 90 deg of roll; Ctrl+C during calibration
    was misreported; bad YAML gave a traceback; and README config values could go
    stale.
- Moving to a local machine (2026-11-17/18), at the owner's request: four pieces
  beyond the PRD's milestones, each built test-first by its own agent, reviewed
  adversarially by another, and fixed before merging.
  - `eyeos run --view`, a live tracking view: a marker on each pupil, the shift
    arrow that drives the cursor, a mini screen map and the gate lights.
  - `eyeos record`, a labelled session recorder for `eval/run_eval.py`, which now
    takes the recording's `--config`.
  - `eval/public/`, an MPIIFaceGaze harness: license-gated download, a landmark
    cache, the shipping calibration and 11 variants. It first ran on the real
    dataset on 2026-11-18 (see the constraints below).
  - `docs/LOCAL_SETUP.md` and the setup scripts for Windows and Linux/macOS.
  - The reviews found 20 issues (1 high, 8 medium, 11 low). All were fixed;
    what is still untested is listed below. One more fix came out of it: a config
    that repeats a key (a second pasted `gates:` section, which plain YAML would let
    silently drop the first one's no-click zones) is now refused.
- Still open: the owner-only items (tag, repo description and topics, branch
  protection), hardware verification and a real recorded session.
- The real MPIIFaceGaze run (2026-11-18): download, landmark extraction for all 15
  participants and evaluation of the 12 variants that existed then, results in
  `eval/results/mpiifacegaze.json` and `docs/PUBLIC_DATASETS.md` ("Real run"). The real
  files matched the format the parsers expected, so no code changed. The shipping
  calibration's mean error is 258 px (6.25°), and it fails the harness's Gate 2 check
  (a leave-one-out over 45 single photos, not the statistic live Gate 2 computes) for
  all 15 participants.
- Audit of that run (2026-11-18): an adversarial multi-agent audit found no geometry
  or units bug. It found stale and overstated text: the README still said the harness
  had never run on the real data, and the docs called the numbers "pessimistic for
  live use" and treated the harness's validation error as live Gate 2. Those texts
  were corrected, and the docs now also reason that `ridge-10` is close to a constant
  prediction and that EyeOS's head-pose "degrees" are uncalibrated. The audit also
  asked for a constant-prediction baseline, a live-style calibration variant and
  diagnostics (a same-day split, the dataset's own head pose). These were added after
  the run (the `centre` and `calibration-mean` baselines, `eyeos-live9`, and the
  `diagnose` command). A second run on the real data (2026-11-18, commit `b0f817a`)
  ran them: the 12 original variants reproduced bit for bit, `eyeos-live9` fails live
  Gate 2's statistic for all 15 people too, and the diagnostics point mainly at
  per-frame landmark noise in the iris offsets, with head motion and day-to-day drift
  secondary (`docs/PUBLIC_DATASETS.md`, "Second run").

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
- **Local setup** (`docs/LOCAL_SETUP.md`, `scripts/setup_windows.ps1`,
  `scripts/setup_unix.sh`, `scripts/check_setup.py`).
  - `setup_unix.sh` ran end to end in a fresh clone on Ubuntu 24.04 with Python 3.12.
    `check_setup.py` passed there with the real OpenCV 4.11 and MediaPipe 0.10.21.
  - `setup_windows.ps1` has never run: this container has no PowerShell, and running
    a downloaded one was not permitted. Its tests are static only (plain ASCII, the
    same extras and steps as the Unix script).
  - None of the guide's Windows, webcam or real-cursor steps has run. Their expected
    results come from the code and its tests.
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
- **Tracking view** (`tracking_view.py`, `eyeos run --view`). Its geometry is pure and
  unit-tested; its OpenCV window is tested against the fake `cv2` in
  `tests/fake_capture.py`, which records the drawing and window calls. `annotate()` has
  one optional smoke test with a real OpenCV.
  - During development the window also ran with OpenCV 4.11 (Qt backend) and MediaPipe
    0.10.21 on Xvfb, from ad-hoc scripts outside the repo: `eyeos run --view` on a
    generated face-free video and on the synthetic fixture, a real `q` keypress (XTest)
    and a real `WM_DELETE_WINDOW` close.
  - After review, with the same OpenCV on Xvfb: real arrow-key presses (XTest) leave
    the view open; `QT_QPA_PLATFORM=wayland` or `offscreen` is set to `xcb` and the run
    completes; a real `opencv-python-headless` 4.11 fails with a one-line error before
    calibration starts. With `$WAYLAND_DISPLAY` set and no X server at all, `--view`
    exits 2 with a message instead of aborting.
  - It has never run with a webcam, a real face, a window manager, a real Wayland
    desktop, GTK, macOS or Windows, or alongside the Tk windows (`--calibrate`,
    `--overlay`): no environment used here had tkinter and OpenCV together. Its cost
    per frame (a frame copy, the drawing, `imshow` and `waitKeyEx(1)`) was not
    measured, so how much it slows a live session is unknown.
- **Session recorder** (`recorder.py`, `eyeos record`). Tested against the fake tkinter
  and synthetic frames. Outside CI it also ran on real Tk 9.0 under Xvfb (no window
  manager) with a paced synthetic camera: all 21 dots were drawn on their labelled
  pixels and Esc left no file. The real `eyeos record` command, on a synthetic CSV
  source, exited 0, and the `run_eval` command it printed ran as printed. Without a
  window manager `-fullscreen` did nothing and the window got no keyboard focus, so the recorder also sets its geometry and calls `focus_force()`,
  and refuses to record unless its canvas covers the screen. Never run with a real
  webcam, on Windows, or on a physical screen.
- **Eval.** The README's main Results table comes from a generated synthetic landmark
  sequence (`eval/fixtures/`, made by `eval/generate_synthetic.py`). Its MPIIFaceGaze
  table comes from the committed `eval/results/mpiifacegaze.json`: real webcam
  photos, but not a live session (see the next point). Live-session numbers are
  marked "not measured", pending recordings, and so is the PRD's 5-minute reading
  false-click test. `eyeos record` now records a labelled session from a webcam
  (`docs/RECORDING.md`), but none has been recorded yet.
- **Public dataset (MPIIFaceGaze).** `eval/public/` runs the landmark pipeline and
  calibration on MPIIFaceGaze (`docs/PUBLIC_DATASETS.md`). The dataset's host was not
  reachable from the container it was built in, so it was built against a fake dataset
  with a fake detector. A later cloud session (2026-11-18) could reach the host and ran
  the real download, extraction and evaluation; the numbers are in
  `eval/results/mpiifacegaze.json` and `docs/PUBLIC_DATASETS.md` ("Real run"). A
  second session the same day reran it with the baselines, `eyeos-live9` and
  `diagnose` (`eval/results/mpiifacegaze-diagnostics.json`; "Second run"); the first
  run's numbers reproduced exactly. Anything added to the harness after commit
  `b0f817a` has not run on the real data. The
  automated tests still use only the fake dataset: the real one is non-commercial and
  never committed, so CI cannot run it.

Everything else runs and is tested in this container with no external hardware:
features, calibration, filter, gates, hotkey latch logic, the dry-run pipeline and the
eval on synthetic data.
