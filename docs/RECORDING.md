# RECORDING.md: real-data eval numbers from your own webcam

PRD §5 asks for eval numbers from real recordings. The README's numbers come from a
generated synthetic sequence, because the cloud build had no camera. `eyeos record`
records a labelled session on your own machine. `eval/run_eval.py` turns it into the
same metrics. Only face landmarks are saved, never video.

**Status.** `eyeos record` is tested in CI against a fake tkinter and synthetic frames.
Outside CI it has run on real Tk 9.0 under Xvfb, a virtual X display with no window
manager, from a separate Python 3.12 environment. A paced synthetic "camera" stood in
for the webcam. In that run all 21 dots were drawn on their labelled pixels, Esc left
no file, and the command it prints ran as printed. It has **not** run with a real
webcam, on Windows, or on a physical screen, so your first session is itself a
result. If something fails, note it in `docs/PLAN.md`.

## What you need

- The webcam setup from [`DEMO.md`](DEMO.md): Python 3.10–3.12 and
  `pip install -e ".[webcam]"`.
- A `config.yaml` with `input.source: webcam`. `screen.width_px` and
  `screen.height_px` must be the size your OS *reports* for the primary screen; with
  Windows display scaling, 1920x1080 at 125% reports 1536x864. `eyeos record` refuses
  a mismatch, like `eyeos run --calibrate`.
- About 3 minutes with the defaults, or 7 with a 5-minute reading stage.

## Record

From the repository root:

```bash
eyeos record --config config.yaml --out recordings/session1.csv
```

| Option | Default | What it does |
|---|---|---|
| `--dot-s` | 2.0 | Seconds each dot is shown. |
| `--settle-s` | 0.6 | Seconds at the start of each dot, and of the reading screen, left out of the eval while your eyes get there. |
| `--reading-s` | 60 | Seconds of reading. PRD §5's click false-trigger test uses 5 minutes: pass `--reading-s 300` to match it. |
| `--no-reading` | off | Skip the reading stage. |

`recordings/` is gitignored. `eyeos record` never replaces an existing file: pick a
new `--out` for each session. When it finishes it prints where the file went and the
command to evaluate it. Exit codes: 0 saved, 1 cancelled, 2 error.

## What each stage asks you to do

| Stage | On screen | What you do | Rows in the CSV |
|---|---|---|---|
| Intro | What follows, and a live camera check ("face found, eyes open") | Sit as you normally work; press **space** | Not recorded |
| Calibration | 9 dots, one at a time, 2 s each | Look at each dot until the next appears | `calibration`, with the dot as target |
| Validation | 12 more dots, between the calibration ones | Same | `on_target`, with the dot as target |
| Look-away 1 | Instructions; press **space**, then do it | Look away from the screen, at something to one side (turning your head is fine), until the beep | `looking_away`, `away_mode` `off_screen` |
| Look-away 2 | Same | Close your eyes until the beep | `looking_away`, `eyes_closed` |
| Look-away 3 | Same | Lean out of the camera's view until the beep | `looking_away`, `no_face` |
| Reading | Instructions, **space**, then a public-domain passage (*Alice's Adventures in Wonderland*, 1865, chapter I) | Read at your normal pace; start again from the top if you finish; it ends by itself | `just_reading`, no target |
| Done | Where the file was saved | Nothing: the window closes after 3 s (or press space) | |

Each look-away prompt gives you 2 s after space to do it, then records 4 s, then
rings the system bell. The bell is Tk's `bell()`, and whether you hear it depends on
your system's sound settings (not tested here). If your sound is off, count to ten
before looking back. Coming back late does no harm, because the next screen waits for
space.

Rows that aren't measured get segment `settle`, which `run_eval` ignores: the first
`--settle-s` of each dot and of reading, the 2 s look-away lead-ins, and screens
waiting for space. They stay in the file so the pipeline replays one unbroken stream.

**Esc**, or closing the window, cancels at any point before the done screen. Nothing
is written. If the camera stops mid-session, the window closes and `eyeos record`
exits with "camera stopped during recording" and nothing saved. If the camera can't
open at all, it fails before the window appears.

The recorder never moves the mouse or clicks. It has no output adapter, and the only
keys it reads are space and Esc in its own window.

## Privacy: what the file contains

Each row is one camera frame:

- the frame's timestamp;
- the face count and MediaPipe's detection-confidence floor;
- 15 named face points (irises, eye corners and lids, nose tip, chin, forehead, face
  edges), as x, y, z;
- the four label columns.

That is 52 columns in all. No camera image is kept, in memory or on disk. The rows
are held in memory until the session ends, then written to a temporary file next to
`--out` and renamed into place, so a failed write leaves no partial file.

Landmarks still describe your face's geometry. Commit a session CSV only if you're
comfortable with that (PRD §5); `recordings/` is gitignored by default.

## Evaluate it

Run this from the repository root, with the **same config** you recorded with:

```bash
python -m eval.run_eval --session recordings/session1.csv --config config.yaml \
    --out recordings/session1-results.json
```

The config matters: the pipeline clamps the cursor to `screen.width_px` x
`screen.height_px`, and the gates use your thresholds. Without `--config`, run_eval
uses the built-in 1920x1080 defaults.

What the numbers are, and what they are not:

- **`calibration_validation_error_px`** is a leave-one-out error over every
  calibration *frame*. Other frames of the same dot stay in the fit, so it tends to
  read lower (more optimistic) than the error `eyeos run --calibrate` reports, which
  averages 10 frames per dot and leaves out a whole dot at a time. The fit uses only frames the live
  calibration would use: one face, both eyes open.
- **`mean_pixel_error_raw_px`** and **`mean_pixel_error_filtered_px`** are averaged
  over the measured validation frames (`on_target`). The raw error includes any
  validation frame with a face, a blink included.
- **`looking_away_by_mode`** gives each prompt's frame count and how many frames
  still allowed cursor movement.
  - `no_face` is Gate 1's job.
  - `off_screen` is Gate 2's job only if your head turns more than 15° from its
    calibration pose. An eyes-only glance passes every gate by design (a known
    negative result, see the README).
  - `eyes_closed` is Gate 1's job once the blink-hold grace window has run out.
    That takes about 0.9 s in the default confirmation mode: 0.4 s hold plus 0.5 s
    grace. Close your eyes within about a second of pressing space and the grace
    window ends before the 2 s lead-in does.
- **`gate_false_accept_rate`** aggregates only the Gate 1 modes, which for a
  recording means `no_face`.
- **`click_false_trigger_rate`** comes from the reading stage. The eval replays the
  session without a keyboard, so only blink-hold clicks can fire. In
  `confirmation_mode: keypress` it is 0 by construction.
- **`mean_angular_error_deg`** is not measured. That needs the viewing distance and
  the screen's physical size, which a session doesn't record.
- The JSON's **`notes`** block was written for the committed synthetic fixture. Its
  remarks about 10 fps, the 30 s reading stand-in and injected noise don't describe
  a recording.

Report what comes out, including bad numbers (STANDARDS §1.4). Keep the command you
ran next to the numbers.

## If something goes wrong

- **"config screen.width_px/height_px is ... but the screen reports ..."**: set them
  to the reported size.
- **"the recording window does not cover the screen"**: run it on the primary screen.
  An always-on-top taskbar, or a window manager that ignores fullscreen requests, can
  cause this. Each row's target is a screen position, so the recorder refuses to
  record when the dots wouldn't be drawn where the file says.
- **"could not open webcam device 0"**: change `input.device_index`.
- **"input.source is 'synthetic', not 'webcam'"**: that warning means you're
  replaying a file, which can't follow the dots. Use it only to try the recorder out.
- **"... already exists"**: pass a new `--out`.

## Not tested

- A real webcam and real MediaPipe landmarks with the recorder. Real numbers stay
  "not measured" in the README until a session is recorded and evaluated.
- Windows, macOS, a physical screen, and Tk 8.6 (the version Windows' Python ships).
  The real-Tk run was Tk 9.0 on Linux under Xvfb.
- Focus: the window calls `focus_force()` so space and Esc reach it. Under Xvfb that
  was needed; whether Windows ever withholds focus from it is untested.
- Whether the bell is audible.
- How well the fixed 0.6 s settle window and 2 s lead-in match real reaction times.
