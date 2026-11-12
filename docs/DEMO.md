# DEMO.md — recording the real EyeOS demo

This cloud build container has no webcam, no display, no keyboard and no Windows
desktop, so the live demo couldn't be recorded here (see `docs/PLAN.md`'s
"Cloud-instance constraints"). This is the script to record it on real hardware, and
it should take about 5 minutes. None of these steps has been run on real hardware
yet, so if one fails, that failure is itself worth recording in `docs/PLAN.md`.

## What you need

- Windows, with a webcam.
- **Python 3.10–3.12.** The `webcam` extra pins `mediapipe<0.10.30`, the last line
  with the legacy Solutions API, and those wheels stop at 3.12.

## Setup (once)

```bash
git clone https://github.com/mit37/eyeos.git
cd eyeos
pip install -e ".[webcam,windows-fallback]"
copy config.example.yaml config.yaml
```

Edit `config.yaml` (it is gitignored):

```yaml
screen:
  width_px: 1920   # the size Windows *reports* for the primary screen, after display
  height_px: 1080  # scaling (1920x1080 at 125% reports 1536x864); eyeos refuses a mismatch
input:
  source: webcam
  device_index: 0  # change if Windows picks the wrong camera
output:
  backend: windows_sendinput
  kill_hotkey: esc
  confirm_key: space   # optional: hold to confirm a click
```

## 1. Calibrate and run, in one command

```bash
eyeos run --config config.yaml --calibrate --overlay
```

1. A fullscreen screen shows 9 dots, one at a time. Look at each dot and press
   **space** when your gaze is steady. If no face or open eyes are found, the same
   dot is asked for again. **Esc** or closing the window cancels: `eyeos run` prints
   "calibration cancelled" and exits without starting the session. If the camera or
   MediaPipe can't start, `eyeos run` stops with that error before any dot is shown;
   if the camera stops mid-calibration, it stops with "camera stopped during
   calibration".
2. The terminal prints the calibration's validation error. Gate 2 refuses to move the
   cursor if it is 60 px or more (`gates.calibration_max_error_px`).
3. The session starts with the gate-lights overlay. Every frame's gate decisions go
   to `logs/session-<UTC time>.jsonl`. Closing the overlay window only hides the
   lights: the session keeps running until you press **Esc**.

If no keyboard listener can start (pynput missing, or no input permission), `eyeos
run` refuses to drive the cursor at all, by design.

## 2. Narrate while recording (about 2 minutes)

1. Look around the screen. The cursor follows your gaze, and Gates 1, 2 and 4 are green.
2. Look away from the camera or cover one eye. The cursor freezes and the Gate 1
   light goes red.
3. Hold your gaze on one spot. The Gate 3 light goes green once the dwell time (0.8 s)
   is reached.
4. On a dwelled target, hold a deliberate blink for at least 0.4 s, or press the
   confirm key. A click fires where you were looking.
5. Press **Esc** (the kill hotkey) mid-movement. The Gate 4 light goes red, nothing
   more is emitted, and the session ends: the terminal prints "kill switch (esc)
   pressed: session ended".

Expect to struggle with step 3 if the gaze is noisy. The synthetic eval suggests
the 20 px dwell radius (`gates.dwell_jitter_px`) may be too tight for webcam noise
(see README's Results). Note what you see; that is a real result.

## 3. Record

- **GIF**: capture the overlay window during step 2's narration with a screen
  recorder (convert with `ffmpeg`), ≤8 MB; 15–30 s is plenty.
- Save it to `docs/demo.gif` and link it at the top of `README.md`, replacing the
  "No demo GIF yet" note.

## 4. Replaying a recording (optional)

If you also saved the raw webcam video, you can replay the real pipeline over it:

```yaml
# config.yaml
input:
  source: video_file
  video_path: recordings/session1.mp4
output:
  backend: dry_run
```

```bash
eyeos run --config config.yaml
```

This gives you the gates' decisions on real footage, in the session log. It does
**not** yet give accuracy numbers. `eval/run_eval.py` needs each frame's true
on-screen target (the `target_x`, `target_y` and `segment` columns in
`eval/fixtures/synthetic_session.csv`), and a plain recording doesn't have them. A
real-accuracy eval needs footage taken while known targets are shown, plus a
labelling step, and that is not built yet. Replaying without `--calibrate` also
means Gate 2 never passes, because calibrations aren't saved between sessions.

Recordings must **not** be committed: `.gitignore` excludes `recordings/`, `*.mp4`,
`*.mov`, `*.mkv` and `*.webm`. If you're comfortable with it, commit only derived
landmark CSVs, which don't contain your face.
