# DEMO.md — recording the real EyeOS demo

This cloud build container has no webcam, no display, and no Windows desktop, so the
live demo couldn't be recorded here (see `docs/PLAN.md`'s "Cloud-instance
constraints"). This is the script to record it on real hardware — should take about
5 minutes.

## What you need

- Windows, with a webcam.
- Python 3.10+.

## Setup (once)

```bash
git clone https://github.com/mit37/eyeos.git
cd eyeos
pip install -e ".[webcam,windows-fallback]"
copy config.example.yaml config.yaml
```

Edit `config.yaml`:

```yaml
input:
  source: webcam
output:
  backend: windows_sendinput
```

## 1. Calibrate (about 1 minute)

Run the calibration UI (fullscreen, 9 dots). Look at each dot and press space when
your gaze feels steady on it; press it 9 times in the printed dot order. Escape
cancels.

```bash
python -c "
from eyeos.calibration import Calibration
from eyeos.calibration_ui import CalibrationUI
from eyeos.config import load_config
from eyeos.input_sources.webcam import WebcamInputSource
from eyeos.features import extract_features

config = load_config('config.yaml')
cal = Calibration(config.screen.width_px, config.screen.height_px)
source = WebcamInputSource(device_index=config.input.device_index)
frames = source.frames()

def capture(target_px):
    frame = next(frames)
    if frame.landmarks is not None:
        cal.add_sample(target_px, extract_features(frame.landmarks))

ui = CalibrationUI(config.screen.width_px, config.screen.height_px, capture_sample=capture)
if ui.run():
    cal.fit()
    print('calibration validation error (px):', cal.validation_error_px)
"
```

(This is intentionally a short inline script, not a saved-calibration CLI flag — see
`docs/PLAN.md` for why persisting calibrations to disk was left out of this build.)

## 2. Run it live, with the gate overlay (about 2 minutes)

Show the four gate lights (green = passing) while you move your gaze around, look
away, and hold a blink to click:

Combine this with step 1 in one script (calibration isn't persisted to disk in this
build — see `docs/PLAN.md` — so `cal` needs to stay in memory between the two):

```bash
python -c "
from eyeos.config import load_config
from eyeos.input_sources.webcam import WebcamInputSource
from eyeos.output.windows_sendinput import WindowsSendInputOutput
from eyeos.overlay import GateOverlay
from eyeos.pipeline import Pipeline
# cal = ... (the Calibration fit in step 1)

config = load_config('config.yaml')
overlay = GateOverlay()
output = WindowsSendInputOutput()
pipeline = Pipeline(
    config, WebcamInputSource(), output, calibration=cal, on_status=overlay.update
)
pipeline.run()
"
```

Narrate for the recording:

1. Look around the screen — cursor follows your gaze (Gate 1/2/4 green, moving).
2. Look away from the camera / cover one eye — cursor freezes, Gate 1 light goes red.
3. Hold still on one spot for under a second, then dwell past it — Gate 3 light goes
   green once the dwell time is reached.
4. Hold a deliberate blink (or press a key, if you wired keypress confirm) on a
   stable, dwelled target — a click fires.
5. Trigger the kill hotkey (Esc by default) mid-movement — everything freezes
   immediately, Gate 4 light goes red.

## 3. Record

- **GIF** (CLI-adjacent tools, `vhs`/`asciinema`+`agg`, or a screen recording
  converted with `ffmpeg`): capture the overlay window during step 2's narration,
  ≤8 MB, 15-30s is plenty.
- Save it to `docs/demo.gif` and link it at the top of `README.md`, replacing the "No
  demo GIF yet" note.

## 4. Feed the recording back into the eval harness (optional but ideal)

If you also recorded the raw webcam video during step 2 (`ffmpeg` screen/webcam
capture, or OBS), you can run the *real* pipeline against it instead of a live
camera, and get real (not synthetic) eval numbers:

```yaml
# config.yaml
input:
  source: video_file
  video_path: recordings/session1.mp4
```

```bash
eyeos run --config config.yaml --log logs/session1.jsonl
```

The recorded video itself should **not** be committed (see `.gitignore` — `*.mp4` is
excluded). If you're comfortable with it, extract just the derived landmark CSV
(`eyeos.input_sources.video_file` + `eyeos.synthetic_format.landmarks_to_row` per
frame) and commit *that* instead, the same way `eval/fixtures/synthetic_session.csv`
was generated synthetically — that gives `eval/run_eval.py` a real-accuracy
counterpart to the synthetic numbers, without publishing your face.
