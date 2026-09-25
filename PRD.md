# PRD — EyeOS (v2 rebuild)

**Repo:** `mit37/eyeos`
**Description:** Webcam eye tracking as an input device: MediaPipe gaze moves the Windows cursor behind a four-gate safety model.
**Topics:** `eye-tracking`, `mediapipe`, `computer-vision`, `accessibility`, `hci`, `python`, `windows`
**Priority:** Tier 3 (Lab). Keep the scope tight, and make the safety model the interesting part.

---

## 1. Pitch

A regular webcam can estimate roughly where you're looking. That's too noisy to trust as a mouse, *unless* the system knows when not to act. EyeOS moves the cursor with your gaze, but only through **four safety gates**. Clicking requires deliberate dwell plus confirmation, and one hotkey kills it instantly.

## 2. The four gates (the core design)

| Gate | Passes when | Fails → |
|---|---|---|
| 1. Presence | exactly one face, detection confidence ≥ threshold, both irises visible | cursor frozen |
| 2. Calibration validity | a calibration exists for this session, its validation error < X px, head pose within ±Y° of the calibration pose | "recalibrate" prompt; cursor frozen |
| 3. Stability / intent | gaze is inside the same target region for the dwell time (default 800 ms) with low jitter | no click (movement allowed) |
| 4. Kill switch & limits | kill hotkey not pressed; cursor speed ≤ limit; clicks ≤ N per 10 s; never click inside a user-defined "no-click zone" (e.g. the taskbar close area) | all output stops |

Moving the cursor needs gates 1, 2 and 4. Clicking needs **all four** plus a confirmation (a blink-hold ≥ 400 ms *or* a keypress, configurable). Every gate decision is logged with its reason, and the overlay shows the four gate lights live.

## 3. Pipeline

Webcam (OpenCV) → MediaPipe Face Landmarker (iris landmarks) → features (iris center relative to the eye corners, both eyes; head pose from the landmarks) → **9-point calibration** → a per-user regression (ridge/polynomial) mapping features → screen coordinates → smoothing (One Euro filter) → gates → output (`SendInput` via ctypes on Windows; a `pynput` fallback on other OSes, documented as less tested).

## 4. Goals / non-goals

**Goals:** the pipeline above; a calibration UI (a fullscreen Tkinter/PySide6 dots screen); overlay with gate lights; config file; session logs.
**Non-goals:** medical/assistive certification claims (the README states it's an experiment, not an accessibility product); typing keyboards; multi-monitor calibration beyond the primary screen (future work).

## 5. Evaluation (generated)

- **Offline, from recorded sessions:** `eval/` takes a recorded video + the known on-screen target positions (Mitansh records 3 sessions of looking at the calibration/validation dots; the videos stay local, and only derived landmark CSVs are committed if he's comfortable).
- Metrics: mean angular/pixel error after calibration, error with the One Euro filter vs without, jitter, gate false-accept rate on "looking away" segments, and click false-trigger rate on a 5-minute "just reading" segment (target: 0).
- The cloud CI runs the eval on a small committed **synthetic landmark sequence** (generated), so the pipeline is tested without video.

## 6. Milestones

1. Scaffold, CI, config, an input-source interface (webcam | video file | synthetic landmarks).
2. Landmark → features + head pose (unit-tested on fixture landmarks).
3. Calibration + regression + validation error.
4. One Euro filter (tested against the reference behavior).
5. The four gates as a pure state machine with a fake clock (the most tests: every gate transition).
6. Output adapters (Windows SendInput + fallback) behind an interface; dry-run mode prints actions instead of moving the mouse.
7. Overlay + calibration UI.
8. Eval, README with gate diagram, `docs/DEMO.md`, tag v2.0.0.

## 7. Cloud-instance constraints

No webcam or Windows desktop in the cloud. Everything runs from synthetic/recorded inputs in dry-run mode; the live demo is recorded by Mitansh.

## 8. Definition of Done

- [ ] Gate state machine fully tested (≥40 tests), including the kill switch and no-click zones
- [ ] Dry-run mode runs the whole pipeline on a video file
- [ ] Eval numbers generated (synthetic in CI; real ones from Mitansh's recordings or marked pending)
- [ ] README states "experiment, not an assistive-tech product"; CI green; tag v2.0.0
