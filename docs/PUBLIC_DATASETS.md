# Public-dataset evaluation

EyeOS's own eval (`eval/run_eval.py`) runs on a synthetic landmark fixture. This harness
runs the same landmark pipeline and calibration on a public dataset of real webcam
photos, so the shipping calibration can be measured on real faces, and alternative
models can be refitted and compared on the same data with one command each.

## Status: what has and has not been run

- **Supported dataset:** MPIIFaceGaze only.
- **Tested:** everything under `tests/test_public_mpiifacegaze.py`, on a small fake
  dataset written to a temporary folder. Its "images" are small JSON files that a fake
  detector turns into landmarks, so no face images are involved, and OpenCV and
  MediaPipe are not needed. Two optional tests use the real OpenCV and MediaPipe when
  they are installed (Rodrigues against `cv2.Rodrigues`, and no face found in a blank
  image) and skip otherwise. The downloader is tested against a fake server; no test
  uses the network.
- **Smoke-tested with the real libraries** (OpenCV 4.11, MediaPipe 0.10.21, Python 3.11,
  Linux): `extract` and `evaluate` ran end to end on four blank generated JPEGs. MediaPipe
  found no face in them, as expected, and `evaluate` reported every person as skipped.
- **Run on the real dataset (2026-11-18):** the real download, landmark extraction
  for all 15 participants and evaluation of every variant, in a cloud container. The
  harness was built where the dataset's host was unreachable; a later session could
  reach it. Results are in `eval/results/mpiifacegaze.json`, and the run, with an
  honest reading of the numbers, is described in "Real run" below.
- **The expected file format was checked against the real files** on that run, and
  matched (see "Expected format" and "Real run"). No parser changed. The readers still
  fail loudly, naming the file and showing the offending line or keys, if a copy of the
  data differs.
- **The automated tests still use only the fake dataset.** The real data is
  non-commercial and never committed, so CI cannot run it.
- **Windows:** the downloader uses only the Python standard library so that it can run
  on Windows. That is untested: it has only run on Linux, against the fake server.

## MPIIFaceGaze

15 participants (`p00` … `p14`), about 3000 photos each from their own laptop webcam,
each labelled with the on-screen point the person was looking at, plus 3D labels (face
centre and gaze target in camera coordinates).

**License: CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0,
<https://creativecommons.org/licenses/by-nc-sa/4.0/>): non-commercial use only, credit the
authors, and share anything derived from the data under the same license. The photos
show real people's faces. They, and the landmark caches derived from them, live in
`datasets/`, which is gitignored; never commit them.

**Cite:**

- X. Zhang, Y. Sugano, M. Fritz, A. Bulling. "It's Written All Over Your Face: Full-Face
  Appearance-Based Gaze Estimation." CVPR Workshops (CVPRW), 2017.
- X. Zhang, Y. Sugano, M. Fritz, A. Bulling. "Appearance-Based Gaze Estimation in the
  Wild." CVPR, 2015.

### Commands

```bash
pip install -e ".[public-eval]"   # scipy, OpenCV, mediapipe<0.10.30 (Python 3.10-3.12)

# 1. Download (prints the license and citation; refuses without --accept-license, exit 2).
#    Streams with progress, resumes a cut-off download, CRC-checks the zip, extracts to
#    datasets/MPIIFaceGaze and fails unless a pXX/pXX.txt came out. A rerun skips what
#    is done.
python scripts/download_mpiifacegaze.py --accept-license [--dest datasets]
#    (scripts/download_mpiifacegaze.sh is a thin wrapper around the same script)

# 2. Extract MediaPipe landmarks once (slow; cached, restartable, parallel by person).
python -m eval.public.mpiifacegaze extract \
    --root datasets/MPIIFaceGaze --cache datasets/cache/mpiifacegaze --workers 4

# 3. Evaluate (fast; rerun as often as you like).
python -m eval.public.mpiifacegaze evaluate \
    --root datasets/MPIIFaceGaze --cache datasets/cache/mpiifacegaze \
    --out eval/results/mpiifacegaze.json
```

Useful options:

- `extract --persons p00,p01` limits the run to some participants.
- `extract --limit N` takes the first N images per participant (a quick trial run). A
  later run without `--limit` redoes that participant in full. A `--limit` run after a
  full one keeps the full cache: without `--force`, a cache is never replaced by a
  shorter one.
- `extract --force` redoes participants that are already cached (with `--limit`, it
  replaces a full cache with the first N images).
- `evaluate --variants eyeos,with-head-pose,global-lopo` compares only some models.
- `evaluate --persons ...` and `evaluate --k-per-dot K` are also available.

`evaluate` prints a short table and writes the full results JSON.

### Expected format

The layout, from the dataset's `readme.txt`, checked against the real files on
2026-11-18 (all 37,667 lines have 28 fields; see "Real run" for details).

| File | Content |
|---|---|
| `pXX/pXX.txt` | one line per image, 28 whitespace-separated fields: `[0]` image path relative to `pXX/` (e.g. `day01/0005.jpg`); `[1:3]` gaze target on screen in pixels; `[3:15]` six 2D facial landmarks; `[15:18]` head rotation (Rodrigues); `[18:21]` head translation; `[21:24]` face centre `fc` (camera coordinates, mm); `[24:27]` 3D gaze target `gt` (camera coordinates, mm); `[27]` `left` or `right` |
| `pXX/Calibration/screenSize.mat` | `width_pixel`, `height_pixel`, `width_mm`, `height_mm` |
| `pXX/Calibration/monitorPose.mat` | `rvects`, `tvecs`: screen to camera (`rvecs` is accepted too) |
| `pXX/Calibration/Camera.mat` | `cameraMatrix`, `distCoeffs`: not used |

The dataset's own landmarks and head pose (fields 3 to 20) are parsed but not used:
EyeOS gets its landmarks from MediaPipe and estimates its own head pose, as it does live.

## What `extract` does

For each image: `cv2.imread`, then MediaPipe's legacy
`solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
refine_landmarks=True)` with Gate 1's confidence threshold, then
`landmarks_from_mediapipe(..., aspect=width/height)`, the same conversion the live
webcam source uses. Each participant becomes one CSV in `eyeos.synthetic_format`'s
columns plus `image`, `target_x/y`, `fc_x/y/z` and `gt_x/y/z`.

A CSV is written under a temporary name and renamed when complete, so an interrupted run
never leaves a cache that looks finished. Without `--force`, a participant is skipped
when their cache starts with the requested images (it holds exactly them, or more), and
a cache is never replaced by a shorter one; the run prints that the cache was kept. A
cache that is shorter than the request, or holds other images, is redone.

## What `evaluate` does (the protocol)

The protocol (`eval/public/protocol.py`) treats each participant the way a live EyeOS
session would treat a user, as far as still photos allow:

1. **Usable frames.** A frame counts only if Gate 1 could pass it: exactly one face and
   both irises visible. Detection rate = usable / all images.
2. **Calibration set.** The screen is the participant's own (`screenSize.mat`). For each
   of EyeOS's 9 calibration dots (`eyeos.calibration.target_points` on that screen), the
   K usable images (default 5) whose true target is nearest the dot stand in for
   "looking at the dot". No image is used twice. They are added as calibration samples
   with their **true** targets. The mean distance from those images' targets to their
   dot is recorded.
3. **Fit.** `eyeos.calibration.Calibration` with its shipping defaults (or a variant).
4. **Test set.** Every other usable image. Each prediction is rounded and clamped to the
   screen exactly as `Pipeline._to_screen_pixel` does for the cursor (a test checks the
   two agree).
5. **Metrics per participant.**
   - Pixel error: mean, median and p90 in pixels, and as a % of the screen diagonal.
   - Angular error in degrees: the angle between `(P_pred - fc)` and `(gt - fc)`, with
     `P_pred` mapped to camera coordinates by the screen model (next section).
   - The calibration's validation error (Calibration's leave-one-out RMS) and whether it
     passes Gate 2 (`< gates.calibration_max_error_px`, 60 px by default).
   - The share of test frames whose head pose is within Gate 2's limit
     (`calibration_max_head_pose_delta_deg`, 15° by default) of the calibration pose.
6. **Across participants.** For pixel and angular error: the mean of the per-person means
   (each person weighs the same), the median and p90 of the per-person means, and the
   median and p90 of all test frames pooled. Also the number of persons that pass
   Gate 2, the mean detection rate and the mean head-pose share.

Everything is deterministic: rerunning on the same cache gives the same JSON, apart from
the date and the git commit.

### The screen-model check

Angular error needs the predicted screen pixel in camera coordinates. For each
participant, every annotated target is mapped with
`gt' = R(rvects) @ [x_px * width_mm / width_pixel, y_px * height_mm / height_pixel, 0] + tvecs`
(Rodrigues in numpy) and compared with the file's `gt`. The median `|gt' - gt|` in mm is
reported per participant.

- Within 5 mm: that model is used.
- Otherwise, mirrored x and/or y axes, a transposed R, a negated t and the inverse pose
  (`inverse-pose`: `gt' = R^T (p - t)`, in case `monitorPose.mat` holds camera to screen
  rather than screen to camera) are tried in every combination (32 in all), and the best
  one is used if it is within 5 mm. The JSON records which one and every candidate's
  median error.
- If none fits, that participant's angular error is `null`, with the reason.

### Limits of this protocol

- **Static photos, not live video.** The One Euro filter, dwell (Gate 3), blink
  handling and click confirmation are not exercised. Every error is one frame's raw
  calibration output, rounded and clamped like the cursor. Live jitter, latency and
  click safety need a live recording (`eyeos record`).
- **MediaPipe runs in static-image mode**, detecting every photo from scratch. Live,
  EyeOS runs it in tracking mode, which can behave differently on the same face.
- **The screens are not the user's.** Each participant used their own laptop screen, so
  pixel errors are in that screen's pixels; compare across screens with the % of the
  diagonal. Gate 2's 60 px threshold is a fixed number of pixels, which is a larger share
  of a small laptop screen than of a 1920x1080 monitor.
- **The K nearest images only approximate looking at a calibration dot.** Nobody looked
  at EyeOS's dots: the calibration images were taken while the participant looked at
  points near them (`calibration_mean_distance_px` says how near). Their true targets are
  used, so that offset does not bias the fit, but the spread of the calibration points
  differs from a real 9-dot session.
- **One calibration across many sittings.** The images come from many days (`dayNN`
  folders), so a calibration fitted on some of them is tested on photos from other
  sittings, lighting and postures. A live calibration is used in the same sitting.
- **EyeOS's head pose is a simplified estimate** (`eyeos.features`), not the dataset's
  head pose, so the head-pose share measures what Gate 2 would see, not true pose.
- **Variants have slightly different test sets.** A variant with more images per dot
  holds more images out for calibration.
- **Nothing here changes EyeOS.** The variants are experiments for comparison. Changing
  a shipping default is a separate decision, made in `eyeos/` with its own tests.

## Variants: refit and compare on the same data

`--variants` (default: all) picks models from `eval/public/models.py`. All are
closed-form least squares, so they are deterministic.

| Variant | Model |
|---|---|
| `eyeos` | shipping: `eyeos.calibration.Calibration` defaults (affine in the 4 iris offsets, ridge 1e-3) |
| `eyeos-poly2` | `Calibration(degree=2)`: adds squares and pairwise products |
| `ridge-0.001`, `ridge-0.1`, `ridge-1`, `ridge-10` | `Calibration(ridge_lambda=...)`; `ridge-0.001` restates the default |
| `per-dot-1`, `per-dot-5`, `per-dot-10`, `per-dot-20` | shipping model on K images per dot; `per-dot-5` restates the default K |
| `with-head-pose` | experiment: ridge (1e-3, standardised features) on the iris offsets plus EyeOS's yaw, pitch and roll |
| `global-lopo` | experiment: leave-one-person-out ridge trained on every other participant's usable frames (iris offsets plus head pose to screen fractions), then a per-user affine correction fitted on the participant's own 9-dot calibration set |

Models that are not a `Calibration` report the same leave-one-out RMS as
`Calibration.validation_error_px`, so Gate 2 judges them alike. The ridge strength of
the two experiments was fixed before any results existed and has not been tuned.

To add a variant, write a class with `name`, `description`, `k_per_dot` (None = the
run's default), `needs_other_persons` and
`fit(features, targets_px, context) -> model`, where the model has `predict(features)`
returning screen points and a `validation_error_px`. Register it in `build_variants()`,
and add a test in `tests/test_public_mpiifacegaze.py`. A variant that sets
`needs_other_persons` receives the other participants in `context.other_persons`; the
protocol never includes the participant being evaluated.

## The results JSON

`--out` gets: `dataset`, `license`, `citation`, `generated` (UTC date and time),
`git_commit` and `git_dirty`, `config` (paths, persons, variants, K, the Gate 2
thresholds, the screen-model tolerance), `screen_model` (per person), `detection` (per
person and in aggregate), `variants` (for each: description, K, `aggregate`, and
`persons` with every per-person metric, or `skipped` with the reason), and `notes` that
restate the protocol's limits next to the numbers.

## Real run (2026-11-18)

The first run on the real MPIIFaceGaze. Every number here is in
`eval/results/mpiifacegaze.json`, which the commands below regenerate.

### Setup

- **Date:** 2026-11-18. **Code:** commit `d3661ee` (the JSON's `git_commit`;
  `git_dirty` false). No code changed during the run.
- **Machine:** cloud container, 4 CPUs (Intel Xeon @ 2.10 GHz), no GPU, Linux.
- **Libraries:** Python 3.12.3, mediapipe 0.10.21, opencv-contrib-python 4.11.0.86,
  numpy 1.26.4, scipy 1.17.1. pip resolved these by itself from
  `pip install -e ".[dev,public-eval]"`; no pins were needed.
- **Data:** `MPIIFaceGaze.zip`, 942.3 MB, 38,278 files; 15 participants, 37,667
  annotated images (1,498 to 2,929 each, not "about 3000" each).

### Commands and wall times

```bash
python3.12 -m venv venv && . venv/bin/activate
pip install -e ".[dev,public-eval]"
python scripts/download_mpiifacegaze.py --accept-license --dest datasets
# trial: p00, first 200 images
python -m eval.public.mpiifacegaze extract --root datasets/MPIIFaceGaze \
    --cache datasets/cache/mpiifacegaze --persons p00 --limit 200
python -m eval.public.mpiifacegaze evaluate --root datasets/MPIIFaceGaze \
    --cache datasets/cache/mpiifacegaze --persons p00 --out /tmp/trial.json
# full run
python -m eval.public.mpiifacegaze extract --root datasets/MPIIFaceGaze \
    --cache datasets/cache/mpiifacegaze --workers 4
python -m eval.public.mpiifacegaze evaluate --root datasets/MPIIFaceGaze \
    --cache datasets/cache/mpiifacegaze --out eval/results/mpiifacegaze.json
```

The download took 4 min 22 s. The 200-image trial extraction took 2.4 s. The full
extraction took 1 min 40 s wall time (5 min 13 s CPU) for 37,667 images with 4
workers, about 375 images a second in all. `evaluate` (all 12 variants) took 6.5 s, and a
second run gave a JSON identical to the first apart from `generated`.

### Format: what the real files hold

Nothing broke a parser. What we saw, for the next person:

- `pXX/pXX.txt`: every one of the 37,667 lines has exactly 28 whitespace-separated
  fields, `\n` line endings, `left`/`right` in the last field. The on-screen target and
  the six 2D landmarks are whole numbers; the 3D values have six decimals.
- `monitorPose.mat` holds `rvects` and `tvecs`, each (3, 1) float64. The dataset's own
  `readme.txt` calls the first one `rvecs`; the files say `rvects`, the name the parser
  tries first (it accepts both).
- `screenSize.mat`: `width_pixel` and `height_pixel` are stored as uint16 (1, 1),
  `width_mm` and `height_mm` as float64 (1, 1). The parser reads both as floats.
- `Camera.mat` also holds `retval` and 100 `rvecs`/`tvecs` (the camera calibration's own
  views), besides `cameraMatrix` and `distCoeffs`. Unused.
- The dataset root also holds a `readme.txt` (ignored: only `pXX` folders are read).
- Every image is 1280x720. The screens are 1280x800 (6 people), 1440x900 (8 people),
  both 286.5 x 179.0 mm, and 1680x1050 at 331.7 x 207.3 mm (p06).

### Detection and the screen model

- **Detection rate:** 37,575 of 37,667 images usable (exactly one face, both irises
  visible): 99.68 % on average per person. 67 images had no face (58 of them p12's,
  9 p07's) and 25 had a face but irises not visible. No image had more than one face.
  p12 is lowest at 96.4 %; 8 participants are at 100 %.
- **Screen model:** the documented model (`default`: `R(rvects) @ [x_mm, y_mm, 0] +
  tvecs`) reproduces the annotated 3D gaze target for all 15 participants, with a
  median residual of at most 5e-7 mm (rounding of the file's six decimals). No
  alternative candidate was needed, and every participant has an angular error.

### Results (all 15 participants scored for every variant)

"Mean", "median" and "p90" are over the 15 per-person means (each person weighs the
same). Pooled values are over all test frames.

| Variant | mean px | median px | p90 px | mean ° | median ° | p90 ° | pooled median px | % diag | Gate 2 |
|---|---|---|---|---|---|---|---|---|---|
| eyeos (shipping) | 258.4 | 257.7 | 324.0 | 6.25 | 6.00 | 7.61 | 230.5 | 15.8 | 0/15 |
| eyeos-poly2 | 259.2 | 256.6 | 324.6 | 6.26 | 5.99 | 7.59 | 230.7 | 15.8 | 0/15 |
| ridge-0.001 | 258.4 | 257.7 | 324.0 | 6.25 | 6.00 | 7.61 | 230.5 | 15.8 | 0/15 |
| ridge-0.1 | 277.7 | 269.4 | 336.0 | 6.73 | 6.46 | 8.17 | 265.2 | 17.0 | 0/15 |
| ridge-1 | 363.7 | 366.0 | 419.0 | 8.88 | 8.63 | 10.24 | 366.0 | 22.2 | 0/15 |
| ridge-10 | 428.8 | 433.0 | 469.7 | 10.56 | 10.24 | 11.83 | 429.4 | 26.1 | 0/15 |
| per-dot-1 | 299.6 | 297.5 | 350.5 | 7.28 | 6.88 | 9.83 | 267.0 | 18.3 | 0/15 |
| per-dot-5 | 258.4 | 257.7 | 324.0 | 6.25 | 6.00 | 7.61 | 230.5 | 15.8 | 0/15 |
| per-dot-10 | 253.6 | 249.0 | 319.9 | 6.13 | 5.85 | 7.48 | 227.5 | 15.5 | 0/15 |
| per-dot-20 | 250.6 | 247.8 | 316.8 | 6.07 | 5.75 | 7.40 | 224.7 | 15.3 | 0/15 |
| with-head-pose | 223.7 | 232.1 | 270.4 | 5.37 | 5.32 | 6.26 | 197.7 | 13.6 | 0/15 |
| global-lopo | 219.9 | 220.1 | 268.6 | 5.28 | 5.04 | 6.32 | 196.6 | 13.4 | 0/15 |

For the shipping `eyeos` model, the per-person mean ranges from 182 px (p02, 4.50°)
to 348 px (p13, 9.20°). The outliers are p13 (348 px, 9.20°, and only 65 % of its test
frames within Gate 2's head-pose limit of the calibration pose, the lowest of anyone),
p07 (330 px, 6.43°) and p08 (315 px, 7.68°, 75 % within the head-pose limit). p13 is also
where the head-pose variants help most: `with-head-pose` 218 px and `global-lopo` 219 px.
The calibration's leave-one-out validation error averages 319 px; Gate 2 needs < 60 px.

### Reading the results honestly

- **The shipping calibration would not pass Gate 2 for any participant.** Its
  validation error is 224 to 422 px (3.7 to 7 times the 60 px limit). A live EyeOS session
  on these photos would refuse to start control, which is the gate doing its job, not
  a pass.
- **About 6° (258 px, 16 % of the screen diagonal) is too coarse for clicking.** On a
  1440x900 laptop screen that is a radius of roughly a fifth of the screen width: enough
  to tell which region of the screen someone looks at, not to hit a button.
- **This setup is harder than a live session, so these numbers are pessimistic for
  live use, by an amount this run cannot measure.** One calibration is fitted on images
  from many days and postures and tested on all the others; MediaPipe runs from scratch
  on each still photo; there is no smoothing. A live session calibrates and runs in one
  sitting. Only a recorded live session (`eyeos record`) can say how much better it is.
- **Head pose matters.** Adding EyeOS's own yaw, pitch and roll (`with-head-pose`)
  cuts the mean error by 13 % (258 to 224 px), and learning from the other 14 people
  first (`global-lopo`) by 15 % (to 220 px). The gain is biggest for the people whose
  head moved most (p13, p04, p08). Neither variant brings anyone under Gate 2.
- **More calibration images help a little; more model capacity and more
  regularisation do not.** 20 images per dot instead of 5 gains 3 % (251 px), 1 image
  per dot loses 16 % (300 px). The quadratic model is no better than the affine one.
  Stronger ridge makes things steadily worse (ridge 10: 429 px).
- **Nothing was tuned on this data.** Every variant, threshold and the ridge strengths
  were fixed before this run; this is a single evaluation, not a model selection.
  Changing EyeOS's shipping model based on it (for example adding head pose) would be a
  separate decision, made in `eyeos/` with its own tests.
- **Not measured here:** how the error compares with published appearance-based
  methods on this dataset (their protocols differ, and no such number is produced by
  this repo), and any constant-prediction baseline (no such variant exists yet).

## Adding another dataset

1. A downloader `scripts/download_<name>.py` (plus a `.sh` wrapper), modelled on
   `scripts/download_mpiifacegaze.py`: print the license, refuse without
   `--accept-license`, write only under the gitignored `datasets/`. For datasets that
   need registration, have it check for the files the user downloaded by hand instead.
2. A module `eval/public/<name>.py` that parses the dataset defensively (fail loudly
   with the file and the offending line) and builds one `protocol.PersonData` per
   participant: the screen size in pixels, the true on-screen targets, one
   `FaceLandmarks` (or None) per image and, if the dataset has them, the face centres,
   3D gaze targets and a pixel-to-camera function for angular error. Reuse
   `MediaPipeImageDetector` and the cache format from `mpiifacegaze.py`, so landmarks
   are produced exactly as live.
3. Call `protocol.evaluate(persons, variants)` and add the dataset's metadata to the
   JSON, as `mpiifacegaze.evaluate_dataset` does.
4. Tests on a fake dataset written to `tmp_path`, with an injected fake detector, and no
   network.
5. A section here: license, citation, commands, and what the protocol cannot measure on
   that dataset.

## Other datasets (not supported yet)

You have to get each of these yourself: register, request access or accept a license
with their authors. None has a loader here, and nothing in this repo has been run on any
of them. Read each license when you get access; do not assume it allows commercial use.

| Dataset | What it is | What a loader would need |
|---|---|---|
| ETH-XGaze | high-resolution lab photos from many synchronised cameras, with large head poses | its standard labels are gaze directions; a loader would have to recover on-screen targets from the setup's geometry, and lab cameras are far from a laptop webcam |
| GazeCapture | photos from phone and tablet front cameras, crowdsourced | targets are given relative to the camera, on small handheld screens; the 9-dot protocol would need adapting to each device's screen |
| EYEDIAP | RGB and depth video of people following screen targets and floating 3D targets | it is video, so it could also exercise the One Euro filter; a loader would have to read the video and its target annotations |
| Columbia Gaze | still photos at fixed head poses and fixed gaze directions | labels are discrete gaze angles, not screen points; it fits a "looking at the camera or not" check better than screen calibration |
| Gaze360 | gaze in the wild, all around the camera, often far from it | no screen at all, and many faces are small or turned away; little of it resembles someone sitting at a laptop |
| UnityEyes | synthetic eye-region images from a renderer you download and run | eye crops only, with no full face, so MediaPipe's face mesh cannot run on them; it would need a different landmark path |
