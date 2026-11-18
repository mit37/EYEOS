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
- **Run on the real dataset once (2026-11-18):** the real download, landmark extraction
  for all 15 participants and evaluation of the 12 variants that existed then, in a
  cloud container. The harness was built where the dataset's host was unreachable; a
  later session could reach it. Results are in `eval/results/mpiifacegaze.json`, and
  the run, with an honest reading of the numbers, is described in "Real run" below.
  The variants and the `diagnose` command added since have not run on the real data.
- **The expected file format was checked against the real files** on that run, and
  matched (see "Expected format" and "Real run"). No parser changed. The readers still
  fail loudly, naming the file and showing the offending line or keys, if a copy of the
  data differs.
- **The automated tests still use only the fake dataset.** The real data is
  non-commercial and never committed, so CI cannot run it.
- **Windows:** the downloader uses only the Python standard library so that it can run
  on Windows. That is untested: it has only run on Linux, against the fake server.

## MPIIFaceGaze

15 participants (`p00` … `p14`), 1,498 to 2,929 photos each (37,667 in all) from their
own laptop webcam, each labelled with the on-screen point the person was looking at, plus
3D labels (face centre and gaze target in camera coordinates).

**License: CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0,
<https://creativecommons.org/licenses/by-nc-sa/4.0/>): non-commercial use only, credit the
authors, and share anything derived from the data under the same license. The photos
show real people's faces. They, and the landmark caches derived from them, live in
`datasets/`, which is gitignored; never commit them. Only metrics computed from the
data are committed: `eval/results/mpiifacegaze.json` and the numbers quoted from it in
README.md and this document, all shared under the dataset's license (see "The results
JSON").

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
   - The calibration's validation error (Calibration's leave-one-out RMS over the
     single calibration photos) and whether it is under Gate 2's limit
     (`< gates.calibration_max_error_px`, 60 px by default). Live Gate 2 applies that
     limit to a different leave-one-out; see "Limits of this protocol".
   - The share of test frames whose head pose is within Gate 2's limit
     (`calibration_max_head_pose_delta_deg`, 15° by default, in EyeOS's head-pose units;
     see "Limits of this protocol") of the calibration pose.
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
- **The Gate 2 figure is not the statistic live Gate 2 computes.** Here the validation
  error is Calibration's leave-one-out over the 45 single calibration photos (K = 5 per
  dot), each added at its true target; when one photo is held out, the other 4 near its
  dot stay in the fit. Live, Gate 2 judges the leave-one-out over 9 samples, each the
  average of up to 10 frames, added at the dot (`eyeos/calibration_session.py`). The two
  respond to noise differently and can disagree in either direction, so the "0/15"
  below is 0 of 15 under the 45-photo statistic, not a measured live refusal. A
  live-style variant, `eyeos-live9`, was added to the harness after this run (see
  "Variants"); it has not run on the real data yet.
- **One calibration across many sittings.** The images come from many days (`dayNN`
  folders), so a calibration fitted on some of them is tested on photos from other
  sittings, lighting and postures. A live calibration is used in the same sitting.
- **EyeOS's head pose is a simplified, uncalibrated estimate** (`eyeos.features`), not
  the dataset's head pose. Yaw and pitch are nose-offset ratios multiplied by fixed,
  hand-picked constants (`_YAW_SCALE_DEG`, `_PITCH_SCALE_DEG`), so their "degrees" are
  EyeOS's own measure. Roll is in real degrees: the tilt, in the image, of the line
  between the eye centres. An audit's check outside this repo, which rotated one real
  MediaPipe face mesh by known angles, suggests yaw reads about 1.5 times and pitch
  about 0.4 times the true angle. That is speculation until EyeOS's estimate is
  compared with MPIIFaceGaze's own head pose, which `diagnose` (added after this run)
  does in its `head_pose_agreement` section; it has not run on the real data yet. Gate
  2's 15° limit applies to the largest change on any of the three axes, so it is real
  degrees of roll but EyeOS's measure of yaw and pitch, a different true angle on each
  axis. Every head-pose share here is what Gate 2 would see live, in those units.
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

### Baselines and the live-style calibration

These variants were added after the first real run (2026-11-18), so the committed
`eval/results/mpiifacegaze.json` has no numbers for them yet; the next run on the
dataset will.

Two baselines ignore the image. They put every other error on a scale: how much of it
is left once a model has used the photo at all. The third variant builds the shipping
calibration the way a live session does, so its validation error is the statistic live
Gate 2 judges, computed on these photos: not a live result (see below).

| Variant | Model | Its `validation_error_px` |
|---|---|---|
| `centre` | baseline: predicts the screen centre (W/2, H/2) for every frame | the RMS distance of the calibration targets to the centre: the leave-one-out of a constant that ignores the data |
| `calibration-mean` | baseline: predicts the mean of the calibration targets for every frame | the true leave-one-out RMS of that mean predictor |
| `eyeos-live9` | the shipping `Calibration`, fitted on 9 samples: each dot's K images averaged into one sample, placed at the dot | the leave-one-out over those 9 samples: the statistic live Gate 2 judges, on these photos |

All three go through the same protocol as `eyeos`: the same calibration and test
frames, the same rounding and clamping, so their pixel and angular errors compare
directly with `eyeos` and every other row fitted at the run's K. (`per-dot-1`,
`per-dot-10` and `per-dot-20` calibrate on another number of photos per dot, so they
are tested on other frames.)

**Why `eyeos-live9`.** Every other variant's `validation_error_px` is the leave-one-out
over its 9 × K calibration photos, each labelled with its own true target, with the
other photos of the same dot left in the fit. Live Gate 2 judges something else.
`eyeos.calibration_session` averages up to 10 usable frames per dot (`SAMPLE_FRAMES`)
into one feature vector (`collect_dot_features`: the 4 iris offsets and the 3 head-pose
angles) and adds it at the dot itself, so the leave-one-out runs over 9 samples. They
are different statistics, and nothing makes them agree on pass or fail. `eyeos-live9`
does what live calibration does: it averages exactly as `collect_dot_features` does
(a test checks the two bit for bit, and another the whole fit against
`calibrate_interactively` on the same frames), so its `passes_gate2` applies live
Gate 2's statistic and threshold to these photos. It is still not a live session: each
dot's K photos are separate moments, often from different days, not up to 10 frames
of one look; and their true targets are on average `calibration_mean_distance_px`
from the dot. A live user does not look exactly at the dot either, but by how much is
not measured anywhere in this repo.

**In the README.** `scripts/update_readme.py` lists the baselines first, says how far
`eyeos`'s mean error is below `centre`'s, counts for each variant the persons it beats
`eyeos` for, and, when the results have `eyeos-live9`, gives its statistic and pass
count next to the 45-photo one, with why it is still not a live result. The commands
it prints to rebuild the file keep the run's paths only when they are repo-relative
(else they use the documented ones), and name `--persons` or `--variants` when the run
did not cover them all. With a results file that predates the three variants, it
leaves out what needs them.

**The environment record.** A different MediaPipe or OpenCV version can give different
landmarks, so a result is only reproducible with them known. `extract` now writes
`<cache>/manifest.json`: for each person it (re)extracts, the date, the image counts,
the detector and its FaceMesh settings, Python, the platform, and the mediapipe and
OpenCV versions. A person kept from an earlier run keeps its entry. `evaluate` writes
an `environment` block into the results: its own Python, platform and numpy, scipy,
OpenCV and mediapipe versions, and under `extract` the manifest entries of the
evaluated persons. Each version comes from the library's pip distribution, or, for one
installed without pip metadata (conda, a system package), from the module's
`__version__`; it is null when neither is found. `extract` is null for a cache made before
manifests existed, such as the 2026-11-18 run's, and a person the manifest does not
list is null. Neither the manifest nor the environment block holds paths; the results
file's `config` does (`root` and `cache`, as given to `evaluate`).

## The results JSON

`--out` gets: `dataset`, `license`, `citation`, `generated` (UTC date and time),
`git_commit` and `git_dirty`, `config` (paths, persons, variants, K, the Gate 2
thresholds, the screen-model tolerance), `screen_model` (per person), `detection` (per
person and in aggregate), `variants` (for each: description, K, `aggregate`, and
`persons` with every per-person metric, or `skipped` with the reason), and `notes` that
restate the protocol's limits next to the numbers.

**Its license.** The `license` field names the dataset's license, CC BY-NC-SA 4.0. The
file holds only counts, error metrics of EyeOS's own models (per person and in
aggregate) and run metadata: no images, landmarks or annotations. Whether such metrics
count as material adapted from the dataset is not settled here. To be safe, they are
marked as derived from MPIIFaceGaze and shared under CC BY-NC-SA 4.0 (non-commercial,
with the dataset's citation) wherever they appear in this repo: in
`eval/results/mpiifacegaze.json` (and any other results file computed from the data),
in the MPIIFaceGaze section of README.md's Results, in the MPIIFaceGaze results
quoted in this document, and in the few figures other docs and docstrings repeat from
them. The rest of the repo is under the MIT License.

## Real run (2026-11-18)

The first run on the real MPIIFaceGaze. The metrics in the tables, the detection counts
and the screen-model residuals are in `eval/results/mpiifacegaze.json`, which the
commands below regenerate; counts and percentages in the prose are computed from it.
The setup and format observations and the wall times were recorded by hand during the
run: no script produces them, and this run's JSON, made before `evaluate` recorded its
environment, does not record library versions.

### Setup

- **Date:** 2026-11-18. **Code:** commit `d3661ee` (the JSON's `git_commit`;
  `git_dirty` false). No code changed during the run.
- **Machine:** cloud container, 4 CPUs (Intel Xeon @ 2.10 GHz), no GPU, Linux.
- **Libraries:** Python 3.12.3, mediapipe 0.10.21, opencv-contrib-python 4.11.0.86,
  numpy 1.26.4, scipy 1.17.1. pip resolved these by itself from
  `pip install -e ".[dev,public-eval]"`; no pins were needed.
- **Data:** `MPIIFaceGaze.zip`, 942.3 MB, 38,278 files; 15 participants, 37,667
  annotated images (1,498 to 2,929 each; this doc and the downloader said "about 3000"
  each before the run).

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

### Results (all 15 participants scored for each of the 12 variants)

"Mean", "median" and "p90" are over the 15 per-person means (each person weighs the
same). With 15 values, the p90 is interpolated between the 3rd and the 2nd worst person,
so one or two people move it. Pooled values are over all test frames.

`ridge-0.001` and `per-dot-5` are the shipping configuration under other names (ridge
1e-3, K = 5), so their rows repeat `eyeos` exactly; the JSON's per-person records for the
three are identical. Read them as one result, not three.

| Variant | mean px | median px | p90 px | mean ° | median ° | p90 ° | pooled median px | % diag | Gate 2 (harness LOO) |
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
The calibration's leave-one-out validation error (the harness's 45-photo statistic)
averages 319 px; Gate 2 needs < 60 px.

### Reading the results honestly

- **The shipping calibration fails the harness's Gate 2 check for every participant**
  (0/15 under the 45-photo statistic). Its validation error is 224 to 422 px, 3.7 to 7
  times the 60 px limit. Live Gate 2 computes a different statistic (see "Limits of
  this protocol"), so this does not show that a live session would refuse these
  people, though the margin is wide. Refusing to start control is the gate doing its
  job, not a failure of it.
- **About 6° (258 px, 16 % of the screen diagonal) is too coarse for clicking.** On a
  1440x900 laptop screen that is a radius of roughly a fifth of the screen width: enough
  to tell which region of the screen someone looks at, not to hit a button.
- **How these numbers compare with live use is not measured, in either direction.**
  This setup differs from a live session in ways that push both ways.
  - Harder here: each calibration is chosen from all of a person's photos, taken over
    many days and postures, and tested on all the others, while a live session
    calibrates and runs in one sitting; nothing is smoothed; and photos whose head pose
    live Gate 2 would freeze are still scored (on average 90 % of test photos are within
    the head-pose limit, 65 % for p13: the JSON's `head_pose_within_limit_share`).
  - Easier here: the calibration photos are drawn from the same multi-day pool as the
    test photos, so the fit has already seen some of the days and postures it is
    scored on, while a live calibration is taken once, at the start, and never samples
    any drift later in the sitting; the calibration has 45 photos, while a live one has
    9 samples, each an average of up to 10 frames from one moment (with 9 single
    photos, `per-dot-1`, the mean error here is 16 % higher); and every photo is
    1280x720, while the live webcam source requests no resolution and gets the camera
    driver's default, which may be lower.
  - Not known either way: each calibration photo is labelled with the target shown when
    it was taken, a live sample with the dot shown. Both carry the person's fixation
    error, and nothing here compares the two. And MediaPipe runs on each photo in
    static-image mode, but live in tracking mode; how that changes per-frame accuracy
    is not measured.
  - The cross-sitting part can be measured without new data: the landmark cache keeps
    each image's `dayNN` folder, so a calibration tested within one day can be compared
    with the cross-day numbers here (a day can hold more than one sitting).
    `diagnose`, added after this run, makes that same-day split (`within_day`); it has
    not run on the real data yet.
    The rest needs a recorded live session (`eyeos record`).
- **Head pose matters.** Adding EyeOS's own yaw, pitch and roll (`with-head-pose`)
  cuts the mean error by 13 % (258 to 224 px), and learning from the other 14 people
  first (`global-lopo`) by 15 % (to 220 px). Both are better than `eyeos` for 15 of 15
  people. Between themselves they are equivalent within noise: `global-lopo` is better
  than `with-head-pose` for 8 of 15. Neither brings anyone under Gate 2's limit. The
  gain from head pose is largest for p13 (130 px, 37 %) and p04 (64 px, 26 %), the 1st
  and 4th lowest by share of test frames within the head-pose limit. Beyond them the
  relation is loose: p08, 2nd lowest, gains 49 px (15 %: 3rd of 15 by absolute gain,
  5th by relative gain); p06, 3rd lowest, gains 18 px (6 %); p00, with 95 % within the
  limit, gains 43 px (23 %). (All counts and gains here are computed from the JSON's
  per-person mean pixel errors: a gain is `eyeos` minus `with-head-pose`, and the
  relative gain divides it by `eyeos`. The share is `eyeos`'s
  `head_pose_within_limit_share`, in EyeOS's own head-pose units.)
- **More calibration images help a little; more model capacity and more
  regularisation do not.** 20 images per dot instead of 5 gains 3 % (251 px), 1 image
  per dot loses 16 % (300 px). The quadratic model is no better than the affine one.
  Stronger ridge makes things steadily worse (ridge 0.1: 278 px, ridge 1: 364 px,
  ridge 10: 429 px), but that is not a regularisation trade-off; see the next point.
- **`ridge-10` is close to a "guess the screen centre" baseline, and that is the scale
  to read the shipping 258 px against.** `Calibration` applies the ridge to the raw iris
  offsets, which are in eye half-widths and spread little across the screen, and never
  to the intercept (`eyeos/calibration.py`, `_ridge_fit`). A large lambda therefore
  shrinks the slopes toward zero and the prediction toward a constant near the mean
  calibration target, which is about the middle of the screen. Against `ridge-10`
  itself, the shipping model's mean error is 40 % lower (258 against 429 px) and lower
  for all 15 people, but for p13 only 13 % lower (348 against 402 px), computed from
  the JSON's per-person mean pixel errors. That `ridge-10` stands in for a constant is
  reasoning from the code and the numbers above, not a measurement: the
  constant-prediction baselines `centre` and `calibration-mean` were added to the
  harness after this run and have not run on the real data yet.
- **Nothing was tuned on this data.** Every variant, threshold and the ridge strengths
  were fixed before this run; this is a single evaluation, not a model selection.
  Changing EyeOS's shipping model based on it (for example adding head pose) would be a
  separate decision, made in `eyeos/` with its own tests.
- **Not measured here:** how the error compares with published appearance-based
  methods on this dataset (their protocols differ, and no such number is produced by
  this repo), and a measured constant-prediction baseline (`ridge-10` approximates one,
  as above; the baseline variants added since have not run on the real data).

### Diagnostics: where the error comes from

**This has not been run on the real data yet.** The `diagnose` command was added after
the run above. It is tested only on fake datasets (`tests/test_public_diagnostics.py`)
whose cause of error is known: noise-free faces, iris offsets that shift with head yaw,
a different iris bias on each day, pure landmark noise, and EyeOS head angles that are a
known multiple of the dataset's. Until a session with the data runs it and commits
the JSON, nothing here says where the 258 px mean in the table above comes from; this
section says how to read the answer when it exists.

```bash
python -m eval.public.mpiifacegaze diagnose --root datasets/MPIIFaceGaze \
    --cache datasets/cache/mpiifacegaze --out eval/results/mpiifacegaze-diagnostics.json
# optional: --persons p00,p01  --k-per-dot 5  --min-day-images 60
```

It needs no new extraction. Each image's day is the `dayNN` folder in the cache's
`image` column. The cache does not keep the dataset's head pose, so `diagnose` reads
`head_rvec` and `head_tvec` again from `pXX.txt` under `--root` and matches them to the
cache by image path. A cached image that is missing from `pXX.txt`, or has another
target there, stops the run with an error. It prints one line per person and writes a
JSON in which each section says what question it answers (`sections`) and each metric
has a one-line description (`metrics`). `aggregate` gives the mean, median, min and max
of each per-person metric. The JSON has no timestamp and rounds every number to 10
significant digits, so last-digit floating-point noise is very unlikely to change the
file between runs on the same cache, code and libraries. Very unlikely is not
impossible, and another numpy or BLAS build can move values further, so the JSON
records the Python, numpy and scipy versions (`environment`).

Two sections fit on a person's frames and score on those same frames (b and g). They
are bounds and explanations, not results. Only d and e follow the evaluation protocol.

- **a. `feature_sanity`: are the iris offsets sane?** Per eye, the median, p5 and p95
  of the x and y offsets; the correlation of the two eyes' x offsets (and y); the
  correlation of each eye's x with the target's x; EyeOS roll p5 to p95. Both eyes turn
  together, so their correlation should be near +1. Landmark noise pulls it toward 0.
  An eye whose x correlates with the target's x with the opposite sign
  (`target_x_signs_match` false) has its axis mirrored: its corners are labelled the
  wrong way round. An iris paired with the other eye's corners passes that check, since
  both offsets still move with the target; it shows instead as a median x offset of
  several half-widths, far outside the eye (`x_medians_within_eye` false; a healthy
  median is well inside ±1).
- **b. `in_sample_ceiling`: how well could one calibration do across all days?** The
  shipping model (affine in the 4 iris offsets, same ridge strength) is fitted on all of
  a person's valid frames, from every day, and scored on the same frames. The same is
  done for each eye alone and with EyeOS's head pose added. This is about the lowest
  error a single affine map of these features reaches when one calibration has to serve
  every day, which is the `cross_day` setting. It is an optimistic bound for that
  setting only. It does not bound a calibration made within one sitting: a per-day
  offset (sitting, lighting, a per-day landmark bias) raises it, yet a per-day
  calibration absorbs it. `per_day` fits the same model on each day that `within_day`
  uses, on that day's own frames, and averages over those days; that is the bound for
  a per-sitting calibration. All-days ceiling high but `per_day` low means one
  calibration cannot serve every day, though a calibration per sitting might (check
  `within_day`). `per_day` high as well means the features limit an affine calibration
  even within a day (a day folder can still hold several sittings). Neither is Gate 2's
  number: Gate 2 judges the leave-one-out RMS over the 45 calibration images, which
  `cross_day.validation_error_px` and `within_day.validation_error_px_mean` report. One
  eye far worse than the other points to that eye.
- **c. `constant_baselines`: the yardstick.** The error of always predicting the
  screen centre, and of always predicting the person's own mean target. A model close
  to these has learned little from the eyes.
- **d. `cross_day` and `within_day`: is it drift between sittings?** `cross_day` is the
  standard protocol for the shipping calibration and reproduces `evaluate`'s `eyeos`
  numbers. `within_day` runs the same protocol (K images per dot, 5 by default, nearest
  true target, the shipping calibration) inside each day folder with enough valid images
  (60 by default), and tests on that day's other images. It reports the mean error over
  those days, the mean leave-one-out validation error (Gate 2's input), how many days
  pass Gate 2, and how many days were used. Days with too few images are left out, so
  `within_over_cross_mean_px` divides by `cross_day_mean_px_on_used_days`: the
  cross-day calibration's error on the test frames of the same days, averaged per day
  as the within-day mean is. `cross_day_mean_px` (every test frame) is kept beside it;
  a large gap between the two means the skipped days score differently. Within each
  day the two protocols still test slightly different frames, because each leaves out
  its own calibration images. Within-day far below cross-day means drift
  between sittings is a large share of the error; a live session calibrates in the same
  sitting. Both high means the features are the bottleneck. A day folder can hold
  several sittings, and a day has fewer images, so its calibration images lie further
  from the dots (`calibration_mean_distance_px`). The two protocols are not matched in
  size, and within-day is not a live session either.
- **e. `error_by_head_pose`: does the error grow as the head moves?** The cross-day
  test error binned by EyeOS's head-pose distance from the calibration pose (0-5, 5-10,
  10-15, 15-20 and over 20, in EyeOS's own "degrees", Gate 2's measure), and the error
  on the frames within Gate 2's head-pose limit only, the frames a live session would
  act on. Error rising across the bins points to head motion.
- **f. `head_pose_agreement`: are EyeOS's head angles real degrees?** EyeOS's yaw, pitch
  and roll are regressed on the dataset's head rotation, per person: slope, intercept
  and correlation per axis, plus all 9 correlations. |slope| is the scale of EyeOS's
  "degrees" (1 means real degrees), which sets what Gate 2's 15-degree limit means. The
  dataset's rotation is read as R = Ry(yaw) Rx(pitch) Rz(roll), head to camera, after
  undoing a half turn of the head frame if that brings the person's mean rotation closer
  to the identity (a head frame lined up with the camera's). That reading has not been
  checked against the real files. So the sign of a slope is not interpreted, and a
  large off-diagonal correlation would show an axis mix-up.
- **g. `variance_split`: head motion, the day, or landmark noise?** The share (R²) of
  the all-days in-sample residual from b that a linear fit explains: on the dataset's
  head pose (angles and translation), on the day folder alone, and on both. Read the
  shares beyond the target, not the raw ones. Noisy iris offsets shrink the fit toward
  the screen centre, so its residual grows with the target (`r2_target`, the control).
  Anything that follows the target then explains part of the residual, even with no
  head effect on the iris offsets at all; people turn their head a little toward where
  they look, so the raw `r2_head_pose` can be high from landmark noise alone. A test
  checks exactly that case: on a fake with pure landmark noise and a head that turns
  toward the target, the raw share is far above chance and the share beyond the target
  stays near 0. `r2_head_pose_beyond_target` (R² on target and head pose, minus `r2_target`) is the
  share to read as head motion the iris offsets do not capture;
  `r2_day_beyond_target` likewise for something constant within a day but different
  between days. Both low means per-frame landmark noise, or effects that are not linear
  in these. Regressors unrelated to the residual score about k/n by chance (k
  regressors, n frames), which is small when n is in the thousands. The terminal
  summary shows the two beyond-target shares (`pose|tgt`, `day|tgt`).

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
