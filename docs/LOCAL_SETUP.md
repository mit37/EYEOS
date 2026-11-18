# LOCAL_SETUP.md — running EyeOS on your own Windows laptop

EyeOS was built in a cloud container with no webcam, no display, no keyboard and no
Windows. This guide moves it to your laptop so you can run the tests, try it live,
record your own labelled sessions and re-run the evaluations as often as you like.
It assumes you are new to Python tooling: every command is spelled out, in
PowerShell. Everything runs on the CPU.

## What has been checked, and what hasn't

Be clear about this before trusting any "expected" result below.

- **Run for real (Linux, not Windows).** In a fresh clone of this branch on Ubuntu
  24.04 with Python 3.12:
  - `bash scripts/setup_unix.sh`, the Linux/macOS twin of the Windows script, exited 0.
    Its setup check passed with the real OpenCV 4.11 and MediaPipe 0.10.21 (FaceMesh
    ran on a blank frame). The whole test suite passed. `.venv` came to 1.1 GB. pip
    took every package from its cache, so the download time was not measured.
  - `eyeos run --config config.example.yaml` printed `processed 485 frames: 0 moves,
    0 clicks`, and `python -m eval.run_eval --session eval/fixtures/synthetic_session.csv`
    exited 0.
  - A SIGINT (what Ctrl+C sends) during a dry run ended it with exit code 130 and
    `eyeos: interrupted`, and every frame processed so far was in the session log.
- **Checked without Windows.** pip resolves the `dev` and `webcam` extras for 64-bit
  Windows to binary wheels on Python 3.10, 3.11 and 3.12 (MediaPipe 0.10.21, OpenCV
  4.11.0.86, NumPy 1.26.4), with and without the `public-eval` extra that another
  branch adds. On Python 3.13 it fails with `No matching distribution found for
  mediapipe<0.10.30,>=0.10`. This used `pip install --dry-run --platform win_amd64`,
  which evaluates environment markers as Linux, so pynput was checked separately: it
  is a pure-Python wheel.
- **Never run.** `scripts/setup_windows.ps1` has never been run: the container had no
  PowerShell. The test suite has never run on Windows. Nothing here has run with a
  webcam, a real display or a real keyboard, so every "Expect" below comes from the
  code and its tests, not from a real session. If what you see differs, that is a real
  result: note it in `docs/PLAN.md`.
- **Built alongside this guide.** `eyeos run --view`, `eyeos record` and the
  MPIIFaceGaze commands were built at the same time on other branches, then merged.
  The commands below match their `--help`, and a test checks that every `eyeos run`
  flag used here exists. What each of them has and hasn't run is in its own doc:
  `docs/DEMO.md` ("Tracking view"), `docs/RECORDING.md` and `docs/PUBLIC_DATASETS.md`.

## 1. Install Git and Python 3.12

Open **PowerShell** (Start menu, type "PowerShell"; no need to run it as
administrator) and run:

```powershell
winget install Git.Git
winget install Python.Python.3.12
```

Then **close PowerShell and open a new window**, so it picks up the new `PATH`, and
check:

```powershell
git --version
py -0                # lists the Pythons the py launcher knows; 3.12 should be there
py -3.12 --version
```

**Why 3.12 and not the newest Python.** EyeOS reads the face through MediaPipe's
legacy Solutions API (`mediapipe.solutions.face_mesh`), which MediaPipe removed in
0.10.30. So the `webcam` extra pins `mediapipe<0.10.30`, and those versions have no
wheels for Python 3.13 or newer (and on Windows, only for 64-bit x86). EyeOS itself
needs Python 3.10 or newer, which leaves 3.10–3.12. On Python 3.13, pip stops with
`No matching distribution found for mediapipe<0.10.30,>=0.10`. Having a newer Python
installed as well is fine: the `py` launcher picks the version you ask for.

## 2. Get the code

```powershell
cd $HOME
git clone -b claude/new-session-cawn9l https://github.com/mit37/eyeos.git
cd eyeos
```

`claude/new-session-cawn9l` is the branch this work lives on until it is merged.
After the merge, `main` has it all: `git switch main`, then `git pull`. If the
repository is private, Git opens a browser window to sign in to GitHub the first time.

## 3. Create the environment and install EyeOS

A **virtual environment** (`.venv`) is a private copy of Python inside the project
folder. Everything EyeOS installs goes there, so it can't clash with anything else on
the laptop, and deleting the folder undoes it all.

### With the script

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

`-ExecutionPolicy Bypass` lets this one script run without changing any Windows
setting. It finds Python 3.12 (or 3.11, 3.10) through `py`, creates `.venv`, installs
EyeOS, runs `scripts\check_setup.py` and the tests, and prints the next steps. It has
**never been run** (see above). If it fails, do the same by hand:

### By hand

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,webcam,windows-fallback,public-eval]"
python scripts\check_setup.py
```

- If `Activate.ps1` fails with "running scripts is disabled on this system", allow
  local scripts for your user once, answer `Y`, and activate again:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- Once activated, the prompt starts with `(.venv)`, and `python`, `pip`, `pytest` and
  `eyeos` all mean the ones inside `.venv`. **Activate again in every new PowerShell
  window** (`cd $HOME\eyeos`, then `.venv\Scripts\Activate.ps1`).
- Keep the quotes in `".[...]"`: without them PowerShell reads the commas as a list.
- `-e` (editable) means your edits and every `git pull` take effect without
  reinstalling. Run the `pip install` line again only when `pyproject.toml` changes.

What each extra brings:

| Extra | What it installs | Needed for |
|---|---|---|
| `dev` | pytest, ruff, black, mypy | the checks in step 4 |
| `webcam` | OpenCV, MediaPipe < 0.10.30 | anything with a camera |
| `windows-fallback` | pynput | the kill switch (Esc) and confirm key. Without it EyeOS refuses to move the real cursor |
| `public-eval` | scipy, OpenCV, MediaPipe | the MPIIFaceGaze evaluation (step 9) |

`scripts\check_setup.py` checks what the tests can't: the tests use fake OpenCV and
MediaPipe modules, so they pass even when the real ones are broken. It checks your
Python version, that `eyeos` comes from this folder, that OpenCV imports, that
MediaPipe has the Solutions API and runs FaceMesh once on a blank frame, and that
pynput and scipy are installed. It opens no camera and no window. Expect every line to
be `[ok]` and `All required checks passed.`

## 4. Run the checks and the dry-run demo

```powershell
pytest -q
ruff check .
black --check .
mypy eyeos
eyeos run --config config.example.yaml
```

Expect:

- `pytest -q`: `N passed`, nothing failed. On Windows some are skipped on purpose:
  the tests that run the Linux/macOS setup script, and the one that checks the
  SendInput backend refuses to start off Windows.
- `ruff check .` prints `All checks passed!`, `black --check .` ends with
  `files would be left unchanged.`, and `mypy eyeos` prints `Success: no issues found`.
- `eyeos run --config config.example.yaml` replays the committed synthetic session
  with no camera. It prints a `[dry-run] ...` line per action (none here) and then
  `processed 485 frames: 0 moves, 0 clicks` and `session log: logs\session-<time>.jsonl`.
  No calibration was given, so Gate 2 correctly refuses every move.

## 5. A safe first live run: nothing moves the mouse

Make your own config (it is gitignored, so it never gets committed):

```powershell
Copy-Item config.example.yaml config.yaml
notepad config.yaml
```

Change these keys **in the sections the file already has**, and leave the rest. The
snippets in this guide show only the keys to change. Don't paste one in as a second
`screen:`, `gates:` or `output:` section: plain YAML keeps only the last copy, which
would drop everything set in the first (a no-click zone, say), so EyeOS refuses a
config with a repeated key and names it, `duplicate key 'gates'`, with its line.

```yaml
screen:
  width_px: 1536 # the size Windows reports, after display scaling (see below)
  height_px: 864
input:
  source: webcam
  device_index: 0 # try 1 or 2 if the wrong camera opens
output:
  backend: dry_run # prints actions instead of moving the mouse
```

**Screen size.** EyeOS is not DPI-aware, so `screen` must be the size Windows reports
*after* display scaling: Settings > System > Display shows the resolution and the
scale, and the reported size is resolution ÷ scale (1920x1080 at 125% reports
1536x864). If you get it wrong, EyeOS stops before drawing anything and prints the
size it saw, either `config screen.width_px/height_px is ... but the screen reports
WxH` (the calibration screen) or `config screen is ... but Windows reports WxH` (the
SendInput backend). Copy those numbers into `config.yaml`.

First look at the tracking view on its own. It draws a marker on each pupil and an
arrow for its shift from the eye's centre, the shift that moves the cursor:

```powershell
eyeos run --config config.yaml --view
```

With no calibration there is no cursor yet: Gate 2 stays red and the small screen map
stays empty. Press **q** in the window to hide it and Ctrl+C in the terminal to stop.
Then run with the calibration dots, the gate lights and the tracking view together:

```powershell
eyeos run --config config.yaml --calibrate --overlay --view
```

That combination puts Tk's windows and OpenCV's window in one process and has never
run anywhere (see `docs/DEMO.md`). If it hangs or crashes, drop `--view` and report
what you saw.

1. A fullscreen screen shows 9 dots, one at a time. Look at each one and press
   **space** when your gaze is steady. Esc cancels.
2. The terminal prints `calibration validation error: N px`. Gate 2 accepts a
   calibration only below 60 px; above that, stop and calibrate again.
3. The gate-lights window and the tracking window ("EyeOS tracking") open. What the
   tracking view draws is explained in `docs/DEMO.md` ("Tracking view"); press **q**
   in it, or close it, to hide it.
4. The terminal prints `[dry-run] move_cursor(x, y)` lines: that is where the cursor
   *would* go. The real mouse does not move.
5. **Stop with Ctrl+C in the terminal.** A dry run starts no keyboard listener, so
   Esc does nothing here. Ctrl+C prints `eyeos: interrupted`. The session log keeps
   every frame up to that point.

Section A of the checklist at the end lists what to try in this mode and what to
expect.

## 6. The real cursor: keypress mode first

Keypress mode is the safest real mode: a click needs your gaze to dwell **and** a
deliberate key press. Pick a confirm key nobody types with. The listener is global and
doesn't swallow keys, so with `space` you would click every time you typed a space
anywhere while a dwell holds. As in step 5, change these keys where they already are
in `config.yaml`:

```yaml
gates:
  confirmation_mode: keypress
output:
  backend: windows_sendinput
  kill_hotkey: esc
  confirm_key: f9
```

Then set the no-click zone for **your** screen, by editing the example's zone in
place. The example's zone (x 1860–1920) is for a 1920-pixel-wide screen and is
off-screen on a narrower one. For a 1536x864 screen, the 60x40 px top-right corner
(where a maximised window's close button is) is:

```yaml
gates:
  no_click_zones:
    - x_min: 1476 # your width minus 60
      y_min: 0
      x_max: 1536 # your width
      y_max: 40
```

**Test Esc first, before anything else.** Start the session, and as soon as the
cursor starts following your gaze, press **Esc**:

```powershell
eyeos run --config config.yaml --calibrate --overlay
```

Expect the overlay to close and `kill switch (esc) pressed: session ended`. Only once
that works, go through section B of the checklist (one click with F9, the re-dwell,
the no-click zone). Use a harmless target to click on: an empty Notepad window, where
a click only moves the text cursor.

One Windows rule to know (from Microsoft's documentation of `SendInput`, not tested
here): a normal program can't send input to a program running as administrator, so
clicks on an elevated window are silently dropped. Whether the Esc listener still
hears Esc while an administrator window has focus wasn't tested either. Keep elevated
windows out of your tests.

## 7. Blink-hold clicking

Blink-hold is the least proven part of EyeOS (see the README: eleven review rounds,
none of its thresholds tuned on real recordings), so try it only after keypress mode
works. A click is a dwell, then your eyes held closed for at least 0.4 s.

```yaml
gates:
  confirmation_mode: blink_hold
```

Run it the same way (`eyeos run --config config.yaml --calibrate --overlay`) and go
through section C of the checklist.

## 8. Record and evaluate your own sessions

The README's numbers come from a synthetic session. Your own recordings give real
ones. `eyeos record` shows targets at known places and saves the face landmarks
(numbers, not images) with the true target of every frame, in the CSV format the eval
reads:

```powershell
eyeos record --config config.yaml --out recordings/session1.csv
eyeos record --config config.yaml --out recordings/session2.csv --dot-s 2.0 --settle-s 0.6 --reading-s 300
python -m eval.run_eval --session recordings/session1.csv --config config.yaml --out recordings/session1-results.json
```

- `--dot-s` is how long each target stays up, `--settle-s` how much of its start is
  dropped while your eyes move there, `--reading-s` the length of the "just reading"
  part (the PRD asks for 5 minutes, 300 s), and `--no-reading` skips it. The recorder
  explains each stage on screen; `docs/RECORDING.md` has the details and what it has
  and hasn't run on. When it finishes it prints the `run_eval` command to use.
- Save recordings under `recordings/`, which is gitignored. PowerShell accepts both `/`
  and `\` in these paths.
- `run_eval` prints the metrics and writes them to `--out`, with the same keys as the
  README's Results: `calibration_validation_error_px`, `mean_pixel_error_raw_px` and
  `mean_pixel_error_filtered_px`, `steady_state_jitter_raw_px` and
  `steady_state_jitter_filtered_px`, `on_target_dwell_satisfied_frames`,
  `just_reading_false_click_count` and `click_false_trigger_rate`. Pass the config
  you recorded with (`--config`): without it, `run_eval` uses the built-in defaults,
  a 1920x1080 screen, and the errors of any other screen size come out wrong.
- Record a few sessions (different lighting, glasses on and off, sitting closer or
  further) and compare their result files. A bad number is a real result: report it.

## 9. A public dataset: MPIIFaceGaze, and evaluating over and over

MPIIFaceGaze is a public dataset of laptop-webcam face images, each labelled with the
screen point the person was looking at. EyeOS's pipeline can be scored on it without
recording anything.

**License.** It is licensed CC BY-NC-SA 4.0: non-commercial use only, credit the
authors, share anything derived from it under the same license. The images show real
people's faces, so they and the landmark caches made from them stay out of git
(`datasets/` is gitignored). Only error metrics computed from the data are committed,
in `eval/results/mpiifacegaze.json` and quoted from it in the README and
`docs/PUBLIC_DATASETS.md`; they are shared under the dataset's license
(`docs/PUBLIC_DATASETS.md`, "The results JSON"). The downloader
prints the license and citation and downloads nothing until you pass
`--accept-license`.

```powershell
python scripts/download_mpiifacegaze.py --accept-license --dest datasets
python -m eval.public.mpiifacegaze extract --root datasets/MPIIFaceGaze --cache datasets/cache/mpiifacegaze --persons p00,p01 --limit 200
python -m eval.public.mpiifacegaze extract --root datasets/MPIIFaceGaze --cache datasets/cache/mpiifacegaze --workers 4
python -m eval.public.mpiifacegaze evaluate --root datasets/MPIIFaceGaze --cache datasets/cache/mpiifacegaze --out eval/results/mpiifacegaze.json
```

- `extract` runs MediaPipe over every image once and caches the landmarks. It is the
  slow step, and it can be stopped and restarted. On the one real run it took about
  1 min 40 s for all 37,667 images with 4 workers, on a 4-CPU cloud container
  (`docs/PUBLIC_DATASETS.md`, "Real run"); take that only as a rough guide, since a
  laptop's CPU and disk differ. The first `extract` line is a quick trial on two
  people and 200 images each; the second does everything, with 4 worker processes.
- `evaluate` doesn't run MediaPipe again: it reuses the cache, so it can be re-run as
  often as you like.

**The "train over and over" loop.** The models here are small regressions from
landmark features to the screen, such as EyeOS's own per-person calibration, not
neural networks. So "training" means: change something, re-run `evaluate`, compare.
`docs/PUBLIC_DATASETS.md` explains the protocol and each variant, and `evaluate
--help` lists their names (`eyeos` is the shipping calibration, `eyeos-poly2` a
second-order version of it). Each run with `--variants` scores only the variants you
name; give each run its own `--out` and compare the files:

```powershell
python -m eval.public.mpiifacegaze evaluate --root datasets/MPIIFaceGaze --cache datasets/cache/mpiifacegaze --out eval/results/run-a.json --variants eyeos
python -m eval.public.mpiifacegaze evaluate --root datasets/MPIIFaceGaze --cache datasets/cache/mpiifacegaze --out eval/results/run-b.json --variants eyeos-poly2
git diff --no-index eval/results/run-a.json eval/results/run-b.json
```

The cache holds MediaPipe's raw landmarks, and `evaluate` turns them into features
each run, so a change to `eyeos/features.py`, the calibration or a variant
(`eval/public/models.py`) needs only another `evaluate`. Only a change to the
detector itself needs `extract --force`.

## 10. Keep working with Claude Code

- **Locally:** install Claude Code by following the official instructions at
  <https://code.claude.com/docs>, then start it in the `eyeos` folder. It reads
  `CLAUDE.md` (this repo's rules) by itself, and it can run the tests, the setup check
  and the dry runs, while you do the parts that need your face, camera and keyboard.
- **Or keep using cloud sessions:** they push their work to GitHub branches. On the
  laptop, `git pull` brings in new commits on the branch you are on, and `git fetch`
  then `git switch <branch>` moves you to another branch.
- After any pull that changes `pyproject.toml`, run the `pip install -e ...` line from
  step 3 again (inside the activated `.venv`).

## 11. CPU only is fine

No GPU is needed for anything in this repo. MediaPipe's face mesh runs on the CPU
(its log says `Created TensorFlow Lite XNNPACK delegate for CPU`). The gaze "models"
are per-user ridge regressions: fitting one to the synthetic session's 45 calibration
samples, leave-one-out validation included, took about 3 ms on the build container's
CPU. Deep-learning gaze models, such as networks trained on MPIIFaceGaze images,
would want a GPU to train. They are future work and not part of this repo.

## Troubleshooting

- **`No matching distribution found for mediapipe<0.10.30,>=0.10`** during `pip
  install`: the `.venv` was made with Python 3.13 or newer. Delete it
  (`Remove-Item -Recurse -Force .venv`) and recreate it with `py -3.12 -m venv .venv`.
- **"needs MediaPipe's legacy Solutions API (mediapipe<0.10.30); installed mediapipe X
  doesn't have it"** when `eyeos run` starts: that Python has a MediaPipe that is too
  new. This happens when `eyeos` runs from a Python other than `.venv`'s (check that
  the prompt starts with `(.venv)`, and that `where.exe eyeos` lists `.venv` first), or
  after a hand-run `pip install mediapipe`. Activate `.venv` and run
  `python scripts\check_setup.py`: it names the Python and the MediaPipe it found.
- **Screen size refused**: see "Screen size" in step 5. Copy the size from the error
  into `config.yaml`, and move the no-click zones with it.
- **`could not open webcam device 0`**: try `device_index: 1` or `2` (laptops with a
  Windows Hello IR camera have more than one), close other apps using the camera
  (Teams, Zoom, the Camera app), and check Settings > Privacy & security > Camera >
  "Let desktop apps access your camera" (Windows 11's wording). If the camera opens
  but gives no frames, the session ends at once, or `--calibrate` stops with `the
  input source gave no frames, so there is nothing to calibrate on`.
- **A setting seems to be ignored** (the no-click zone does nothing, the mode didn't
  change): check `config.yaml` for a section that appears twice, such as two
  `gates:` lines. YAML keeps only the last one, without an error.
- **`refusing to drive the cursor without a kill switch: the kill hotkey needs pynput
  ...`**: the keyboard listener could not start. If pynput isn't installed, run the
  `pip install -e ...` line again. If it is (`check_setup.py` says so), security
  software may be blocking it: pynput watches the keyboard globally, the way a
  keylogger would, and some antivirus tools block or quarantine that. Allow it for the
  `.venv` Python only if you trust this code. Not tested here.
- **`DLL load failed` when importing `cv2` or `mediapipe`**: a common cause on Windows
  is a missing Microsoft Visual C++ Redistributable. Install the current one from
  Microsoft and try again. Not tested here.
- **`check_setup.py` warns that `eyeos` comes from another checkout**: you installed
  EyeOS from a different folder too. Run the `pip install -e ...` line from this
  folder.
- **`check_setup.py` warns that `eyeos` is a copy installed without -e**: the `pip
  install` line ran without `-e`, so `.venv` holds a frozen copy that `eyeos run` uses
  and that neither your edits nor `git pull` change. Run the `pip install -e ...` line
  again, with the `-e`.
- **Your recordings never leave the machine.** `eyeos/` contains no networking code.
  Session logs (`logs/`) hold timestamps, gate decisions and their reasons, and cursor
  coordinates. `eyeos record` writes a CSV of landmark numbers (the format `run_eval`
  reads), not camera images. `logs/`, `recordings/`, `datasets/`, `config.yaml` and
  video files are gitignored.
  Nothing reaches GitHub unless you `git add` and `git push` it, so run `git status`
  before every commit.

## How to test it end to end

Tick each box; where you see something else, write it down. That is a real result.

**Automated checks** (step 4)

- [ ] `python scripts\check_setup.py`: every line `[ok]`, then `All required checks passed.`
- [ ] `pytest -q`: `N passed` and nothing failed (a few skipped on Windows is expected).
- [ ] `ruff check .`, `black --check .` and `mypy eyeos` are clean.
- [ ] `eyeos run --config config.example.yaml --no-log` prints `processed 485 frames: 0 moves, 0 clicks`.

**A. Dry run with the webcam: the gate lights** (step 5: `output.backend: dry_run`,
`eyeos run --config config.yaml --calibrate --overlay --view`)

- [ ] After calibration, `calibration validation error` is below 60 px. Sitting still
      and looking at the screen, Gates 1, 2 and 4 are green, and `[dry-run]
      move_cursor` lines follow your gaze while the real mouse stays put.
- [ ] Lean out of the camera's view: Gate 1 goes red, `expected exactly one face, saw
      0`. Have a second person lean in: `saw 2`. The `move_cursor` lines stop.
- [ ] Keep looking at the screen and turn your head about 20° to one side: Gate 1
      stays green, Gate 2 goes red, `head pose drifted 20.3deg from calibration pose
      (limit 15.0deg) — recalibrate` (your number will differ). Turn back: green again.
- [ ] Hold your gaze on one spot: Gate 3 goes green (`dwell satisfied`) after 0.8 s.
      If it never does, your gaze jitters more than the 20 px dwell radius. The
      README's Results predicts this from synthetic noise; seeing it for real is worth
      recording.
- [ ] Keep your head still and glance past the screen edge with your eyes only. **No
      light changes**, and the gaze sits on the screen edge. EyeOS doesn't catch this
      (a known negative result in the README).
- [ ] Ctrl+C ends the session with `eyeos: interrupted`.

**B. The real cursor in keypress mode** (step 6)

- [ ] Start, and press **Esc** as soon as the cursor moves: the overlay closes and the
      terminal prints `kill switch (esc) pressed: session ended`. Don't go on until
      this works.
- [ ] Dwell on an empty Notepad window until Gate 3 is green, then press **F9** once:
      exactly one click, where you were looking. The session log has one line with
      `"click_fired": true`.
- [ ] Hold F9 down: still one click, no repeat. Release it, dwell for another 0.8 s
      and press F9 again: a second click. Pressing F9 without a fresh dwell does
      nothing.
- [ ] Look into the top-right no-click zone: Gate 4 goes red (`gaze is inside a
      no-click zone`), the cursor stops, and F9 does nothing.
- [ ] Esc ends the session.

**C. Blink-hold** (step 7)

- [ ] Dwell until Gate 3 is green, then close your eyes deliberately for about half a
      second: one click where you were looking. While your eyes are closed, Gate 1
      reads like `ok (eyes closed 0.27s: blink-hold grace)`.
- [ ] Blink normally while dwelling: no click.
- [ ] Keep your eyes closed for more than about a second: Gate 1 goes red when the
      grace window runs out, and there is still only one click.
- [ ] Esc ends the session.

**D. The 5-minute reading test** (PRD §5: false clicks while just reading; the target
is 0). Use a dry run, so a false click can't do anything, and the default `either`
mode, so blink-holds can click (again, edit these keys in place):

```yaml
gates:
  confirmation_mode: either
output:
  backend: dry_run
```

```powershell
eyeos run --config config.yaml --calibrate --log logs/reading.jsonl
```

Read an article normally for at least 5 minutes, then press Ctrl+C. Count the clicks
the gates fired (each would have been a false click):

<!-- false-click-count -->
```powershell
python -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1])]; t=r[-1]['timestamp_s']-r[0]['timestamp_s']; print(sum(x['click_fired'] for x in r), 'clicks in', round(t), 's over', len(r), 'frames')" logs/reading.jsonl
```

- [ ] It prints `0 clicks in N s over M frames`, with N at least 300. Any other count
      is a real negative result: write it down in `docs/PLAN.md` with the command that
      produced it. STANDARDS.md §1.4 says it belongs in the README's Results too.

## On Linux or macOS

`bash scripts/setup_unix.sh` does the same as the Windows script: it looks for
`python3.12`, `python3.11`, `python3.10`, then `python3`, makes `.venv`, installs the
same extras and runs the same checks. It ran end to end on Ubuntu 24.04 (see the top
of this guide), never on macOS. On Linux, pynput builds a package (`evdev`) from
source, which needs a C compiler and the Python headers (Ubuntu:
`sudo apt install build-essential python3.12-dev`). Off Windows the real cursor needs
`output.backend: pynput`, the less-tested fallback, which has never run here.
