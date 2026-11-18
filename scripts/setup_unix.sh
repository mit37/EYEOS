#!/usr/bin/env bash
# setup_unix.sh: one-time EyeOS setup on Linux or macOS. scripts/setup_windows.ps1 does
# the same on Windows; docs/LOCAL_SETUP.md explains every step.
#
#   bash scripts/setup_unix.sh
#
#   1. finds Python 3.12, 3.11 or 3.10 (python3.12, python3.11, python3.10, then python3
#      and python if their version fits). mediapipe<0.10.30 has no wheels for 3.13+;
#   2. creates .venv in the repo root, or reuses one made with a supported Python;
#   3. installs EyeOS in editable mode with the extras dev,webcam,windows-fallback,public-eval;
#   4. runs scripts/check_setup.py (the real OpenCV + MediaPipe) and the test suite;
#   5. prints the next steps.
#
# Environment variables:
#   EYEOS_PYTHON=/path/to/python3.12   use this interpreter instead of searching PATH
#   EYEOS_SKIP_TESTS=1                 don't run the test suite
#   EYEOS_SETUP_DRY_RUN=1              only print the interpreter and the commands
#
# Where it has run: end to end on Ubuntu 24.04 with Python 3.12, in a fresh clone (see
# docs/LOCAL_SETUP.md, "What has been checked"). It has never run on macOS. Written for
# bash 3.2 (macOS's /bin/bash) too, but only run with bash 5.

set -euo pipefail

EXTRAS="dev,webcam,windows-fallback,public-eval"
SUPPORTED=" 3.12 3.11 3.10 "

die() {
    printf '\nsetup_unix.sh: error: %s\n' "$*" >&2
    exit 1
}

say() { printf '==> %s\n' "$*"; }

# Print a command, then run it (only print it in a dry run). Arguments with spaces or
# brackets are printed in double quotes, so the line can be copied into bash or zsh.
run() {
    line="+"
    for arg in "$@"; do
        case "$arg" in
            *[\ \[\]]*) line="$line \"$arg\"" ;;
            *) line="$line $arg" ;;
        esac
    done
    printf '%s\n' "$line"
    if [ -z "${EYEOS_SETUP_DRY_RUN:-}" ]; then
        "$@"
    fi
}

# "3.12" for a working interpreter, nothing otherwise.
py_version() {
    "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true
}

is_supported() {
    case "$SUPPORTED" in *" $1 "*) return 0 ;; esac
    return 1
}

why_312="EyeOS's webcam extra needs mediapipe<0.10.30 (the last versions with the legacy \
Solutions API), which has wheels for Python 3.10-3.12 only."

# --- 1. the repo root ----------------------------------------------------------------
script_dir=${BASH_SOURCE[0]%/*}
if [ "$script_dir" = "${BASH_SOURCE[0]}" ]; then script_dir=.; fi
cd "$script_dir/.." || die "cannot change to the repo root ($script_dir/..)"
[ -f pyproject.toml ] || die "no pyproject.toml in $PWD: run this script from an EyeOS checkout \
(bash scripts/setup_unix.sh)"

# --- 2. a Python 3.10-3.12 -----------------------------------------------------------
PY=""
PYV=""
if [ -n "${EYEOS_PYTHON:-}" ]; then
    PYV=$(py_version "$EYEOS_PYTHON")
    [ -n "$PYV" ] || die "EYEOS_PYTHON=$EYEOS_PYTHON does not run"
    is_supported "$PYV" || die "EYEOS_PYTHON=$EYEOS_PYTHON is Python $PYV, not 3.10-3.12. $why_312"
    PY=$EYEOS_PYTHON
else
    seen=""
    for name in python3.12 python3.11 python3.10 python3 python; do
        path=$(command -v "$name" 2>/dev/null) || continue
        version=$(py_version "$path")
        if [ -n "$version" ] && is_supported "$version"; then
            PY=$path
            PYV=$version
            break
        fi
        seen="$seen; $name is ${version:-not runnable}"
    done
    if [ -z "$PY" ]; then
        die "no Python 3.10-3.12 found on PATH (${seen#; }). $why_312 Install Python 3.12 \
(your package manager, Homebrew's python@3.12, or python.org), then run this again, or \
point EYEOS_PYTHON at it."
    fi
fi

# --- 3. .venv ------------------------------------------------------------------------
VENV_PY=.venv/bin/python
if [ -e "$VENV_PY" ]; then
    venv_version=$(py_version "$VENV_PY")
    if [ -z "$venv_version" ] || ! is_supported "$venv_version"; then
        die ".venv already exists and holds Python ${venv_version:-that does not run}, not \
3.10-3.12. Remove it (rm -rf .venv) and run this again."
    fi
    say "reusing .venv (Python $venv_version)"
else
    say "creating .venv with $PY (Python $PYV)"
    run "$PY" -m venv .venv || die "could not create .venv. On Debian/Ubuntu the venv module \
is a separate package: sudo apt install python$PYV-venv"
fi

# --- 4. install, check, test ---------------------------------------------------------
say "installing EyeOS with the extras $EXTRAS (.venv ends up about 1 GB)"
run "$VENV_PY" -m pip install --upgrade pip || die "could not upgrade pip (no network?)"
run "$VENV_PY" -m pip install -e ".[$EXTRAS]" || die "pip could not install EyeOS; its error \
is above. On Linux, pynput (the windows-fallback extra) builds evdev from source, which needs \
a C compiler and the Python headers (Debian/Ubuntu: sudo apt install build-essential \
python$PYV-dev)."

say "checking the live stack (OpenCV, MediaPipe, pynput, scipy)"
run "$VENV_PY" scripts/check_setup.py || die "the setup check failed: the [FAIL] lines above \
say what to fix (docs/LOCAL_SETUP.md, Troubleshooting)"

if [ -z "${EYEOS_SKIP_TESTS:-}" ]; then
    say "running the test suite"
    run "$VENV_PY" -m pytest -q || die "some tests failed (output above). The install itself \
worked; a failing test on a new machine is worth reporting."
fi

# --- 5. next steps -------------------------------------------------------------------
# printf, not cat: everything up to here uses shell builtins and the chosen Python only.
printf '%s\n' \
    "" \
    "Setup finished. Next steps (docs/LOCAL_SETUP.md has the details):" \
    "  source .venv/bin/activate                  # in every new terminal" \
    "  eyeos run --config config.example.yaml     # dry run on the synthetic fixture, no camera" \
    "  cp config.example.yaml config.yaml         # your own config (gitignored)" \
    "  # In config.yaml set input.source: webcam and screen.width_px/height_px, keep" \
    "  # output.backend: dry_run, then a first live run that never moves the mouse:" \
    "  eyeos run --config config.yaml --calibrate --overlay --view"
