<#
.SYNOPSIS
    One-time EyeOS setup on Windows: Python 3.12, .venv, the extras, the checks.

.DESCRIPTION
    UNTESTED: this script has never been run. It was written in a Linux cloud container
    with no PowerShell and no Windows, so not one line of it has executed. Its Linux and
    macOS twin, scripts/setup_unix.sh, does the same steps and has run end to end on
    Ubuntu. If a step here fails, docs/LOCAL_SETUP.md (section 3, "By hand") does the
    same thing one command at a time.

    From the repo root, it:
      1. finds Python 3.12, 3.11 or 3.10 through the py launcher (py -3.12, ...).
         EyeOS needs mediapipe<0.10.30, which has no wheels for Python 3.13 or newer,
         and whose Windows wheels are for 64-bit x86 Python (win-amd64) only;
      2. creates .venv, or reuses one made with a supported Python;
      3. installs EyeOS in editable mode with the extras
         dev,webcam,windows-fallback,public-eval;
      4. runs scripts\check_setup.py (the real OpenCV + MediaPipe) and the test suite;
      5. prints the next steps.

.PARAMETER SkipTests
    Don't run the test suite.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1

    -ExecutionPolicy Bypass applies to this one run only and changes no setting.
#>
[CmdletBinding()]
param(
    [switch]$SkipTests
)

Set-StrictMode -Version 2.0
# Not "Stop": with it, Windows PowerShell 5.1 turns any stderr line from a native
# program (a pip warning, py's "not found") into a terminating error. Each native
# command's exit code is checked by hand instead.
$ErrorActionPreference = "Continue"

$Extras = "dev,webcam,windows-fallback,public-eval"
$Candidates = @("3.12", "3.11", "3.10")
$Why312 = "EyeOS needs Python 3.10-3.12: its webcam extra needs mediapipe<0.10.30 " +
    "(the last versions with the legacy Solutions API), which has no wheels for " +
    "Python 3.13 or newer."

function Fail([string]$Message) {
    Write-Host ""
    Write-Host "setup_windows.ps1: error: $Message" -ForegroundColor Red
    exit 1
}

function Say([string]$Message) {
    Write-Host "==> $Message" -ForegroundColor Cyan
}

# --- 1. the repo root ---------------------------------------------------------------
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot
if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot "pyproject.toml"))) {
    Fail ("no pyproject.toml in $RepoRoot. Run this script from an EyeOS checkout: " +
        "powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1")
}

# --- 2. a Python 3.10-3.12 through the py launcher ----------------------------------
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Fail ("the Python launcher 'py' was not found. Install Python 3.12 with " +
        "'winget install Python.Python.3.12', then open a NEW PowerShell window " +
        "(so it sees the new PATH) and run this again.")
}

$PyVersion = $null
foreach ($v in $Candidates) {
    $null = & py "-$v" -c "import sys" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $PyVersion = $v
        break
    }
}
if (-not $PyVersion) {
    $listed = (& py -0 2>&1 | Out-String).Trim()
    Fail ("the py launcher found no Python 3.10-3.12. It lists:`n$listed`n$Why312 " +
        "Install Python 3.12 with 'winget install Python.Python.3.12', then open a " +
        "new PowerShell window and run this again.")
}

$Platform = (& py "-$PyVersion" -c "import sysconfig; print(sysconfig.get_platform())" | Out-String).Trim()
if ($Platform -ne "win-amd64") {
    Fail ("py -$PyVersion is a '$Platform' Python, but mediapipe<0.10.30 has Windows " +
        "wheels for 64-bit x86 Python (win-amd64) only, so pip would fail. Install the " +
        "64-bit x86 Python 3.12 from python.org and run this again.")
}

# --- 3. .venv -------------------------------------------------------------------------
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $VenvPython) {
    $venvVersion = (& $VenvPython -c "import sys; print('%d.%d' % sys.version_info[:2])" | Out-String).Trim()
    if ($Candidates -notcontains $venvVersion) {
        Fail (".venv already exists and holds Python '$venvVersion', not 3.10-3.12. " +
            "Delete it (Remove-Item -Recurse -Force .venv) and run this again.")
    }
    Say "reusing .venv (Python $venvVersion)"
} else {
    Say "creating .venv with Python $PyVersion"
    Write-Host "+ py -$PyVersion -m venv .venv"
    & py "-$PyVersion" -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "could not create .venv with py -$PyVersion (output above)" }
}

# --- 4. install, check, test ----------------------------------------------------------
Say "installing EyeOS with the extras $Extras (.venv took about 1 GB on Linux)"
Write-Host "+ .venv\Scripts\python.exe -m pip install --upgrade pip"
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Fail "could not upgrade pip (no network?)" }

Write-Host "+ .venv\Scripts\python.exe -m pip install -e `".[$Extras]`""
& $VenvPython -m pip install -e ".[$Extras]"
if ($LASTEXITCODE -ne 0) {
    Fail ("pip could not install EyeOS; its error is above. 'No matching distribution " +
        "found for mediapipe' means the wrong Python: $Why312")
}

Say "checking the live stack (OpenCV, MediaPipe, pynput, scipy)"
Write-Host "+ .venv\Scripts\python.exe scripts\check_setup.py"
& $VenvPython scripts\check_setup.py
if ($LASTEXITCODE -ne 0) {
    Fail ("the setup check failed: the [FAIL] lines above say what to fix " +
        "(docs\LOCAL_SETUP.md, Troubleshooting)")
}

if (-not $SkipTests) {
    Say "running the test suite"
    Write-Host "+ .venv\Scripts\python.exe -m pytest -q"
    & $VenvPython -m pytest -q
    if ($LASTEXITCODE -ne 0) {
        Fail ("some tests failed (output above). The install itself worked. The suite " +
            "had never run on Windows before, so a failure here is worth reporting.")
    }
}

# --- 5. next steps --------------------------------------------------------------------
Write-Host ""
Write-Host "Setup finished. Next steps (docs\LOCAL_SETUP.md has the details):" -ForegroundColor Green
Write-Host "  .venv\Scripts\Activate.ps1                   # in every new PowerShell window"
Write-Host "  eyeos run --config config.example.yaml       # dry run on the synthetic fixture, no camera"
Write-Host "  Copy-Item config.example.yaml config.yaml    # your own config (gitignored)"
Write-Host "  # In config.yaml set input.source: webcam and screen.width_px/height_px, keep"
Write-Host "  # output.backend: dry_run, then a first live run that never moves the mouse:"
Write-Host "  eyeos run --config config.yaml --calibrate --overlay --view"
Write-Host "If Activate.ps1 is refused ('running scripts is disabled on this system'), run once:"
Write-Host "  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned"
