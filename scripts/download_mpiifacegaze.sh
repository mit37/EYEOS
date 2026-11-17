#!/usr/bin/env bash
# Thin wrapper (STANDARDS §2 names dataset fetchers scripts/download_*.sh). The work is
# done by download_mpiifacegaze.py, which needs only the Python standard library, so on
# Windows run that directly: python scripts\download_mpiifacegaze.py --accept-license
#
#   scripts/download_mpiifacegaze.sh --accept-license [--dest datasets]
#
# Set PYTHON to choose the interpreter (default: python3).
set -euo pipefail
exec "${PYTHON:-python3}" "$(dirname "$0")/download_mpiifacegaze.py" "$@"
