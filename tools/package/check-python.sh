#!/usr/bin/env bash
#
# check-python.sh — gate the staged interpreter tree: exactly the pinned CPython.
#
#     bash tools/package/check-python.sh PYTHON_DIR X.Y.Z
#
# PYTHON_DIR is a staged python-build-standalone tree (bin/pythonX.Y, lib/...), X.Y.Z the
# version .python-version pins. It fails if bin/pythonX.Y does not run, or reports any
# version but X.Y.Z.
# Exit status: 0 clean, 1 with the problem on stderr, 2 on bad usage.
set -euo pipefail

usage() { echo "usage: check-python.sh PYTHON_DIR X.Y.Z" >&2; exit 2; }
[ $# -eq 2 ] || usage
DIR="$1" VERSION="$2"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || usage
[ -d "$DIR" ] || { echo "check-python: not a directory: $DIR" >&2; exit 2; }
MINOR="${VERSION%.*}"

PY="$DIR/bin/python$MINOR"
staged="$("$PY" -I -c 'import platform; print(platform.python_version())' 2>/dev/null)" \
    || { echo "ERROR: the staged interpreter $PY cannot run" >&2; exit 1; }
[ "$staged" = "$VERSION" ] || {
    echo "ERROR: the staged interpreter is CPython $staged, .python-version pins $VERSION" >&2
    exit 1
}
echo "    ok: CPython $VERSION"
