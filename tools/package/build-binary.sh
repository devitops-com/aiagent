#!/usr/bin/env bash
#
# build-binary.sh — produce `dist/aiagent-install.sh`, a self-extracting
# **makeself** installer carrying a relocatable, sourceless-precompiled CPython
# 3.x with aiagent + every runtime dependency.
#
# The heavy tree is zstd-compressed and decompressed at install time by a BUNDLED
# static zstd, so target hosts need neither Python nor zstd. `.py` sources are
# dropped (sourceless `.pyc` only). makeself provides SHA256 integrity. x86-64.
#
# Build deps: uv, makeself, curl, gcc/make (to build the static zstd once).
# Run `make lock` first (produces requirements.txt for the constraints step).
set -euo pipefail
export PYTHONNOUSERSITE=1   # hermetic build: never satisfy deps from the user site

APP="aiagent"
ARCH="$(uname -m)"
[ "$ARCH" = "x86_64" ] || { echo "ERROR: this build targets x86_64 only (got $ARCH)" >&2; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY_VERSION="$(tr -d '[:space:]' < "$ROOT/.python-version")"
[ -n "$PY_VERSION" ] || { echo "ERROR: cannot read .python-version" >&2; exit 1; }
PY_NODOT="${PY_VERSION/./}"

DIST="$ROOT/dist"
STAGE="$DIST/.build"
MKDIR="$DIST/.mkself"
CONSTRAINTS="$DIST/constraints.txt"
STARTUP_IN="$ROOT/tools/package/startup.sh.in"
OUT="$DIST/$APP-install.sh"
REQ="$ROOT/requirements.txt"
CACHE="$ROOT/.cache/aiagent-build"
ZSTD_BIN="$CACHE/zstd-static-$ARCH"
ZSTD_VERSION="1.5.6"
ZSTD_SHA256="8c29e06cf42aacc1eafc4077ae2ec6c6fcb96a626157e0593d5e82a34fd403c1"

VERSION="$(grep -m1 -E '^version[[:space:]]*=' pyproject.toml | cut -d'"' -f2)"
[ -n "$VERSION" ] || { echo "ERROR: cannot read version from pyproject.toml" >&2; exit 1; }

command -v makeself >/dev/null || { echo "ERROR: makeself not installed (apt-get install makeself)" >&2; exit 1; }
[ -f "$REQ" ] || { echo "ERROR: $REQ missing — run 'make lock' first" >&2; exit 1; }
if grep -qiE '^(torch|nvidia-)' "$REQ"; then
    echo "ERROR: torch/nvidia-* in requirements.txt; aiagent must stay torch-free" >&2; exit 1
fi

echo "==> $APP $VERSION -> makeself installer (CPython $PY_VERSION, sourceless, zstd -19)"

# --- 1. Clean ------------------------------------------------------------
rm -rf "$STAGE" "$MKDIR" "$CONSTRAINTS" build src/*.egg-info
mkdir -p "$STAGE" "$MKDIR" "$CACHE"

# --- 2. Build the aiagent wheel -----------------------------------------
echo "==> Building $APP wheel"
uv build --wheel -o "$DIST" >/dev/null
WHEEL="$(ls "$DIST"/${APP}-${VERSION}-*.whl)"

# --- 3. Pin dependency versions from the project lock -------------------
grep -E '^[A-Za-z0-9._-]+==' "$REQ" | sed -E 's/[[:space:]]*\\$//' > "$CONSTRAINTS"

# --- 4. Stage a standalone, relocatable CPython -------------------------
uv python install "$PY_VERSION" >/dev/null 2>&1 || true
PYBIN="$(uv python find "$PY_VERSION")"
BASEP="$("$PYBIN" -c 'import sys; print(sys.base_prefix)')"
echo "==> Staging interpreter from $BASEP"
cp -a "$BASEP" "$STAGE/python"
PY="$STAGE/python/bin/python${PY_VERSION}"
rm -f "$STAGE/python/lib/python${PY_VERSION}/EXTERNALLY-MANAGED"

# --- 5. Prune unused stdlib (incl. Tcl/Tk v8 AND v9) --------------------
( cd "$STAGE/python/lib/python${PY_VERSION}" && rm -rf \
    test tkinter turtledemo idlelib lib2to3 ensurepip \
    config-${PY_VERSION}-*-linux-gnu 2>/dev/null || true )
# tkinter is gone, so Tcl/Tk is dead weight. The standalone ships v9
# (libtcl9tk9.0.so, tcl9/, tk9/) — match v8 AND v9 (the old glob missed v9).
( cd "$STAGE/python/lib" && rm -rf \
    tcl8* tk8* tcl9* tk9* Tix* itcl* tdbc* thread* libtcl* libtk* 2>/dev/null || true )

# --- 5b. Reset site-packages to a clean baseline (keep only pip) --------
( cd "$STAGE/python/lib/python${PY_VERSION}/site-packages" && for d in *; do
    case "$d" in pip|pip-*|__pycache__) ;; *) rm -rf "$d" ;; esac
  done )

# --- 6. Install aiagent + deps into the staged interpreter -------------
echo "==> Installing $APP + dependencies"
"$PY" -m pip install --no-input --disable-pip-version-check --no-warn-script-location \
    --no-compile "$WHEEL" -c "$CONSTRAINTS" >/dev/null
SP="$STAGE/python/lib/python${PY_VERSION}/site-packages"
( cd "$SP" && rm -rf pip setuptools wheel pkg_resources _distutils_hack 2>/dev/null || true )
find "$SP" -name direct_url.json -delete 2>/dev/null || true

# --- 6b. Strip dead weight ---------------------------------------------
# Keep numpy / tokenizers / tiktoken — needed for future ML features (RAG).
[ -d "$SP/torch" ] && { echo "ERROR: torch leaked into the bundle" >&2; exit 1; }
echo "==> Stripping dead weight (dep CLIs, headers, hf_xet, dep tests)"
( cd "$STAGE/python/bin" && for f in *; do
    case "$f" in python${PY_VERSION}|python3|python|${APP}) ;; *) rm -f "$f" ;; esac
  done )
rm -rf "$STAGE/python/include" "$STAGE/python/share"
rm -f "$STAGE/python/lib"/libpython*.a
find "$SP" -type d -name tests -prune -exec rm -rf {} + 2>/dev/null || true
# hf_xet: HuggingFace Hub Xet download accelerator — aiagent never hits the Hub.
rm -rf "$SP/hf_xet" "$SP"/hf_xet-*.dist-info 2>/dev/null || true

# --- 7. Sanity-check the staged interpreter ----------------------------
# NB: do NOT `strip` libpython — it corrupts PBS symbol-version tables.
"$PY" -c "import sqlite3, ssl, ctypes" \
    || { echo "ERROR: staged interpreter is not functional" >&2; exit 1; }

# --- 8. Precompile EVERYTHING (unchecked-hash, relative paths) ----------
echo "==> Precompiling all modules"
"$PY" -m compileall -q -f -j 0 -s "$STAGE/python" --invalidation-mode unchecked-hash "$STAGE/python" >/dev/null 2>&1 || true

# --- 8b. Drop .py sources: relocate to sourceless .pyc, delete .py ------
echo "==> Dropping .py sources (sourceless .pyc)"
"$PY" -s - "$STAGE/python" "$PY_NODOT" <<'PYEOF'
import os, sys
root, tag = sys.argv[1], sys.argv[2]
suffix = f".cpython-{tag}.pyc"
for dp, _dn, fn in os.walk(root):
    if os.path.basename(dp) != "__pycache__":
        continue
    parent = os.path.dirname(dp)
    for f in fn:
        if f.endswith(suffix):
            os.replace(os.path.join(dp, f), os.path.join(parent, f[:-len(suffix)] + ".pyc"))
for dp, _dn, fn in os.walk(root, topdown=False):
    for f in fn:
        if f.endswith(".py"):
            os.remove(os.path.join(dp, f))
    if os.path.basename(dp) == "__pycache__":
        try:
            os.rmdir(dp)
        except OSError:
            pass
PYEOF

# --- 8c. Sanity on the sourceless tree ---------------------------------
"$PY" -s -c "import aiagent, dspy" \
    || { echo "ERROR: sourceless bundle not importable" >&2; exit 1; }

# --- 9. Obtain a static zstd (cached across builds) --------------------
if ! "$ZSTD_BIN" --version >/dev/null 2>&1; then
    echo "==> Building static zstd $ZSTD_VERSION (cached at $ZSTD_BIN)"
    ztmp="$(mktemp -d)"
    curl -fsSL -o "$ztmp/z.tgz" \
        "https://github.com/facebook/zstd/releases/download/v${ZSTD_VERSION}/zstd-${ZSTD_VERSION}.tar.gz"
    echo "$ZSTD_SHA256  $ztmp/z.tgz" | sha256sum -c - >/dev/null \
        || { echo "ERROR: zstd source checksum mismatch" >&2; exit 1; }
    tar -C "$ztmp" -xzf "$ztmp/z.tgz"
    make -C "$ztmp/zstd-${ZSTD_VERSION}/programs" zstd \
        HAVE_ZLIB=0 HAVE_LZMA=0 HAVE_LZ4=0 ZSTD_LEGACY_SUPPORT=0 \
        LDFLAGS=-static -j"$(nproc)" >/dev/null 2>&1
    strip "$ztmp/zstd-${ZSTD_VERSION}/programs/zstd"
    cp "$ztmp/zstd-${ZSTD_VERSION}/programs/zstd" "$ZSTD_BIN"
    rm -rf "$ztmp"
fi
cp "$ZSTD_BIN" "$MKDIR/zstd"
chmod +x "$MKDIR/zstd"

# --- 10. Compress the heavy payload (zstd -19, multithreaded) ----------
mkdir -p "$STAGE/doc"
cp -p README.md LICENSE CHANGELOG.md "$STAGE/doc/" 2>/dev/null || true
echo "==> Compressing payload (zstd -19 -T0)"
tar -C "$STAGE" -cf - python doc | "$MKDIR/zstd" -19 -T0 -q -o "$MKDIR/bundle.tar.zst"

# --- 11. Generate the makeself startup script (baked-in version/py) ----
{
    printf '%s\n' '#!/bin/sh'
    printf 'AIAGENT_VERSION=%s\n' "$VERSION"
    printf 'PYVER=%s\n' "$PY_VERSION"
    cat "$STARTUP_IN"
} > "$MKDIR/startup.sh"
chmod +x "$MKDIR/startup.sh"

# --- 12. Assemble the self-extracting installer with makeself ----------
# --nox11: run the startup script inline (no xterm). --nocomp: the payload is
# already zstd-compressed. --sha256: integrity check on extraction.
echo "==> Assembling makeself installer"
rm -f "$OUT"
makeself --nox11 --nocomp --sha256 --tar-quietly \
    "$MKDIR" "$OUT" "aiagent $VERSION installer" ./startup.sh >/dev/null
rm -rf "$STAGE" "$MKDIR"

# --- 13. Smoke test: install to a temp prefix + run --------------------
# Exercises the two failure modes that shipped in v0.1.0:
#   (a) Typer/Click make_metavar: render usage/help for an ARG-bearing command.
#   (b) sourceless skill loader: actually load the built-in skill entry .pyc.
echo "==> Smoke test (install to a temp prefix + run)"
TPREFIX="$DIST/.smoketest"
rm -rf "$TPREFIX"
AIAGENT_PREFIX="$TPREFIX" sh "$OUT" >/dev/null
"$TPREFIX/bin/$APP" --help >/dev/null
# (a) An Arguments panel forces make_metavar(ctx) on a positional; a pre-8.2
# Typer/Click pair crashes the render here. Capture the output (do NOT pipe into
# grep -q: under `set -o pipefail` its early exit SIGPIPEs the producer and
# fails the pipeline) and assert the arg metavar (SKILL) actually rendered.
run_help="$("$TPREFIX/bin/$APP" run --help)" \
    || { echo "ERROR: 'run --help' crashed (Typer/Click make_metavar break?)" >&2; exit 1; }
case "$run_help" in
    *SKILL*) ;;
    *) echo "ERROR: 'run --help' did not render the SKILL argument" >&2; exit 1 ;;
esac
"$TPREFIX/bin/$APP" eval --help >/dev/null
skills_out="$("$TPREFIX/bin/$APP" skills list)"
case "$skills_out" in
    *extract*) ;;
    *) echo "ERROR: built-in skills not discoverable from the bundle" >&2; exit 1 ;;
esac
# (b) load the built-in entry modules through the actual (sourceless) loader —
# build_module() imports the .pyc and constructs the dspy.Module (no router).
# Run from an ON-DISK file so __main__ has a real __file__: litellm imports
# python-dotenv, whose find_dotenv() walks the stack for a frame whose file
# exists on disk and asserts if none do. A sourceless tree has no such frame
# except a real script file — exactly the installed `aiagent` console script's
# situation — so a stdin-heredoc probe asserts here where a file probe does not.
PROBE="$DIST/.skill-load-probe.py"
cat > "$PROBE" <<'PYEOF'
from aiagent.config import load_settings
from aiagent.skills.registry import load_registry
from aiagent.skills.loader import build_module
reg, _ = load_registry(load_settings())
for name in ("chat", "extract"):
    build_module(reg.get(name))
PYEOF
if ! "$TPREFIX/bin/python${PY_VERSION}" -s "$PROBE" >/dev/null; then
    echo "ERROR: built-in skills fail to load from the sourceless bundle" >&2
    rm -f "$PROBE"; exit 1
fi
rm -f "$PROBE"
"$TPREFIX/bin/$APP" doctor --offline >/dev/null
rm -rf "$TPREFIX"
echo "    ok (--help, run/eval --help render, skills list -> extract, sourceless skill load, doctor --offline)"

# --- 14. Report --------------------------------------------------------
SIZE="$(du -h "$OUT" | cut -f1)"
echo ""
echo "Built installer:"
echo "  $OUT  ($SIZE)"
echo ""
echo "Install on any linux-x86_64 host (no Python, no zstd required):"
echo "  ./aiagent-install.sh                             # -> ~/.local"
echo "  AIAGENT_PREFIX=/usr/local ./aiagent-install.sh   # system / devai image install"
echo "  aiagent --help"
