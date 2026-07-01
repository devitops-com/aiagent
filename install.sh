#!/bin/sh
# aiagent bootstrap installer.
#
# The shipped installer (aiagent-install.sh) is a makeself self-extracting
# archive: it finds its embedded payload by seeking within its own file, so it
# CANNOT be piped straight into a shell. This tiny bootstrap downloads it to a
# temp file and runs it — which is what makes the one-liner work:
#
#     curl -fsSL https://github.com/devitops-com/aiagent/releases/latest/download/install.sh | sh
#
# Environment:
#   AIAGENT_PREFIX   install prefix (default ~/.local); honored by the installer
#   AIAGENT_VERSION  release tag to install (default: latest), e.g. v0.1.0
#
# Custom prefix with the pipe form:
#   curl -fsSL .../install.sh | AIAGENT_PREFIX=/usr/local sh
set -eu

REPO="devitops-com/aiagent"
ASSET="aiagent-install.sh"
VERSION="${AIAGENT_VERSION:-latest}"

# The bundle is a linux-x86_64 build; fail fast anywhere else.
OS="$(uname -s)"
ARCH="$(uname -m)"
if [ "$OS" != "Linux" ] || [ "$ARCH" != "x86_64" ]; then
    echo "aiagent: unsupported platform ${OS}/${ARCH}; only Linux x86_64 is supported" >&2
    exit 1
fi

if command -v curl >/dev/null 2>&1; then
    dl() { curl -fsSL "$1" -o "$2"; }
elif command -v wget >/dev/null 2>&1; then
    dl() { wget -qO "$2" "$1"; }
else
    echo "aiagent: need curl or wget to download the installer" >&2
    exit 1
fi

if [ "$VERSION" = "latest" ]; then
    URL="https://github.com/${REPO}/releases/latest/download/${ASSET}"
else
    URL="https://github.com/${REPO}/releases/download/${VERSION}/${ASSET}"
fi

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT INT TERM

echo "aiagent: downloading ${URL}" >&2
if ! dl "$URL" "$TMP"; then
    echo "aiagent: download failed (${URL})" >&2
    exit 1
fi

echo "aiagent: running installer" >&2
# </dev/null: the makeself installer is non-interactive; detach it from the
# (already-consumed) pipe stdin used when this bootstrap is run via `curl | sh`.
sh "$TMP" </dev/null
