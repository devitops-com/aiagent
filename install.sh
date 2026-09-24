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
#   TMPDIR           where the installer is downloaded and unpacked (default
#                    /var/tmp, never /tmp: that is often a small RAM-backed tmpfs)
#   AIAGENT_VERIFY   1: before running the downloaded installer, verify its GitHub
#                    artifact attestation (the release workflow built it from this
#                    repository) with `gh attestation verify`. Needs the GitHub CLI
#                    (gh) on PATH, logged in. Any failure, or no gh, stops here: the
#                    installer is not run. Default 0 (no check). Releases up to
#                    v0.3.1 were built locally and have no attestation, so they fail it.
#
# Custom prefix, or verification, with the pipe form:
#   curl -fsSL .../install.sh | sudo AIAGENT_PREFIX=/usr/local sh
#   curl -fsSL .../install.sh | AIAGENT_VERIFY=1 sh
set -eu

REPO="devitops-com/aiagent"
ASSET="aiagent-install.sh"
VERSION="${AIAGENT_VERSION:-latest}"
VERIFY="${AIAGENT_VERIFY:-0}"
TMPDIR="${TMPDIR:-/var/tmp}"
export TMPDIR   # the makeself installer unpacks its payload under $TMPDIR too

# Fail closed: a verification that was asked for is never skipped quietly.
case "$VERIFY" in
    0|1) ;;
    *) echo "aiagent: AIAGENT_VERIFY must be 1 or 0, got '${VERIFY}'" >&2; exit 1 ;;
esac
if [ "$VERIFY" = 1 ] && ! command -v gh >/dev/null 2>&1; then
    echo "aiagent: AIAGENT_VERIFY=1 needs the GitHub CLI (gh) on PATH: https://cli.github.com" >&2
    exit 1
fi

# The bundle is a linux-x86_64 build; fail fast anywhere else.
OS="$(uname -s)"
ARCH="$(uname -m)"
if [ "$OS" != "Linux" ] || [ "$ARCH" != "x86_64" ]; then
    echo "aiagent: unsupported platform ${OS}/${ARCH}; only Linux x86_64 is supported" >&2
    exit 1
fi

# HTTPS only, redirects included (GitHub redirects release downloads to its CDN).
# wget has no option for that (--https-only applies to recursive downloads only),
# so it is just the fallback for hosts without curl; AIAGENT_VERIFY=1 checks what
# either one downloaded.
if command -v curl >/dev/null 2>&1; then
    dl() { curl --proto '=https' --tlsv1.2 -fsSL "$1" -o "$2"; }
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

TMP="$(mktemp -p "$TMPDIR" aiagent-install.XXXXXX)"
trap 'rm -f "$TMP"' EXIT INT TERM

echo "aiagent: downloading ${URL}" >&2
if ! dl "$URL" "$TMP"; then
    echo "aiagent: download failed (${URL})" >&2
    exit 1
fi

if [ "$VERIFY" = 1 ]; then
    echo "aiagent: verifying the build provenance attestation (gh attestation verify)" >&2
    if ! gh attestation verify "$TMP" --repo "$REPO" >&2; then
        echo "aiagent: attestation verification failed; the installer was not run" >&2
        exit 1
    fi
fi

echo "aiagent: running installer" >&2
# </dev/null: the makeself installer is non-interactive; detach it from the
# (already-consumed) pipe stdin used when this bootstrap is run via `curl | sh`.
sh "$TMP" </dev/null
