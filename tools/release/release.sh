#!/usr/bin/env bash
#
# release.sh — cut a release of aiagent: tag it, and let CI build, attest and publish it.
#
#     bash tools/release/release.sh [X.Y.Z]        (make release [VERSION=X.Y.Z])
#
# Flow: guard the repo state -> with X.Y.Z, set pyproject.toml's [project] version
# to it -> promote the CHANGELOG's [Unreleased] section to the version -> ONE
# commit "chore: release vX.Y.Z" (both files) -> annotated tag -> push commit and
# tag atomically. Nothing is built here and no release is created here:
# the pushed tag triggers .github/workflows/release.yml, which builds the installer
# from the tag (full smoke test; it bundles the promoted CHANGELOG), attests
# dist/aiagent-install.sh and install.sh (GitHub artifact attestations can only be
# made inside GitHub Actions) and publishes the GitHub release. This script then
# follows that run with `gh run watch` and prints the release URL.
#
# The version is read from pyproject.toml (the single source of truth, same as
# tools/package/build-binary.sh). Without X.Y.Z the current version is released.
# X.Y.Z (digits, no leading zeros) bumps it in the release commit itself, so no
# separate bump commit (and no CI runs for one) is needed; it may equal the current
# version (a plain release) but not be lower (compared as numbers). Fill in the
# CHANGELOG's [Unreleased] section before running this.
#
# Guards (all must pass before anything mutates): X.Y.Z well-formed and not lower,
# on `main`, clean working tree (untracked files and assume-unchanged /
# skip-worktree entries included), local == origin/main, neither the tag nor the
# GitHub release exists yet, and a non-empty [Unreleased] section.
#
# Non-interactive use: set AIAGENT_RELEASE_ASSUME_YES=1 to skip the prompt.
# AIAGENT_RELEASE_WATCH_WAIT: how many seconds to wait for the tag's workflow run
# to show up (default 60); without one, the script says where to follow it and
# still succeeds.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BRANCH="main"
CHANGELOG="CHANGELOG.md"
PYPROJECT="pyproject.toml"
WORKFLOW="release.yml"
WATCH_WAIT="${AIAGENT_RELEASE_WATCH_WAIT:-60}"
WATCH_POLL=3

[ $# -le 1 ] || { echo "usage: release.sh [X.Y.Z]" >&2; exit 2; }

# The version is [project]'s (tomllib's ["project"]["version"], what the tag check in
# release.yml compares the tag with): another table's `version` key is not it. Step 4
# bumps the same line.
IN_PROJECT='/^\[/ { in_project = ($0 ~ /^\[project\][[:space:]]*$/) }'
VERSION_LINE='in_project && /^version[[:space:]]*=/'
# `|| true` keeps a failed read from tripping `set -e`/pipefail before the explicit
# emptiness check below can emit a friendly diagnostic.
CURRENT_VERSION="$(awk "$IN_PROJECT $VERSION_LINE"' { print; exit }' "$PYPROJECT" \
    | cut -d'"' -f2 || true)"
[ -n "$CURRENT_VERSION" ] || { echo "ERROR: cannot read version from $PYPROJECT" >&2; exit 1; }

# --- The version to release ---------------------------------------------------
XYZ='^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$'
# version_lt A B: A < B, both X.Y.Z, compared part by part as numbers (1.9.10 > 1.9.2).
version_lt() {
    local -a a b
    local i
    IFS=. read -r -a a <<<"$1"
    IFS=. read -r -a b <<<"$2"
    for i in 0 1 2; do
        [ "${a[i]}" -eq "${b[i]}" ] || { [ "${a[i]}" -lt "${b[i]}" ]; return; }
    done
    return 1
}
VERSION="$CURRENT_VERSION"
if [ $# -eq 1 ]; then
    [[ "$1" =~ $XYZ ]] \
        || { echo "ERROR: '$1' is not a version X.Y.Z (digits, no leading zeros)" >&2; exit 1; }
    [[ "$CURRENT_VERSION" =~ $XYZ ]] \
        || { echo "ERROR: the current version $CURRENT_VERSION ($PYPROJECT) is not X.Y.Z" >&2; exit 1; }
    if version_lt "$1" "$CURRENT_VERSION"; then
        echo "ERROR: version $1 is lower than the current version $CURRENT_VERSION" >&2; exit 1
    fi
    VERSION="$1"
fi
TAG="v$VERSION"

# --- cleanup / abort safety --------------------------------------------------
# Before the release commit lands, any failure should leave the tree pristine:
# revert the version bump and the CHANGELOG promotion, remove the awk-rewrite temp files.
PREPARED=()  # the files the release commit is to carry, once rewritten
COMMITTED=0
cleanup() {
    rm -f "$CHANGELOG.tmp" "$PYPROJECT.tmp"
    if [ "${#PREPARED[@]}" -gt 0 ] && [ "$COMMITTED" != "1" ]; then
        # Restore from HEAD (not the index) so the rewrites are reverted even
        # after they were `git add`ed; `git checkout HEAD --` also unstages them.
        git checkout HEAD -- "${PREPARED[@]}" 2>/dev/null || true
        echo "==> Reverted ${PREPARED[*]} (release aborted before commit)" >&2
    fi
    return 0
}
trap cleanup EXIT

# --- 0. Tooling --------------------------------------------------------------
command -v gh  >/dev/null || { echo "ERROR: gh CLI not installed" >&2; exit 1; }
command -v git >/dev/null || { echo "ERROR: git not installed" >&2; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "ERROR: gh is not authenticated (run: gh auth login)" >&2; exit 1; }

# --- 1. Guards ---------------------------------------------------------------
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[ "$CURRENT_BRANCH" = "$BRANCH" ] \
    || { echo "ERROR: not on '$BRANCH' (currently on '$CURRENT_BRANCH')" >&2; exit 1; }

# --untracked-files=all overrides a status.showUntrackedFiles=no config: what was
# tested locally must be exactly what the tag points at (CI builds from the tag).
[ -z "$(git status --porcelain --untracked-files=all)" ] \
    || { echo "ERROR: working tree is not clean; commit or stash first" >&2; exit 1; }
# git status cannot see edits to files marked assume-unchanged (lower-case tag in
# `git ls-files -v`) or skip-worktree (S).
HIDDEN="$(git ls-files -v | grep -E '^([a-z]|S) ' || true)"
[ -z "$HIDDEN" ] || {
    echo "ERROR: files hidden from git status (assume-unchanged / skip-worktree):" >&2
    printf '%s\n' "$HIDDEN" | cut -c3- | sed 's/^/  /' >&2
    echo "       clear with: git update-index --no-assume-unchanged --no-skip-worktree FILE" >&2
    exit 1
}

if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
    echo "ERROR: tag $TAG already exists locally" >&2; exit 1
fi
if git ls-remote --exit-code --tags origin "$TAG" >/dev/null 2>&1; then
    echo "ERROR: tag $TAG already exists on origin" >&2; exit 1
fi
if gh release view "$TAG" >/dev/null 2>&1; then
    echo "ERROR: a GitHub release for $TAG already exists" >&2; exit 1
fi

echo "==> Fetching origin/$BRANCH"
git fetch --quiet origin "$BRANCH"
LOCAL_HEAD="$(git rev-parse HEAD)"
REMOTE_HEAD="$(git rev-parse "origin/$BRANCH")"
[ "$LOCAL_HEAD" = "$REMOTE_HEAD" ] \
    || { echo "ERROR: local $BRANCH differs from origin/$BRANCH; push or pull first" >&2; exit 1; }

# --- 2. The [Unreleased] section must have something to release --------------
# The release workflow publishes the promoted section as the release notes.
bash tools/release/release-notes.sh Unreleased "$CHANGELOG" >/dev/null \
    || { echo "ERROR: nothing to release" >&2; exit 1; }

# --- 3. Confirm --------------------------------------------------------------
REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || echo '?')"
if [ "$VERSION" = "$CURRENT_VERSION" ]; then
    VERSION_SHOWN="$VERSION"
    BUMPS=""
else
    VERSION_SHOWN="$CURRENT_VERSION -> $VERSION"
    BUMPS="sets $PYPROJECT's version, "
fi
cat <<EOF

About to release:
  repo    : $REPO
  version : $VERSION_SHOWN
  tag     : $TAG
  branch  : $BRANCH  ($(git rev-parse --short HEAD))

This ${BUMPS}promotes the CHANGELOG, commits, tags and pushes. The pushed tag starts the
$WORKFLOW workflow, which builds, attests and publishes the GitHub release.
It is outward-facing and hard to undo.
EOF
if [ "${AIAGENT_RELEASE_ASSUME_YES:-0}" != "1" ]; then
    if [ -t 0 ]; then
        printf 'Continue? [y/N] '; read -r reply
    # The /dev/tty node always exists; without a controlling terminal opening it
    # fails (and set -e would abort without a word), so probe it instead.
    elif { : > /dev/tty; } 2>/dev/null; then
        printf 'Continue? [y/N] ' > /dev/tty; read -r reply < /dev/tty
    else
        echo "ERROR: not a TTY; set AIAGENT_RELEASE_ASSUME_YES=1 to proceed non-interactively" >&2
        exit 1
    fi
    case "$reply" in
        [yY]|[yY][eE][sS]) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
fi

# --- 4. Bump the version, promote the CHANGELOG -------------------------------
# The [project] version line read above, only its first one: sub() swaps the quoted
# value alone, so the line keeps its spacing and any comment.
if [ "$VERSION" != "$CURRENT_VERSION" ]; then
    awk -v ver="$VERSION" "$IN_PROJECT $VERSION_LINE"' && !done {
            sub(/"[^"]*"/, "\"" ver "\""); done = 1
        }
        { print }
    ' "$PYPROJECT" > "$PYPROJECT.tmp"
    mv "$PYPROJECT.tmp" "$PYPROJECT"
    PREPARED+=("$PYPROJECT")
    echo "==> Bumped $PYPROJECT: $CURRENT_VERSION -> $VERSION"
fi

# Keep a fresh empty '## [Unreleased]', move its content under '## [VERSION] - DATE'.
RELEASE_DATE="$(date +%F)"
awk -v ver="$VERSION" -v date="$RELEASE_DATE" '
    !done && /^## \[Unreleased\]/ {
        print "## [Unreleased]"; print ""; print "## [" ver "] - " date
        done=1; next
    }
    { print }
' "$CHANGELOG" > "$CHANGELOG.tmp"
mv "$CHANGELOG.tmp" "$CHANGELOG"
PREPARED+=("$CHANGELOG")
echo "==> Promoted $CHANGELOG: [Unreleased] -> [$VERSION] - $RELEASE_DATE"

# --- 5. Commit + annotated tag (local only) ----------------------------------
# Nothing is on the remote yet: the recovery for a failure here is to undo the
# local commit (which also restores the promoted CHANGELOG and the version) and start over.
git add "${PREPARED[@]}"
git commit -m "chore: release $TAG" >/dev/null
COMMITTED=1
if ! git tag -a "$TAG" -m "aiagent $TAG"; then
    echo "ERROR: creating tag $TAG failed. The release commit is local-only (not pushed)." >&2
    echo "       Undo it and retry from a clean state:" >&2
    echo "         git reset --hard HEAD~1" >&2
    exit 1
fi
echo "==> Committed and tagged $TAG"

# --- 6. Push commit + tag together -------------------------------------------
# --atomic: origin takes both or neither, so main never carries a release commit
# without its tag (and the release workflow never runs for a tag main does not have).
echo "==> Pushing $BRANCH and $TAG to origin"
if ! git push --atomic origin "$BRANCH" "$TAG"; then
    cat >&2 <<EOF
ERROR: push failed. The release commit and tag $TAG exist locally, but nothing
       was pushed. Choose one:
       (a) Abort and undo everything local, then start over:
             git tag -d $TAG && git reset --hard HEAD~1
       (b) Fix the remote state and push again; the tag starts the release workflow:
             git push --atomic origin $BRANCH $TAG
EOF
    exit 1
fi

# --- 7. Follow the release workflow the tag started --------------------------
# Past this point $TAG is pushed and the release is in CI's hands; `make release`
# cannot resume (the tag guard blocks it). GitHub creates the run a few seconds
# after the push.
echo "==> $TAG pushed; the $WORKFLOW workflow builds, attests and publishes it"
RELEASE_COMMIT="$(git rev-parse HEAD)"
run_id=""
waited=0
while :; do
    run_id="$(gh run list --workflow "$WORKFLOW" --branch "$TAG" --commit "$RELEASE_COMMIT" \
        --event push --limit 1 --json databaseId --jq '.[0].databaseId // empty' 2>/dev/null \
        || true)"
    case "$run_id" in
        ''|*[!0-9]*) run_id="" ;;
        *) break ;;
    esac
    [ "$waited" -lt "$WATCH_WAIT" ] 2>/dev/null || break
    sleep "$WATCH_POLL"
    waited=$((waited + WATCH_POLL))
done

if [ -z "$run_id" ]; then
    echo "==> No $WORKFLOW run for $TAG showed up yet. Follow it at"
    echo "      https://github.com/$REPO/actions/workflows/$WORKFLOW"
    echo "    The release appears at https://github.com/$REPO/releases/tag/$TAG when it succeeds."
    exit 0
fi

echo "==> Watching run $run_id (Ctrl-C stops watching, not the release)"
if ! gh run watch "$run_id" --exit-status; then
    cat >&2 <<EOF
ERROR: the release workflow (run $run_id) failed: $TAG is pushed but not published.
       See why:           gh run view $run_id --log-failed
       Transient failure: gh run rerun $run_id --failed
       Needs a code fix:  drop the tag (git push origin :refs/tags/$TAG && git tag -d $TAG),
                          push the fix to $BRANCH, then tag it again and push the tag:
                            git tag -a $TAG -m 'aiagent $TAG' && git push origin $TAG
                          (make release would refuse: [Unreleased] is empty now)
EOF
    exit 1
fi

echo ""
echo "Released aiagent $TAG"
gh release view "$TAG" --json url -q .url 2>/dev/null || true
