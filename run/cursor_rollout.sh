#!/bin/bash
# Cursor operator wrapper: rebuild the top image by default, line-based inspect
# progress, then the log viewer.
set -eu
cd "$(dirname "$0")/.."

export PATH="/Applications/Docker.app/Contents/Resources/bin:${PATH}"

rebuild=1
pass=()
for arg in "$@"; do
    case "$arg" in
        --rebuild)
            rebuild=1
            ;;
        --no-rebuild)
            rebuild=0
            ;;
        *)
            pass+=("$arg")
            ;;
    esac
done

if [ ! -x .venv/bin/python ]; then
    echo "missing .venv; create it and install run/requirements.txt" >&2
    exit 1
fi

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

if [ "$rebuild" -eq 1 ]; then
    # Top image only (task.py, arena, grader). Bases still need ./build.sh.
    PLATFORM="${PLATFORM:-linux/amd64}"
    echo "rebuilding beat-stockfish:local (${PLATFORM})"
    docker build --platform "$PLATFORM" -t beat-stockfish:local .
else
    echo "skipping rebuild (--no-rebuild)"
fi

# "full" is a Textual TUI; Cursor only captures line-oriented stdout.
export INSPECT_DISPLAY=plain

.venv/bin/python run/rollout.py "${pass[@]}"

echo "starting inspect view on http://127.0.0.1:7575"
exec .venv/bin/inspect view --log-dir logs --port 7575
