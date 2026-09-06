#!/usr/bin/env bash
# Keep this deployment tracking the GitHub repo.
#
#   ./auto-update.sh once       check now; pull and restart if there is a push
#   ./auto-update.sh watch      keep checking in the foreground
#   ./auto-update.sh start      keep checking in the background
#   ./auto-update.sh stop       stop the background watcher
#   ./auto-update.sh status     is the watcher running, and are we current?
#
# Pull-based on purpose: a webhook needs a public endpoint and a self-hosted
# GitHub runner means handing this machine a repo token and a remote-execution
# surface. Polling git gets the same outcome with neither.
#
# The containers are rebuilt and restarted ONLY when the stack is already
# running; a stopped stack just gets the new code and stays stopped.

set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

INTERVAL="${UPDATE_INTERVAL:-10}"        # seconds between checks
PIDFILE=".auto-update.pid"
LOGFILE="auto-update.log"

say() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

check_once() {
    if ! git diff --quiet || ! git diff --cached --quiet; then
        say "local changes present - not touching this checkout"
        return 0
    fi
    git fetch -q origin || { say "fetch failed (offline?); will try again"; return 0; }

    local here there
    here="$(git rev-parse HEAD)"
    there="$(git rev-parse origin/main)"
    [ "$here" = "$there" ] && return 0

    say "update found: ${here:0:7} -> ${there:0:7}"
    if ! git merge-base --is-ancestor HEAD origin/main; then
        say "local history has diverged from origin/main - resolve by hand"
        return 0
    fi
    git pull -q --ff-only origin main || { say "pull failed"; return 0; }
    say "pulled $(git log -1 --format='%h %s' | cut -c1-70)"

    # ps prints a header row even with nothing running, so count real
    # container lines, not just any output.
    if [ "$(docker compose ps --status running --format '{{.Service}}' 2>/dev/null | grep -c . || true)" -gt 0 ]; then
        say "stack is running - rebuilding and restarting"
        # --remove-orphans: a push that retires a service from the compose file
        # must also retire its running container, or it lingers forever.
        if docker compose up -d --build --remove-orphans >>"$LOGFILE" 2>&1; then
            say "restarted on the new version"
        else
            say "rebuild FAILED - the old containers may still be running; see $LOGFILE"
        fi
    else
        say "stack is not running - code updated, nothing restarted"
    fi
}

case "${1:-once}" in
  once)
    check_once ;;
  watch)
    say "watching origin/main every ${INTERVAL}s (ctrl-c to stop)"
    while true; do check_once; sleep "$INTERVAL"; done ;;
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        say "already running (pid $(cat "$PIDFILE"))"; exit 0
    fi
    nohup "$0" watch >>"$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
    say "watcher started (pid $!, every ${INTERVAL}s, log: $LOGFILE)" ;;
  stop)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        kill "$(cat "$PIDFILE")" && rm -f "$PIDFILE" && say "watcher stopped"
    else
        rm -f "$PIDFILE"; say "watcher was not running"
    fi ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        say "watcher running (pid $(cat "$PIDFILE"))"
    else
        say "watcher not running"
    fi
    git fetch -q origin 2>/dev/null || true
    if [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main 2>/dev/null)" ]; then
        say "checkout is current with origin/main"
    else
        say "an update is available - run: ./auto-update.sh once"
    fi ;;
  *) echo "usage: $0 {once|watch|start|stop|status}" >&2; exit 2 ;;
esac
