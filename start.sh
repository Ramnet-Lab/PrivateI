#!/usr/bin/env bash
# One-shot setup and launch.
#
#   ./start.sh                 set everything up and open the app
#   ./start.sh --no-browser    do not open a browser at the end
#
# Models run through Docker Model Runner - a Docker Desktop feature that
# executes them natively on this machine's GPU. Safe to re-run: it never
# overwrites .env and never re-downloads a model it already has.

set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

OPEN_BROWSER=1
AUTO_UPDATE=1
for arg in "$@"; do
  case "$arg" in
    --no-browser)  OPEN_BROWSER=0 ;;
    --no-auto-update) AUTO_UPDATE=0 ;;
    -h|--help)     sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  B=$'\033[1m'; R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'; C=$'\033[36m'; Z=$'\033[0m'
else
  B=""; R=""; G=""; Y=""; C=""; Z=""
fi
step() { printf '\n%s==> %s%s\n' "$B$C" "$*" "$Z"; }
ok()   { printf '    %sok%s   %s\n' "$G" "$Z" "$*"; }
warn() { printf '    %swarn%s %s\n' "$Y" "$Z" "$*"; }
fail() { printf '\n%sERROR%s %s\n\n' "$R" "$Z" "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

OS="$(uname -s)"
case "$OS" in
  Darwin) PLATFORM=mac ;;
  Linux)  PLATFORM=linux ;;
  *) fail "This script supports macOS and Linux. Detected: $OS" ;;
esac

# --- Docker ---------------------------------------------------------------
# A fresh Docker Desktop install puts the CLI in places the current shell has
# not seen, so widen PATH before every probe and clear bash's command cache.
widen_path() {
  PATH="$PATH:/usr/local/bin:/opt/homebrew/bin:$HOME/.docker/bin"
  hash -r 2>/dev/null || true
}
widen_path

install_docker_mac() {
  step "Installing Docker Desktop"
  if ! have brew; then
    fail "Docker is not installed and Homebrew is not available.
    Download Docker Desktop from https://www.docker.com/products/docker-desktop
    install it, open it once to accept the licence, then run ./start.sh again."
  fi
  warn "Homebrew needs your password to finish this install."
  brew install --cask docker-desktop || fail "Docker Desktop install failed. Install it manually from
    https://www.docker.com/products/docker-desktop then run ./start.sh again."
  widen_path
}

install_docker_linux() {
  step "Installing Docker Engine"
  have curl || fail "Please install curl, or install Docker yourself: https://docs.docker.com/engine/install/"
  printf '    This runs the official Docker install script with sudo. Continue? [y/N] '
  read -r reply
  case "$reply" in [yY]*) ;; *) fail "Install Docker yourself: https://docs.docker.com/engine/install/" ;; esac
  tmp="$(mktemp)"; curl -fsSL https://get.docker.com -o "$tmp"
  sudo sh "$tmp" || fail "Docker install failed."
  rm -f "$tmp"
  sudo systemctl enable --now docker 2>/dev/null || true
  widen_path
  docker info >/dev/null 2>&1 || warn "You may need to add yourself to the docker group:
    sudo usermod -aG docker \"$USER\"   then log out and back in."
}

start_docker_mac() {
  step "Starting Docker Desktop"
  open -a Docker 2>/dev/null || fail "Could not launch Docker Desktop. Open it from Applications, then re-run."
  printf '    waiting for the Docker daemon (first launch asks you to accept a licence)'
  waited=0
  until docker info >/dev/null 2>&1; do
    [ "$waited" -ge 180 ] && { printf '\n'; fail "Docker did not become ready in 3 minutes.
    Open Docker Desktop, finish any setup or licence prompt it shows, then re-run ./start.sh"; }
    printf '.'; sleep 3; waited=$((waited + 3))
  done
  printf '\n'
}

step "Checking Docker"
if ! have docker; then
  if [ "$PLATFORM" = mac ]; then
    [ -d /Applications/Docker.app ] || install_docker_mac
  else
    install_docker_linux
  fi
  widen_path
  have docker || fail "Docker is installed but the 'docker' command is not on this shell's PATH yet.
    Close this terminal, open a new one, and run ./start.sh again."
fi
if ! docker info >/dev/null 2>&1; then
  [ "$PLATFORM" = mac ] && start_docker_mac || fail "The Docker daemon is not running. Start it and re-run.
    On Linux:  sudo systemctl start docker"
fi
docker info >/dev/null 2>&1 || fail "Docker is still not reachable."
ok "Docker $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo running)"

if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
elif have docker-compose; then
  COMPOSE="docker-compose"
  warn "Using the old docker-compose v1. If anything behaves oddly, upgrade to Docker Compose v2."
else
  fail "Docker Compose is not available. Update Docker Desktop, or install the compose plugin."
fi

# --- .env -----------------------------------------------------------------
step "Configuration"
if [ ! -f .env ]; then
  cp .env.example .env
  ok "created .env from .env.example"
else
  ok ".env already exists (left untouched)"
fi

env_get() { sed -n "s/^$1=//p" .env | head -1; }
env_set() {
  if grep -q "^$1=" .env; then
    tmp="$(mktemp)"; awk -v k="$1" -v v="$2" -F= '
      $1==k {print k "=" v; next} {print}' .env > "$tmp" && mv "$tmp" .env
  else
    printf '%s=%s\n' "$1" "$2" >> .env
  fi
}
env_set_if_blank() { [ -n "$(env_get "$1")" ] || env_set "$1" "$2"; }

if [ -z "$(env_get NEO4J_PASSWORD)" ]; then
  # Alphanumeric only: a '/' in the password breaks NEO4J_AUTH=neo4j/<pw>, and
  # '$' breaks compose interpolation. Assigned on its own line because a
  # failure inside $( ) in argument position does not abort under set -e.
  if have openssl; then
    NEW_PW="$(openssl rand -hex 24)"
  else
    NEW_PW="$(LC_ALL=C tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 40)"
  fi
  [ -n "$NEW_PW" ] || fail "Could not generate a database password."
  env_set NEO4J_PASSWORD "$NEW_PW"
  ok "generated a random NEO4J_PASSWORD"
else
  ok "NEO4J_PASSWORD already set"
fi

env_set_if_blank TEXT_MODEL  "ai/gemma4"
env_set_if_blank VLM_MODEL   "ai/gemma4"
env_set_if_blank EMBED_MODEL "ai/nomic-embed-text-v1.5"
APP_PORT="$(env_get APP_PORT)"; APP_PORT="${APP_PORT:-8080}"
ok "models: $(env_get TEXT_MODEL) (text/vision), $(env_get EMBED_MODEL) (embeddings)"

# --- ports ------------------------------------------------------------------
port_busy() {
  if have lsof; then lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
  else ! (exec 3<>/dev/tcp/127.0.0.1/"$1") 2>/dev/null; fi
}
ALREADY_UP=0
$COMPOSE ps --status running 2>/dev/null | grep -q . && ALREADY_UP=1
if [ "$ALREADY_UP" = 0 ]; then
  for p in "$APP_PORT" 7474 7687; do
    port_busy "$p" && warn "port $p is already in use by something else - change it in .env if startup fails" || true
  done
else
  ok "this stack is already running; leaving its ports alone"
fi

# --- Docker Model Runner ------------------------------------------------------
# The models execute on the host through Model Runner, with the GPU - never
# inside the Docker VM. So there is no VM memory requirement beyond the app and
# the database, and no model service in docker-compose.yml at all.
step "Checking Docker Model Runner"
if ! docker model version >/dev/null 2>&1; then
  fail "This Docker installation has no Model Runner.
    On macOS and Windows it ships with Docker Desktop 4.40 or newer - update
    Docker Desktop. On Linux install the plugin:
    https://docs.docker.com/ai/model-runner/"
fi
if ! docker model status >/dev/null 2>&1; then
  warn "Model Runner is off; turning it on"
  docker desktop enable model-runner >/dev/null 2>&1 || true
  waited=0
  until docker model status >/dev/null 2>&1; do
    [ "$waited" -ge 60 ] && fail "Could not enable Model Runner.
    Turn it on in Docker Desktop > Settings > AI, then run ./start.sh again."
    sleep 3; waited=$((waited + 3))
  done
fi
ok "Model Runner is running (models execute on this machine's GPU)"

step "Checking resources"
DISK_GB="$(df -Pk . | awk 'NR==2 {print int($4/1048576)}')"
[ "${DISK_GB:-0}" -lt 20 ] && warn "only ${DISK_GB}GB free on this disk; the models need about 8GB" \
                           || ok "${DISK_GB}GB free disk"

# --- build and start ----------------------------------------------------------
step "Building images (first run takes a few minutes)"
$COMPOSE build

step "Starting services"
$COMPOSE up -d

wait_healthy() {
  name="$1"; limit="${2:-180}"; waited=0
  printf '    waiting for %s' "$name"
  until [ "$($COMPOSE ps --format json "$name" 2>/dev/null | grep -c '"Health":"healthy"' || true)" -ge 1 ]; do
    [ "$waited" -ge "$limit" ] && { printf '\n'; fail "$name did not become healthy in ${limit}s.
    See what it says:  $COMPOSE logs $name"; }
    printf '.'; sleep 3; waited=$((waited + 3))
  done
  printf ' %sok%s\n' "$G" "$Z"
}
wait_healthy neo4j 240
wait_healthy app 180

# --- models ---------------------------------------------------------------------
step "Downloading models (one time, about 8GB)"
pulled=""
for m in "$(env_get TEXT_MODEL)" "$(env_get VLM_MODEL)" "$(env_get EMBED_MODEL)"; do
  [ -n "$m" ] || continue
  case " $pulled " in *" $m "*) continue ;; esac
  pulled="$pulled $m"
  if docker model list 2>/dev/null | awk '{print $1}' | grep -qx "${m#docker.io/}" ; then
    ok "$m is already here"
  else
    docker model pull "$m" || fail "Could not pull $m. Check your connection and re-run ./start.sh"
  fi
done
ok "models ready:$pulled"

# --- prove it actually answers ---------------------------------------------------
step "Checking the model answers (loads it into the GPU on first use)"
if docker model run "$(env_get TEXT_MODEL)" "Say OK." >/dev/null 2>&1; then
  ok "the model loaded and answered"
else
  warn "The model did not answer a test prompt. The app will still open, but"
  warn "processing will fail. See:  docker model status  and  docker model logs"
fi

# --- done -----------------------------------------------------------------------
URL="http://127.0.0.1:${APP_PORT}"
step "Ready"
printf '    %s%s%s\n\n' "$B" "$URL" "$Z"
printf '    Drop documents on the page and they process automatically.\n'
printf '    Stop with:      %s down\n' "$COMPOSE"
printf '    See progress:   %s logs -f app\n\n' "$COMPOSE"

# Keep this deployment current: a background watcher pulls any push to the
# repo and restarts the running containers on the new version. Skip with
# --no-auto-update, stop later with ./auto-update.sh stop.
if [ "$AUTO_UPDATE" = 1 ] && [ -d .git ] && [ -x ./auto-update.sh ]; then
  ./auto-update.sh start 2>/dev/null | tail -1 | sed 's/^/    /' || true
fi

if [ "$OPEN_BROWSER" = 1 ]; then
  if [ "$PLATFORM" = mac ]; then open "$URL" 2>/dev/null || true
  elif have xdg-open; then xdg-open "$URL" >/dev/null 2>&1 || true
  fi
fi
