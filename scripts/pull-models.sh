#!/bin/sh
# Fetch the models named in .env onto this machine's Docker Model Runner.
#
# One implementation, called from three places: the Makefile's `pull` target
# (which `up` and `rebuild` depend on), ./start.sh, and ./auto-update.sh. The
# Windows launcher keeps its own copy because it cannot run this one.
#
# Nothing can do this from inside the image. The models live in the host's
# Model Runner, the container has no Docker socket and the image has no docker
# CLI, so "pull on build" can only mean "pull on the host, beside the build".
#
# The policy in one sentence: a missing embedding model is fatal and a missing
# text model is a warning, because the operator can point text at somebody
# else's endpoint on the settings page and can never do that for embeddings.
#
#   SKIP_MODEL_PULL=1   fetch nothing and say so - for a machine that runs
#                       every model somewhere else on purpose.

set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

say()  { printf '    %s\n' "$*"; }
warn() { printf '    warn %s\n' "$*" >&2; }
fail() { printf '\nERROR %s\n\n' "$*" >&2; exit 1; }

if [ "${SKIP_MODEL_PULL:-0}" = 1 ]; then
  say "SKIP_MODEL_PULL=1 - not fetching any model"
  exit 0
fi

# A missing .env is not "nothing to do": it means the caller skipped setup.
# The loop this replaces answered that case by pulling nothing and exiting 0.
[ -f .env ] || fail "There is no .env, so there are no model names to read.
    Run  make setup  (or ./start.sh) first."

# A trailing comment, a stray space, and the trailing CR of a .env saved on
# Windows are none of them part of a model name. The loop this replaces used
# 'cut -d= -f2' and turned all three into one.
env_get() {
  sed -n "s/^$1=//p" .env | head -1 | sed 's/[[:space:]]*#.*$//' | tr -d '[:space:]'
}
TEXT_MODEL="$(env_get TEXT_MODEL)"
EMBED_MODEL="$(env_get EMBED_MODEL)"

# A machine with no Model Runner is a legitimate machine: one pointed at an
# external endpoint on the settings page needs no local text model, and
# refusing to start it would be refusing to start an app that works. Say so.
if ! docker model version >/dev/null 2>&1; then
  warn "This Docker has no Model Runner, so no model was fetched."
  warn "The app will still start, but it can only answer against an endpoint"
  warn "set on its settings page, and chat will fall back to keyword matching"
  warn "because embeddings always run on this machine. To run models here,"
  warn "update Docker Desktop to 4.40 or newer, or install the Linux plugin:"
  warn "https://docs.docker.com/ai/model-runner/"
  exit 0
fi
if ! docker model status >/dev/null 2>&1; then
  docker desktop enable model-runner >/dev/null 2>&1 || true
  waited=0
  until docker model status >/dev/null 2>&1; do
    if [ "$waited" -ge 30 ]; then
      warn "Model Runner is installed but off, so nothing was fetched."
      warn "Turn it on in Docker Desktop > Settings > AI, then run: make pull"
      exit 0
    fi
    sleep 3; waited=$((waited + 3))
  done
fi

# 'docker model inspect' is the only presence test that answers for the name as
# it is actually written in .env, and it is local-only - it never reaches the
# registry, which is what lets a machine that already holds its models start
# with no network at all. Do not read the columns of 'docker model list'
# instead: that column prints the short name with the ai/ prefix and the tag
# removed, so it matched no name anybody would realistically configure, and
# every single start went to the registry as a result.
have_model() { docker model inspect "$1" >/dev/null 2>&1; }

# Say what is about to be downloaded before downloading it. A first `make up`
# that silently fetches several GB reads as a hang.
missing=''
for m in "$EMBED_MODEL" "$TEXT_MODEL"; do
  [ -n "$m" ] || continue
  case " $missing " in *" $m "*) continue ;; esac
  have_model "$m" || missing="$missing $m"
done
if [ -n "$missing" ]; then
  say "Downloading onto this machine, once, several GB:$missing"
  say "Leave it running - each model shows its own progress below."
fi

fetch() {                       # fetch KEY NAME FATAL
  key="$1"; name="$2"; hard="$3"
  if [ -z "$name" ]; then
    say "$key is not set in .env - nothing to fetch for it"
    return 0
  fi
  if have_model "$name"; then
    say "$key: $name is already here"
    return 0
  fi
  if docker model pull "$name"; then
    say "$key: $name is ready"
    return 0
  fi
  if [ "$hard" = 1 ]; then
    fail "Could not pull $name, named by $key in .env.
    Passages cannot be indexed without it, and the chat page would answer
    from keyword matching instead without saying so. Fix the connection and
    run  make pull  again - or, to run without semantic search on purpose,
    leave EMBED_MODEL empty in .env."
  fi
  warn "Could not pull $name, named by $key in .env - carrying on."
  warn "Documents will fail to process until it is here, unless the settings"
  warn "page points at another endpoint that serves a model."
  return 0
}

# Embeddings are pinned to this machine - app/pipeline/embed.py builds its
# client with allow_override=False - so no settings page can rescue a missing
# one, and a missing one does not stop a document from processing. It leaves
# the passages unindexed, after which chat and reports fall back to keyword
# matching without saying so. That silence is why this one is fatal, and why
# it runs first: the failure should land before the much larger text model
# has been downloaded.
fetch EMBED_MODEL "$EMBED_MODEL" 1

# The text model is the opposite case. In external mode the resolver ignores
# TEXT_MODEL entirely, so on that machine this download is never called on;
# and when it is needed and absent, the first request already says what to
# pull and where. Warn, never block.
fetch TEXT_MODEL "$TEXT_MODEL" 0

# --- runner context size --------------------------------------------------------
# Model Runner has no small fixed default. Measured on a machine with no runtime
# config at all, it loads each model at that model's OWN trained size:
# ai/gemma4:12b at 262144, ai/qwen3-embedding:4b at 40960. So a number written
# here does not remove a ceiling - it imposes one, and the reason to impose one
# is memory. The KV cache is allocated eagerly at load, so a window costs its
# full size even on short prompts.
#
# Nor does this backend truncate. llama.cpp ships with context shift disabled,
# so an over-long request comes back as HTTP 400 exceed_context_size_error
# naming both numbers, measured at 0.064s before any prefill. The danger here is
# a window too SMALL for a real prompt, not a window that eats one silently.
#
# The budget, measured on this pipeline: real prompts run 15k-78k tokens, so
# 131072 covers the largest observed with room to spare at about 11 GiB
# resident. The embedder is the one that was actually wasting memory - 40960
# tokens of cache to embed passages of roughly 250 - so capping it at 8192
# frees more than the text model gains. Net, the pair costs LESS than the
# defaults did.
CTX="$(env_get TEXT_NUM_CTX)"
if [ -z "$CTX" ]; then
  # No explicit value: size it from this machine rather than assuming a laptop.
  total_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
  [ "$total_kb" -gt 0 ] 2>/dev/null || total_kb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 ))
  gib=$(( total_kb / 1024 / 1024 ))
  if   [ "$gib" -ge 32 ]; then CTX=131072
  elif [ "$gib" -ge 24 ]; then CTX=65536
  elif [ "$gib" -ge 16 ]; then CTX=32768
  else                         CTX=16384
  fi
  say "sizing the context from ${gib}GB of RAM: $CTX tokens"
fi
ECTX="$(env_get EMBED_NUM_CTX)"
[ -n "$ECTX" ] || ECTX=8192

set_ctx() {                     # set_ctx MODEL SIZE MODE LABEL
  name="$1"; size="$2"; mode="$3"; label="$4"
  [ -n "$name" ] || return 0
  have_model "$name" || return 0
  if ! docker model configure --help >/dev/null 2>&1; then
    warn "This Docker cannot set a model's context size, so $name loads at its"
    warn "own trained context. That is not a cap - it may be far larger than"
    warn "needed and cost gigabytes of cache. Update Docker Desktop to size it."
    return 0
  fi
  # Skip when it already matches: configuring reloads several gigabytes of
  # weights, and a build should not pay that to write the number it already has.
  #
  # Keyed on the MODE, not the model name. Model Runner records the config
  # against the blob, and two tags can share one - ai/gemma4:12b is stored as
  # docker.io/ai/gemma4:latest because they are the same bytes - so a name
  # match finds nothing and the build reconfigures every time. One entry per
  # mode is exactly what this script writes, so the mode identifies it.
  current=$(docker model configure show 2>/dev/null | awk -v want="$mode" '
    /"Mode":/       { m=$0; sub(/.*"Mode": *"/,"",m); sub(/".*/,"",m) }
    /"context-size":/ { if (m==want) { c=$0; gsub(/[^0-9]/,"",c); print c; exit } }')
  if [ "$current" = "$size" ]; then
    say "$label: context already $size tokens"
    return 0
  fi
  if docker model configure --context-size "$size" --mode "$mode" "$name" >/dev/null 2>&1; then
    say "$label: context set to $size tokens"
  else
    warn "Could not set the context size for $name. It will load at its own"
    warn "trained context. Try by hand:"
    warn "  docker model configure --context-size $size --mode $mode $name"
  fi
}

set_ctx "$TEXT_MODEL"  "$CTX"  completion TEXT_MODEL
set_ctx "$EMBED_MODEL" "$ECTX" embedding  EMBED_MODEL
