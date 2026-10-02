#!/bin/sh
# Starts the project from this folder: installs uv if it's missing (after asking), creates the
# config files on the first run, then runs it with the locked dependencies. Arguments are
# passed on.

# --- project settings ---------------------------------------------------------
project="SolTrade"
# Space-separated SAMPLE:TARGET pairs, copied on the first run.
configs="config.json.sample:config.json .env.sample:.env"
# Shown after the first run creates the config files.
first_run_hint="Set SOLTRADE_PRIVATE_KEY in .env and the tokens to trade (secondary_mints) in config.json, then run this again. See the README's Getting started."
entry="main.py"
# -----------------------------------------------------------------------------

set -eu
cd "$(dirname "$0")"

find_uv() {
  command -v uv >/dev/null 2>&1 && return 0
  # where the official installer puts uv, for a shell that started before it was installed
  for bin in "${XDG_BIN_HOME:-}" "$HOME/.local/bin"; do
    if [ -n "$bin" ] && [ -x "$bin/uv" ]; then
      PATH="$bin:$PATH"
      return 0
    fi
  done
  return 1
}

if ! find_uv; then
  echo "$project runs with uv (https://docs.astral.sh/uv/), which is not installed."
  printf 'Install it now with the official installer? [y/N] '
  read -r answer || answer=
  case "$answer" in
    [yY] | [yY][eE][sS]) ;;
    *)
      echo "uv is required. Install it from https://docs.astral.sh/uv/ and run this again."
      exit 1
      ;;
  esac
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  else
    wget -qO- https://astral.sh/uv/install.sh | sh
  fi
  if ! find_uv; then
    echo "uv was installed but can't be found. Open a new terminal and run this again."
    exit 1
  fi
fi

created=
for pair in $configs; do
  sample="${pair%%:*}"
  target="${pair#*:}"
  if [ ! -e "$target" ]; then
    cp "$sample" "$target"
    created="${created:+$created and }$target"
  fi
done
if [ -n "$created" ]; then
  echo "Created $created from the samples. $first_run_hint"
  exit 0
fi

exec uv run --frozen "$entry" "$@"
