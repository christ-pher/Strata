#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ "${1:-}" = "--no-start" ]; then
  exec ./setup-orca-v100.sh
fi
if [ ! -f strata-orca-iq3_xxs.json ] || [ ! -f data/runtime/packs/orca-iq3_xxs/orca-prepared.done ]; then
  ./setup-orca-v100.sh
fi
printf '%s\n' "$$" > data/runtime/orca-server.pid
exec .venv/bin/python -m serve.server --engine strata --config "$PWD/strata-orca-iq3_xxs.json" --port 8080 "$@"
