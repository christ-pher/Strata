#!/bin/sh
set -eu
cd "$(dirname "$0")"
export STRATA_MODEL_HOME="${STRATA_MODEL_HOME:-/opt/models/Strata}"
quant=IQ3_XXS
case "${1:-}" in
  IQ3_XXS|IQ4_XS) quant=$1; shift ;;
esac
key=$(printf '%s' "$quant" | tr '[:upper:]' '[:lower:]')
config="strata-orca-$key.json"
parts=3
if [ "$quant" = IQ3_XXS ]; then parts=2; fi
shard="$STRATA_MODEL_HOME/models/orca-$key/Qwen3.8-Flash-Next-Uncensored-${quant}-00001-of-0000${parts}.gguf"
if [ "${1:-}" = "--no-start" ]; then
  exec ./setup-orca-v100.sh "$quant"
fi
if [ ! -f "$config" ] || [ ! -f "$shard" ] || [ ! -f "$STRATA_MODEL_HOME/runtime/packs/orca-$key/orca-prepared.done" ]; then
  ./setup-orca-v100.sh "$quant"
fi
printf '%s\n' "$$" > "$STRATA_MODEL_HOME/runtime/orca-server.pid"
exec .venv/bin/python -m serve.server --engine strata --config "$PWD/$config" --port 8080 "$@"
