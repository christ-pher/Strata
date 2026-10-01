#!/bin/sh
# Prepare or start any supported quantization using the local Volta build.
set -eu
cd "$(dirname "$0")"
export STRATA_MODEL_HOME="${STRATA_MODEL_HOME:-/opt/models/Strata}"
model=${1:-IQ2_XS}
if [ "$#" -gt 0 ]; then shift; fi
family=qwen
case "$model" in
  orca|orca-iq3_xxs) exec ./run-orca-v100.sh "$@" ;;
  Q2_0|IQ2_XS|IQ3_XXS|IQ3_S) ;;
  IQ1_M|coder|Coder) model=IQ1_M; family=coder ;;
  *) echo 'Usage: ./run-v100.sh [IQ2_XS|Q2_0|IQ3_XXS|IQ3_S|coder|orca] [setup options]' >&2; exit 2 ;;
esac
export XDG_CONFIG_HOME="$STRATA_MODEL_HOME/runtime/user-config"
export STRATA_PREFILL_RING=8
exec ./setup.sh --family "$family" --model "$model" --data-dir "$STRATA_MODEL_HOME/runtime" \
  --models-dir "$STRATA_MODEL_HOME/models" --context 32768 --kv int8 --vision none --host 0.0.0.0 --yes "$@"
