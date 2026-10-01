#!/bin/sh
# One launcher for preparing, starting, and stopping supported models.
set -eu
cd "$(dirname "$0")"
export STRATA_MODEL_HOME="${STRATA_MODEL_HOME:-/opt/models/Strata}"
export STRATA_UNIFIED_LAUNCHER=1
export XDG_CONFIG_HOME="$STRATA_MODEL_HOME/runtime/user-config"
export STRATA_PREFILL_RING=8
model=${1:-IQ2_XS}
if [ "$#" -gt 0 ]; then shift; fi
family=qwen
case "$model" in
  stop) exec .venv/bin/python tools/stop_server.py "$@" ;;
  orca|orca-iq3_xxs) family=orca; model=IQ3_XXS ;;
  orca-iq4_xs) family=orca; model=IQ4_XS ;;
  Q2_0|IQ2_XS|IQ3_XXS|IQ3_S) ;;
  IQ1_M|coder|Coder) model=IQ1_M; family=coder ;;
  swift-iq2_xs) family=swift; model=IQ2_XS ;;
  swift-iq3_xxs) family=swift; model=IQ3_XXS ;;
  -h|--help) echo 'Usage: ./run-v100.sh [IQ2_XS|Q2_0|IQ3_XXS|IQ3_S|coder|swift-iq2_xs|swift-iq3_xxs|orca|orca-iq4_xs|stop] [setup options]'; exit 0 ;;
  *) echo "Unknown model: $model (see ./run-v100.sh --help)" >&2; exit 2 ;;
esac
exec ./setup.sh --family "$family" --model "$model" --data-dir "$STRATA_MODEL_HOME/runtime" \
  --models-dir "$STRATA_MODEL_HOME/models" --context 32768 --kv int8 --vision none --host 0.0.0.0 --yes "$@"
