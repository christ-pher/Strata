#!/bin/sh
# Download and prepare a supported Orca compatibility model.
set -eu
cd "$(dirname "$0")"
export STRATA_MODEL_HOME="${STRATA_MODEL_HOME:-/opt/models/Strata}"
quant=${1:-IQ3_XXS}
case "$quant" in
  IQ3_XXS) parts=2 ;;
  IQ4_XS) parts=3 ;;
  *) echo 'Usage: ./setup-orca-v100.sh [IQ3_XXS|IQ4_XS]' >&2; exit 2 ;;
esac
key=$(printf '%s' "$quant" | tr '[:upper:]' '[:lower:]')
export STRATA_ORCA_QUANT="$quant"
command -v hf >/dev/null 2>&1 || { echo 'Hugging Face CLI (hf) is required.' >&2; exit 1; }
revision=main
if [ "$quant" != IQ3_XXS ]; then revision=0434906af7b5202b676d43f108cf4f73d25691ef; fi
hf download orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF \
  --revision "$revision" --include "*-${quant}-*" --local-dir "$STRATA_MODEL_HOME/models/orca-$key" --max-workers 2
export STRATA_GGUF_PY="$PWD/third_party/llama.cpp/gguf-py"
pack="$STRATA_MODEL_HOME/runtime/packs/orca-$key"
shard="$STRATA_MODEL_HOME/models/orca-$key/Qwen3.8-Flash-Next-Uncensored-${quant}-00001-of-0000${parts}.gguf"
if [ ! -f "$pack/orca-prepared.done" ]; then
  .venv/bin/python tools/iq_pack.py --gguf "$shard" --out "$pack" --compat-bf16
  touch "$pack/orca-prepared.done"
fi
.venv/bin/python - <<'PY'
import json, os
from pathlib import Path
root=Path.cwd()
model_home=Path(os.environ['STRATA_MODEL_HOME'])
quant=os.environ['STRATA_ORCA_QUANT']
key=quant.lower()
parts=2 if quant=='IQ3_XXS' else 3
pack=model_home/f'runtime/packs/orca-{key}'
shard=model_home/f'models/orca-{key}'/f'Qwen3.8-Flash-Next-Uncensored-{quant}-00001-of-0000{parts}.gguf'
mtp=model_home/'runtime/mtp/rt'
if not (mtp/'experts.bin').exists():
    raise SystemExit('Prepare the original model first with ./run-v100.sh IQ2_XS --no-start to create MTP assets.')
cfg={'exe':str(root/'engine/strata'),'cwd':str(root),'args':[
 '--pack',str(pack),'--native',str(shard),'--ple-gguf',str(shard),
 '--expert-profile',str(root/'data/expert-profile.bin'),'--expert-cache','auto',
 '--prefill','auto','--spec','4','--spec-min-p','0.5','--mtp',str(mtp),
 '--max-context','262144','--kv','int8'],
 'tokenizer':str(pack/'tokenizer'),'model_name':f'orcarouter-qwen3.8-flash-next-uncensored-{key}',
 'log':str(root/f'strata-orca-{key}.log'),'lib_dirs':['/usr/local/cuda-12.8/lib64'],
 'host':'0.0.0.0','port':8080,'gpu':0,'env':{'STRATA_PREFILL_RING':'8'}}
if quant=='IQ4_XS':
    cfg['args'] += ['--ple-io','ram']
(root/f'strata-orca-{key}.json').write_text(json.dumps(cfg,indent=1))
print(f'Orca {quant} prepared. Start with ./run-orca-v100.sh {quant} after allocating enough RAM.')
PY
