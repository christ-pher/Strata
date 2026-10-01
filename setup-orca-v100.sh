#!/bin/sh
# Download and prepare the documented Orca IQ3_XXS compatibility model.
set -eu
cd "$(dirname "$0")"
command -v hf >/dev/null 2>&1 || { echo 'Hugging Face CLI (hf) is required.' >&2; exit 1; }
hf download orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF \
  --include '*IQ3_XXS*' --local-dir "$PWD/data/models-download/orca-iq3_xxs" --max-workers 2
export STRATA_GGUF_PY="$PWD/third_party/llama.cpp/gguf-py"
pack="$PWD/data/runtime/packs/orca-iq3_xxs"
shard="$PWD/data/models-download/orca-iq3_xxs/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf"
if [ ! -f "$pack/orca-prepared.done" ]; then
  .venv/bin/python tools/iq_pack.py --gguf "$shard" --out "$pack" --compat-bf16
  touch "$pack/orca-prepared.done"
fi
.venv/bin/python - <<'PY'
import json
from pathlib import Path
root=Path.cwd()
pack=root/'data/runtime/packs/orca-iq3_xxs'
shard=root/'data/models-download/orca-iq3_xxs/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf'
mtp=root/'data/runtime/mtp/rt'
if not (mtp/'experts.bin').exists():
    raise SystemExit('Prepare the original model first with ./run-v100.sh IQ2_XS --no-start to create MTP assets.')
cfg={'exe':str(root/'engine/strata'),'cwd':str(root),'args':[
 '--pack',str(pack),'--native',str(shard),'--ple-gguf',str(shard),
 '--expert-profile',str(root/'data/expert-profile.bin'),'--expert-cache','auto',
 '--prefill','512','--spec','4','--spec-min-p','0.5','--mtp',str(mtp),
 '--max-context','32768','--kv','int8'],
 'tokenizer':str(pack/'tokenizer'),'model_name':'orcarouter-qwen3.8-flash-next-uncensored-iq3_xxs',
 'log':str(root/'strata-orca-iq3_xxs.log'),'lib_dirs':['/usr/local/cuda-12.8/lib64'],
 'host':'0.0.0.0','port':8080,'gpu':0,'env':{'STRATA_PREFILL_RING':'8'}}
(root/'strata-orca-iq3_xxs.json').write_text(json.dumps(cfg,indent=1))
print('Orca prepared. Start with ./run-orca-v100.sh (stop any other model first).')
PY
