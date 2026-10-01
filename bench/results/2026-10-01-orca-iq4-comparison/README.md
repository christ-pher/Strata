# Orca IQ4_XS on the V100 VM

Source: `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF`, revision
`0434906af7b5202b676d43f108cf4f73d25691ef`.
Three GGUF shards total 97,473,155,200 bytes (90.78 GiB).
Headers report 60.94 GiB of routed experts, 26.82 GiB of IQ4_NL PLE,
and 3.01 GiB of other weights. Runtime buffers and dense compatibility
conversion add to these figures.

The configuration uses the existing IQ4_XS/IQ4_NL expert kernels, IQ4_NL
PLE loader, automatic prompt chunks, 262,144-token context, INT8 KV,
RAM-backed PLE and MTP speculation. The uncensored model's small control
weights are prepared with `--compat-bf16`; that conversion is separate
from routed expert quantization. Full-length context remains untested.

The original IQ kernels and PLE implementation were restored. EPYC AVX2
build detection remains. The generic native manifest supports per-role
shards because this IQ4 model splits layers 10 and 42 across GGUF files.
Tests use IQ4_XS/IQ4_NL fixtures, including truncated-shard rejection.
Shared loader/PLE/GEMM/CPU regression tests and actual installed IQ3_S
expert parity pass on the restored build.

Prepare: `./run-v100.sh orca-iq4_xs --no-start`.
Launch: `./run-v100.sh orca-iq4_xs`.
Stop: `./run-v100.sh stop`.

Saved-prompt comparison command (requires other servers to be stopped):

```sh
.venv/bin/python tools/v100_bench.py \
  --models iq3_s orca-iq4_xs \
  --prompts bench/results/2026-10-01-v100-avx2-native/prompts.json \
  --out bench/results/orca-iq4-comparison
```

Throughput measurements do not establish coding quality or refusal behavior.
Compare saved responses and replay practical coding tasks separately.

## First validation, 2026-10-01

All three shards passed exact size and SHA-256 verification. Actual layers
11 and 12 passed expert CPU/GPU parity. The live API reports context 262,144.
The first saved coding prompt (1,124 input tokens, 512 generated tokens,
temperature zero, thinking disabled, no reused prompt tokens) measured
364.6 prompt tokens/s and 56.6 generated tokens/s, with 92.1% expert-cache
hits. This is one startup sample, not the full repeated benchmark. The
server was left running for the user's practical quality test. The raw
reply and timing record is `bench/results/2026-10-01-orca-iq4-comparison/quick-benchmark.json`.

The subsequent practical task reached approximately 21K tokens of conversation
history and completed; the user reported good results. Logged response decode
rates near the end were 51.5–63.2 tokens/s. These used conversation reuse and
different prompts, so they are not additional controlled benchmark samples.
The server was stopped at the user's request after the session. No full
repeated IQ3_S/IQ4_XS comparison was run tonight; the command above is ready
for a later session. The prepared model and its 262K context config remain.
