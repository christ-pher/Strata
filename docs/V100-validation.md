# V100 validation — 2026-09-30

Local checkout: Strata 0.1.27 with the Volta adaptations documented in V100.md.
Hardware: Tesla PG500-216, compute capability 7.0, 32 GB VRAM; 94 GiB system
RAM; 24 virtual CPUs without AVX2. Toolkit: CUDA 12.8.93; driver: 580.178.04.

## Build and numerical checks

- Complete `strata` build for CUDA architecture 70 passed.
- `cuobjdump --list-elf engine/strata` confirmed sm70 cubins.
- Eight focused CTest checks passed: device arena, QSA, BF16 GEMV, GR, INT8 KV,
  Q4 KV, BF16 GEMM CPU-reference parity, and scalar expert arithmetic.
- Real IQ2_XS expert parity passed on layers 0, 1, 2, 3, 20 and 47 using the
  existing native expert harness. Its CPU and GPU calculations use different
  activation quantizers; this is tolerance-based parity, not bitwise equality.
- The installer accepts sm70, rejects sm60, and selects the installed CUDA 12.8.

## Actual API requests

IQ2_XS, INT8 KV, 32,768 context, MTP speculation 4, auto prompt chunks,
`STRATA_PREFILL_RING=8`, one GPU, local OpenAI-compatible endpoint.

| Request | Input tokens | Output tokens | Result | Wall time |
| --- | ---: | ---: | --- | ---: |
| Add 17 + 25, answer only | 28 | 3 | `42` | 0.823 s |
| Exact phrase retrieval after repeated observation text | 2,891 | 9 | `maple-river-742` | 3.805 s |

The longer request reported 822.8 prompt tokens/s and 37.2 generated tokens/s,
with 6 of 6 MTP drafts accepted. The arithmetic request accepted 3 of 3 drafts.
These are smoke tests with very short outputs, not sustained throughput benchmarks
or a model-quality evaluation.

Startup loaded 33.02 GiB of experts into RAM, placed 19,345 experts in the GPU
cache, and initialized the draft layer successfully. Auto prefill selected 8,192
tokens. The test server was stopped after validation; start with `./run-v100.sh`.
Only IQ2_XS was downloaded and tested end to end. The launcher also supports Q2_0,
IQ3_XXS, IQ3_S and Coder IQ1_M, downloading their weights when selected.

Orca IQ3_XXS was subsequently installed and tested with a PLE loader correction.
The original IQ2_XS API smoke test also passed with that correction. See
[Orca V100 validation](ORCA-V100.md).
