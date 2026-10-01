# V100 VM after EPYC host CPU passthrough — 2026-10-01

The VM now exposes AMD EPYC 7402 with AVX2, FMA and F16C; no AVX-512. RAM remains approximately 94 GiB and the GPU remains a 32 GiB Volta. The native ggml backend was rebuilt with compiler-cache reuse bypassed. Runtime logs confirm AVX-2 expert dispatch.

| Model | Previous decode tokens/s | AVX2 decode tokens/s | Change | Previous prompt tokens/s | AVX2 prompt tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| IQ2_XS | 55.9 | 58.0 | +3.7% | 725.7 | 726.8 |
| IQ3_S | 47.5 | 56.9 | +19.8% | 692.5 | 694.0 |

Replayed the exact saved prompts from the [previous benchmark](../2026-10-01-v100/README.md): two passes through three tasks, 512 generated tokens per request, one excluded 128-token warm-up per model. Configurations match the baseline: 262K context, INT8 KV with 32K resident cells, MTP speculation 4, automatic GPU expert cache and prefill, prefill ring 8, thinking off, temperature 0 and no prompt reuse. Models ran sequentially. No user-facing server is left running by the benchmark.

These are observed before/after results from one run, not isolated CPU microbenchmarks or confidence intervals. Shared host load and adaptive expert caching can affect rates; small differences may be run-to-run variation. This does not measure Orca throughput or model quality. Orca files are no longer installed, so actual-weight parity used the installed original IQ3_S layers 0 and 1, including IQ3_XXS/IQ4_NL and IQ2_S/Q2_0 expert formats.

Validation: rebuilt engine and ggml CPU objects; compiled feature probes return enabled for AVX2/FMA/F16C; BF16 GEMM and automatic-dispatch expert arithmetic tests passed; actual-weight expert CPU/GPU parity reported zero failures; 23 CPU-build and setup-rope regression tests passed.

[Raw measurements](results.json), [hardware](hardware.json), [expert parity](expert-parity.txt) and [saved prompts](prompts.json).

Reproduce:

```sh
.venv/bin/python tools/v100_bench.py \
  --prompts bench/results/2026-10-01-v100/prompts.json \
  --out bench/results/my-avx2-run
```
