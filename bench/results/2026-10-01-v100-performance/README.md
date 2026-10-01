# Orca IQ4_XS tuning on the V100 VM — 2026-10-01

The retained engine change is bounded background expert-usage recording. The saved Orca configuration now uses
16 CPU workers (changed from 8 at the user’s request); the benefit was repeatable in text-only runs but not established with vision loaded. Vision
remains enabled in the saved config. The BF16 weight-reuse experiment was reverted after a neutral full-model
prompt result. Other model configs, model precision, KV settings and speculative decoding are unchanged.

## Hardware and baseline

Origin was fetched and `main` was clean and equal to `origin/main` at `d3ba706` before edits.
`restore/performance-baseline-20261001` preserves that source baseline; all work is on `perf/v100-20261001`.
Hardware: 32 guest vCPUs, AMD EPYC 7402 with AVX2/FMA/F16C, 125 GiB RAM, Tesla PG500-216 / Volta with 32 GiB VRAM,
180 W power limit, NVIDIA 580.178.04, CUDA 12.8 build targeting sm70. Release/native CPU build retained.
The GPU power limit, clocks and VM settings were not changed. [Hardware and binary provenance](hardware.json).

## Method

Twelve full runs, each with one excluded 128-token warm-up, then two passes through the exact saved three prompts:
1,124, 4,226 and 8,348 prompt tokens, 256 output tokens per request. Total: 72 measured requests and 18,432 output
tokens. Temperature zero, thinking off, prompt cache disabled, ring 8, automatic prompt chunks (8192), shipped
expert profile, adaptive cache, INT8 KV, 262K configured context / 32K resident cells, MTP speculation 4.
Models/variants ran sequentially through temporary localhost servers. All measured requests reported zero
reused prompt tokens. The final installed-build smoke check is separate from these throughput averages.

Rates below are arithmetic means of complete-request engine rates. These are workload measurements, not
confidence intervals, quality evaluations or evidence for untested model/context sizes. Cache adaptation and
CPU/GPU rounding can change generated text and draft acceptance. Saved replies and raw timings are retained.
`source_commit` in run files identifies the harness checkout; the actual executable is identified by SHA-256.
Baseline runs use the original `d3ba706` executable. GEMM/usage comparisons use the preserved `9ffda9a` experimental
executable, explicitly toggling GEMM reuse and recording. The installed build restores the original GEMM code.

## CPU workers (original engine, text only)

| Workers | Prompt mean tok/s | Decode mean tok/s |
| --- | ---: | ---: |
| 31, first run | 537.5 | 55.27 |
| 31, repeat | 635.5 | 56.83 |
| 24 | 642.9 | 59.72 |
| 16 | 640.6 | 59.05 |
| 8, first run | 640.0 | 59.88 |
| 8, repeat | 642.0 | 59.88 |

Eight workers averaged 6.8% higher decode throughput than the two 31-worker runs combined (5.4% above the later
31-worker control). Eight, sixteen and twenty-four workers were close. The first 31-worker prompt pass was cold;
its lower prompt average is not evidence of a worker benefit. Apply the 8-worker setting only to the tested Orca
config; other models retain automatic worker selection. Long-context/cold-cache workloads remain untested.

## Volta BF16 conversion reuse

The experiment reused fitting weights across token tiles inside the existing 64 MiB scratch allocation. It added
no VRAM allocation, changed no matrix tile shapes and retained FP32 arithmetic. Narrow SIMT and oversized weights
retained the original paths. Both modes passed numerical checks, including output strides, beta accumulation,
tile boundaries and BF16 values outside FP16 range.

Isolated T=8192/K=2560 GEMMs improved by about 2.5–3.8% for N=48/128/512/2560. Ten timed calls per shape after three
warm-ups, CUDA events; [baseline](gemm-baseline.txt), [reuse](gemm-reuse.txt), [microbenchmark source](bf16_gemm_bench.cpp).
Full-model prompt throughput was 642.43 tok/s disabled versus 641.85 enabled (-0.09%). This is neutral within
variation. Decode means were 59.60 and 60.30; this prompt-only change does not substantiate a decode gain.
The experiment was reverted (`3d22561`); `experiment/volta-bf16-reuse-20261001` preserves its commit `9ffda9a`.

## Background recording overhead

Same experimental executable, eight workers, GEMM reuse enabled in both cases:

| Recording | Prompt mean tok/s | Decode mean tok/s |
| --- | ---: | ---: |
| Off | 641.85 | 60.300 |
| On | 643.80 | 60.283 |

The 0.03% decode difference is below observed variation: no measurable overhead in this test. This is not proof
of zero overhead under every request rate/storage condition. The live snapshot was roughly 161 KiB while sampled,
and successfully produced a complete 24,576-pair candidate profile. Benchmark counters stayed in an isolated
build directory and were not installed as a production profile.

## Vision residency (text requests, original engine)

| Mode / workers | Prompt mean tok/s | Decode mean tok/s | Expert slots |
| --- | ---: | ---: | ---: |
| Text / 8, run 1 | 640.0 | 59.88 | 10101 |
| Text / 8, run 2 | 642.0 | 59.88 | 10101 |
| GPU vision / 8, run 1 | 623.3 | 59.08 | 9548 |
| GPU vision / 8, run 2 | 550.8 | 56.67 | 9548 |
| GPU vision / 31 | 622.2 | 57.72 | 9548 |

Vision occupied about 1.37 GiB and reduced expert capacity by 553 slots (5.5%). Across the two eight-worker runs
per mode, vision decode averaged 3.4% lower. Second-pass prompt averages were 629.50 with vision versus 643.45
text-only (2.2% lower, still zero prompt reuse). All-request prompt averages were 587.06 versus 640.96 (8.4% lower),
but the first pass of the vision repeat was unusually slow. A cleanup-time vmstat check showed transient swapping;
this is a possible cold-page/memory-pressure confound, not an isolated explanation. No GPU clock-event throttling
was reported at that check. Do not present 8.4% as a stable vision penalty.

With vision, the combined eight-worker decode average was 57.88 versus the single 31-worker run's 57.72: too close
and variable to establish a worker gain. These requests measure retaining the GPU encoder and reduced cache
capacity. They do not measure image encoding or answering image questions. Vision was actually loaded before
cache sizing, not just left as an engine flag.

## Settings and independent restores

Usage collection is enabled for Orca IQ4_XS at `/opt/engines/Strata/logs/expert-usage/orca-iq4_xs/` on subsequent
launches. It records expert IDs/counts, not prompts or replies. No candidate profile is applied automatically.
See [recording scope and profile generation](../../../docs/PERFORMANCE-TUNING.md).

Restore only recording (workers and vision remain):

```sh
.venv/bin/python tools/v100_tuning.py --restore bench/results/2026-10-01-v100-performance/restore-recording.json
```

Restore only worker count (recording and vision remain):

```sh
.venv/bin/python tools/v100_tuning.py --restore bench/results/2026-10-01-v100-performance/restore-workers.json
```

For occasional text-only sessions:

```sh
.venv/bin/python tools/v100_tuning.py strata-orca-iq4_xs.json --text-only
./run-v100.sh orca-iq4_xs
```

The tool prints a field-specific restore command that re-enables the saved vision encoder without undoing
workers or recording. Setting changes apply on restart. Stop a running server before relaunching. Restores
refuse to overwrite a setting subsequently edited elsewhere. Source changes are separately committed;
`restore/before-orca-recording-20261001` and `restore/before-orca-workers-20261001` mark the activation boundaries.
Original binary, build metadata and full Orca config: `build/performance-restore-20261001/` (ignored local backup).

## Raw evidence and validation

[Summary index](summary.json) lists per-run result files, rates, cache capacities, token counts and hashes.
Each run directory contains exact prompts, responses, isolated configuration and `engine-summary.txt`; full engine
and server logs remain local/ignored. Final checks: asynchronous count/flush test, profile merging/deduplication,
base-ranking guard, benchmark vision/environment isolation, independent field restores, and BF16 GPU numerics.
The installed engine was smoke-tested with vision, eight workers and recording, using isolated test counters.
The abrupt benchmark teardown interrupted the final pending snapshot, leaving a temporary file; the prior atomic
snapshot remained readable. Ordinary ongoing collection does not wait for disk writes, and pending data can be
lost on abrupt termination as documented.
Temporary servers were stopped after testing.

## Subsequent worker preference

The user selected 16 workers after reviewing the results. The saved Orca configuration was changed from 8 to 16;
recording and vision were retained. Restore just this adjustment to return to 8:

```sh
.venv/bin/python tools/v100_tuning.py --restore bench/results/2026-10-01-v100-performance/restore-workers16.json
```

To return from 16 to the original automatic worker selection, restore to 8 first with the command above,
then apply `restore-workers.json`. Worker changes take effect on the next engine launch.
