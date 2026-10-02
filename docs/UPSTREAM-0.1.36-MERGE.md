# Upstream 0.1.36 integration

Merged tag `v0.1.36` (`36fa455`) into the V100 fork on 2026-10-02. The only conflict was README.md: retained the fork instructions and updated README.original.md to the release README. Local engine, setup, server and performance changes merged without conflicts.

## Where the speed comes from

The fused prompt path groups token/expert assignments on the GPU, quantizes activations once per token, reads weights directly from resident or streamed expert slots, and combines gate/up, SwiGLU and intermediate quantization. This removes CPU grouping synchronization, weight gathering and separate intermediate passes. Smaller buffers allow a 1024-slot Q2_0 streaming ring (512 for native IQ), improving transfer overlap. Q2_0 uses this by default; native IQ requires `STRATA_PF_FUSED=1`.

Upstream measured Q2_0 prompt rates of 1294 -> 1570 at 4K, 2170 -> 2653 at 32K and 2123 -> 2468 tokens/s at 128K on an RTX 5070 12 GB, Ryzen 5 7600 and 64 GB DDR5-5200, Windows. These are upstream measurements, not this VM's. The fused kernels require sm80+. They change intermediate rounding; they do not promise identical output to the old path.

Cluster decode divides QSA block top-k and greedy vocabulary argmax across cooperating thread blocks, keeping identical selected IDs/tokens. It requires sm90+; upstream measured Q2_0 decode 89 -> 93.5 at 4K and 64.5 -> 76.4 tokens/s at 128K on that same PC. See [upstream measurements and switches](DETAILS.md#speed-measured).

Our Tesla PG500-216 is Volta sm70. Both optimizations select their previous fallback paths here. The saved Orca config still uses `STRATA_PREFILL_RING=8`; no switch can make these newer GPU instructions available on Volta. Do not attribute upstream's speed percentages to this VM.

## Other changes and our retained work

Release adds accurate cancelled-prompt accounting, draft-head diagnostics, update scripts, and optional learned-cache persistence. `expert_profile_save` saves residency and learned routing rankings and reuses the profile at the next start. It remains disabled in our config. Our background usage recorder remains enabled and records decode/verification frequencies to separate session snapshots; it does not automatically replace the shipped profile. These mechanisms can coexist but serve different purposes.

Retained: CUDA 12.8/sm70 build, BF16-to-FP32 projection fallback, AVX2/scalar expert handling, CPU rebuild detection, Orca shard/PLE compatibility, external model storage, unified launcher, usage recording and performance tools, and config-controlled reasoning guard. The current saved Orca config is text-only with automatic worker selection (the explicit 16-worker override was removed at the user’s request after validation), INT8 KV, 262144 context / 32768 resident KV cells, MTP speculation 4, and the shipped expert profile. Earlier benchmark notes describe vision enabled at that earlier date.

Our BF16 weight-reuse experiment remains reverted: its isolated GEMM gains did not improve full-model prompt speed. Worker tuning and vision/cache measurements are documented in [the prior performance report](../bench/results/2026-10-01-v100-performance/README.md). Usage collection alone does not establish a throughput improvement; a learned profile still needs held-out comparison before activation.

## Validation

On this VM: CUDA 12.8/sm70 Release build succeeded; 64 CTest checks passed and two fused-kernel checks skipped because this GPU is below sm80. `ple_parity` and `expert_multi_test` were excluded for the unavailable model fixtures and AVX-512 requirements documented in the previous integration. The cluster parity selftest passed its supported fallback checks; it cannot validate sm90 hardware here.

Setup/tooling: 274 tests passed. Server: 171 tests, successful with nine fixture/platform skips. Local Python: eight tests passed. Git whitespace and launcher/update shell syntax checks passed. These checks do not validate newer-GPU, Windows or HIP hardware behavior.

The installed binary is engine/strata with matching source metadata. Previous binary and metadata remain in build/pre-upstream-0.1.36-engine/. No saved model settings were changed.

Actual Orca IQ4_XS expert CPU/GPU parity passed at layers 0, 11, 42 and 47 with zero failures.

A matched localhost API smoke used 16 workers for both versions and the same saved 1124-token code-review prompt, one excluded 128-token warm-up, 128 generated tokens, temperature zero, thinking off and prompt reuse disabled. Old engine prompt/decode: 395.3 / 58.8 tokens/s; new engine: 395.5 / 58.7. Both returned successful responses with identical reply text. One request per version is insufficient to establish a stable gain or regression. Engine hashes, exact configs and replies are in the [baseline](../bench/results/2026-10-02-upstream-0136-baseline/results.json) and [updated](../bench/results/2026-10-02-upstream-0136-after/results.json) records. The source_commit field is the pre-merge harness HEAD; the after record additionally identifies the integrated upstream tag. Benchmark usage snapshots went to the configured usage directory and should not be treated as independent user-workload evidence. Temporary servers and engines were stopped; no inference server was left running.
