# Upstream 0.1.35 integration

Merged upstream main at `d9ab843` (2026-10-02), advancing the fork from
0.1.32. Releases 0.1.33, 0.1.34 and 0.1.35 are included, together with the
existing V100 performance work and the config-only thinking repetition guard.

Upstream adds cancellation when a client disconnects, stricter request parsing,
per-request and total speculative-draft metrics, independent vision GPU
selection, atomic setup config writes, setup recommendations and risk handling,
GGUF choice diagnostics, an optional setup MCP server, CUDA fallback fixes,
and Windows/Linux HIP improvements. The new upstream setup MCP server is
available as code and documentation; it was not configured or launched here.

The fork retains its CUDA 12 / sm70 build, BF16 projection fallback, AVX2/scalar
expert dispatch and CPU rebuild detection, external model storage, unified
V100 launcher, Orca shard support and BF16 compatibility conversion, expert
usage collection and benchmark tools. README.original.md now contains the
current upstream README; README.md retains the fork's instructions and VM
measurements and identifies the updated baseline.

Integration adjustments:

- Keep the unified launcher while adopting atomic config writes.
- Recognize Orca's two- and three-shard GGUF names in upstream choice diagnostics.
- Restore the selected CUDA toolkit version in the merged install path, including
  its platform install arguments.
- Retain Volta refusal of sm75 tensor instructions even with STRATA_QSA_WARP set,
  alongside upstream compute-capability emulation and Turing behavior.
- Extend the new setup MCP fallback metadata to include the fork's Orca models.
- Correct upstream golden-config normalization on Linux so `strata-*.log` is
  not mistaken for the executable, and explicitly model Windows executable
  names in the Windows HIP archive test. Recommended configs remain unchanged.

Validation on the V100 / EPYC AVX2 VM:

- CUDA 12.8 / sm70 Release engine and full test targets built successfully.
- Tooling unittest discovery: 268 tests passed.
- Server unittest discovery: 162 tests successful, with 9 platform/fixture skips.
- Local Python regressions: 8 tests passed.
- CTest: all 62 supported checks passed, including expert-cache per-layer
  admission, hybrid KV attention, BF16 GEMM and scalar experts.
- Actual Orca IQ4_XS expert CPU/GPU parity at layers 0, 11, 42 and 47: zero failures.
- Python syntax, launcher shell syntax and Git whitespace checks passed.

`ple_parity` and `expert_multi_test` were excluded from CTest: the former
requires unavailable original-model PLE/dense/capture fixtures and the latter
requires AVX-512 VNNI/VBMI, which this guest does not expose. Neither is counted
as a pass. This validation does not establish Windows/AMD hardware behavior,
full-context model quality, or prevention of paraphrased reasoning loops.

The rebuilt engine is installed in the ignored engine/ directory with matching
source metadata. The prior engine and metadata are retained in
build/pre-upstream-0.1.35-engine/. Generated configs, model assets, binaries and
logs remain local and excluded from version control. Orca retains sampled
settings and `reasoning_repetition_guard: true`, without the fixed thinking cap.

A temporary localhost-only server with the installed Orca config started with
engine 0.1.35 and repetition intervention enabled. A sampled, non-thinking
OpenAI API request for 17 + 25 returned `42` and finish reason `stop`. The test
server and its engine were stopped after validation; no inference server was
left running. This short smoke is a functional check, not a throughput sample.
