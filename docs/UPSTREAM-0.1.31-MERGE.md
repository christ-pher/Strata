# Upstream 0.1.31 integration

Merged upstream `main` at `9259cad` on 2026-10-01, retaining the V100 / CUDA 12 setup, BF16 GEMM fallback, CPU feature rebuild detection, scalar dispatch, external model storage, and Orca compatibility conversion.

New packs use upstream's comma-separated per-role v4 manifest format. The loader also accepts the fork's existing three whitespace-separated shard names (`-` means the primary shard), so installed Orca packs need no conversion.

Upstream's prompt-attention dispatch checked only the GPU major capability, incorrectly admitting Volta 7.0 to the Turing 7.5 tensor-core kernel. The merged implementation checks both major and minor capability and returns to the existing attention fallback below sm75. The hybrid KV parity test verifies this refusal on V100.

Validation on this VM:

- CUDA 12.8 / sm70 engine and test build completed.
- Tooling unittest discovery: 122 tests passed.
- Server unittest discovery: 100 tests, successful with 7 skips.
- CTest: all 53 supported checks passed, including BF16 GEMM, hybrid KV, expert layout and legacy manifest compatibility.
- Actual Orca IQ4_XS expert CPU/GPU parity at layers 0, 11 and 47: zero failures.
- Python syntax, launcher shell syntax and Git whitespace checks passed.

The initial full CTest run also attempted two unavailable checks: `ple_parity` requires the original Q2_0 PLE table and dense/capture fixtures, and `expert_multi_test` requires AVX-512 VNNI/VBMI. Those checks are excluded from the supported run above; they are not reported as passes. The initial hybrid KV failure led to the dispatch fix described above.

The rebuilt local engine was installed in the ignored `engine/` directory with matching source metadata. No inference server was started for this validation, and no Strata server was left running.
