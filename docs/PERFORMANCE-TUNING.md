# V100 performance tuning

Baseline: `restore/performance-baseline-20261001` (`d3ba706`), verified equal to origin/main before edits.
Local binary, build metadata and Orca config backups: `build/performance-restore-20261001/` (ignored).
Changes are on `perf/v100-20261001`; each optimization is committed separately for independent `git revert`.

## Background expert usage

Set `STRATA_EXPERT_USAGE_DIR` to a directory dedicated to one model/quantization. Empty/unset disables recording.
For persistent recording, set it in the saved config's `env` object using `tools/v100_tuning.py --usage-dir`.
Config values take precedence over inherited environment variables. The current Orca config is enabled after
validation; other models remain opt-in. Each setting gets a field-specific restore record under the ignored
build directory, so restoring recording does not undo a later worker or vision change.

The engine counts main-model decode/verification expert selections already delivered to the CPU pool, including
GPU hits and rejected verification positions. It does not add device transfers or capture prompts, replies,
token IDs, batched prompt routing, or the separate MTP draft layer. This is a workload frequency estimate,
not an exact count of accepted output tokens. The normal expert cache still adapts during service.

Counters occupy 192 KiB for 48 x 512 experts. At each completed/cancelled request, a copy goes to a background
writer; one pending snapshot is retained even if storage is slow. Each engine session has its own cumulative
`usage-*.tsv` file, atomically replaced on Linux. Storage is bounded per session, rather than growing per token.
A crash can lose the current request or a pending snapshot; completed snapshots remain readable. Failed writes
report to the engine log and do not stop inference. No profile is applied automatically.

Periodically, preferably after collecting a range of practical tasks:

```sh
.venv/bin/python tools/make_profile.py --no-base \
  --usage /path/to/model-usage/usage-*.tsv --out /path/to/candidate-profile.bin
```

Include each session snapshot once; do not also include copied earlier snapshots from the same session.
Keep different models and quantizations separate. The tool rejects incompatible dimensions and the default
base-preserving mode, which would hide new rankings behind the complete shipped profile. Duplicate paths
are counted only once. Compare a candidate against the shipped profile on held-out tasks before selecting
it with `--expert-profile`. Retain snapshots and profiles outside Git; archive old session files as needed.

## Controlled experiments

`tools/v100_bench.py` supports `--workers`, `--vision none|gpu`, `--exe`, and repeated `--engine-env NAME=VALUE`.
It uses temporary localhost servers and leaves saved launcher configs unchanged. Vision comparisons load the
actual GPU encoder before the engine sizes its expert cache. Text requests measure the cost of keeping vision
resident, not image-encoding throughput. Prompt reuse is disabled, but adaptive expert caching remains active.
All variants must replay the same warm-up, prompt order, output budget and repetitions. Record engine hashes
and startup cache capacity, and repeat close results before deciding on defaults.
