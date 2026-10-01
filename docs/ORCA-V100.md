# Orca IQ3_XXS on this V100 VM

This setup uses `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF`, specifically the
two IQ3_XXS shards. `run-v100.sh orca --no-start` downloads them and runs the documented
`--compat-bf16` conversion. The pack includes Orca's own tokenizer and references
its original expert weights. The shared MTP runtime is reused or prepared automatically.

```sh
cd /opt/engines/Strata
./run-v100.sh orca --no-start    # prepare/resume the download and pack
./run-v100.sh orca     # start the prepared model
./run-v100.sh stop    # stop Orca, including a background instance
```

`./run-v100.sh orca` also starts it; append `--no-start` to prepare only. Stop the current server with Ctrl+C before
switching models. To return to the original, use `./run-v100.sh IQ2_XS`.

The UI is http://10.0.20.99:8080 and API base URL is
http://10.0.20.99:8080/v1. LAN access is unauthenticated, as authorized.

Configuration: `strata-orca-iq3_xxs.json`; log: `strata-orca-iq3_xxs.log`.
Defaults are 32K context, INT8 KV, automatic prompt chunks, text input, MTP spec 4,
and automatic GPU expert caching. The PLE table is in the first Orca shard.

Weights: `/opt/models/Strata/models/orca-iq3_xxs/` (85.2 GB decimal).
Compatibility pack: `/opt/models/Strata/runtime/packs/orca-iq3_xxs/` (about 1.5 GiB).
The original IQ2_XS installation is retained. Orca's expert arena needs about
49.8 GiB RAM, plus runtime buffers; the VM's 94 GiB is sufficient for one model.

## V100 validation — 2026-09-30

Both 85.2 GB shards downloaded successfully. Packing converted 460 tensors to
BF16 (1.39 GiB), leaving expert and PLE table bytes unchanged. Eight packing
unit tests passed. Real expert parity passed for layers 0 and 1 on the scalar
CPU path and V100 (IQ3_XXS gate/up, IQ4_NL down).

Startup exposed a loader conflict: native projection discovery included Orca's
original quantized PLE key, while the compatibility pack stored a BF16 key.
The engine now keeps the packed key resident and preserves its BF16 form.
Native overrides remain available for quantized keys. The original IQ2_XS
model was restarted with the changed engine and still answered `42` correctly.

Orca loaded a 49.80 GiB expert arena and 12,861 GPU-cache entries. Total system
RAM use was about 53 GiB, with about 40 GiB available. The existing 94 GiB VM is
sufficient; a RAM increase was not needed. Disk space remaining after packing
was about 23 GiB, with the original IQ2_XS files retained.

Actual unauthenticated API requests to `http://10.0.20.99:8080/v1` passed:

| Request | Prompt tokens | Result | Wall time | Accepted MTP drafts |
| --- | ---: | --- | ---: | --- |
| Add 17 + 25, answer only | 28 | `42` | 1.43 s | 3 / 3 |
| Exact phrase after repeated observation text | 1,852 | `cobalt-leaf-913` | 5.66 s | 6 / 6 |

These short-output smoke tests validate serving and batched prompt processing;
they are not sustained throughput or model-quality benchmarks. Raw smoke-test
results are saved in `/opt/models/Strata/runtime/validation/orca-smoke.json`.
