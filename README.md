# Strata for Tesla V100 / Volta

A personal fork of [Niko1221/Strata](https://github.com/Niko1221/Strata), adapted to run **Qwen3.8-Flash-Next on a 32 GiB Volta GPU inside a Linux VM**, with EPYC 7402 host CPU passthrough (AVX2, FMA and F16C).

This fork is based on **upstream 0.1.31 (`9259cad`)**. It keeps Strata's browser UI, OpenAI-compatible and Anthropic-compatible APIs, expert caching, and MTP speculative decoding, with local changes for the V100 and this VM's CPU capabilities.

The unchanged upstream README is preserved in [README.original.md](README.original.md). Its performance figures describe upstream hardware; our measurements on this VM are in the [personal benchmarks](#personal-benchmarks-and-validation) below.

## What changed in this fork

- **Volta / sm70 support:** build-time and runtime GPU checks accept compute capability 7.0, and setup recognizes the V100.
- **CUDA 12 selection:** setup selects a CUDA 12 toolkit for Volta, even when CUDA 13 is also installed. CUDA 13 cannot compile this GPU target. On Linux, setup selects a compatible GCC version when the default compiler is too new for the toolkit.
- **BF16 prompt projections on older GPUs:** BF16 inputs expand to FP32 for tiled SGEMM, with a SIMT path for narrow products. This lets the V100 execute projections that use newer GPU features upstream.
- **CPU feature handling:** expert dispatch selects AVX2 automatically on this EPYC VM and retains scalar fallback for guests without SIMD. Setup detects CPU changes and rebuilds native ggml objects with compiler-cache reuse bypassed.
- **Orca compatibility:** the PLE loader retains the packed BF16 key produced by the compatibility conversion. Dedicated scripts prepare and launch Orca IQ3_XXS and IQ4_XS, including experts split across GGUF shards.
- **External model storage:** weights, prepared packs, MTP assets, and local settings live under `/opt/models/Strata`, outside the source checkout.
- **V100 launchers and numerical checks:** reproducible launch defaults plus BF16 GEMM and scalar expert regression tests.

The 0.1.31 merge also retains compatibility with this fork's existing whitespace-separated v4 Orca manifests and corrects upstream prompt-attention dispatch to distinguish Volta sm70 from Turing sm75. New packs use upstream's comma-separated per-role shard format.

Upstream 0.1.31 functionality is retained, including its newer prompt processing, idle unload, conversation-cache options, and rope-scaling controls. The validated target for this fork is the single-GPU Linux VM below; other upstream hardware and platforms retain their own setup documentation.

## The VM this runs on

Guest-visible configuration checked on 2026-10-01:

| Component | Configuration |
| --- | --- |
| Operating system | Ubuntu 24.04.5 LTS, x86_64 |
| Virtualization | QEMU/KVM |
| CPU | 24 vCPUs; guest reports `AMD EPYC 7402 24-Core Processor` |
| CPU instruction sets | AVX2, FMA and F16C exposed; no AVX-512 |
| System memory | 160 GiB assigned, approximately 157 GiB usable RAM |
| Swap | 8 GiB |
| GPU | Tesla PG500-216, Volta / V100-class, compute capability 7.0 |
| GPU memory | 32,768 MiB (32 GiB) |
| NVIDIA driver | 580.178.04 |
| CUDA build toolkit | 12.8.93 |
| Guest root filesystem | Approximately 492 GiB; repository and model storage share this filesystem |
| Source checkout | `/opt/engines/Strata` |
| Model storage | `/opt/models/Strata` |

These describe the VM, not the physical host CPU or storage medium. Missing guest SIMD affects CPU expert execution, so results from modern desktop CPUs should not be assumed to apply here.

## Install and start

Use a current NVIDIA driver, a **CUDA 12 toolkit that supports sm70**, and enough RAM and disk space for the model you select. The verified toolkit is CUDA 12.8. The launcher prepares the Python environment, builds the local engine, downloads the selected weights, and prepares the pack on first use.

Clone this fork into a directory you can write to:

```sh
git clone https://github.com/christ-pher/Strata.git
cd Strata
mkdir -p /opt/models/Strata
./run-v100.sh IQ2_XS
```

The storage root must be writable by your user. To use another location:

```sh
STRATA_MODEL_HOME="$HOME/models/Strata" ./run-v100.sh IQ2_XS
```

For the existing VM checkout:

```sh
cd /opt/engines/Strata
./run-v100.sh IQ3_S
```

Wait for the `ready` message and keep the terminal or SSH session open. Stop with **Ctrl+C** before starting another model on the same port. First preparation downloads tens of gigabytes, and startup loads a large expert arena into RAM; subsequent starts reuse the prepared files.

The V100 launcher defaults to **IQ2_XS, 32,768 context, INT8 KV, text input, and port 8080**, with `STRATA_PREFILL_RING=8`. It binds to `0.0.0.0`; for access confined to the VM, append `--host 127.0.0.1`. Use `--api-key` when you want authenticated network access. Existing model configurations can retain previously saved context settings; the VM's latest IQ3_S smoke check used 262,144 context with 32,768 resident KV cells.

Common commands:

```sh
./run-v100.sh IQ3_S --no-start              # download and prepare only
./run-v100.sh IQ2_XS --context 65536        # change context
./run-v100.sh IQ2_XS --host 127.0.0.1       # listen only inside the VM
./run-v100.sh IQ2_XS --family swift         # Swift fine-tune
./run-v100.sh coder                        # pruned coding model, IQ1_M
```

## Model choices

| Choice | Launcher | Notes |
| --- | --- | --- |
| Original Q2_0 | `./run-v100.sh Q2_0` | Most compressed of the original installer choices |
| Original IQ2_XS | `./run-v100.sh IQ2_XS` | Default; validated on this VM |
| Original IQ3_XXS | `./run-v100.sh IQ3_XXS` | Higher precision than IQ2_XS |
| Original IQ3_S | `./run-v100.sh IQ3_S` | Validated with the upgraded 0.1.30 engine |
| Coder IQ1_M | `./run-v100.sh coder` | Pruned, coding-focused model |
| Swift 1.5 | Append `--family swift` | Q2_0, IQ2_XS, or IQ3_XXS |
| Orca IQ3_XXS | `./run-v100.sh orca` | Separate compatibility preparation; see below |
| Orca IQ4_XS | `./run-v100.sh orca-iq4_xs` | Validated at 262K configured context; see [IQ4_XS notes](docs/ORCA-IQ4-XS.md) |

The launcher selects the supported repositories and shards. A matching quantization name or `.gguf` extension alone does not establish compatibility: the engine expects the supported `qwen4exp` architecture and tensor formats. Models for a different Qwen architecture and Safetensors/AWQ/GPTQ/EXL2 files are not direct replacements.

Plan for both model shards, generated packs, MTP assets, and runtime memory. IQ2_XS downloads total about **63.4 GiB** and IQ3_S about **77.9 GiB** in logical file sizes; physical disk usage can differ. The large PLE lookup shard stays on disk, while expert execution uses system RAM and the GPU cache. More VRAM helps expert caching, but system RAM remains necessary.

### Orca compatibility setup

```sh
./run-v100.sh IQ2_XS --no-start  # prepare the original MTP assets first
./setup-orca-v100.sh            # download and pack Orca IQ3_XXS
./run-orca-v100.sh
./stop-orca-v100.sh
```

The Orca scripts require the `hf` CLI and use the documented BF16 compatibility conversion. To prepare or launch IQ4_XS, pass `IQ4_XS` to the setup or run script. IQ4_XS passed actual expert parity and a saved-prompt benchmark on the current AVX2 build. See [IQ4_XS validation](docs/ORCA-IQ4-XS.md) and the [historical Orca V100 notes](docs/ORCA-V100.md).

## Browser and API usage

- **Browser:** `http://127.0.0.1:8080` inside the VM, or `http://<vm-address>:8080` from another machine when listening on all interfaces.
- **OpenAI-compatible base URL:** `http://<vm-address>:8080/v1`.
- **Anthropic-compatible endpoint:** `http://<vm-address>:8080/v1/messages`.
- **Terminal chat:** `.venv/bin/python chat.py`.

The browser includes chat and live monitoring. Thinking can be off, low, medium, or high; reasoning tokens count toward an API request's output budget. The V100 launcher disables vision by default; image support is not part of the current VM validation. Requests are served one at a time.

Example from inside the VM, with thinking disabled:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"strata","messages":[{"role":"user","content":"What is 17 + 25?"}],"temperature":0,"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
```

If authentication is configured, include an `Authorization: Bearer <your-key>` header.

## Storage and version control

```text
/opt/engines/Strata/             source checkout
/opt/models/Strata/
  models/                      downloaded model shards
  runtime/
    packs/                     prepared model packs and tokenizers
    mtp/                       MTP weights and prepared runtime
    user-config/               local setup settings
  backups/                     previous local engine backup
```

The existing VM has ignored compatibility symlinks at `data/models-download` and `data/runtime`. New installations use the external directories through the launchers. Generated `strata-*.json` configurations, logs, the local engine, builds, Python environment, model files, credentials, and caches are excluded from Git. Small upstream profiles, vocabularies, and the experimental projection fixture remain tracked because they are project data.

To update this fork:

```sh
git pull --ff-only
./run-v100.sh IQ3_S
```

Setup detects changed engine sources and rebuilds locally when needed. To integrate newer upstream changes into your own development checkout, configure `upstream` once and merge its changes while reviewing any conflicts with the V100 adaptations:

```sh
git remote add upstream https://github.com/Niko1221/Strata.git  # once per clone
git fetch upstream
git merge upstream/main
```

Commit local work before merging. On the existing VM, `origin` already points to this fork and `upstream` to the original repository. External model files can be reused across source updates.

## Personal benchmarks and validation

After EPYC host CPU passthrough and a native CPU rebuild, replaying the same saved workload measured **58.0 tokens/s for IQ2_XS (+3.7%)** and **56.9 tokens/s for IQ3_S (+19.8%)** average decode. Prompt throughput was essentially unchanged. See the [AVX2 comparison and validation](bench/results/2026-10-01-v100-avx2-native/README.md). The table below preserves the earlier no-AVX2 baseline.

Measured on **2026-10-01**, using this VM's 32 GiB Volta GPU, 24 vCPUs without AVX2, approximately 94 GiB RAM, and the fork's **0.1.30** engine built with CUDA 12.8.

| Model | Prompt average (tokens/s) | Prompt peak (tokens/s) | Decode average (tokens/s) | Decode peak (tokens/s) |
| --- | ---: | ---: | ---: | ---: |
| IQ2_XS | 725.7 | 880.3 | 55.9 | 60.1 |
| IQ3_S | 692.6 | 845.3 | 47.5 | 49.8 |

Each model ran **six measured requests**: a source-code review (**1,124 prompt tokens**), a deployment/troubleshooting plan (**4,226**), and an architecture/performance analysis (**8,348**), each repeated twice. Every request generated **512 tokens**, giving **3,072 output tokens per model**. The material came from repository documentation and source excerpts; exact prompts and responses are saved with the results.

Settings were identical: **262,144 context, INT8 KV with 32,768 resident cells, automatic prefill and expert caching, MTP speculation 4, `STRATA_PREFILL_RING=8`, temperature 0, and thinking disabled**. One 128-token warm-up per model was excluded. Prompt reuse was disabled (`--prompt-cache 0`), and every measured request reported zero cached prompt tokens. GPU expert caching remained enabled and adapted across requests.

**Average** is the arithmetic mean of the six engine-reported request rates. **Peak** is the fastest complete measured request for that phase, not an instantaneous maximum; prompt and decode peaks can come from different requests. Timings exclude model loading and HTTP/tokenization overhead. Responses were capped at 512 tokens to provide a consistent decode workload. These are single-VM throughput measurements for these prompts, not a model-quality evaluation or a prediction for other workloads.

[Per-request table and reproduction instructions](bench/results/2026-10-01-v100/README.md) · [Raw measurements and responses](bench/results/2026-10-01-v100/results.json) · [Exact prompts](bench/results/2026-10-01-v100/prompts.json)

The update also passed a full sm70 build, eight focused GPU/scalar numerical checks, the server/tokenizer suite (80 tests with two platform skips), and 19 setup rope tests. Earlier functional checks are in [V100 validation](docs/V100-validation.md).

## Troubleshooting and technical details

- **Compile fails for sm70:** check that setup is using CUDA 12 rather than CUDA 13. See [V100 build instructions](docs/V100.md).
- **CPU expert execution:** host CPU passthrough exposes AVX2, FMA and F16C, enabling automatic SIMD dispatch. Historical benchmarks below used the earlier generic guest CPU without AVX2.
- **Startup uses substantial RAM:** close other workloads or select a smaller model; watch available RAM and swapping.
- **Port 8080 is occupied:** stop the existing server before switching models.
- **Download or preparation interrupted:** rerun the same launcher to resume or reuse completed work.
- **Unexpected model errors:** use a supported release and inspect the local `strata-<model>.log`.

Strata keeps frequently used experts on the GPU, executes other experts on the CPU using system RAM, reads PLE lookup rows from disk, and uses MTP to draft tokens for verification. See the [upstream technical details](docs/DETAILS.md), [paper](docs/paper/Strata-Paper.pdf), and [original README](README.original.md) for the broader design, upstream hardware guidance, Docker, multi-GPU, and AMD support.

## Credits and licenses

The original Strata engine and documentation are by [Niko1221/Strata](https://github.com/Niko1221/Strata). This fork adds the local V100/VM adaptations described above; the Volta port draws on the work discussed in upstream [PR #130](https://github.com/Niko1221/Strata/pull/130) and [PR #139](https://github.com/Niko1221/Strata/pull/139).

Models are by Qwen, with supported compressed releases by ISTA-DASLab and fine-tunes by UkisAI and OrcaRouter. Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp); upstream credits also acknowledge Splash, ninfer, and HyperQwen.

Code is covered by the [MIT License](LICENSE), subject to the licenses of included components. The web font uses the SIL Open Font License; the experimental projection vector carries the Qwen Community License 1.0. Model files are external to this repository and retain their own licenses. See the [upstream credits and licenses](docs/DETAILS.md#credits-and-licenses).
