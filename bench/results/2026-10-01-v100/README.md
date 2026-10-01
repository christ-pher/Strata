# V100 VM benchmark — 2026-10-01

| Model | Prompt average (tokens/s) | Prompt peak (tokens/s) | Decode average (tokens/s) | Decode peak (tokens/s) |
| --- | ---: | ---: | ---: | ---: |
| IQ2_XS | 725.7 | 880.3 | 55.9 | 60.1 |
| IQ3_S | 692.6 | 845.3 | 47.5 | 49.8 |

## Workload and measurements

Identical prompts and model settings, one excluded 128-token warm-up per model, then two passes through three tasks. All 12 measured requests generated 512 tokens, ended at the output limit, and reused zero prompt tokens. MTP and adaptive GPU expert caching remained enabled. Models ran sequentially, IQ2_XS first. This measures an already loaded model, not cold startup or answer quality.

| Model | Task | Pass | Prompt tokens | Output tokens | Prompt tokens/s | Decode tokens/s | Request wall time (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IQ2_XS | code_review | 1 | 1124 | 512 | 513.1 | 52.5 | 11.96 |
| IQ2_XS | deployment_plan | 1 | 4226 | 512 | 749.1 | 53.9 | 15.17 |
| IQ2_XS | architecture_analysis | 1 | 8348 | 512 | 798.8 | 57.2 | 19.45 |
| IQ2_XS | code_review | 2 | 1124 | 512 | 573.4 | 60.1 | 10.49 |
| IQ2_XS | deployment_plan | 2 | 4226 | 512 | 839.3 | 53.4 | 14.64 |
| IQ2_XS | architecture_analysis | 2 | 8348 | 512 | 880.3 | 58.3 | 18.31 |
| IQ3_S | code_review | 1 | 1124 | 512 | 473.0 | 42.6 | 14.41 |
| IQ3_S | deployment_plan | 1 | 4226 | 512 | 725.7 | 46.7 | 16.82 |
| IQ3_S | architecture_analysis | 1 | 8348 | 512 | 772.2 | 49.1 | 21.29 |
| IQ3_S | code_review | 2 | 1124 | 512 | 523.7 | 48.0 | 12.85 |
| IQ3_S | deployment_plan | 2 | 4226 | 512 | 815.4 | 48.7 | 15.71 |
| IQ3_S | architecture_analysis | 2 | 8348 | 512 | 845.3 | 49.8 | 20.21 |

Summary averages are arithmetic means of request rates; peaks are maximum request rates. Request wall times include client/server overhead, while the throughput columns use engine phase timings. Larger prompts can achieve higher prefill throughput because fixed overhead is amortized. Second passes also benefit from adaptive expert placement, despite prefix reuse being disabled.

## Configuration

Ubuntu 24.04.5 LTS, QEMU/KVM, 24 vCPUs without AVX2/AVX-512, approximately 94 GiB RAM, Tesla PG500-216 (sm70) with 32 GiB VRAM, driver 580.178.04, CUDA 12.8.93. See [hardware.json](hardware.json).

Both models: 262,144 context; INT8 KV; 32,768 resident KV cells; automatic expert cache and prefill; speculation 4; minimum draft probability 0.5; prefill ring 8. Thinking off, temperature 0, output budget 512, prompt checkpoints off. Configurations are in [iq2_xs-config.json](iq2_xs-config.json) and [iq3_s-config.json](iq3_s-config.json).

## Artifacts and reproduction

[results.json](results.json) contains every response, usage count, phase timing, summary, warm-up, engine build metadata and binary SHA-256. [prompts.json](prompts.json) preserves the exact workload. The recorded source commit identifies the engine source baseline; the benchmark harness is added with this results commit. Server/engine logs stay local and are ignored by Git.

Rerun from the repository with the two installed model configurations and a free localhost port:

```sh
.venv/bin/python tools/v100_bench.py \
  --prompts bench/results/2026-10-01-v100/prompts.json \
  --out bench/results/my-v100-run
```

The script starts temporary servers on 127.0.0.1:18080 and stops them after each model. The command above replays the saved prompts exactly. Without `--prompts`, the harness regenerates excerpts from the current source checkout, which can change between revisions. This run used the saved prompts in this directory.
