"""Benchmark installed V100 models through temporary localhost API servers.

Run from the repository: .venv/bin/python tools/v100_bench.py
Results include exact prompts, responses, configurations and engine timings.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from strata_tokenizer import Tokenizer


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def corpus(files):
    return "\n\n".join(f"FILE: {name}\n{(ROOT / name).read_text()}" for name in files)


def make_prompts(tokenizer):
    cases = [
        ("code_review", 1024, ["serve/frontend.py", "serve/server.py"],
         "Review this Python server excerpt. Explain its responsibilities, identify concrete edge cases "
         "visible in the excerpt, and propose improvements with example Python code. "
         "Use detailed paragraphs and a numbered review; distinguish evidence from assumptions."),
        ("deployment_plan", 4096, ["README.original.md", "docs/V100.md", "docs/DETAILS.md"],
         "Using the documentation below, write a detailed deployment and troubleshooting plan for a "
         "single Volta GPU Linux VM with 94 GiB RAM and an EPYC 7402 exposing AVX2. Cover model preparation, memory, "
         "storage, API configuration, lifecycle and verification. Explain tradeoffs and give shell examples."),
        ("architecture_analysis", 8192,
         ["docs/DETAILS.md", "src/core/expert_cache.cpp", "src/prefill/prefill.cpp"],
         "Analyze the supplied Strata design notes and source excerpts. Explain expert placement, CPU "
         "fallback, prompt processing, MTP decoding and memory tradeoffs. Then propose a reproducible "
         "performance investigation for a V100 VM. Give a substantial technical report with concrete "
         "hypotheses, measurements, expected bottlenecks and limitations. Do not invent measurements."),
    ]
    result = []
    for name, target, files, task in cases:
        text = corpus(files)
        ids = tokenizer.encode(text)
        # Supplied material is an excerpt, not repeated padding.
        excerpt = tokenizer.decode(ids[:target])
        result.append({"name": name, "target_source_tokens": target, "source_files": files,
                       "prompt": task + "\n\nBEGIN MATERIAL\n" + excerpt + "\nEND MATERIAL\n\n" + task})
    return result


def ask(port, prompt, max_tokens):
    body = {"model": "strata", "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as response:
        value = json.load(response)
    return value, time.monotonic() - start


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "bench/results/2026-10-01-v100")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--prompts", type=Path, help="replay a saved prompts.json instead of generating excerpts")
    ap.add_argument("--models", nargs="+", default=["iq2_xs", "iq3_s"],
                    help="model configuration names, e.g. iq3_s orca-iq4_xs")
    args = ap.parse_args()
    if args.repeats < 1 or args.max_tokens < 128:
        ap.error("use at least one repeat and an output budget of at least 128 tokens")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise SystemExit("Results already exist; use another output directory.")
    token_path = ROOT / "data/runtime/packs/iq2_xs/tokenizer"
    vocab = json.loads((token_path / "vocab.json").read_text())
    tokens = [""] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    tok = Tokenizer(tokens, (token_path / "merges.txt").read_text().split("\n"),
                    json.loads((token_path / "token_type.json").read_text()))
    prompts = json.loads(args.prompts.read_text()) if args.prompts else make_prompts(tok)
    save(out / "prompts.json", prompts)
    result = {"started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "engine_sha256": hashlib.sha256((ROOT / "engine/strata").read_bytes()).hexdigest(),
              "engine_build": json.loads((ROOT / "engine/BUILD.json").read_text()),
              "method": {"repeats": args.repeats, "max_tokens": args.max_tokens,
                         "temperature": 0, "thinking": False, "prompt_cache": 0,
                         "prefill_ring": 8, "warmup_max_tokens": 128,
                         "average": "arithmetic mean of measured request rates",
                         "peak": "maximum complete-request rate; not instantaneous"},
              "models": {}}
    save(out / "results.json", result)
    for model in args.models:
        source = json.loads((ROOT / f"strata-{model}.json").read_text())
        # Only engine configuration is copied; authentication/local UI settings are omitted.
        cfg = {k: source[k] for k in ("exe", "args", "cwd", "tokenizer", "model_name", "lib_dirs") if k in source}
        cfg["args"] = list(cfg["args"]) + ["--prompt-cache", "0"]
        cfg.update(host="127.0.0.1", port=args.port, log=str(out / f"{model}-engine.log"))
        config_path = out / f"{model}-config.json"
        save(config_path, cfg)
        rows = []
        entry = {"configuration": cfg, "requests": rows}
        result["models"][model] = entry
        env = dict(os.environ, STRATA_PREFILL_RING="8")
        print(f"Loading {model}", flush=True)
        with (out / f"{model}-server.log").open("w") as log:
            proc = subprocess.Popen([str(ROOT / ".venv/bin/python"), "-m", "serve.server", "--engine", "strata",
                                     "--config", str(config_path), "--host", "127.0.0.1", "--port", str(args.port)],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + 240
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        raise RuntimeError(f"{model} server exited: {proc.returncode}")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2) as r:
                            if r.status == 200:
                                break
                    except OSError:
                        time.sleep(1)
                else:
                    raise RuntimeError("Server readiness timed out")
                warmup, elapsed = ask(args.port, "Write a detailed explanation of how a mixture-of-experts "
                                      "language model shares inference work between a GPU, system RAM and CPU.", 128)
                entry["warmup"] = {"response": warmup, "wall_seconds": elapsed}
                print(f"{model} warmup complete", flush=True)
                for repeat in range(1, args.repeats + 1):
                    for case in prompts:
                        response, elapsed = ask(args.port, case["prompt"], args.max_tokens)
                        timings = response["timings"]
                        if timings["cache_n"] != 0:
                            raise RuntimeError("Prompt reuse must be zero")
                        if response["usage"]["completion_tokens"] < 128:
                            raise RuntimeError("Output too short for a sustained decode sample")
                        row = {"case": case["name"], "repeat": repeat, "wall_seconds": elapsed,
                               "response": response}
                        rows.append(row)
                        save(out / "results.json", result)
                        print(f"{model} {case['name']} #{repeat}: {timings['prompt_n']} prompt tokens, "
                              f"{timings['predicted_n']} output tokens; "
                              f"prefill {timings['prompt_per_second']:.1f}, "
                              f"decode {timings['predicted_per_second']:.1f} tok/s", flush=True)
                entry["summary"] = {}
                for label, key in (("prompt", "prompt_per_second"), ("decode", "predicted_per_second")):
                    rates = [r["response"]["timings"][key] for r in rows]
                    entry["summary"][label] = {"average_tps": statistics.mean(rates), "peak_tps": max(rates),
                                               "min_tps": min(rates), "samples": len(rates)}
                save(out / "results.json", result)
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        print(f"{model} finished: {entry['summary']}", flush=True)
    result["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    save(out / "results.json", result)


if __name__ == "__main__":
    main()
