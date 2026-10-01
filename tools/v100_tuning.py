"""Apply one runtime tuning setting with a field-specific restore record.

Saved model configs are local/ignored, so Git alone cannot restore their settings.
Each operation changes only CPU workers or the expert-usage directory. Restoring one
operation preserves all unrelated settings, including changes made afterward.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MISSING = {"missing": True}


def current(cfg, kind):
    if kind == "usage":
        return cfg.get("env", {}).get("STRATA_EXPERT_USAGE_DIR", MISSING)
    args = cfg["args"]
    positions = [i for i, value in enumerate(args) if value == "--pool-workers"]
    if len(positions) > 1 or (positions and positions[0] + 1 >= len(args)):
        raise ValueError("ambiguous --pool-workers arguments")
    return args[positions[0] + 1] if positions else MISSING


def assign(cfg, kind, value):
    if kind == "usage":
        if value == MISSING:
            cfg.get("env", {}).pop("STRATA_EXPERT_USAGE_DIR", None)
        else:
            cfg.setdefault("env", {})["STRATA_EXPERT_USAGE_DIR"] = value
        return
    args = cfg["args"]
    if "--pool-workers" in args:
        at = args.index("--pool-workers")
        del args[at:at + 2]
    if value != MISSING:
        args.extend(["--pool-workers", str(value)])


def save(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path, nargs="?")
    actions = ap.add_mutually_exclusive_group(required=True)
    actions.add_argument("--workers", type=int)
    actions.add_argument("--usage-dir", help="empty string disables usage recording")
    actions.add_argument("--restore", type=Path)
    ap.add_argument("--restore-dir", type=Path, default=ROOT / "build/performance-restore-20261001")
    a = ap.parse_args()
    if a.restore:
        record = json.loads(a.restore.read_text())
        path = Path(record["config"])
        cfg = json.loads(path.read_text())
        if current(cfg, record["kind"]) != record["after"]:
            raise SystemExit("Setting changed since this restore point; refusing to overwrite it")
        assign(cfg, record["kind"], record["before"])
        save(path, cfg)
        print(f"Restored {record['kind']} in {path}; restart the server to apply")
        return
    if a.config is None:
        ap.error("config is required when applying a setting")
    if a.workers is not None and a.workers < 1:
        ap.error("workers must be positive")
    path = a.config.resolve()
    cfg = json.loads(path.read_text())
    kind = "workers" if a.workers is not None else "usage"
    value = str(a.workers) if kind == "workers" else a.usage_dir
    before = current(cfg, kind)
    if before == value:
        print("Setting already applied")
        return
    a.restore_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    restore = a.restore_dir / f"{path.stem}-{kind}-{stamp}.json"
    save(restore, {"config": str(path), "kind": kind, "before": before, "after": value})
    assign(cfg, kind, value)
    save(path, cfg)
    print(f"Applied {kind}; restart the server to apply. Restore with:\n"
          f".venv/bin/python tools/v100_tuning.py --restore {restore}")

if __name__ == "__main__":
    main()
