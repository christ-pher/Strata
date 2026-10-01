"""Stop servers launched with configs from this checkout (Linux)."""
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parent.parent


def server_config(argv):
    module = any(argv[i:i + 2] == ["-m", "serve.server"] for i in range(1, len(argv) - 1))
    script = len(argv) > 1 and argv[1] == str(ROOT / "serve/server.py")
    if not (module or script):
        return None
    try:
        config = Path(argv[argv.index("--config") + 1])
    except (ValueError, IndexError):
        return None
    return config if config.is_absolute() and config.parent == ROOT and config.name.startswith("strata-") and config.suffix == ".json" else None


def main():
    if len(sys.argv) != 1:
        raise SystemExit("Usage: ./run-v100.sh stop")
    stopped = 0
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = proc.joinpath("cmdline").read_bytes().decode().split("\0")
            config = server_config(argv)
            if config is None or proc.stat().st_uid != os.getuid():
                continue
            children = (proc / "task" / proc.name / "children").read_text().split()
            targets = [int(proc.name)]
            for child in children:
                try:
                    command = Path(f"/proc/{child}/cmdline").read_bytes().split(b"\0")[0]
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    continue
                if command in (str(ROOT / "engine/strata").encode(), str(ROOT / "build/strata").encode()):
                    targets.append(int(child))
            for pid in targets:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            print(f"Stopped {config.stem} server and its engine.")
            stopped += 1
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    if not stopped:
        print("No running Strata server found for this checkout.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
