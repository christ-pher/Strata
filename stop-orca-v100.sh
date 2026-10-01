#!/bin/sh
set -eu
cd "$(dirname "$0")"
export STRATA_MODEL_HOME="${STRATA_MODEL_HOME:-/opt/models/Strata}"
exec .venv/bin/python - <<'PY'
import os,signal
from pathlib import Path
pidfile=Path(os.environ['STRATA_MODEL_HOME'])/'runtime/orca-server.pid'
if not pidfile.exists():
    raise SystemExit('No Orca launcher PID is recorded.')
pid=int(pidfile.read_text().strip())
try:
    command=Path(f'/proc/{pid}/cmdline').read_bytes()
except FileNotFoundError:
    pidfile.unlink(); print('Orca is already stopped.'); raise SystemExit(0)
if b'serve.server' not in command or not any(name in command for name in
        (b'strata-orca-iq3_xxs.json', b'strata-orca-iq4_xs.json')):
    raise SystemExit('Recorded PID is not the Orca server; no process was stopped.')
children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
engine=str(Path.cwd()/'engine/strata').encode()
engine_pids=[]
for child in children:
    try:
        if Path(f'/proc/{child}/cmdline').read_bytes().split(b'\0')[0]==engine:
            engine_pids.append(int(child))
    except FileNotFoundError: pass
for target in [pid,*engine_pids]:
    try: os.kill(target,signal.SIGTERM)
    except ProcessLookupError: pass
pidfile.unlink()
print('Stopped Orca server and its engine.')
PY
