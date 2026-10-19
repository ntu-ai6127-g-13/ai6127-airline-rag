"""Monitor one reproducible weak-only training run, with heartbeat, deadline and STOP."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--control-dir', type=Path, required=True)
    parser.add_argument('--max-minutes', type=float, default=90)
    args = parser.parse_args()
    if args.output_dir.exists() or args.control_dir.exists(): raise ValueError('Preserve prior runs; choose new directories')
    if args.max_minutes <= 0: raise ValueError('Deadline must be positive')
    args.control_dir.mkdir(parents=True)
    command = [sys.executable, '-m', 'src.train_classifiers', 'train-all', '--output-dir', str(args.output_dir.resolve()), '--seed', '42']
    started = time.monotonic(); state = {'status':'running', 'command':command, 'gold_used_for_training':False,
        'started_at':datetime.now(timezone.utc).isoformat(), 'max_minutes':args.max_minutes}
    def flush():
        state.update(elapsed_seconds=round(time.monotonic()-started, 2), heartbeat_at=datetime.now(timezone.utc).isoformat())
        p=args.control_dir/'monitor.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state, indent=2));tmp.replace(p)
    with (args.control_dir/'training.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state['pid']=process.pid;flush()
        try:
            while process.poll() is None:
                if (args.control_dir/'STOP').exists() or time.monotonic()-started > args.max_minutes*60:
                    state['status']='stopped' if (args.control_dir/'STOP').exists() else 'timed_out'
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL);process.wait()
                    break
                flush();time.sleep(5)
            if state['status']=='running': state['status']='completed' if process.returncode==0 else 'failed'
            state['returncode']=process.returncode
        except BaseException:
            if process.poll() is None: os.killpg(process.pid, signal.SIGTERM);process.wait(timeout=10)
            state['status']='failed';raise
        finally: flush()
    print(json.dumps(state), flush=True)
    if state['status']!='completed': raise SystemExit(1)


if __name__=='__main__':main()
