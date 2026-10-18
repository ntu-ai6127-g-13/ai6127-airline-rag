"""Run Streamlit with a small crash-restart supervisor for local demos."""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8517)
    args = parser.parse_args()
    command = [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
               "--server.headless", "true", "--server.address", "127.0.0.1",
               "--server.port", str(args.port), "--server.fileWatcherType", "none",
               "--browser.gatherUsageStats", "false"]
    stopping = False
    child: subprocess.Popen | None = None

    def stop(_signal, _frame) -> None:
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, stop)
    while not stopping:
        print(f"Starting query app on http://127.0.0.1:{args.port}", flush=True)
        child = subprocess.Popen(command, cwd=ROOT)
        return_code = child.wait()
        if stopping:
            break
        print(f"Query app exited with code {return_code}; restarting in 2 seconds", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
