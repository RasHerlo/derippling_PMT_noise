"""Restart the v4 dest batch until one run walks every discovered job.

Native crashes (Windows 0xC0000005) kill python without a traceback.
Skip-existing plus this loop finishes the remaining dest folders.
Stops after a run that logs a summary row for every job (errors still count
as walked; those dests stay on the catch-up list).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = r"E:\Rasmus-Guillermo\ECF1"
DEFAULT_OUT = r"F:\CollectedData"
SLEEP_SEC = 20.0


def _runs_dir(out: Path) -> Path:
    return out / ".defringe_v4_runs"


def _latest_summary(out: Path) -> Path | None:
    paths = sorted(_runs_dir(out).glob("20*/summary.jsonl"))
    return paths[-1] if paths else None


def _run_progress(summary: Path) -> tuple[int, int]:
    """Return (n_jobs_from_jobs_json, max_i_in_this_summary)."""
    jobs_path = summary.parent / "jobs.json"
    n_jobs = 0
    if jobs_path.is_file():
        try:
            jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
            n_jobs = len(jobs) if isinstance(jobs, list) else 0
        except (OSError, json.JSONDecodeError):
            n_jobs = 0
    max_i = 0
    try:
        for line in summary.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            max_i = max(max_i, int(rec.get("i") or 0))
    except OSError:
        pass
    return n_jobs, max_i


def _log(path: Path, msg: str) -> None:
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}  {msg}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
    print(line, end="", flush=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--root" not in argv:
        argv.extend(["--root", DEFAULT_ROOT])
    if "--out" not in argv:
        argv.extend(["--out", DEFAULT_OUT])
    out = Path(argv[argv.index("--out") + 1])
    log_path = _runs_dir(out) / "watchdog.log"
    cmd = [sys.executable, "-u", "-m", "batch_defringe.process_v4", *argv]
    _log(log_path, f"watchdog start  cmd={' '.join(cmd)}")
    while True:
        t0 = time.time()
        _log(log_path, "launching process_v4")
        rc = subprocess.call(cmd, cwd=str(_REPO))
        dt = time.time() - t0
        summary = _latest_summary(out)
        n_jobs, max_i = _run_progress(summary) if summary is not None else (0, 0)
        _log(
            log_path,
            f"process_v4 exited rc={rc} after {dt:.0f}s  "
            f"this_run max_i={max_i}/{n_jobs}  summary={summary}",
        )
        if n_jobs > 0 and max_i >= n_jobs:
            _log(log_path, "full job list walked; watchdog stopping")
            return 0 if rc == 0 else rc
        _log(log_path, f"incomplete run (crash or early stop); retry in {SLEEP_SEC:.0f}s")
        time.sleep(SLEEP_SEC)


if __name__ == "__main__":
    raise SystemExit(main())
