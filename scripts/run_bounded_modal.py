"""Launch only the approved one-shot app with a 20-minute total local wall limit.

No detached deployment or retries. On timeout, interrupt the client and stop only
its uniquely observed app ID; never enumerate or modify other apps.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT_SECONDS = 1200


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approve-paid-run", action="store_true", help="Explicitly allow this paid cloud run"
    )
    parser.add_argument(
        "--profile", help="Optional existing Modal profile; otherwise use the active profile"
    )
    args = parser.parse_args()
    if not args.approve_paid_run:
        parser.error(
            "Cloud usage is billable. Supply --approve-paid-run only after approving the scope/cost."
        )
    cli = shutil.which("modal")
    if cli is None:
        parser.error("Modal CLI is not installed; no automatic installation or login is attempted.")
    profile = ["--profile", args.profile] if args.profile else []
    log = ROOT / "outputs/modal-reference.log"
    state_path = ROOT / "outputs/modal-reference-status.json"
    started = time.monotonic()
    state = {
        "phase": "starting",
        "wall_timeout_seconds": LIMIT_SECONDS,
        "scope": {
            "gpu": "A10G:1",
            "cpu_hard_limit": 4,
            "memory_hard_limit_mib": 32768,
            "retries": 0,
            "persistent_deployment": False,
            "volumes": False,
        },
    }

    def update(**kwargs):
        state.update(kwargs)
        state["elapsed_seconds"] = time.monotonic() - started
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)

    if (ROOT / "outputs/reference-bf16").exists() or log.exists():
        raise FileExistsError("reference output/log already exists; refusing a duplicate paid run")
    log.parent.mkdir(exist_ok=True)
    command = [
        cli,
        "run",
        *profile,
        "scripts/modal_reference.py",
        "--output",
        "outputs/reference-bf16",
    ]
    with log.open("x") as stream:
        client = subprocess.Popen(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        update(phase="running", client_pid=client.pid)
        try:
            code = client.wait(timeout=LIMIT_SECONDS)
            update(phase="artifacts_received" if code == 0 else "failed", exit_code=code)
        except subprocess.TimeoutExpired:
            update(phase="stopping_after_wall_timeout")
            os.killpg(client.pid, signal.SIGINT)
            try:
                client.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(client.pid, signal.SIGTERM)
                try:
                    client.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(client.pid, signal.SIGKILL)
                    client.wait(timeout=10)
            app_ids = set(re.findall(r"\bap-[A-Za-z0-9]+\b", log.read_text()))
            if len(app_ids) == 1:
                app_id = app_ids.pop()
                stopped = subprocess.run(
                    [cli, "app", "stop", app_id, "--yes", *profile],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=30,
                    check=False,
                )
                update(phase="timed_out", app_id=app_id, explicit_stop_exit_code=stopped.returncode)
            else:
                update(
                    phase="timed_out",
                    cleanup_note="client stopped; no unique app ID found for explicit stop",
                )
            raise SystemExit(124)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
