"""Wait for the owned download, then convert and run one known-target smoke test.

This is a development orchestrator, not a daemon/service. It exits on the first
error, does not retry inference, and never overwrites converted/output directories.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wait_for_weights(source, headers, *, download_pid, timeout=7200):
    start = time.monotonic()
    while True:
        ready = all(
            (source / shard).is_file() and (source / shard).stat().st_size == info["file_bytes"]
            for shard, info in headers.items()
        )
        if ready:
            return
        if time.monotonic() - start > timeout:
            raise TimeoutError("weight download did not finish within the bounded wait")
        try:
            os.kill(download_pid, 0)
        except ProcessLookupError as exc:
            raise RuntimeError(
                "download process exited before complete weight shards appeared; inspect models/download.log"
            ) from exc
        time.sleep(10)


def verify_smoke(result, truth):
    if not result.get("valid"):
        raise ValueError("real-model output did not pass the strict GAM parser")
    points = [point for prediction in result["predictions"] for point in prediction["pixels"] or []]
    if len(points) != 1 or len(points[0]) != 2:
        raise ValueError("smoke scene requires exactly one predicted point")
    error = math.dist(points[0], truth["center_pixels"])
    if error > truth["tolerance_pixels"]:
        raise ValueError(f"target center error {error:.2f}px exceeds smoke tolerance")
    return {
        "predicted_pixels": points[0],
        "center_error_pixels": error,
        "tolerance_pixels": truth["tolerance_pixels"],
        "note": "One synthetic image only; not official-checkpoint parity or a benchmark.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-pid", type=int, required=True)
    args = parser.parse_args()
    source = ROOT / "models/groundingpi-hf"
    model = ROOT / "models/groundingpi-mlx-4bit"
    output = ROOT / "outputs/first-run"
    status_file = ROOT / "models/first-run-status.json"
    status = {"pid": os.getpid(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    def update(phase, **extra):
        status.update(phase=phase, **extra)
        temporary = status_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_file)
        print(f"[{status['started_at']}] {phase}", flush=True)

    try:
        if args.download_pid <= 1:
            raise ValueError("download PID must identify the session-owned download process")
        if model.exists() or output.exists():
            raise FileExistsError(
                "first-run model/output directory already exists; refusing overwrite"
            )
        headers = {
            p.name.removesuffix(".json"): json.loads(p.read_text())
            for p in (ROOT / "upstream/headers").glob("*.json")
        }
        if len(headers) != 2:
            raise ValueError("expected the two audited header snapshots")
        update("waiting_for_download", download_pid=args.download_pid)
        wait_for_weights(source, headers, download_pid=args.download_pid)
        update("converting_language_4bit")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mlx_grounding_pi.convert",
                "--source",
                str(source),
                "--output",
                str(model),
                "--bits",
                "4",
                "--group-size",
                "64",
            ],
            cwd=ROOT,
            check=True,
        )
        update("running_real_model_smoke_test")
        truth = json.loads((ROOT / "examples/synthetic_scene.json").read_text())
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mlx_grounding_pi.generate",
                "--model",
                str(model),
                "--image",
                str(ROOT / "examples/synthetic_scene.png"),
                "--phrase",
                truth["phrase"],
                "--task",
                "point",
                "--max-tokens",
                "128",
                "--output",
                str(output),
            ],
            cwd=ROOT,
            check=True,
        )
        result = json.loads((output / "result.json").read_text())
        metrics = verify_smoke(result, truth)
        update("completed", smoke=metrics, result_path=str(output / "result.json"))
    except Exception as exc:
        update("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
