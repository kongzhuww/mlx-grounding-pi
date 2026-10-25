"""One-shot, explicitly authorized Modal BF16 reference. Run with existing Modal CLI.

Scope: one A10G, hard CPU/memory limits, 1200-second execution timeout, no retries,
no secrets/volumes/deployment. Uploaded paths are an explicit source/test allowlist.
"""

import io
import re
import shutil
import zipfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.1",
        "transformers==5.20.2",
        "numpy==2.5.4",
        "Pillow==13.0.0",
        "safetensors==0.8.0",
        "huggingface_hub==2.3.0",
    )
    .env(
        {
            "TORCHDYNAMO_DISABLE": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "PYTHONPATH": "/work:/work/scripts",
        }
    )
)
# Never mount the repository root/home, local models, outputs, environment or credentials.
for directory in ["mlx_grounding_pi", "upstream", "examples"]:
    image = image.add_local_dir(
        ROOT / directory, "/work/" + directory, copy=True, ignore=["**/__pycache__/**", "**/*.pyc"]
    )
image = image.add_local_file(
    ROOT / "scripts/prepare_evaluation.py", "/work/scripts/prepare_evaluation.py", copy=True
)
image = image.add_local_file(
    ROOT / "scripts/modal_reference.py", "/work/scripts/modal_reference.py", copy=True
)
app = modal.App("groundingpi-bf16-reference-once")


@app.function(
    image=image,
    gpu="A10G",
    cpu=(4, 4),
    memory=(32768, 32768),
    timeout=1200,
    max_containers=1,
    min_containers=0,
    buffer_containers=0,
    scaledown_window=2,
    retries=0,
    single_use_containers=True,
    include_source=False,
)
def run_reference():
    import os
    import subprocess
    import sys
    from pathlib import Path

    os.chdir("/work")
    sys.path.insert(0, "/work")
    print(
        "Authorized scope: one A10G; CPU 4/4; memory 32768/32768 MiB; timeout 1200s; retries 0",
        flush=True,
    )
    subprocess.run([sys.executable, "scripts/prepare_evaluation.py"], check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_grounding_pi.download",
            "--output",
            "models/groundingpi-hf",
            "--weights",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_grounding_pi.reference",
            "--root",
            "/work",
            "--source",
            "models/groundingpi-hf",
            "--cases",
            "examples/evaluation_cases.json",
            "--output",
            "outputs/reference-bf16",
        ],
        check=True,
    )
    # Return only named result artifacts; never weight shards or the container filesystem.
    output = Path("outputs/reference-bf16")
    files = sorted(output.iterdir())
    if len(files) != 11:
        raise RuntimeError("expected one report and ten case artifacts")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for path in files:
            if not (path.name == "report.json" or re.fullmatch(r"[a-z0-9-]+\.npz", path.name)):
                raise ValueError("unexpected reference artifact name")
            z.write(path, path.name)
    print("Reference export finished; returning result artifacts, not model weights", flush=True)
    return archive.getvalue()


@app.local_entrypoint()
def main(output: str = "outputs/reference-bf16"):
    target = Path(output)
    if target.exists():
        raise FileExistsError(target)
    payload = run_reference.remote()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        members = z.infolist()
        if len(members) != 11 or sum(item.file_size for item in members) > 200 * 1024**2:
            raise ValueError("unexpected reference archive size")
        if any(
            not (m.filename == "report.json" or re.fullmatch(r"[a-z0-9-]+\.npz", m.filename))
            for m in members
        ):
            raise ValueError("unsafe artifact archive member")
        target.mkdir(parents=True, exist_ok=False)
        for member in members:
            with z.open(member) as source, (target / member.filename).open("xb") as out:
                shutil.copyfileobj(source, out)
    print(f"Saved official BF16 artifacts to {target.resolve()}")
