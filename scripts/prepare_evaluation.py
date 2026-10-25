"""Fetch two bounded public-photo fixtures, verifying pinned content hashes.

Images stay in ignored outputs/, not in the distributed package. No credentials,
model downloads or cloud resources are involved.
"""

import hashlib
import json
import tempfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MAX_IMAGE_BYTES = 20 * 1024**2


def main():
    manifest = ROOT / "examples/evaluation_cases.json"
    for asset in json.loads(manifest.read_text())["assets"]:
        path = (manifest.parent / asset["path"]).resolve()
        if not path.is_relative_to(ROOT / "outputs"):
            raise ValueError("photo fixtures must stay under outputs/")
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
                raise ValueError(f"existing fixture digest mismatch: {path}")
            print(f"Verified existing fixture: {path.name}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(asset["url"], timeout=60) as response:
            payload = response.read(MAX_IMAGE_BYTES + 1)
        if len(payload) > MAX_IMAGE_BYTES:
            raise ValueError("fixture exceeds bounded download size")
        if hashlib.sha256(payload).hexdigest() != asset["sha256"]:
            raise ValueError("source image changed; refusing an unpinned fixture")
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as f:
            f.write(payload)
            temp = Path(f.name)
        if path.exists():
            raise FileExistsError(path)
        temp.rename(path)
        print(f"Downloaded and verified: {path.name}")


if __name__ == "__main__":
    main()
