"""Download and SHA-verify the historical Ouro checkpoint, never silently use main."""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path
from core import DEFAULT_MODEL_DIR, MODEL_ID, REVISION, MODEL_SHA256, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    directory = args.model_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=str(directory))
    weight = directory / "model.safetensors"
    actual = sha256_file(weight)
    if actual != MODEL_SHA256:
        raise RuntimeError(f"Wrong pinned model SHA256: {actual}; expected {MODEL_SHA256}")
    files = {str(p.relative_to(directory)): sha256_file(p) for p in directory.rglob("*") if p.is_file() and not p.name.endswith(".lock") and ".cache" not in p.parts}
    provenance = dict(repo_id=MODEL_ID, revision=REVISION, checkpoint_sha256=actual, model_dir=str(directory), files=files, python=platform.python_version())
    write_json(directory / "PINNED_DOWNLOAD_VERIFIED.json", provenance)
    print(json.dumps({"verified": True, "checkpoint": str(weight), "sha256": actual}, indent=2))

if __name__ == "__main__":
    main()
