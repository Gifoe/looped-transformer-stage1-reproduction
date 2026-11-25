"""Fetch and verify an official, revision-pinned Ouro checkpoint."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest, sha1
import json
import os
from pathlib import Path
import time

import requests


ALLOWED = {"config.json", "configuration_ouro.py", "modeling_ouro.py", "model.safetensors",
           "generation_config.json", "special_tokens_map.json", "tokenizer.json",
           "tokenizer_config.json", "tokenizer.model", "vocab.json", "merges.txt"}
REQUIRED = {"config.json", "configuration_ouro.py", "modeling_ouro.py",
            "model.safetensors", "tokenizer.json", "tokenizer_config.json"}


def checksum(path, heavy):
    with path.open("rb") as f:
        if heavy:
            return file_digest(f, "sha256").hexdigest()
        return sha1(f"blob {path.stat().st_size}\0".encode() + f.read()).hexdigest()


def verified(p, record):
    weight = p.name in {"model.safetensors", "model.safetensors.partial"}
    expected = record["lfs"]["sha256"] if weight else record["blobId"]
    return (p.is_file() and p.stat().st_size == record["size"]
            and checksum(p, weight) == expected)


def download_segment(url, fd, first, last, expected, receipt):
    if receipt.is_file():
        return
    wanted = f"bytes {first}-{last}/{expected}"
    for attempt in range(5):
        start = first
        try:
            with requests.get(url, stream=True, headers={"Range": f"bytes={first}-{last}"},
                              timeout=(18, 90)) as response:
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get("Content-Range") != wanted:
                    raise ValueError(f"Unexpected byte-range response at offset {first}")
                for buf in response.iter_content(2 * 2**20):
                    if start + len(buf) > last + 1 or os.pwrite(fd, buf, start) != len(buf):
                        raise ValueError(f"Invalid byte write at offset {start}")
                    start += len(buf)
        except requests.RequestException as exc:
            print("RANGE RETRY", first, attempt + 1, type(exc).__name__, flush=True)
            time.sleep(3 * (attempt + 1))
            continue
        if start == last + 1:
            receipt.touch()
            print("RANGE COMPLETE", first, last, flush=True)
            return
    raise RuntimeError(f"Unable to download range {first}-{last}")


def ranged_download(target, temp, source, record, workers):
    ledger = temp.parent / "MODEL_RANGES"
    marker = ledger / "LAYOUT.json"
    if marker.exists():
        plan = json.loads(marker.read_text(encoding="utf-8"))
        if not temp.is_file():
            raise ValueError("The range ledger has no corresponding partial artifact")
    else:
        prefix = temp.stat().st_size if temp.exists() else 0
        plan = {"prefix_bytes": prefix if prefix < record["size"] else 0,
                "bytes": record["size"], "sha256": record["lfs"]["sha256"]}
        ledger.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(plan) + "\n", encoding="utf-8")
    if plan["bytes"] != record["size"] or plan["sha256"] != record["lfs"]["sha256"]:
        raise ValueError("Cannot reuse range receipts from another checkpoint")
    stride = 96 * 2**20
    fd = os.open(temp, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.ftruncate(fd, plan["bytes"])
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(download_segment, source, fd, first, min(first + stride, plan["bytes"]) - 1,
                                   plan["bytes"], ledger / f"{first}.ok")
                       for first in range(plan["prefix_bytes"], plan["bytes"], stride)]
            for result in pending:
                result.result()
    finally:
        os.close(fd)
    if not verified(temp, record):
        raise ValueError("Completed byte ranges do not match the locked weight SHA-256")
    temp.replace(target)


def fetch(destination, mirror, model, record, range_workers):
    name = record["rfilename"]
    p = destination / name
    if verified(p, record):
        return
    temp = p.with_suffix(p.suffix + ".partial")
    source = f'{mirror}/{model["repository"]}/resolve/{model["revision"]}/{name}'
    if name == "model.safetensors" and range_workers > 1:
        ranged_download(p, temp, source, record, range_workers)
        print("VERIFIED", name, p.stat().st_size, flush=True)
        return
    for attempt in range(5):
        if verified(temp, record):
            temp.replace(p)
            return
        start = temp.stat().st_size if temp.exists() else 0
        if start >= record["size"]:
            start = 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        try:
            with requests.get(source, headers=headers, stream=True, timeout=(15, 90)) as response:
                response.raise_for_status()
                resume = start and response.status_code == 206
                if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                    raise ValueError(f"Unexpected retry byte range for {name}")
                with temp.open("ab" if resume else "wb") as stream:
                    for data in response.iter_content(4 * 2**20):
                        stream.write(data)
        except requests.RequestException as exc:
            print("RETRY", name, attempt + 1, type(exc).__name__, flush=True)
            time.sleep(5 * (attempt + 1))
            continue
        if verified(temp, record):
            temp.replace(p)
            print("VERIFIED", name, p.stat().st_size, flush=True)
            return
    raise RuntimeError(f"Could not download the locked artifact {name}")


def run(a):
    plan = json.loads(a.plan.read_text(encoding="utf-8"))[a.name]
    url = f'{a.mirror}/api/models/{plan["repository"]}/revision/{plan["revision"]}'
    response = requests.get(url, params={"blobs": "true"}, timeout=30)
    response.raise_for_status()
    body = response.json()
    if body["sha"] != plan["revision"]:
        raise ValueError("Unexpected repository revision")
    files = {r["rfilename"]: r for r in body["siblings"] if r["rfilename"] in ALLOWED}
    if not REQUIRED <= files.keys():
        raise ValueError("Missing official model files")
    s = files["model.safetensors"]
    if s["lfs"]["sha256"] != plan["weight_sha256"] or s["size"] != plan["weight_bytes"]:
        raise ValueError("Model weight hash or size differs from the checkpoint plan")
    target = a.out / a.name
    target.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(fetch, target, a.mirror, plan, r, a.range_workers) for r in files.values()]
        for future in pending:
            future.result()
    config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    if config["hidden_size"] != 2048:
        raise ValueError("The historical T1 vectors do not fit the official model width")
    lock = dict(**plan, files={r["rfilename"]: dict(bytes=r["size"], blob_id=r["blobId"],
                    sha256=r.get("lfs", {}).get("sha256")) for r in files.values()},
                hidden_size=config["hidden_size"], layers=config["num_hidden_layers"])
    temp = target / "MODEL_LOCK.json.partial"
    temp.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    temp.replace(target / "MODEL_LOCK.json")
    print("READY", a.name, plan["revision"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mirror", default="https://hf-mirror.com")
    p.add_argument("--range-workers", type=int, default=1)
    run(p.parse_args())
