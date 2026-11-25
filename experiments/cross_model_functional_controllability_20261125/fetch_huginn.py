"""Verify and download the officially released Huginn-0125 artifacts."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import time

import requests

from fetch_ouro import checksum, download_segment
from huginn_control import REPOSITORY, REVISION, sha256


EXPECTED = {
    "model-00001-of-00004.safetensors": "5b461584b74249e87eabec34b59a13e188791dd8a7ccf2c7be95bb7fa8383e43",
    "model-00002-of-00004.safetensors": "44bac332b204abb687b2820eb0fadcb5795fdb0cfc06b3b6f2ac5241eec5c591",
    "model-00003-of-00004.safetensors": "07160dce9da01918d0b0bf2159ffe11d007e1fbe661d687ad4a7b0e65bde4e84",
    "model-00004-of-00004.safetensors": "40ce4f159938857dbbfa1020962045e1b96303ff902cfa76051ce54d2d51dea4",
}
FILES = set(EXPECTED) | {
    "model.safetensors.index.json", "config.json", "generation_config.json", "raven_config_minimal.py",
    "raven_modeling_minimal.py", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
}


def fetch(mirror, root, item):
    name = item["rfilename"]
    target = root / name
    temporary = target.with_suffix(target.suffix + ".partial")
    weight = name in EXPECTED
    expected = EXPECTED[name] if weight else item["blobId"]

    def verified(path):
        return path.is_file() and path.stat().st_size == item["size"] and checksum(path, weight) == expected

    for attempt in range(6):
        if verified(target):
            return name, expected if weight else sha256(target)
        if verified(temporary):
            temporary.replace(target)
            print("HUGINN VERIFIED", name, target.stat().st_size, flush=True)
            return name, expected if weight else sha256(target)
        previous = temporary.stat().st_size if temporary.exists() else 0
        offset = previous if previous < item["size"] else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            url = f"{mirror}/{REPOSITORY}/resolve/{REVISION}/{name}"
            with requests.get(url, headers=headers, stream=True, timeout=(20, 100)) as response:
                response.raise_for_status()
                resume = offset != 0 and response.status_code == 206
                if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError(f"Wrong Huginn resume offset for {name}")
                with temporary.open("ab" if resume else "wb") as stream:
                    for block in response.iter_content(4 << 20):
                        stream.write(block)
        except requests.RequestException as error:
            print("HUGINN RETRY", name, offset, type(error).__name__, flush=True)
            time.sleep(3 * (attempt + 1))
    if not verified(temporary):
        raise ValueError(f"Huginn checkpoint artifact was not verified: {name}")
    temporary.replace(target)
    return name, expected if weight else sha256(target)


def ranged_weight(root, mirror, name, size, workers):
    target = root / name
    digest = EXPECTED[name]
    if target.is_file() and target.stat().st_size == size and sha256(target) == digest:
        return name, digest
    partial = target.with_suffix(target.suffix + ".partial")
    ledger = root / "RANGES" / name
    ledger.mkdir(parents=True, exist_ok=True)
    manifest = ledger / "LAYOUT.json"
    if manifest.is_file():
        plan = json.loads(manifest.read_text(encoding="utf-8"))
        if not partial.is_file() or plan["sha256"] != digest or plan["bytes"] != size:
            raise ValueError(f"Huginn byte-range receipts differ from the checkpoint for {name}")
        prefix = plan["prefix"]
    else:
        prefix = partial.stat().st_size if partial.is_file() else 0
        if prefix == size and sha256(partial) == digest:
            partial.replace(target)
            return name, digest
        prefix = prefix if prefix < size else 0
        manifest.write_text(json.dumps(dict(prefix=prefix, bytes=size, sha256=digest)) + "\n", encoding="utf-8")
    stride = 8 << 20
    url = f"{mirror}/{REPOSITORY}/resolve/{REVISION}/{name}"
    fd = os.open(partial, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.ftruncate(fd, size)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            work = [pool.submit(download_segment, url, fd, start, min(start+stride, size)-1, size,
                                ledger / f"{start}.ok") for start in range(prefix, size, stride)]
            for result in work:
                result.result()
    finally:
        os.close(fd)
    if sha256(partial) != digest:
        raise ValueError(f"The assembled Huginn shard SHA-256 differs for {name}")
    partial.replace(target)
    print("HUGINN VERIFIED", name, target.stat().st_size, flush=True)
    return name, digest


def run(a):
    if a.metadata is None:
        response = requests.get(f"{a.mirror}/api/models/{REPOSITORY}/revision/{REVISION}",
                                params={"blobs": "true"}, timeout=30)
        response.raise_for_status()
        info = response.json()
    else:
        info = json.loads(a.metadata.read_text(encoding="utf-8"))
    if info["sha"] != REVISION:
        raise ValueError("Huginn repository revision differs")
    jobs = {r["rfilename"]: r for r in info["siblings"] if r["rfilename"] in FILES}
    if jobs.keys() != FILES or any(jobs[name]["lfs"]["sha256"] != digest for name, digest in EXPECTED.items()):
        raise ValueError("Unexpected artifact listing for the Huginn official release")
    a.out.mkdir(parents=True, exist_ok=True)
    existing = sum(512 * path.stat().st_blocks for path in a.out.glob("*.safetensors*"))
    required = sum(jobs[name]["size"] for name in EXPECTED)
    if shutil.disk_usage(a.out).free + existing < required + (256 << 20):
        raise OSError("Insufficient space for the four verified official Huginn weights")
    assets = jobs.values() if a.range_workers == 1 else [x for key, x in jobs.items() if key not in EXPECTED]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        files = dict(pool.map(lambda item: fetch(a.mirror, a.out, item), assets))
    if a.range_workers > 1:
        for name in sorted(EXPECTED, key=lambda key: jobs[key]["size"]):
            key, value = ranged_weight(a.out, a.mirror, name, jobs[name]["size"], a.range_workers)
            files[key] = value
    lock = a.out / "MODEL_LOCK.json"
    temp = lock.with_suffix(".partial")
    temp.write_text(json.dumps({"revision": REVISION, "repository": REPOSITORY, "files": files},
                               indent=2) + "\n", encoding="utf-8")
    temp.replace(lock)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mirror", default="https://hf-mirror.com")
    p.add_argument("--metadata", type=Path)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--range-workers", type=int, default=1)
    run(p.parse_args())
