"""Publish a packaged research directory using GitHub's Git database API."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest, sha1
import json
from pathlib import Path
import sys
import time

import requests


REPOSITORY = "Gifoe/looped-transformer-stage1-reproduction"
FEATURE = "experiments/ouro_loop_aware_functional_identification_20261125"
BRANCH = "research/ouro-loop-aware-functional-identification"
API = f"https://api.github.com/repos/{REPOSITORY}"


def request(secret, method, path, data=None):
    headers = {"Authorization": f"Bearer {secret}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    for sample in range(5):
        try:
            response = requests.request(method, API + path, headers=headers, json=data, timeout=(12, 45))
        except requests.RequestException:
            if sample == 4:
                raise
        else:
            if response.ok:
                return response.json()
            propagating = response.status_code == 422 and any(
                text in response.text for text in ("is not a valid blob", "is not a valid tree",
                                                    "Tree SHA does not exist", "Object does not exist")
            )
            if response.status_code not in {429, 500, 502, 503, 504} and not propagating:
                raise RuntimeError(f"GitHub rejected {method} {path} ({response.status_code})")
        time.sleep(2 ** sample)
    raise TimeoutError(f"GitHub request exhausted retries: {method} {path}")


def manifest(directory):
    file = directory / "RELEASE_MANIFEST.json"
    release = json.loads(file.read_text(encoding="utf-8"))
    if release["official_benchmarks"] != 12:
        raise ValueError("The requested publication requires all twelve official benchmarks")
    out = {}
    for item in release["files"]:
        path = (directory / item["path"]).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Unexpected artifact {item['path']}")
        with path.open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                raise ValueError(f"Changed artifact {item['path']}")
        if path.stat().st_size >= 96 * 2**20:
            raise ValueError(f"GitHub requires a smaller artifact shard: {item['path']}")
        out[item["path"]] = path
    out[file.name] = file
    return out


def upload(secret, name, path):
    contents = path.read_bytes()
    expected = sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest()
    result = request(secret, "POST", "/git/blobs", dict(encoding="base64",
                     content=base64.b64encode(contents).decode("ascii")))
    if result["sha"] != expected:
        raise ValueError(f"GitHub blob differs from packaged bytes {name}")
    return {"path": f"{FEATURE}/{name}", "type": "blob", "mode": "100755" if name.endswith(".sh") else "100644",
            "sha": expected}


def run(a, secret):
    directory = a.directory.resolve()
    files = manifest(directory)
    if a.verify_only:
        print(f"VERIFIED {len(files)} release blobs")
        return
    if not secret:
        raise ValueError("Pass GitHub authorization on standard input")
    ref = f"/git/ref/heads/{BRANCH}"
    prior = request(secret, "GET", ref)["object"]["sha"]
    if prior != a.expected_head:
        raise ValueError(f"Research branch moved from expected commit {a.expected_head} to {prior}")
    start = request(secret, "GET", f"/git/commits/{prior}")["tree"]["sha"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(upload, secret, name, path) for name, path in files.items()]
        rows = [future.result() for future in pending]
    layer = start
    for i in range(0, len(rows), 100):
        layer = request(secret, "POST", "/git/trees", dict(base_tree=layer, tree=rows[i:i+100]))["sha"]
    commit = request(secret, "POST", "/git/commits", dict(
        message="Publish complete Loop-Aware Functional Identification study and official benchmarks",
        tree=layer, parents=[prior]
    ))["sha"]
    head = request(secret, "GET", ref)["object"]["sha"]
    if head != prior:
        raise ValueError("Research branch changed during upload")
    request(secret, "PATCH", f"/git/refs/heads/{BRANCH}", dict(sha=commit, force=False))
    if request(secret, "GET", ref)["object"]["sha"] != commit:
        raise ValueError("Final GitHub commit was not visible on the research branch")
    print(f"PUBLISHED {commit} {len(rows)} artifacts", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", required=True, type=Path)
    p.add_argument("--expected-head")
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    if not a.verify_only and not a.expected_head:
        p.error("--expected-head is required when publishing")
    run(a, "" if a.verify_only else sys.stdin.readline().strip())
