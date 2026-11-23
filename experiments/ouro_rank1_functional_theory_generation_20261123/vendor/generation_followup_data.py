"""Fetch the official full test splits and verify pinned file digests."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import pyarrow.parquet as pq
import requests

SOURCES = {
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main/test-00000-of-00001.parquet", 1319),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be", "data/test-00000-of-00001.parquet", 12032),
    "humaneval": ("evalplus/humanevalplus", "d32357cf319e50e9c8d8dab5ea876c72b0fd321b", "data/test-00000-of-00001-5973903632b82d40.parquet", 164),
    "mbpp": ("evalplus/mbppplus", "b2d74c91837c3f2a20c1299ae98133cbe7cfa077", "data/test-00000-of-00001-d5781c9c51e02795.parquet", 378),
}
HUBS = ("https://hf-mirror.com", "https://huggingface.co")


def fetch(path):
    for retry in range(3):
        for host in HUBS:
            try:
                response = requests.get(host + path, timeout=(9, 40))
                response.raise_for_status()
                return response.content
            except requests.RequestException as error:
                last = error
        time.sleep(2 + retry)
    raise last


def digest(data):
    return hashlib.sha256(data).hexdigest()


def obtain(root, item):
    name, (repo, revision, source, n) = item
    out = root / f"{name}.parquet"
    meta = json.loads(fetch(f"/api/datasets/{repo}/revision/{revision}?blobs=true"))
    assert meta["sha"] == revision
    pointer = next(x for x in meta["siblings"] if x["rfilename"] == source)
    expected = (pointer.get("lfs") or {}).get("sha256")
    data = out.read_bytes() if out.exists() else fetch(f"/datasets/{repo}/resolve/{revision}/{source}")
    if expected is not None:
        assert digest(data) == expected
    else:
        prefix = f"blob {len(data)}\0".encode()
        assert hashlib.sha1(prefix + data).hexdigest() == pointer["blobId"]
    out.write_bytes(data)
    receipt = pq.ParquetFile(out).metadata
    assert receipt.num_rows == n, (name, receipt.num_rows, n)
    record = dict(dataset=repo, revision=revision, source=source, split="test", rows=n,
                  sha256=digest(data), columns=receipt.schema.names)
    (root / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(name, n, record["columns"], flush=True)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda item: obtain(root, item), SOURCES.items()))
